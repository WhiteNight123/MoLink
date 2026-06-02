"""Shared utilities for cross-host benchmark scripts.

Provides common constants, process management, health-check helpers,
and GPU monitoring used by bench_distributed.py, bench_docker.py, etc.
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import threading
import time

import pynvml

# ── Remote machine config ──────────────────────────────────────────────────

LOCAL_IP = "10.130.151.13"
REMOTE_HOST = "10.130.151.15"
REMOTE_PORT = "15301"
REMOTE_USER = "gpu2"

SSH_CMD = [
    "ssh", "-o", "StrictHostKeyChecking=no",
    "-p", REMOTE_PORT, f"{REMOTE_USER}@{REMOTE_HOST}",
]

# Local paths
LOCAL_MODEL = "/home/emnets-2/gxq/Qwen3-14B"
LOCAL_MOLINK_DIR = "/home/emnets-2/gxq/molink-measurement/MoLink"
LOCAL_TOKENIZER = LOCAL_MODEL
LOCAL_BENCH_CLIENT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "benchmark_client.py"
)

# Remote paths
REMOTE_MODEL = "/home/gpu2/gxq/Qwen3-14B"
REMOTE_MOLINK_DIR = "/home/gpu2/gxq/molink-measurement/MoLink"
REMOTE_MOLINK_PYTHON = "/home/gpu2/miniconda3/envs/molink/bin/python"
REMOTE_VLLM_PYTHON = "/home/gpu2/miniconda3/envs/vllm19/bin/python"
REMOTE_VLLM_BIN = "/home/gpu2/miniconda3/envs/vllm19/bin"

# Ports
HEAD_PORT = 8080
VLLM_FWD_PORT = 8081
TAIL_PORT = 9095
MOLINK_GRPC_HEAD = 50061
MOLINK_GRPC_TAIL = 50062
VLLM_COMMON_MODEL = "/tmp/Qwen3-14B"
RAY_PORT = 6379

# Layer split (Qwen3-14B ~40 layers)
HEAD_END_LAYER = 21
TAIL_START_LAYER = 21

# Benchmark defaults
BENCH_DURATION = 30
RPS = 3.0
INPUT_TOKENS = 1024
OUTPUT_TOKENS = 512
MAX_MODEL_LEN = 4096

# ── Process tracking ──────────────────────────────────────────────────────

procs: list[subprocess.Popen] = []
ssh_tunnel_pid: int | None = None


# ── Logging ───────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Command execution ─────────────────────────────────────────────────────

def run_local(
    cmd: list[str],
    env_extra: dict | None = None,
    background: bool = False,
    gpu: str = "0",
    log_path: str | None = None,
) -> subprocess.Popen | subprocess.CompletedProcess:
    """Run a command locally with MoLink environment set up."""
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": gpu,
        "PYTHONPATH": LOCAL_MOLINK_DIR,
        "NO_PROXY": "*",
        "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "3600",
    }
    if env_extra:
        env.update(env_extra)
    if background:
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            f = open(log_path, "w")
            p = subprocess.Popen(
                cmd, env=env,
                stdout=f, stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
            )
        else:
            p = subprocess.Popen(
                cmd, env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid,
            )
        procs.append(p)
        return p
    return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=30)


def run_remote(cmd: str, background: bool = False,
               log_path: str | None = None) -> subprocess.CompletedProcess:
    """Run a command on the remote machine via SSH."""
    env_prefix = "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; "
    if background:
        if log_path:
            full_cmd = SSH_CMD + [
                f"nohup bash -c '{env_prefix}{cmd}' &>'{log_path}' &"
            ]
        else:
            full_cmd = SSH_CMD + [
                f"nohup bash -c '{env_prefix}{cmd}' &>/dev/null &"
            ]
        return subprocess.run(full_cmd, capture_output=True, text=True, timeout=15)
    return subprocess.run(
        SSH_CMD + [env_prefix + cmd],
        capture_output=True, text=True, timeout=60,
    )


# ── Cleanup ───────────────────────────────────────────────────────────────

def cleanup() -> None:
    """Kill all spawned processes, Ray cluster, SSH tunnels, and remote procs."""
    global ssh_tunnel_pid
    log("Cleaning up...")

    for p in procs:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            pass
    procs.clear()

    try:
        subprocess.run(["ray", "stop", "--force"],
                       capture_output=True, timeout=15)
    except Exception:
        pass
    try:
        subprocess.run(
            SSH_CMD + [f"{REMOTE_VLLM_BIN}/ray stop --force 2>/dev/null; true"],
            capture_output=True, timeout=30,
        )
    except Exception:
        pass

    if ssh_tunnel_pid:
        try:
            os.kill(ssh_tunnel_pid, signal.SIGKILL)
        except Exception:
            pass
        ssh_tunnel_pid = None

    # Clear instrument event logs on both machines so they don't
    # accumulate across repeated runs (append-only files).
    subprocess.run(["rm", "-f", "/tmp/molink_worker_events.log"], capture_output=True)
    subprocess.run(
        SSH_CMD + ["rm -f /tmp/molink_worker_events.log"],
        capture_output=True, timeout=10,
    )

    for port in [HEAD_PORT, VLLM_FWD_PORT, TAIL_PORT, MOLINK_GRPC_HEAD, MOLINK_GRPC_TAIL]:
        try:
            result = subprocess.run(
                ["ss", "-tlnp", f"sport = :{port}"],
                capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.strip().split("\n"):
                for part in line.split(","):
                    if "pid=" in part:
                        pid_s = part.split("pid=")[-1].split(",")[0].split(")")[0]
                        if pid_s.isdigit():
                            try:
                                os.kill(int(pid_s), signal.SIGKILL)
                            except Exception:
                                pass
        except Exception:
            pass

    try:
        subprocess.run(
            SSH_CMD + [
                "pkill -9 -e -f 'vllm|VLLM|molinkv1' 2>/dev/null; "
                "sleep 2; true"
            ],
            capture_output=True, timeout=20,
        )
    except Exception:
        pass

    log("Cleanup done. Waiting for GPU memory release...")
    time.sleep(10)


# ── Health checks ─────────────────────────────────────────────────────────

async def health_check(url: str, timeout: int = 300) -> bool:
    """Wait until an HTTP endpoint returns 200 on /health."""
    import aiohttp
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(
                    f"{url}/health",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as r:
                    if r.status == 200:
                        return True
        except Exception:
            pass
        await asyncio.sleep(5)
    return False


async def health_check_remote(remote_port: int, timeout: int = 300) -> bool:
    """Check health on remote machine via SSH."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = run_remote(f"curl -sf http://localhost:{remote_port}/health")
        if r.returncode == 0:
            return True
        await asyncio.sleep(5)
    return False


# ── GPU Monitoring ────────────────────────────────────────────────────────────

class GPUMonitor:
    """Background GPU utilization sampler for multiple local and remote GPUs."""

    def __init__(self, interval_s=0.1, gpu_indices=None, remote_ssh=None):
        self.interval_s = interval_s
        self.gpu_indices = gpu_indices or [0]
        self.remote_ssh = remote_ssh
        self._stop = threading.Event()
        self._thread = None
        self._data: list[dict] = []
        self._lock = threading.Lock()

        pynvml.nvmlInit()
        self._local_handles = {i: pynvml.nvmlDeviceGetHandleByIndex(i)
                               for i in self.gpu_indices}

    def _sample_local(self) -> dict:
        result = {}
        for idx, handle in self._local_handles.items():
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            result[f"local_{idx}"] = {
                "gpu_util_pct": util.gpu,
                "mem_used_mb": mem.used / (1024 * 1024),
                "mem_total_mb": mem.total / (1024 * 1024),
            }
        return result

    def _sample_remote(self) -> dict:
        if not self.remote_ssh:
            return {}
        try:
            r = subprocess.run(
                self.remote_ssh + [
                    "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total "
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode != 0:
                return {}
            result = {}
            for i, line in enumerate(r.stdout.strip().split("\n")):
                parts = [x.strip() for x in line.split(",")]
                if len(parts) < 3:
                    continue
                result[f"remote_{i}"] = {
                    "gpu_util_pct": float(parts[0]),
                    "mem_used_mb": float(parts[1]),
                    "mem_total_mb": float(parts[2]),
                }
            return result
        except Exception:
            return {}

    def _run(self):
        while not self._stop.is_set():
            ts = time.time()
            record = self._sample_local()
            record.update(self._sample_remote())
            for v in record.values():
                v["ts"] = ts
            with self._lock:
                self._data.append(record)
            self._stop.wait(self.interval_s)

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> list[dict]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        with self._lock:
            return list(self._data)

    def save(self, path):
        data = self.stop()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return data


def gpu_stats(records, node):
    """Compute summary statistics from GPU monitor records."""
    utils = []
    mems = []
    for r in records:
        d = r.get(node)
        if d and d.get("gpu_util_pct", 0) > 0:
            utils.append(d["gpu_util_pct"])
            mems.append(d.get("mem_used_mb", 0))

    if not utils:
        return {"count": 0}

    s = sorted(utils)
    n = len(s)
    sm = sorted(mems)
    ts_list = []
    for r in records:
        for v in r.values():
            if isinstance(v, dict) and v.get("ts"):
                ts_list.append(v["ts"])
                break
    return {
        "count": n,
        "duration_s": ts_list[-1] - ts_list[0] if len(ts_list) > 1 else 0,
        "gpu_util_avg": sum(utils) / n,
        "gpu_util_p50": s[n // 2],
        "gpu_util_p90": s[int(n * 0.9)],
        "gpu_util_p99": s[min(int(n * 0.99), n - 1)],
        "gpu_util_max": s[-1],
        "gpu_util_min": s[0],
        "mem_avg_mb": sum(mems) / n,
        "mem_max_mb": sm[-1] if sm else 0,
    }


def plot_gpu_chart(records, output_path, title="GPU Utilization"):
    """Plot all GPUs on a single time-series chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gpu_keys = set()
    for r in records:
        gpu_keys.update(r.keys())
    gpu_keys = sorted(gpu_keys)

    if not gpu_keys:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.tab10.colors

    for i, key in enumerate(gpu_keys):
        samples = []
        for r in records:
            d = r.get(key)
            if d and d.get("gpu_util_pct", 0) > 0:
                samples.append(d)
        if not samples:
            continue

        t0 = samples[0]["ts"]
        times = [s["ts"] - t0 for s in samples]
        utils = [s["gpu_util_pct"] for s in samples]

        color = colors[i % len(colors)]
        ax.plot(times, utils, linewidth=0.8, color=color, label=key, alpha=0.85)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("GPU Utilization (%)")
    ax.set_title(title)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, loc="upper right")
    ax.yaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)
    ax.set_axisbelow(True)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    base = output_path.replace(".pdf", "").replace(".png", "")
    for fmt in [".pdf", ".png"]:
        fig.savefig(base + fmt, bbox_inches="tight", dpi=150)
    plt.close(fig)
