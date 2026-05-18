"""Shared utilities for cross-host benchmark scripts.

Provides common constants, process management, and health-check helpers
used by bench_distributed.py, bench_profile.py, and bench_vllm_profile.py.
"""

import asyncio
import os
import signal
import subprocess
import sys
import time

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
        p = subprocess.Popen(
            cmd, env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )
        procs.append(p)
        return p
    return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=30)


def run_remote(cmd: str, background: bool = False) -> subprocess.CompletedProcess:
    """Run a command on the remote machine via SSH."""
    env_prefix = "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; "
    if background:
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
