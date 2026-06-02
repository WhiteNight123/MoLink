#!/usr/bin/env python3
"""Batch-level instrumentation benchmark for MoLink and vLLM PP.

Usage:
    python run_batch_instrument.py molink     # MoLink only (Docker mode)
    python run_batch_instrument.py vllm       # vLLM only (Docker mode)
    python run_batch_instrument.py            # Both (Docker mode)
    python run_batch_instrument.py --distributed molink  # Distributed mode
"""

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────

DOCKER_IMAGE = "molink:0.2"
DOCKER_NETWORK = "molink"
SUBNET = "172.26.0.0/16"

CONTAINER_DEFS = [
    {"role": "head", "name": "bench-head", "ip": "172.26.0.10"},
    {"role": "tail", "name": "bench-tail", "ip": "172.26.0.11"},
]

MODEL_PATH = "/gxq/Qwen3-14B"
HOST_TOKENIZER = "/home/emnets-2/gxq/Qwen3-14B"
MOLINK_CODE = "/gxq/molink-measurement/MoLink"
VLLM_SOURCE = "/gxq/molink-measurement/vllm/vllm"
VLLM_SITE = "/opt/conda/envs/vllm19/lib/python3.12/site-packages/vllm"
VOLUMES = ["/mnt/disk1-16/gxq:/data", "/home/emnets-2/gxq:/gxq"]

VLLM19_PYTHON = "/opt/conda/envs/vllm19/bin/python"
VLLM19_BIN = "/opt/conda/envs/vllm19/bin"

HEAD_PORT = 8080
TAIL_PORT = 9095
GRPC_PORTS = {"head": 50061, "tail": 50062}
RAY_PORT = 6379

PORT_MAP = {"head": HEAD_PORT, "tail": TAIL_PORT}

INPUT_TOKENS = 1024
OUTPUT_TOKENS = 32
BENCH_RPS = 3
BENCH_DURATION = 1
MAX_MODEL_LEN = 4096
MOLINK_MAX_CONCURRENT_BATCHES = 3
HEALTH_TIMEOUT = 300
COOLDOWN = 5

BENCH_DIR = Path(__file__).parent.resolve()
BENCHMARK_CLIENT = BENCH_DIR / "benchmark_client.py"
RESULTS_ROOT = BENCH_DIR / "results_batch"

# Distributed mode imports (lazy — only needed with --distributed)
try:
    sys.path.insert(0, str(BENCH_DIR.parent))
    import bench_utils as _bu
except ImportError:
    _bu = None

LOCAL_VLLM_SRC = str(Path(__file__).resolve().parent.parent.parent / "vllm" / "vllm")
REMOTE_VLLM_SITE = "/home/gpu2/miniconda3/envs/vllm19/lib/python3.12/site-packages/vllm"
_VLLM_INSTRUMENTED = [
    "v1/executor/ray_utils.py",
    "v1/executor/ray_executor.py",
    "v1/worker/gpu_worker.py",
    "v1/engine/core.py",
    "entrypoints/openai/completion/serving.py",
]


# ── Helpers ────────────────────────────────────────────────────────────────

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def die(msg):
    log(f"ERROR: {msg}")
    sys.exit(1)


def _dexec(name, cmd, detach=False, env=None):
    args = ["docker", "exec"]
    if detach:
        args.append("-d")
    if env:
        for k, v in env.items():
            args += ["-e", f"{k}={v}"]
    args += [name, "bash", "-c", cmd]
    return subprocess.run(args, capture_output=True, text=True)


def ensure_network():
    r = subprocess.run(["docker", "network", "inspect", DOCKER_NETWORK],
                       capture_output=True, text=True)
    if r.returncode != 0:
        log(f"Creating Docker network: {DOCKER_NETWORK}")
        subprocess.run(
            ["docker", "network", "create", "--subnet", SUBNET, DOCKER_NETWORK],
            check=True,
        )


def start_container(name, ip, gpu_devs, host_port, container_port):
    gpu_list = ",".join(str(g) for g in gpu_devs)
    cmd = ["docker", "run", "-d", "--name", name]
    for vol in VOLUMES:
        cmd += ["-v", vol]
    cmd += [
        "--gpus", f"device={gpu_list}",
        "--shm-size=64g", "--cap-add=NET_ADMIN",
        "--network", DOCKER_NETWORK, "--ip", ip,
        "-p", f"{host_port}:{container_port}",
        DOCKER_IMAGE, "sleep", "infinity",
    ]
    subprocess.run(cmd, check=True)


def cleanup_containers():
    log("Cleaning up containers...")
    for c in CONTAINER_DEFS:
        subprocess.run(["docker", "rm", "-f", c["name"]],
                       capture_output=True, text=True)


# ── Network shaping ───────────────────────────────────────────────────────

def setup_tc(name, bw, delay, peer_ips):
    _dexec(name, "tc qdisc del dev eth0 root 2>/dev/null || true")
    _dexec(name, "tc qdisc add dev eth0 root handle 1: htb default 30")
    _dexec(name, f"tc class add dev eth0 parent 1: classid 1:1 htb rate {bw}")
    _dexec(name, f"tc qdisc add dev eth0 parent 1:1 netem delay {delay}")
    for i, ip in enumerate(peer_ips, 1):
        _dexec(name,
               f"tc filter add dev eth0 parent 1: protocol ip prio {i} "
               f"u32 match ip dst {ip} flowid 1:1")


def setup_network_shaping(nodes, bw="1gbit", delay="5ms"):
    log(f"Setting up network shaping: {bw} / {delay}")
    for node in nodes:
        others = [n["ip"] for n in nodes if n != node]
        setup_tc(node["name"], bw, delay, others)


# ── Service management ────────────────────────────────────────────────────

def stop_services(names):
    log("Stopping services...")
    for n in names:
        _dexec(n,
               "pkill -f 'python.*molinkv1' 2>/dev/null || true; "
               "pkill -f 'python.*vllm' 2>/dev/null || true; "
               "pkill -f 'ray::' 2>/dev/null || true; "
               "pkill -f 'raylet' 2>/dev/null || true; "
               "pkill -f 'VLLM' 2>/dev/null || true; "
               "pkill -f 'molink' 2>/dev/null || true")
    time.sleep(5)


def wait_health(url, timeout=HEALTH_TIMEOUT, container_name=None, log_file=None):
    log(f"Waiting for {url} ...")
    elapsed = 0
    while elapsed < timeout:
        r = subprocess.run(
            ["curl", "-sf", "--noproxy", "localhost", url],
            capture_output=True, text=True,
        )
        if r.returncode == 0:
            log("Healthy!")
            return True
        time.sleep(5)
        elapsed += 5
        if elapsed % 30 == 0:
            log(f"  ... {elapsed}s/{timeout}s")
    if container_name and log_file:
        log(f"=== {container_name} logs ===")
        _dexec(container_name, f"tail -20 {log_file}")
    return False


# ── MoLink ─────────────────────────────────────────────────────────────────

def start_molink(head_node, tail_node, max_ve):
    log("Starting MoLink head...")
    env = {
        "PYTHONPATH": MOLINK_CODE,
        "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "3600",
        "MOLINK_ENABLE_METRICS": "1",
    }
    head_cmd = (
        f"{VLLM19_PYTHON} -m molinkv1.entrypoints.api_server "
        f"--model {MODEL_PATH} --max-model-len {MAX_MODEL_LEN} "
        f"--enforce-eager --no-enable-prefix-caching "
        f"--molink-grpc-port {GRPC_PORTS['head']} "
        f"--molink-start-layer 0 --molink-end-layer 21 "
        f"--molink-enable-metrics --molink-max-concurrent-batches {max_ve} "
        f"--port {HEAD_PORT}"
    )
    _dexec(head_node["name"], head_cmd + " &>/tmp/bench_head.log", detach=True, env=env)
    if not wait_health(f"http://localhost:{HEAD_PORT}/health",
                       container_name=head_node["name"],
                       log_file="/tmp/bench_head.log"):
        die("MoLink head failed to start")
    log("Head ready. Waiting 20s for gRPC...")
    time.sleep(20)

    log("Starting MoLink tail...")
    tail_cmd = (
        f"{VLLM19_PYTHON} -m molinkv1.entrypoints.api_server "
        f"--model {MODEL_PATH} --max-model-len {MAX_MODEL_LEN} "
        f"--enforce-eager --no-enable-prefix-caching "
        f"--molink-grpc-port {GRPC_PORTS['tail']} "
        f"--molink-start-layer 21 --molink-end-layer -1 "
        f"--molink-enable-metrics --molink-max-concurrent-batches {max_ve} "
        f"--molink-initial-peer {head_node['ip']}:{GRPC_PORTS['head']} "
        f"--port {TAIL_PORT}"
    )
    _dexec(tail_node["name"], tail_cmd + " &>/tmp/bench_tail.log", detach=True, env=env)
    if not wait_health(f"http://localhost:{TAIL_PORT}/health",
                       container_name=tail_node["name"],
                       log_file="/tmp/bench_tail.log"):
        die("MoLink tail failed to start")
    log("Tail ready!")
    time.sleep(5)


# ── vLLM ───────────────────────────────────────────────────────────────────

def start_vllm(nodes):
    """Start vLLM with PP=2 using NCCL over the network (fair mode)."""
    # NCCL fairness: disable GPU Direct P2P so NCCL uses TCP via eth0
    fair_env = (
        f"NCCL_P2P_DISABLE=1 "
        f"NCCL_SOCKET_IFNAME=eth0 "
        f"NCCL_IB_DISABLE=1 "
        f"VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE=shm "
        f"VLLM_DISABLE_PYNCCL=1 "
        f"CUDA_VISIBLE_DEVICES=0 "
    )

    # Ray head on head node
    log(f"Starting Ray head in {nodes[0]['name']}...")
    _dexec(nodes[0]["name"],
           f"{fair_env}"
           f"{VLLM19_BIN}/ray start --head "
           f"--node-ip-address={nodes[0]['ip']} "
           f"--port={RAY_PORT} --num-gpus=1",
           detach=True)
    time.sleep(8)

    # Ray worker on tail node
    log(f"Starting Ray worker in {nodes[1]['name']}...")
    _dexec(nodes[1]["name"],
           f"{fair_env}"
           f"{VLLM19_BIN}/ray start "
           f"--address={nodes[0]['ip']}:{RAY_PORT} --num-gpus=1",
           detach=True)
    time.sleep(8)

    # Copy instrumented vLLM to site-packages in both containers
    log("Copying instrumented vLLM code to containers...")
    for node in nodes:
        r = _dexec(node["name"],
                   f"cp {VLLM_SOURCE}/v1/executor/ray_utils.py {VLLM_SITE}/v1/executor/ray_utils.py && "
                   f"rm -f {VLLM_SITE}/v1/executor/__pycache__/ray_utils.cpython*.pyc && "
                   f"cp {VLLM_SOURCE}/v1/executor/ray_executor.py {VLLM_SITE}/v1/executor/ray_executor.py && "
                   f"rm -f {VLLM_SITE}/v1/executor/__pycache__/ray_executor.cpython*.pyc && "
                   f"cp {VLLM_SOURCE}/v1/worker/gpu_worker.py {VLLM_SITE}/v1/worker/gpu_worker.py && "
                   f"rm -f {VLLM_SITE}/v1/worker/__pycache__/gpu_worker.cpython*.pyc && "
                   f"cp {VLLM_SOURCE}/v1/engine/core.py {VLLM_SITE}/v1/engine/core.py && "
                   f"rm -f {VLLM_SITE}/v1/engine/__pycache__/core.cpython*.pyc && "
                   f"cp {VLLM_SOURCE}/entrypoints/openai/completion/serving.py {VLLM_SITE}/entrypoints/openai/completion/serving.py && "
                   f"rm -f {VLLM_SITE}/entrypoints/openai/completion/__pycache__/serving.cpython*.pyc")
        if r.returncode != 0:
            log(f"WARNING: copy may have failed on {node['name']}: {r.stderr}")
        # Clear any previous worker event file
        _dexec(node["name"], "rm -f /tmp/vllm_worker_events.log")

    # vLLM serve
    log(f"Starting vLLM serve (PP=2)...")
    _dexec(nodes[0]["name"],
           f"{fair_env}"
           f"{VLLM19_BIN}/vllm serve "
           f"--model {MODEL_PATH} --port {HEAD_PORT} "
           f"--max-model-len {MAX_MODEL_LEN} "
           f"--pipeline-parallel-size 2 --tensor-parallel-size 1 "
           f"--distributed-executor-backend ray "
           f"--no-enable-prefix-caching --enforce-eager "
           f"&>/tmp/bench_vllm.log",
           detach=True)
    if not wait_health(f"http://localhost:{HEAD_PORT}/health",
                       container_name=nodes[0]["name"],
                       log_file="/tmp/bench_vllm.log"):
        die("vLLM failed to start")
    log("vLLM ready!")
    time.sleep(5)


# ── Log collection ─────────────────────────────────────────────────────────

def collect_logs(system, nodes, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if system == "molink":
        r = _dexec(nodes[0]["name"], "cat /tmp/molink_worker_events.log 2>/dev/null || true")
        if r.stdout.strip():
            (outdir / "head.log").write_text(r.stdout)
        r = _dexec(nodes[1]["name"], "cat /tmp/molink_worker_events.log 2>/dev/null || true")
        if r.stdout.strip():
            (outdir / "tail.log").write_text(r.stdout)
        # Also collect main serve log for request events
        r = _dexec(nodes[0]["name"], "cat /tmp/bench_head.log 2>/dev/null || true")
        if r.stdout.strip():
            (outdir / "molink.log").write_text(r.stdout)
    elif system == "vllm":
        # Collect worker events from both containers (file-based logging)
        r = _dexec(nodes[0]["name"], "cat /tmp/vllm_worker_events.log 2>/dev/null || true")
        if r.stdout.strip():
            (outdir / "head.log").write_text(r.stdout)
        r = _dexec(nodes[1]["name"], "cat /tmp/vllm_worker_events.log 2>/dev/null || true")
        if r.stdout.strip():
            (outdir / "tail.log").write_text(r.stdout)
        # Also collect main serve log for request events
        r = _dexec(nodes[0]["name"], "cat /tmp/bench_vllm.log 2>/dev/null || true")
        if r.stdout.strip():
            (outdir / "vllm.log").write_text(r.stdout)


# ── Benchmark runner ────────────────────────────────────────────────────────

def run_benchmark(system, url, outdir, model=None, tokenizer=None, rps=BENCH_RPS, duration=BENCH_DURATION):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    log(f"--- [{system}] running benchmark ---")
    cmd = [
        sys.executable, str(BENCHMARK_CLIENT),
        "--url", url,
        "--type", "molink" if system == "molink" else "vllm",
        "--input-tokens", str(INPUT_TOKENS),
        "--output-tokens", str(OUTPUT_TOKENS),
        "--rps", str(rps),
        "--duration", str(duration),
        "--model", model or MODEL_PATH,
        "--tokenizer", tokenizer or HOST_TOKENIZER,
        "--output", str(outdir / "result.json"),
    ]
    subprocess.run(cmd, check=True)


# ── Plotting ───────────────────────────────────────────────────────────────

def generate_plot(system, log_dir, outdir):
    """Generate batch-level timeline plot from collected logs."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    plot_script = BENCH_DIR / "plot_batch_comparison.py"
    if not plot_script.exists():
        log(f"WARNING: plot_batch_comparison.py not found, skipping plot")
        return

    log("Generating plot...")
    cmd = [sys.executable, str(plot_script),
           "--system", system,
           "--logdir", str(log_dir),
           "--output", str(outdir)]
    subprocess.run(cmd, check=False)


# ── Distributed mode ────────────────────────────────────────────────────────

def _copy_vllm_instrumented():
    """Copy instrumented vLLM files to local and remote installations."""
    local_site = None
    try:
        import vllm
        local_site = os.path.dirname(vllm.__file__)
    except ImportError:
        pass

    for rel in _VLLM_INSTRUMENTED:
        src = os.path.join(LOCAL_VLLM_SRC, rel)
        if not os.path.exists(src):
            continue
        if local_site:
            dst = os.path.join(local_site, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
        subprocess.run(
            ["scp", "-o", "StrictHostKeyChecking=no",
             "-P", _bu.REMOTE_PORT,
             src, f"{_bu.REMOTE_USER}@{_bu.REMOTE_HOST}:{REMOTE_VLLM_SITE}/{rel}"],
            capture_output=True, text=True, timeout=30,
        )

    # Clear caches
    if local_site:
        for rel in _VLLM_INSTRUMENTED:
            cache_dir = os.path.join(os.path.dirname(os.path.join(local_site, rel)),
                                     "__pycache__")
            if os.path.isdir(cache_dir):
                base = os.path.splitext(os.path.basename(rel))[0]
                for f in os.listdir(cache_dir):
                    if f.startswith(base) and f.endswith(".pyc"):
                        os.remove(os.path.join(cache_dir, f))
    subprocess.run(
        _bu.SSH_CMD + [
            f"find {REMOTE_VLLM_SITE} -name '__pycache__' "
            f"-exec rm -rf {{}} + 2>/dev/null; true"
        ],
        capture_output=True, text=True, timeout=30,
    )
    # Clear worker event files
    subprocess.run(["rm", "-f", "/tmp/vllm_worker_events.log"], capture_output=True)
    subprocess.run(_bu.SSH_CMD + ["rm -f /tmp/vllm_worker_events.log"],
                   capture_output=True, text=True, timeout=10)


def wait_health_remote(port, timeout=HEALTH_TIMEOUT):
    """Wait for health endpoint on remote machine via SSH."""
    log(f"Waiting for remote:{port}/health ...")
    elapsed = 0
    while elapsed < timeout:
        r = subprocess.run(
            _bu.SSH_CMD + [
                "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; "
                f"curl -sf http://localhost:{port}/health"
            ],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            log("Remote healthy!")
            return True
        time.sleep(5)
        elapsed += 5
        if elapsed % 30 == 0:
            log(f"  ... {elapsed}s/{timeout}s")
    return False


def start_molink_distributed(gpu, remote_gpu, log_dir, max_ve):
    """Start MoLink head locally + tail remotely."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log("Starting MoLink head (local)...")
    _bu.run_local(
        [sys.executable, "-m", "molinkv1.entrypoints.api_server",
         "--model", _bu.LOCAL_MODEL,
         "--max-model-len", str(MAX_MODEL_LEN),
         "--enforce-eager", "--no-enable-prefix-caching",
         "--molink-grpc-port", str(GRPC_PORTS["head"]),
         "--molink-start-layer", "0",
         "--molink-end-layer", str(_bu.HEAD_END_LAYER),
         "--molink-max-concurrent-batches", str(max_ve),
         "--port", str(HEAD_PORT)],
        background=True, gpu=gpu,
        log_path=str(log_dir / "server.log"),
    )
    if not wait_health(f"http://localhost:{HEAD_PORT}/health"):
        die("MoLink head failed to start")
    log("Head ready. Waiting for gRPC...")
    time.sleep(20)

    remote_log = f"/tmp/molink_tail_{time.strftime('%H%M%S')}.log"
    log("Starting MoLink tail (remote)...")
    _bu.run_remote(
        f"CUDA_VISIBLE_DEVICES={remote_gpu} "
        f"PYTHONPATH={_bu.REMOTE_MOLINK_DIR} "
        f"VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600 "
        f"NO_PROXY='*' "
        f"{_bu.REMOTE_MOLINK_PYTHON} -m molinkv1.entrypoints.api_server "
        f"--model {_bu.REMOTE_MODEL} "
        f"--max-model-len {MAX_MODEL_LEN} "
        f"--enforce-eager --no-enable-prefix-caching "
        f"--molink-grpc-port {GRPC_PORTS['tail']} "
        f"--molink-start-layer {_bu.TAIL_START_LAYER} --molink-end-layer -1 "
        f"--molink-max-concurrent-batches {max_ve} "
        f"--port {TAIL_PORT} "
        f"--molink-initial-peer {_bu.LOCAL_IP}:{GRPC_PORTS['head']}",
        background=True,
        log_path=remote_log,
    )
    if not wait_health_remote(TAIL_PORT):
        die("MoLink tail failed to start on remote")
    log("Tail ready!")
    time.sleep(5)


def start_vllm_distributed(gpu, remote_gpu, log_dir):
    """Start vLLM PP=2 with Ray across local + remote machines."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log("Starting Ray head locally...")
    r = _bu.run_local(
        ["ray", "start", "--head", "--port", str(RAY_PORT),
         "--num-gpus=1", "--dashboard-host", "0.0.0.0"],
        env_extra={
            "GLOO_SOCKET_IFNAME": "eno1",
            "NCCL_SOCKET_IFNAME": "eno1",
            "NCCL_SHM_DISABLE": "1",
            "NCCL_P2P_DISABLE": "1",
        },
        gpu=gpu,
    )
    if r.returncode != 0:
        die(f"Ray head failed: {r.stderr[:300]}")
    time.sleep(8)

    log("Starting Ray worker on remote...")
    r = subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no",
         "-p", _bu.REMOTE_PORT, f"{_bu.REMOTE_USER}@{_bu.REMOTE_HOST}",
         f"CUDA_VISIBLE_DEVICES={remote_gpu} "
         "RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL=minor "
         "GLOO_SOCKET_IFNAME=enx6c1ff766c0ef "
         "NCCL_SOCKET_IFNAME=enx6c1ff766c0ef "
         "NCCL_SHM_DISABLE=1 "
         "NCCL_P2P_DISABLE=1 "
         f"{_bu.REMOTE_VLLM_BIN}/ray "
         f"start --address={_bu.LOCAL_IP}:{RAY_PORT} --num-gpus=1"],
        capture_output=True, text=True, timeout=60,
    )
    if r.returncode != 0:
        log(f"WARNING: Remote Ray worker issues: {r.stderr[:300]}")
    time.sleep(8)

    log("Copying instrumented vLLM code...")
    _copy_vllm_instrumented()

    log("Starting vLLM PP=2 via Ray...")
    _bu.run_local(
        [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
         "--model", _bu.VLLM_COMMON_MODEL,
         "--host", "0.0.0.0",
         "--port", str(HEAD_PORT),
         "--max-model-len", str(MAX_MODEL_LEN),
         "--pipeline-parallel-size", "2",
         "--tensor-parallel-size", "1",
         "--distributed-executor-backend", "ray",
         "--enforce-eager",
         "--no-enable-prefix-caching"],
        env_extra={
            "MASTER_ADDR": _bu.LOCAL_IP,
            "GLOO_SOCKET_IFNAME": "eno1",
            "NCCL_SHM_DISABLE": "1",
            "NCCL_P2P_DISABLE": "1",
        },
        background=True, gpu=gpu,
        log_path=str(log_dir / "server.log"),
    )
    if not wait_health(f"http://localhost:{HEAD_PORT}/health"):
        die("vLLM failed to start")
    log("vLLM ready!")
    time.sleep(5)


def collect_logs_distributed(system, log_dir):
    """Collect logs from local/remote for distributed mode."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    if system == "molink":
        # Collect instrument events from /tmp/molink_worker_events.log
        local_events = Path("/tmp/molink_worker_events.log")
        if local_events.exists():
            shutil.copy2(str(local_events), str(log_dir / "head.log"))
            log(f"Local head events copied to {log_dir / 'head.log'}")
        else:
            log("WARNING: /tmp/molink_worker_events.log not found locally")

        r = subprocess.run(
            ["scp", "-o", "StrictHostKeyChecking=no",
             "-P", _bu.REMOTE_PORT,
             f"{_bu.REMOTE_USER}@{_bu.REMOTE_HOST}:/tmp/molink_worker_events.log",
             str(log_dir / "tail.log")],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode == 0:
            log(f"Remote tail events copied to {log_dir / 'tail.log'}")
        else:
            log(f"Failed to copy remote events: {r.stderr[:200]}")

    elif system == "vllm":
        local_events = Path("/tmp/vllm_worker_events.log")
        if local_events.exists():
            shutil.copy2(str(local_events), str(log_dir / "head.log"))
        subprocess.run(
            ["scp", "-o", "StrictHostKeyChecking=no",
             "-P", _bu.REMOTE_PORT,
             f"{_bu.REMOTE_USER}@{_bu.REMOTE_HOST}:/tmp/vllm_worker_events.log",
             str(log_dir / "tail.log")],
            capture_output=True, text=True, timeout=30,
        )
        server_log = log_dir / "server.log"
        if server_log.exists():
            shutil.copy2(str(server_log), str(log_dir / "vllm.log"))


# ── Main ───────────────────────────────────────────────────────────────────

def parse_args():
    import argparse
    ap = argparse.ArgumentParser(description="Batch-level instrumentation benchmark")
    ap.add_argument("systems", nargs="*",
                    help="Systems to benchmark: molink, vllm (default: both)")
    ap.add_argument("--distributed", action="store_true",
                    help="Use distributed mode (local head + remote tail via SSH)")
    ap.add_argument("--gpus", default=None, help="GPU devices, e.g. 0,1 (Docker mode)")
    ap.add_argument("--gpu", default="0", help="Local GPU device (distributed mode)")
    ap.add_argument("--remote-gpu", default="0", help="Remote GPU device (distributed mode)")
    ap.add_argument("--network", default="1gbit,10ms",
                    help="Network condition: bandwidth,latency or 'none' (Docker mode)")
    ap.add_argument("--rps", type=int, default=BENCH_RPS,
                    help=f"Requests per second (default: {BENCH_RPS})")
    ap.add_argument("--duration", type=int, default=BENCH_DURATION,
                    help=f"Benchmark duration in seconds (default: {BENCH_DURATION})")
    ap.add_argument("--molink-max-concurrent-batches", type=int,
                    default=MOLINK_MAX_CONCURRENT_BATCHES,
                    help=f"MoLink max concurrent batches (default: {MOLINK_MAX_CONCURRENT_BATCHES})")
    ap.add_argument("--output", default=None, help="Results directory")
    return ap.parse_args()


def main_docker(args, systems):
    """Docker mode: run benchmark in Docker containers with network shaping."""
    # GPU allocation
    if args.gpus:
        gpu_devs = [int(x) for x in args.gpus.split(",")]
    else:
        gpu_devs = [0, 1]
    if len(gpu_devs) < 2:
        die(f"Need 2 GPUs, got {len(gpu_devs)}")

    head_gpus = [gpu_devs[0]]
    tail_gpus = [gpu_devs[1]]

    nodes = [
        {**CONTAINER_DEFS[0], "gpus": head_gpus},
        {**CONTAINER_DEFS[1], "gpus": tail_gpus},
    ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(args.output) if args.output else RESULTS_ROOT / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 50)
    log(f" Batch-Level Instrumentation Benchmark")
    log(f" GPUs: head={head_gpus}, tail={tail_gpus}")
    log(f" Systems: {', '.join(systems)}")
    log(f" Network: {args.network}")
    log(f" Results: {results_dir}")
    log("=" * 50)

    # Save config
    config = {
        "timestamp": timestamp,
        "systems": systems,
        "gpus": gpu_devs,
        "input_tokens": INPUT_TOKENS,
        "output_tokens": OUTPUT_TOKENS,
        "rps": 2,
        "duration_s": 10,
        "network": args.network,
    }
    (results_dir / "config.json").write_text(json.dumps(config, indent=2))

    # Setup Docker
    cleanup_containers()
    ensure_network()

    log("Starting 2 containers (head + tail)...")
    for node in nodes:
        port = PORT_MAP[node["role"]]
        start_container(node["name"], node["ip"], node["gpus"], port, port)
    log("Containers started.")
    time.sleep(5)

    container_names = [n["name"] for n in nodes]

    try:
        # Setup network shaping
        if args.network != "none":
            bw, delay = args.network.split(",")
            setup_network_shaping(nodes, bw, delay)

        for system in systems:
            stop_services(container_names)

            # Copy instrumented MoLink code to containers
            if system == "molink":
                log("Copying instrumented MoLink code...")
                for node in nodes:
                    _dexec(node["name"],
                           f"find {MOLINK_CODE} -name '__pycache__' -type d "
                           f"-exec rm -rf {{}} + 2>/dev/null || true")
                    _dexec(node["name"], "rm -f /tmp/molink_worker_events.log")

            if system == "molink":
                start_molink(nodes[0], nodes[1], args.molink_max_concurrent_batches)
                url = f"http://localhost:{HEAD_PORT}/generate"
            elif system == "vllm":
                start_vllm(nodes)
                url = f"http://localhost:{HEAD_PORT}/v1/completions"
            else:
                die(f"Unknown system: {system}")

            # Run benchmark
            system_dir = results_dir / system
            log_dir = system_dir / "logs"
            plot_dir = system_dir / "plots"
            run_benchmark(system, url, system_dir, rps=args.rps, duration=args.duration)
            collect_logs(system, nodes, log_dir)
            generate_plot(system, log_dir, plot_dir)

            log(f"Done: {system_dir}")
            time.sleep(COOLDOWN)

        stop_services(container_names)

    finally:
        cleanup_containers()

    log("=" * 50)
    log(f" All benchmarks complete! Results: {results_dir}")
    log("=" * 50)


def main_distributed(args, systems):
    """Distributed mode: local head + remote tail via SSH."""
    if _bu is None:
        die("bench_utils not available — install pynvml and dependencies")

    gpu = args.gpu
    remote_gpu = args.remote_gpu

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(args.output) if args.output else RESULTS_ROOT / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 50)
    log(f" Distributed Batch-Level Instrumentation Benchmark")
    log(f" Local GPU: {gpu}, Remote GPU: {remote_gpu}")
    log(f" Systems: {', '.join(systems)}")
    log(f" Results: {results_dir}")
    log("=" * 50)

    config = {
        "timestamp": timestamp,
        "mode": "distributed",
        "systems": systems,
        "local_gpu": gpu,
        "remote_gpu": remote_gpu,
        "input_tokens": INPUT_TOKENS,
        "output_tokens": OUTPUT_TOKENS,
    }
    (results_dir / "config.json").write_text(json.dumps(config, indent=2))

    try:
        for system in systems:
            _bu.cleanup()

            system_dir = results_dir / system
            log_dir = system_dir / "logs"
            plot_dir = system_dir / "plots"

            if system == "molink":
                start_molink_distributed(gpu, remote_gpu, log_dir, args.molink_max_concurrent_batches)
                url = f"http://localhost:{HEAD_PORT}/generate"
                model = _bu.LOCAL_MODEL
                tokenizer = _bu.LOCAL_TOKENIZER
            elif system == "vllm":
                start_vllm_distributed(gpu, remote_gpu, log_dir)
                url = f"http://localhost:{HEAD_PORT}/v1/completions"
                model = _bu.VLLM_COMMON_MODEL
                tokenizer = _bu.LOCAL_TOKENIZER
            else:
                die(f"Unknown system: {system}")

            run_benchmark(system, url, system_dir, model=model, tokenizer=tokenizer, rps=args.rps, duration=args.duration)
            collect_logs_distributed(system, log_dir)
            generate_plot(system, log_dir, plot_dir)

            log(f"Done: {system_dir}")
            time.sleep(COOLDOWN)
    finally:
        _bu.cleanup()

    log("=" * 50)
    log(f" All benchmarks complete! Results: {results_dir}")
    log("=" * 50)


def main():
    args = parse_args()
    systems = list(args.systems) if args.systems else ["molink", "vllm"]
    if args.distributed:
        main_distributed(args, systems)
    else:
        main_docker(args, systems)


if __name__ == "__main__":
    main()
