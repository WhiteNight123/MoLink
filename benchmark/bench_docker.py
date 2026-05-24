#!/usr/bin/env python3
"""Docker-based benchmark for MoLink and vLLM pipeline parallelism.

Manages containers with tc/netem network shaping, runs benchmarks,
and collects logs.

Usage:
    python bench_docker.py                  # MoLink + vLLM
    python bench_docker.py molink           # MoLink only
    python bench_docker.py vllm             # vLLM only
    python bench_docker.py molink011 molink019  # v0.11 vs v0.19

Options:
    --pp N          Pipeline parallel size (default: 2)
    --tp N          Tensor parallel size (default: 1)
    --gpus N,N      GPU devices, comma-separated (default: auto 0,1,...)
    --rps 3 5       RPS values to test (default: 3)
    --network X     Network condition, e.g. 1gbit,5ms (default: 1gbit,5ms)
    --duration N    Seconds per run (default: 30)
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────

DOCKER_IMAGE = "molink:0.2"
DOCKER_NETWORK = "molink"
SUBNET = "172.26.0.0/16"

CONTAINER_DEFS = [
    {"role": "head",   "name": "bench-head",   "ip": "172.26.0.10"},
    {"role": "middle", "name": "bench-middle",  "ip": "172.26.0.12"},
    {"role": "tail",   "name": "bench-tail",    "ip": "172.26.0.11"},
]

MODEL_PATH = "/gxq/Qwen3-14B"
HOST_TOKENIZER = "/home/emnets-2/gxq/Qwen3-14B"
MOLINK_CODE = "/gxq/molink-measurement/MoLink"
VLLM_SOURCE = "/gxq/molink-measurement/vllm/vllm"
VLLM_SITE = "/opt/conda/envs/vllm19/lib/python3.12/site-packages/vllm"
VOLUMES = ["/mnt/disk1-16/gxq:/data", "/home/emnets-2/gxq:/gxq"]

# Container paths per system
VLLM19_PYTHON = "/opt/conda/envs/vllm19/bin/python"
VLLM19_BIN = "/opt/conda/envs/vllm19/bin"
V011_PYTHON = "/opt/conda/envs/molinkv11/bin/python"
V011_PYTHONPATH = "/home/MoLink"
V019_PYTHON = "/opt/conda/envs/molinkv19/bin/python"

HEAD_PORT = 8080
MIDDLE_PORT = 9096
TAIL_PORT = 9095
GRPC_PORTS = {"head": 50061, "middle": 50062, "tail": 50063}
RAY_PORT = 6379

PORT_MAP = {"head": HEAD_PORT, "middle": MIDDLE_PORT, "tail": TAIL_PORT}

# Layer splits for Qwen3-14B (~40 layers)
LAYER_SPLITS = {
    2: [(0, 21), (21, -1)],
    3: [(0, 14), (14, 28), (28, -1)],
}

INPUT_TOKENS = 1024
OUTPUT_TOKENS = 512
MAX_MODEL_LEN = 4096
MAX_CONCURRENT_BATCHES = 2
HEALTH_TIMEOUT = 300
COOLDOWN = 10

BENCH_DIR = Path(__file__).parent.resolve()
BENCHMARK_CLIENT = BENCH_DIR / "benchmark_client.py"
RESULTS_ROOT = BENCH_DIR / "results"

sys.path.insert(0, str(BENCH_DIR.parent))
from bench_utils import GPUMonitor, plot_gpu_chart


# ─── Logging ──────────────────────────────────────────────────────────────────

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def die(msg):
    log(f"ERROR: {msg}")
    sys.exit(1)


# ─── Docker helpers ──────────────────────────────────────────────────────────

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
    # Overlay networks require Swarm mode
    r = subprocess.run(["docker", "info", "--format", "{{.Swarm.LocalNodeState}}"],
                       capture_output=True, text=True)
    if r.stdout.strip() != "active":
        log("Initializing Docker Swarm for overlay network...")
        subprocess.run(["docker", "swarm", "init"], check=True)

    r = subprocess.run(["docker", "network", "inspect", DOCKER_NETWORK],
                       capture_output=True, text=True)
    if r.returncode != 0:
        log(f"Creating Docker overlay network: {DOCKER_NETWORK}")
        subprocess.run(
            ["docker", "network", "create", "--driver", "overlay",
             "--attachable", "--subnet", SUBNET, DOCKER_NETWORK],
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


# ─── Network shaping ─────────────────────────────────────────────────────────

def setup_tc(name, bw, delay, peer_ips):
    _dexec(name, "tc qdisc del dev eth0 root 2>/dev/null || true")
    _dexec(name, "tc qdisc add dev eth0 root handle 1: htb default 30")
    _dexec(name, f"tc class add dev eth0 parent 1: classid 1:1 htb rate {bw}")
    _dexec(name, f"tc qdisc add dev eth0 parent 1:1 netem delay {delay}")
    for i, ip in enumerate(peer_ips, 1):
        _dexec(name,
               f"tc filter add dev eth0 parent 1: protocol ip prio {i} "
               f"u32 match ip dst {ip} flowid 1:1")


def reset_tc(names):
    for n in names:
        _dexec(n, "tc qdisc del dev eth0 root 2>/dev/null || true")


# ─── Service management ──────────────────────────────────────────────────────

def stop_services(names):
    log("Stopping services...")
    for n in names:
        _dexec(n,
               "pkill -f 'python.*molinkv1' 2>/dev/null || true; "
               "pkill -f 'python.*vllm' 2>/dev/null || true; "
               "pkill -f 'ray::' 2>/dev/null || true; "
               "pkill -f 'raylet' 2>/dev/null || true; "
               "pkill -f 'VLLM' 2>/dev/null || true")
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
    # Timeout: dump logs
    if container_name and log_file:
        log(f"=== {container_name} logs ===")
        _dexec(container_name, f"tail -20 {log_file}")
    return False


# ─── MoLink ──────────────────────────────────────────────────────────────────

def _start_molink_node(name, python, pythonpath, grpc_port,
                       start_layer, end_layer, http_port,
                       tp=1, max_batches=MAX_CONCURRENT_BATCHES,
                       initial_peer=None,
                       molink_enabled=False):
    env = {"PYTHONPATH": pythonpath, "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "3600"}
    cmd = (
        f"{python} -m molinkv1.entrypoints.api_server "
        f"--model {MODEL_PATH} "
        f"--max-model-len {MAX_MODEL_LEN} "
        f"--tensor-parallel-size {tp} "
        f"--enforce-eager --no-enable-prefix-caching "
        f"--molink-grpc-port {grpc_port} "
        f"--molink-start-layer {start_layer} "
        f"--molink-end-layer {end_layer} "
        f"--port {http_port}"
    )
    if max_batches:
        cmd += f" --molink-max-concurrent-batches {max_batches}"
    if molink_enabled:
        cmd += " --molink-enabled"
    if initial_peer:
        cmd += f" --molink-initial-peer {initial_peer}"
    log_file = {0: "/tmp/bench_head.log", -1: "/tmp/bench_tail.log"}.get(
        end_layer, "/tmp/bench_middle.log")
    _dexec(name, cmd + f" &>{log_file}", detach=True, env=env)


def start_molink(nodes, tp=1):
    pp = len(nodes)
    splits = LAYER_SPLITS[pp]
    for n in nodes:
        _dexec(n["name"],
               f"find {MOLINK_CODE} -name '__pycache__' -type d "
               f"-exec rm -rf {{}} + 2>/dev/null || true")

    # Head
    _start_molink_node(nodes[0]["name"], VLLM19_PYTHON, MOLINK_CODE,
                       GRPC_PORTS["head"], *splits[0], HEAD_PORT,
                       tp=tp)
    if not wait_health(f"http://localhost:{HEAD_PORT}/health",
                       container_name=nodes[0]["name"],
                       log_file="/tmp/bench_head.log"):
        die("MoLink head failed to start")
    log("Head ready. Waiting 20s for gRPC...")
    time.sleep(20)

    # Middle (PP=3)
    if pp >= 3:
        _start_molink_node(nodes[1]["name"], VLLM19_PYTHON, MOLINK_CODE,
                           GRPC_PORTS["middle"], *splits[1], MIDDLE_PORT,
                           tp=tp,
                           initial_peer=f"{nodes[0]['ip']}:{GRPC_PORTS['head']}")
        if not wait_health(f"http://localhost:{MIDDLE_PORT}/health",
                           container_name=nodes[1]["name"],
                           log_file="/tmp/bench_middle.log"):
            die("Middle node failed")
        log("Middle ready!")
        time.sleep(5)

    # Tail
    if pp == 2:
        prev_ip, prev_grpc = nodes[0]["ip"], GRPC_PORTS["head"]
    else:
        prev_ip, prev_grpc = nodes[-2]["ip"], GRPC_PORTS["middle"]
    _start_molink_node(nodes[-1]["name"], VLLM19_PYTHON, MOLINK_CODE,
                       GRPC_PORTS["tail"], *splits[-1], TAIL_PORT,
                       tp=tp, initial_peer=f"{prev_ip}:{prev_grpc}")
    if not wait_health(f"http://localhost:{TAIL_PORT}/health",
                       container_name=nodes[-1]["name"],
                       log_file="/tmp/bench_tail.log"):
        die("Tail node failed")
    log("Tail ready!")
    time.sleep(5)


def start_molink_v011(nodes):
    _start_molink_node(nodes[0]["name"], V011_PYTHON, V011_PYTHONPATH,
                       GRPC_PORTS["head"], 0, 21, HEAD_PORT,
                       max_batches=None, molink_enabled=True)
    if not wait_health(f"http://localhost:{HEAD_PORT}/health",
                       container_name=nodes[0]["name"],
                       log_file="/tmp/bench_head.log"):
        die("v0.11 head failed")
    log("Head ready. Waiting 20s...")
    time.sleep(20)
    _start_molink_node(nodes[-1]["name"], V011_PYTHON, V011_PYTHONPATH,
                       GRPC_PORTS["tail"], 21, -1, TAIL_PORT,
                       max_batches=None, molink_enabled=True,
                       initial_peer=f"{nodes[0]['ip']}:{GRPC_PORTS['head']}")
    if not wait_health(f"http://localhost:{TAIL_PORT}/health",
                       container_name=nodes[-1]["name"],
                       log_file="/tmp/bench_tail.log"):
        die("v0.11 tail failed")
    log("Tail ready!")
    time.sleep(5)


def start_molink_v019(nodes):
    _start_molink_node(nodes[0]["name"], V019_PYTHON, MOLINK_CODE,
                       GRPC_PORTS["head"], 0, 21, HEAD_PORT)
    if not wait_health(f"http://localhost:{HEAD_PORT}/health",
                       container_name=nodes[0]["name"],
                       log_file="/tmp/bench_head.log"):
        die("v0.19 head failed")
    log("Head ready. Waiting 20s...")
    time.sleep(20)
    _start_molink_node(nodes[-1]["name"], V019_PYTHON, MOLINK_CODE,
                       GRPC_PORTS["tail"], 21, -1, TAIL_PORT,
                       initial_peer=f"{nodes[0]['ip']}:{GRPC_PORTS['head']}")
    if not wait_health(f"http://localhost:{TAIL_PORT}/health",
                       container_name=nodes[-1]["name"],
                       log_file="/tmp/bench_tail.log"):
        die("v0.19 tail failed")
    log("Tail ready!")
    time.sleep(5)


# ─── vLLM ────────────────────────────────────────────────────────────────────

def start_vllm_fair(nodes, tp=1):
    """Start vLLM with NCCL forced through the overlay network stack.

    Sets NCCL_P2P_DISABLE=1 so NCCL avoids GPU Direct P2P and uses TCP
    sockets via eth0 (overlay), where tc/netem shaping is applied. This makes the
    PP tensor transfer path comparable to MoLink's gRPC path.
    """
    pp = len(nodes)
    cuda_devs = ",".join(str(i) for i in range(tp))

    # NCCL fairness env vars:
    # - NCCL_P2P_DISABLE=1:  skip GPU Direct, use TCP over eth0
    # - NCCL_SOCKET_IFNAME=eth0: route through the tc-shaped overlay interface
    # - NCCL_IB_DISABLE=1:  no RDMA bypass
    # - VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE=shm: Ray edges use shm
    # - VLLM_DISABLE_PYNCCL=1:  skip PyNCCL, use stock torch.distributed
    fair_env = (
        f"NCCL_P2P_DISABLE=1 "
        f"NCCL_SOCKET_IFNAME=eth0 "
        f"NCCL_IB_DISABLE=1 "
        f"VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE=shm "
        f"VLLM_DISABLE_PYNCCL=1 "
        f"CUDA_VISIBLE_DEVICES={cuda_devs} "
    )

    # Ray head
    log(f"Starting Ray head (fair) in {nodes[0]['name']}...")
    _dexec(nodes[0]["name"],
           f"{fair_env}"
           f"VLLM_HOST_IP={nodes[0]['ip']} "
           f"{VLLM19_BIN}/ray start --head "
           f"--node-ip-address={nodes[0]['ip']} "
           f"--port={RAY_PORT} --num-gpus={tp}",
           detach=True)
    time.sleep(8)

    # Ray workers
    for node in nodes[1:]:
        log(f"Starting Ray worker (fair) in {node['name']}...")
        _dexec(node["name"],
               f"{fair_env}"
               f"VLLM_HOST_IP={node['ip']} "
               f"{VLLM19_BIN}/ray start "
               f"--address={nodes[0]['ip']}:{RAY_PORT} --num-gpus={tp}",
               detach=True)
        time.sleep(8)

    # vLLM serve
    log(f"Starting vLLM serve (fair, PP={pp} TP={tp})...")
    _dexec(nodes[0]["name"],
           f"{fair_env}"
           f"VLLM_HOST_IP={nodes[0]['ip']} "
           f"{VLLM19_BIN}/vllm serve "
           f"--model {MODEL_PATH} "
           f"--port {HEAD_PORT} "
           f"--max-model-len {MAX_MODEL_LEN} "
           f"--pipeline-parallel-size {pp} "
           f"--tensor-parallel-size {tp} "
           f"--distributed-executor-backend ray "
           f"--no-enable-prefix-caching --enforce-eager "
           f"&>/tmp/bench_vllm.log",
           detach=True)
    if not wait_health(f"http://localhost:{HEAD_PORT}/health",
                       container_name=nodes[0]["name"],
                       log_file="/tmp/bench_head.log"):
        die("vLLM fair failed to start")


# ─── Log collection ──────────────────────────────────────────────────────────

def collect_logs(system, nodes, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if system in ("molink", "molink011", "molink019"):
        for node, log_name in [(nodes[0], "head.log"),
                                (nodes[-1], "tail.log")]:
            r = _dexec(node["name"], f"cat /tmp/bench_{node['role']}.log 2>/dev/null || true")
            if r.stdout.strip():
                (outdir / log_name).write_text(r.stdout)
        if len(nodes) >= 3:
            r = _dexec(nodes[1]["name"], "cat /tmp/bench_middle.log 2>/dev/null || true")
            if r.stdout.strip():
                (outdir / "middle.log").write_text(r.stdout)
    elif system in ("vllm", "vllm_fair"):
        r = _dexec(nodes[0]["name"], "cat /tmp/bench_vllm.log 2>/dev/null || true")
        if r.stdout.strip():
            (outdir / "vllm.log").write_text(r.stdout)


# ─── Benchmark runner ────────────────────────────────────────────────────────

def run_benchmark(system, url, rps, duration, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    endpoint_type = "molink" if "molink" in system else "vllm"
    log(f"--- [{system}] rps={rps} ---")
    cmd = [
        sys.executable, str(BENCHMARK_CLIENT),
        "--url", url,
        "--type", endpoint_type,
        "--input-tokens", str(INPUT_TOKENS),
        "--output-tokens", str(OUTPUT_TOKENS),
        "--rps", str(rps),
        "--duration", str(duration),
        "--model", MODEL_PATH,
        "--tokenizer", HOST_TOKENIZER,
        "--output", str(outdir / "result.json"),
    ]
    subprocess.run(cmd, check=True)


# ─── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    import argparse
    ap = argparse.ArgumentParser(
        description="Docker benchmark for MoLink / vLLM pipeline parallelism")
    ap.add_argument("systems", nargs="*",
                    help="Systems to benchmark: molink vllm molink011 molink019")
    ap.add_argument("--pp", type=int, default=2,
                    help="Pipeline parallel size (default: 2)")
    ap.add_argument("--tp", type=int, default=1,
                    help="Tensor parallel size (default: 1)")
    ap.add_argument("--gpus", default=None,
                    help="GPU devices, e.g. 0,1 or 1,2 (default: auto)")
    ap.add_argument("--rps", nargs="+", type=float, default=[3],
                    help="RPS values (default: 0.5 1 3 5)")
    ap.add_argument("--network", nargs="+",
                    default=["1gbit,1ms"],
                    help="Network conditions: bandwidth,latency or 'none' for no limit")
    ap.add_argument("--duration", type=int, default=30,
                    help="Seconds per run (default: 30)")
    ap.add_argument("--output", default=None,
                    help="Results directory (default: auto-timestamped)")
    return ap.parse_args()


def main():
    args = parse_args()

    # Resolve systems
    systems = list(args.systems)
    if "v011" in systems:
        systems.remove("v011")
        if "molink011" not in systems:
            systems.append("molink011")
        if "molink019" not in systems:
            systems.append("molink019")
    if not systems:
        systems = ["molink", "vllm"]

    pp = args.pp
    tp = args.tp

    # GPU allocation
    if args.gpus:
        gpu_devs = [int(x) for x in args.gpus.split(",")]
    else:
        gpu_devs = list(range(pp * tp))
    if len(gpu_devs) < pp * tp:
        die(f"Need {pp * tp} GPUs, got {len(gpu_devs)}")

    # Split GPUs per node
    node_gpus = [gpu_devs[i * tp:(i + 1) * tp] for i in range(pp)]

    # Build node list: head, [middle,] tail
    roles = ["head", "tail"] if pp == 2 else ["head", "middle", "tail"]
    node_defs = {c["role"]: c for c in CONTAINER_DEFS}
    nodes = []
    for i, role in enumerate(roles):
        nodes.append({**node_defs[role], "gpus": node_gpus[i]})

    # Results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(args.output) if args.output else RESULTS_ROOT / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)

    # Banner
    log("=" * 50)
    log(f" Docker Benchmark  PP={pp}  TP={tp}")
    log(f" GPUs: {gpu_devs}")
    log(f" Systems: {', '.join(systems)}")
    log(f" Network: {args.network}")
    log(f" RPS: {args.rps}")
    log(f" Duration: {args.duration}s per run")
    log(f" Results: {results_dir}")
    log("=" * 50)

    # Save config
    config = {
        "timestamp": timestamp,
        "pp_size": pp, "tp_size": tp,
        "gpus": gpu_devs,
        "systems": systems,
        "network_conditions": args.network,
        "input_tokens": INPUT_TOKENS,
        "output_tokens": OUTPUT_TOKENS,
        "rps_values": args.rps,
        "duration_s": args.duration,
    }
    (results_dir / "config.json").write_text(json.dumps(config, indent=2))

    # Setup Docker
    cleanup_containers()
    ensure_network()

    log(f"Starting {pp} containers...")
    for node in nodes:
        port = PORT_MAP[node["role"]]
        start_container(node["name"], node["ip"], node["gpus"], port, port)
    log("Containers started.")
    time.sleep(5)

    container_names = [n["name"] for n in nodes]

    try:
        for system in systems:
            for net_cond in args.network:
                if net_cond == "none":
                    net_label = "no_limit"
                    reset_tc(container_names)
                else:
                    bw, delay = net_cond.split(",")
                    net_label = f"bw{bw}_delay{delay}"
                    log(f"Network: {bw} / {delay}")
                    for node in nodes:
                        others = [n["ip"] for n in nodes if n != node]
                        setup_tc(node["name"], bw, delay, others)

                stop_services(container_names)

                # Start system
                if system == "molink":
                    start_molink(nodes, tp=tp)
                    url = f"http://localhost:{HEAD_PORT}/generate"
                elif system == "vllm":
                    start_vllm_fair(nodes, tp=tp)
                    url = f"http://localhost:{HEAD_PORT}/v1/completions"
                elif system == "vllm_fair":
                    start_vllm_fair(nodes, tp=tp)
                    url = f"http://localhost:{HEAD_PORT}/v1/completions"
                elif system == "molink011":
                    start_molink_v011(nodes)
                    url = f"http://localhost:{HEAD_PORT}/generate"
                elif system == "molink019":
                    start_molink_v019(nodes)
                    url = f"http://localhost:{HEAD_PORT}/generate"
                else:
                    die(f"Unknown system: {system}")

                # Run benchmarks
                gpu_indices = sorted(set(g for n in nodes for g in n["gpus"]))
                for rps in args.rps:
                    outdir = results_dir / system / net_label / f"rps{int(rps)}"

                    gpu_mon = GPUMonitor(interval_s=0.2, gpu_indices=gpu_indices)
                    gpu_mon.start()
                    run_benchmark(system, url, rps, args.duration, outdir)
                    gpu_data = gpu_mon.save(str(outdir / "gpu_monitor.json"))
                    plot_gpu_chart(gpu_data, str(outdir / "gpu_chart"),
                                  title=f"GPU Utilization — {system} {net_label} rps{int(rps)}")

                    if system in ("molink", "vllm", "vllm_fair", "molink011", "molink019"):
                        collect_logs(system, nodes, str(outdir))
                    log(f"Done: {outdir}/result.json")
                    time.sleep(COOLDOWN)

                stop_services(container_names)
    finally:
        cleanup_containers()

    log("=" * 50)
    log(f" All benchmarks complete! Results: {results_dir}")
    log("=" * 50)


if __name__ == "__main__":
    main()
