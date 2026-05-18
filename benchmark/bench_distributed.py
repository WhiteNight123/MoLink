#!/usr/bin/env python3
"""
Distributed benchmark: MoLink vs vLLM with PP=2.

  MoLink: head (local RTX 4090) + tail (remote RTX 3090) via gRPC
  vLLM:   PP=2 via Ray (local RTX 4090 + remote RTX 3090)

Usage:
    python bench_distributed.py              # both MoLink and vLLM
    python bench_distributed.py molink       # only MoLink
    python bench_distributed.py vllm         # only vLLM
"""

import asyncio
import json
import os
import sys
from datetime import datetime

from bench_utils import (
    BENCH_DURATION, HEAD_END_LAYER, HEAD_PORT, INPUT_TOKENS,
    LOCAL_BENCH_CLIENT, LOCAL_IP, LOCAL_MODEL, LOCAL_MOLINK_DIR, LOCAL_TOKENIZER,
    MAX_MODEL_LEN, MOLINK_GRPC_HEAD, MOLINK_GRPC_TAIL, OUTPUT_TOKENS, RPS,
    RAY_PORT, REMOTE_HOST, REMOTE_MOLINK_DIR, REMOTE_MOLINK_PYTHON, REMOTE_MODEL,
    REMOTE_VLLM_BIN, VLLM_COMMON_MODEL, VLLM_FWD_PORT, TAIL_PORT,
    TAIL_START_LAYER,
    cleanup, health_check, health_check_remote, log, run_local, run_remote,
)

RESULTS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_distributed")


# ── Start MoLink ───────────────────────────────────────────────────────────

async def start_molink():
    """Start MoLink head locally + tail remotely."""
    log("Starting MoLink head (local, layers 0-21, GPU 0)...")
    run_local(
        [sys.executable, "-m", "molinkv1.entrypoints.api_server",
         "--model", LOCAL_MODEL,
         "--max-model-len", str(MAX_MODEL_LEN),
         "--tensor-parallel-size", "1",
         "--enforce-eager",
         "--no-enable-prefix-caching",
         "--molink-grpc-port", str(MOLINK_GRPC_HEAD),
         "--molink-start-layer", "0",
         "--molink-end-layer", str(HEAD_END_LAYER),
         "--molink-max-concurrent-batches", "2",
         "--port", str(HEAD_PORT)],
        background=True, gpu="0",
    )

    if not await health_check(f"http://localhost:{HEAD_PORT}", 300):
        log("ERROR: MoLink head failed to start")
        return None
    log("MoLink head ready.")

    import socket
    for _ in range(30):
        try:
            with socket.create_connection(("127.0.0.1", MOLINK_GRPC_HEAD), timeout=2):
                break
        except Exception:
            await asyncio.sleep(1)
    else:
        log("ERROR: Head gRPC port not ready")
        return None
    log("Head gRPC ready.")
    await asyncio.sleep(10)

    log("Starting MoLink tail (remote, layers 21-end, GPU 0)...")
    run_remote(
        f"CUDA_VISIBLE_DEVICES=0 "
        f"PYTHONPATH={REMOTE_MOLINK_DIR} "
        f"VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600 "
        f"NO_PROXY='*' "
        f"{REMOTE_MOLINK_PYTHON} -m molinkv1.entrypoints.api_server "
        f"--model {REMOTE_MODEL} "
        f"--max-model-len {MAX_MODEL_LEN} "
        f"--tensor-parallel-size 1 "
        f"--enforce-eager "
        f"--no-enable-prefix-caching "
        f"--molink-grpc-port {MOLINK_GRPC_TAIL} "
        f"--molink-start-layer {TAIL_START_LAYER} "
        f"--molink-end-layer -1 "
        f"--molink-max-concurrent-batches 2 "
        f"--port {TAIL_PORT} "
        f"--molink-initial-peer {LOCAL_IP}:{MOLINK_GRPC_HEAD}",
        background=True,
    )

    log("Waiting for remote tail node...")
    if not await health_check_remote(TAIL_PORT, 300):
        log("ERROR: Remote tail health check timed out")
        return None
    log("MoLink tail ready.")
    await asyncio.sleep(5)

    # Functional check
    import aiohttp
    log("Verifying MoLink pipeline...")
    try:
        async with aiohttp.ClientSession() as s:
            async with s.post(
                f"http://localhost:{HEAD_PORT}/generate",
                json={"prompt": "Hello.", "max_tokens": 5, "temperature": 0},
                timeout=aiohttp.ClientTimeout(total=60),
            ) as r:
                if r.status == 200:
                    log("Pipeline OK.")
                else:
                    body = await r.text()
                    log(f"Pipeline check failed: HTTP {r.status}: {body[:200]}")
                    return None
    except Exception as e:
        log(f"Pipeline check failed: {e}")
        return None

    return f"http://localhost:{HEAD_PORT}/generate"


# ── Start vLLM ─────────────────────────────────────────────────────────────

async def start_vllm():
    """Start vLLM PP=2 across local (RTX 4090) + remote (RTX 3090) via Ray."""
    import subprocess

    # 1. Start Ray head on local machine
    log("Starting Ray head on local machine (RTX 4090, GPU 0)...")
    r = run_local(
        ["ray", "start", "--head", "--port", str(RAY_PORT),
         "--num-gpus=1", "--dashboard-host", "0.0.0.0",
         "--dashboard-port", "8265"],
        env_extra={
            "GLOO_SOCKET_IFNAME": "eno1",
            "NCCL_SOCKET_IFNAME": "eno1",
            "NCCL_SHM_DISABLE": "1",
            "NCCL_P2P_DISABLE": "1",
        },
        gpu="0",
    )
    if r.returncode != 0:
        log(f"ERROR: Ray head failed: {r.stderr[:300]}")
        return None
    await asyncio.sleep(3)

    # 2. Start Ray worker on remote machine
    log("Starting Ray worker on remote machine (RTX 3090, GPU 0)...")
    r = subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no",
         "-p", "15301", "gpu2@10.130.151.15",
         "CUDA_VISIBLE_DEVICES=0 "
         "RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL=minor "
         "GLOO_SOCKET_IFNAME=enx6c1ff766c0ef "
         "NCCL_SOCKET_IFNAME=enx6c1ff766c0ef "
         "NCCL_SHM_DISABLE=1 "
         "NCCL_P2P_DISABLE=1 "
         f"{REMOTE_VLLM_BIN}/ray "
         f"start --address={LOCAL_IP}:{RAY_PORT} --num-gpus=1"],
        capture_output=True, text=True, timeout=60,
    )
    if r.returncode != 0:
        log(f"WARNING: Remote Ray worker issues: {r.stderr[:300]}")
    else:
        log("Remote Ray worker started.")

    # Wait for cluster to stabilize
    for attempt in range(10):
        r = run_local(["ray", "status"], gpu="0")
        if "2 nodes" in r.stdout:
            log("Ray cluster ready (2 nodes, 2 GPUs).")
            break
        if attempt == 9:
            log(f"WARNING: Ray cluster may not be ready: {r.stdout[:200]}")
        await asyncio.sleep(2)

    # 3. Start vLLM API server locally
    log("Starting vLLM PP=2 via Ray (local 4090 + remote 3090)...")
    run_local(
        [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
         "--model", VLLM_COMMON_MODEL,
         "--host", "0.0.0.0",
         "--port", str(HEAD_PORT),
         "--max-model-len", str(MAX_MODEL_LEN),
         "--pipeline-parallel-size", "2",
         "--tensor-parallel-size", "1",
         "--distributed-executor-backend", "ray",
         "--enforce-eager",
         "--no-enable-prefix-caching"],
        env_extra={
            "MASTER_ADDR": LOCAL_IP,
            "GLOO_SOCKET_IFNAME": "eno1",
            "NCCL_SHM_DISABLE": "1",
            "NCCL_P2P_DISABLE": "1",
        },
        background=True,
        gpu="0",
    )

    if not await health_check(f"http://localhost:{HEAD_PORT}", 300):
        log("ERROR: vLLM failed to start")
        return None
    log("vLLM ready.")

    return f"http://localhost:{HEAD_PORT}/v1/completions"


# ── Benchmark runner ───────────────────────────────────────────────────────

async def run_benchmark(url, system, model=None, out_file=None):
    """Run benchmark using benchmark_client.py."""
    if out_file is None:
        out_file = f"{RESULTS_ROOT}/{system}_result.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    import subprocess
    cmd = [
        sys.executable, LOCAL_BENCH_CLIENT,
        "--url", url,
        "--type", system,
        "--input-tokens", str(INPUT_TOKENS),
        "--output-tokens", str(OUTPUT_TOKENS),
        "--rps", str(RPS),
        "--duration", str(BENCH_DURATION),
        "--model", model or LOCAL_MODEL,
        "--tokenizer", LOCAL_TOKENIZER,
        "--output", out_file,
    ]

    log(f"Running {system} benchmark (RPS={RPS}, {BENCH_DURATION}s)...")
    env = {**os.environ, "NO_PROXY": "*", "no_proxy": "*"}
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)

    if os.path.isfile(out_file):
        with open(out_file) as f:
            return json.load(f)
    return None


# ── Results comparison ─────────────────────────────────────────────────────

def print_comparison(molink_result, vllm_result):
    print()
    print("=" * 74)
    print("  MoLink vs vLLM  |  PP=2  |  RPS={:.0f}  |  Input={}  |  Output={}".format(
        RPS, INPUT_TOKENS, OUTPUT_TOKENS))
    print("  MoLink: distributed PP=2 (local RTX 4090 + remote RTX 3090)")
    print("  vLLM:   distributed PP=2 via Ray (local RTX 4090 + remote RTX 3090)")
    print("=" * 74)

    def extract(result):
        if not result:
            return None
        r = result.get("results", {})
        ttft = r.get("ttft_s", {})
        tpop = r.get("tpop_s", {})
        return {
            "throughput": r.get("throughput_tokens_per_s", 0),
            "ttft_avg": (ttft.get("avg") or 0) * 1000,
            "ttft_p50": (ttft.get("p50") or 0) * 1000,
            "ttft_p99": (ttft.get("p99") or 0) * 1000,
            "tpot_avg": (tpop.get("avg") or 0) * 1000,
            "tpot_p50": (tpop.get("p50") or 0) * 1000,
            "tpot_p99": (tpop.get("p99") or 0) * 1000,
            "success": r.get("successful_requests", 0),
            "failed": r.get("failed_requests", 0),
        }

    m = extract(molink_result)
    v = extract(vllm_result)

    if not m and not v:
        print("  No results to compare.")
        return

    print(f"  {'Metric':<28} {'MoLink':>16} {'vLLM':>16}")
    print("  " + "-" * 62)

    def row(name, key, fmt=".1f", unit="ms"):
        mv = m[key] if m else 0
        vv = v[key] if v else 0
        print(f"  {name:<28} {mv:>15{fmt}}{unit}  {vv:>15{fmt}}{unit}")

    row("TTFT avg", "ttft_avg")
    row("TTFT p50", "ttft_p50")
    row("TTFT p99", "ttft_p99")
    row("TPOT avg", "tpot_avg", ".2f")
    row("TPOT p50", "tpot_p50", ".2f")
    row("TPOT p99", "tpot_p99", ".2f")
    row("Throughput", "throughput", ".1f", " tok/s")
    print("  " + "-" * 62)

    ms = m["success"] if m else 0
    vs_ = v["success"] if v else 0
    mf = m["failed"] if m else 0
    vf = v["failed"] if v else 0
    print(f"  {'Successful requests':<28} {ms:>16}  {vs_:>16}")
    print(f"  {'Failed requests':<28} {mf:>16}  {vf:>16}")
    print("=" * 74)

    if m and v and m["throughput"] and v["throughput"]:
        ratio = m["throughput"] / v["throughput"]
        print(f"  Throughput ratio: MoLink/vLLM = {ratio:.2f}x")
        if ratio > 1:
            print(f"  MoLink is {((ratio - 1) * 100):.1f}% faster")
        else:
            print(f"  vLLM is {((1/ratio - 1) * 100):.1f}% faster")
    print()


# ── Main ───────────────────────────────────────────────────────────────────

async def main():
    systems = sys.argv[1:] if len(sys.argv) > 1 else ["molink", "vllm"]
    molink_result = None
    vllm_result = None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(RESULTS_ROOT, timestamp)
    os.makedirs(results_dir, exist_ok=True)

    log("=" * 60)
    log(" Distributed Benchmark: PP=2")
    log(f" Local head : {LOCAL_IP} (RTX 4090)")
    log(f" Remote     : {REMOTE_HOST} (2x RTX 3090)")
    log(f" Systems    : {', '.join(systems)}")
    log(f" RPS={RPS}  Duration={BENCH_DURATION}s  Input={INPUT_TOKENS} tok  Output={OUTPUT_TOKENS} tok")
    log(f" Results    : {results_dir}")
    log("=" * 60)

    if "molink" in systems:
        log("\n===== MoLink =====")
        cleanup()
        url = await start_molink()
        if url:
            molink_result = await run_benchmark(
                url, "molink",
                out_file=f"{results_dir}/molink/molink_result.json")
        else:
            log("MoLink failed to start, skipping.")
        cleanup()

    if "vllm" in systems:
        log("\n===== vLLM =====")
        cleanup()
        url = await start_vllm()
        if url:
            vllm_result = await run_benchmark(
                url, "vllm", model=VLLM_COMMON_MODEL,
                out_file=f"{results_dir}/vllm/vllm_result.json")
        else:
            log("vLLM failed to start, skipping.")
        cleanup()

    print_comparison(molink_result, vllm_result)

    # Save combined results
    out = {}
    if molink_result:
        out["molink"] = molink_result
    if vllm_result:
        out["vllm"] = vllm_result
    out_path = f"{results_dir}/distributed_compare.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log(f"Results saved to {out_path}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        cleanup()
