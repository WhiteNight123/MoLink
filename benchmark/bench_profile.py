#!/usr/bin/env python3
"""
Unified pipeline profiler: MoLink and vLLM PP=2.

Runs cross-host profiling benchmarks, collects timing data, and produces
comparative analysis.

Usage:
    python bench_profile.py molink           # MoLink profiling (default)
    python bench_profile.py vllm             # vLLM profiling + MoLink comparison
    python bench_profile.py analyze [DIR]    # Analyze existing cross-host profiles
    python bench_profile.py bottleneck DIR   # Analyze Docker benchmark results
"""

import asyncio
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from bench_utils import (
    BENCH_DURATION, HEAD_END_LAYER, HEAD_PORT, INPUT_TOKENS,
    LOCAL_BENCH_CLIENT, LOCAL_IP, LOCAL_MOLINK_DIR, LOCAL_MODEL,
    MAX_MODEL_LEN, MOLINK_GRPC_HEAD, MOLINK_GRPC_TAIL, OUTPUT_TOKENS, RPS,
    RAY_PORT, REMOTE_HOST, REMOTE_MOLINK_DIR, REMOTE_MOLINK_PYTHON, REMOTE_MODEL,
    REMOTE_PORT, REMOTE_USER, REMOTE_VLLM_BIN, SSH_CMD, TAIL_PORT,
    TAIL_START_LAYER, VLLM_COMMON_MODEL,
    cleanup, health_check, health_check_remote, log, run_local, run_remote,
)

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILER_DIR = "/tmp/molink_profile"
VLLM_PROFILE_DIR = "/tmp/vllm_profile"
RESULTS_ROOT = os.path.join(BENCH_DIR, "results_profile")

NCCL_BANDWIDTH_MB_S = 115.6


# ═══════════════════════════════════════════════════════════════════════════════
#  Shared benchmark runner
# ═══════════════════════════════════════════════════════════════════════════════

async def run_benchmark(url, system, model, out_file, duration, output_tokens):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    cmd = [
        sys.executable, LOCAL_BENCH_CLIENT,
        "--url", url,
        "--type", system,
        "--input-tokens", str(INPUT_TOKENS),
        "--output-tokens", str(output_tokens),
        "--rps", str(RPS),
        "--duration", str(duration),
        "--model", model,
        "--tokenizer", LOCAL_MODEL,
        "--output", out_file,
    ]
    log(f"Running {system} benchmark (RPS={RPS}, {duration}s, output={output_tokens})...")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    if os.path.isfile(out_file):
        with open(out_file) as f:
            return json.load(f)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  MoLink profiling
# ═══════════════════════════════════════════════════════════════════════════════

async def start_molink_profile():
    if os.path.exists(PROFILER_DIR):
        shutil.rmtree(PROFILER_DIR)
    os.makedirs(PROFILER_DIR, exist_ok=True)
    run_remote(f"rm -rf {PROFILER_DIR} && mkdir -p {PROFILER_DIR}")

    log("Starting MoLink head (local, layers 0-21, GPU 0, profiling ON)...")
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
         "--molink-enable-metrics",
         "--port", str(HEAD_PORT)],
        env_extra={"MOLINK_PROFILER_DIR": PROFILER_DIR},
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

    # Sync profiler module to remote
    log("Syncing profiler module to remote...")
    subprocess.run(
        SSH_CMD + [f"mkdir -p {REMOTE_MOLINK_DIR}/molinkv1"],
        capture_output=True, timeout=10,
    )
    for f in ["profiler.py", "__init__.py"]:
        local = f"{LOCAL_MOLINK_DIR}/molinkv1/{f}"
        remote = f"{REMOTE_MOLINK_DIR}/molinkv1/{f}"
        if os.path.exists(local):
            subprocess.run(
                ["scp", "-P", REMOTE_PORT, "-o", "StrictHostKeyChecking=no",
                 local, f"{REMOTE_USER}@{REMOTE_HOST}:{remote}"],
                capture_output=True, timeout=10,
            )
    for f in ["molink_pb2.py", "molink_pb2_grpc.py"]:
        local = f"{LOCAL_MOLINK_DIR}/molinkv1/comm/{f}"
        remote = f"{REMOTE_MOLINK_DIR}/molinkv1/comm/{f}"
        if os.path.exists(local):
            subprocess.run(
                ["scp", "-P", REMOTE_PORT, "-o", "StrictHostKeyChecking=no",
                 local, f"{REMOTE_USER}@{REMOTE_HOST}:{remote}"],
                capture_output=True, timeout=10,
            )
    for f in ["worker_node.py"]:
        local = f"{LOCAL_MOLINK_DIR}/molinkv1/engine/{f}"
        remote = f"{REMOTE_MOLINK_DIR}/molinkv1/engine/{f}"
        if os.path.exists(local):
            subprocess.run(
                ["scp", "-P", REMOTE_PORT, "-o", "StrictHostKeyChecking=no",
                 local, f"{REMOTE_USER}@{REMOTE_HOST}:{remote}"],
                capture_output=True, timeout=10,
            )

    log("Starting MoLink tail (remote, layers 21-end, GPU 0, profiling ON)...")
    run_remote(
        f"export MOLINK_PROFILER_DIR={PROFILER_DIR}; "
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
        f"--molink-enable-metrics "
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


def collect_molink_profiles(out_dir):
    os.makedirs(out_dir, exist_ok=True)

    for f in glob.glob(f"{PROFILER_DIR}/profile_head_*.jsonl"):
        dst = f"{out_dir}/{os.path.basename(f)}"
        shutil.copy2(f, dst)
        log(f"Collected head profile: {dst}")

    r = subprocess.run(
        SSH_CMD + [f"ls {PROFILER_DIR}/profile_tail_*.jsonl 2>/dev/null"],
        capture_output=True, text=True, timeout=10,
    )
    if r.stdout.strip():
        for line in r.stdout.strip().split("\n"):
            remote_file = line.strip()
            if remote_file:
                dst = f"{out_dir}/{os.path.basename(remote_file)}"
                subprocess.run(
                    ["scp", "-P", REMOTE_PORT, "-o", "StrictHostKeyChecking=no",
                     f"{REMOTE_USER}@{REMOTE_HOST}:{remote_file}", dst],
                    capture_output=True, timeout=10,
                )
                log(f"Collected tail profile: {dst}")

    return out_dir


# ═══════════════════════════════════════════════════════════════════════════════
#  vLLM profiling
# ═══════════════════════════════════════════════════════════════════════════════

async def start_vllm_profile():
    if os.path.exists(VLLM_PROFILE_DIR):
        shutil.rmtree(VLLM_PROFILE_DIR)
    os.makedirs(VLLM_PROFILE_DIR, exist_ok=True)

    log("Starting Ray head (local)...")
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

    log("Starting Ray worker (remote)...")
    r = subprocess.run(
        SSH_CMD + [
            f"VLLM_PROFILE_DIR={VLLM_PROFILE_DIR} "
            "CUDA_VISIBLE_DEVICES=0 "
            "RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL=minor "
            "GLOO_SOCKET_IFNAME=enx6c1ff766c0ef "
            "NCCL_SOCKET_IFNAME=enx6c1ff766c0ef "
            "NCCL_SHM_DISABLE=1 "
            "NCCL_P2P_DISABLE=1 "
            f"{REMOTE_VLLM_BIN}/ray "
            f"start --address={LOCAL_IP}:{RAY_PORT} --num-gpus=1"
        ],
        capture_output=True, text=True, timeout=60,
    )
    if r.returncode != 0:
        log(f"WARNING: Remote Ray worker issues: {r.stderr[:300]}")

    for attempt in range(10):
        r = run_local(["ray", "status"], gpu="0")
        if "2 nodes" in r.stdout:
            log("Ray cluster ready (2 nodes, 2 GPUs).")
            break
        if attempt == 9:
            log("WARNING: Ray cluster may not be ready")
        await asyncio.sleep(2)

    log("Starting vLLM PP=2 via Ray with profiling...")
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
            "VLLM_PROFILE_DIR": VLLM_PROFILE_DIR,
        },
        background=True,
        gpu="0",
    )

    if not await health_check(f"http://localhost:{HEAD_PORT}", 300):
        log("ERROR: vLLM failed to start")
        return None
    log("vLLM ready.")

    return f"http://localhost:{HEAD_PORT}/v1/completions"


def collect_vllm_profiles(out_dir):
    os.makedirs(out_dir, exist_ok=True)

    for f in glob.glob(f"{VLLM_PROFILE_DIR}/profile_vllm_rank*.jsonl"):
        dst = f"{out_dir}/{os.path.basename(f)}"
        shutil.copy2(f, dst)
        log(f"Collected local profile: {dst}")

    r = subprocess.run(
        SSH_CMD + [f"ls {VLLM_PROFILE_DIR}/profile_vllm_rank*.jsonl 2>/dev/null"],
        capture_output=True, text=True, timeout=10,
    )
    if r.stdout.strip():
        for line in r.stdout.strip().split("\n"):
            remote_file = line.strip()
            if remote_file:
                dst = f"{out_dir}/{os.path.basename(remote_file)}"
                subprocess.run(
                    ["scp", "-P", REMOTE_PORT, "-o", "StrictHostKeyChecking=no",
                     f"{REMOTE_USER}@{REMOTE_HOST}:{remote_file}", dst],
                    capture_output=True, timeout=10,
                )
                log(f"Collected remote profile: {dst}")

    return out_dir


# ═══════════════════════════════════════════════════════════════════════════════
#  Analysis: cross-host profiling (MoLink JSONL + vLLM JSONL)
# ═══════════════════════════════════════════════════════════════════════════════

def _stats(values):
    if not values:
        return {"count": 0}
    s = sorted(values)
    n = len(s)
    return {
        "count": n, "avg": sum(s) / n, "p50": s[n // 2],
        "p90": s[int(n * 0.9)], "p99": s[min(int(n * 0.99), n - 1)],
        "min": s[0], "max": s[-1],
    }


def _stats_nn(values):
    """Like _stats but returns None for empty input."""
    if not values:
        return None
    return _stats(values)


def _print_stats(name, records, key):
    vals = [r[key] for r in records if key in r]
    s = _stats(vals)
    if s["count"] == 0:
        print(f"  {name:<32} no data")
        return
    print(f"  {name:<32} avg={s['avg']:8.1f}  p50={s['p50']:8.1f}  p90={s['p90']:8.1f}  ms  (n={s['count']})")


def _latest_results_dir(subdir=None):
    """Find the latest timestamp directory that contains subdir under results_profile."""
    if not os.path.isdir(RESULTS_ROOT):
        return None
    dirs = sorted(
        d for d in os.listdir(RESULTS_ROOT)
        if os.path.isdir(os.path.join(RESULTS_ROOT, d))
    )
    if subdir:
        for d in reversed(dirs):
            candidate = os.path.join(RESULTS_ROOT, d, subdir)
            if os.path.isdir(candidate):
                return candidate
        return None
    return os.path.join(RESULTS_ROOT, dirs[-1]) if dirs else None


def analyze_molink(profile_dir=None):
    """Analyze MoLink JSONL profile files and print timing breakdown."""
    if profile_dir is None:
        profile_dir = _latest_results_dir("molink")

    head_records, tail_records = [], []
    for f in glob.glob(f"{profile_dir}/profile_head_*.jsonl"):
        with open(f) as fh:
            for line in fh:
                try: head_records.append(json.loads(line))
                except Exception: pass
    for f in glob.glob(f"{profile_dir}/profile_tail_*.jsonl"):
        with open(f) as fh:
            for line in fh:
                try: tail_records.append(json.loads(line))
                except Exception: pass

    if not head_records and not tail_records:
        print("No profile records found.")
        return

    head_compute = [r for r in head_records if r.get("stage") == "head_compute"]
    head_push = [r for r in head_records if r.get("stage") == "head_push"]
    head_pipeline = [r for r in head_records if r.get("stage") == "head_pipeline"]
    tail_steps = [r for r in tail_records if r.get("stage") == "tail_step"]

    print()
    print("=" * 80)
    print("  MoLink Pipeline Profile Analysis")
    print(f"  Head records: {len(head_records)}  Tail records: {len(tail_records)}")
    print("=" * 80)

    print("\n--- HEAD NODE ---")
    _print_stats("Head compute (layers 0-21)", head_compute, "head_compute_ms")
    _print_stats("  num_tokens", head_compute, "num_tokens")
    if head_compute:
        tensor_sizes = [r.get("tensor_size_mb", 0) for r in head_compute]
        ts = _stats(tensor_sizes)
        if ts["count"] > 0:
            print(f"  {'tensor_size_mb':<32} avg={ts['avg']:8.2f}  p50={ts['p50']:8.2f}  MB")

    print("\n--- HEAD → TAIL TRANSFER ---")
    _print_stats("Pickle scheduler_output", head_push, "pickle_ms")
    _print_stats("Tensor serialization", head_push, "tensor_serialize_ms")
    _print_stats("Total serialize (wall)", head_push, "serialize_wall_ms")
    _print_stats("gRPC send", head_push, "grpc_send_ms")
    if head_push:
        sizes = [r.get("serialized_bytes", 0) / 1024 / 1024 for r in head_push]
        ss = _stats(sizes)
        if ss["count"] > 0:
            print(f"  {'serialized_size_MB':<32} avg={ss['avg']:8.2f}  p50={ss['p50']:8.2f}  MB")

    print("\n--- TAIL NODE ---")
    _print_stats("Queue wait", tail_steps, "queue_wait_ms")
    _print_stats("Deserialize tensors", tail_steps, "deserialize_ms")
    _print_stats("Compute lock wait", tail_steps, "compute_lock_wait_ms")
    _print_stats("Tail compute (layers 21-end)", tail_steps, "compute_ms")
    _print_stats("Serialize result", tail_steps, "serialize_result_ms")
    _print_stats("gRPC send result back", tail_steps, "grpc_send_ms")
    _print_stats("Tail total", tail_steps, "total_ms")

    print("\n--- HEAD WAIT & DESERIALIZE ---")
    _print_stats("Wait for tail result", head_pipeline, "wait_result_ms")
    _print_stats("Deserialize result", head_pipeline, "deserialize_result_ms")

    # Correlated pipeline breakdown
    print("\n--- FULL PIPELINE BREAKDOWN ---")
    head_by_step = {r["step_id"]: r for r in head_pipeline if "step_id" in r}
    tail_by_step = {r["step_id"]: r for r in tail_steps if "step_id" in r}

    if head_by_step and tail_by_step:
        common_steps = sorted(set(head_by_step) & set(tail_by_step))
        if common_steps:
            breakdowns = []
            for sid in common_steps:
                hc = next((r for r in head_compute if r.get("step_id") == sid), None)
                hp = head_by_step[sid]
                ts = tail_by_step[sid]
                head_compute_ms = hc["head_compute_ms"] if hc else 0
                head_push_ms = (hp.get("push_intermediate_ms", 0) if "push_intermediate_ms" in hp
                                else hp.get("serialize_wall_ms", 0) + hp.get("grpc_send_ms", 0))
                total = hp.get("total_pipeline_ms", 0)
                breakdowns.append({
                    "head_compute": head_compute_ms,
                    "head_push": head_push_ms,
                    "tail_queue": ts.get("queue_wait_ms", 0),
                    "tail_deser": ts.get("deserialize_ms", 0),
                    "tail_compute": ts.get("compute_ms", 0),
                    "tail_send": ts.get("grpc_send_ms", 0),
                    "head_wait": hp.get("wait_result_ms", 0),
                    "total": total,
                    "unaccounted": total - head_push_ms - hp.get("wait_result_ms", 0),
                })

            if breakdowns:
                n = len(breakdowns)
                avg = lambda k: sum(b[k] for b in breakdowns) / n
                avg_total = avg("total")
                avg_unaccounted = avg("unaccounted")
                total_sum = sum(avg(k) for k in ["head_compute", "head_push", "tail_queue",
                                                  "tail_deser", "tail_compute", "tail_send", "head_wait"])

                print(f"  {'Stage':<32} {'Avg (ms)':>10} {'% of pipeline':>14}")
                print(f"  {'-'*32} {'-'*10} {'-'*14}")

                def pct_row(name, val):
                    pct = (val / avg_total * 100) if avg_total > 0 else 0
                    print(f"  {name:<32} {val:10.1f} {pct:13.1f}%")

                pct_row("Head compute (layers 0-21)", avg("head_compute"))
                pct_row("Serialize + gRPC to tail", avg("head_push"))
                pct_row("Tail queue wait", avg("tail_queue"))
                pct_row("Tail deserialize", avg("tail_deser"))
                pct_row("Tail compute (layers 21-end)", avg("tail_compute"))
                pct_row("Tail gRPC send back", avg("tail_send"))
                pct_row("Head wait + deserialize", avg("head_wait"))
                if avg_unaccounted > 1:
                    pct_row("Unaccounted (overhead)", avg_unaccounted)
                print(f"  {'='*32} {'='*10}")
                print(f"  {'Total pipeline':<32} {avg_total:10.1f}")
                print(f"  {'Sum of stages':<32} {total_sum:10.1f}")
                print(f"  (based on {n} correlated steps)")

    # Bottleneck analysis
    print("\n--- BOTTLENECK ANALYSIS ---")
    if tail_steps:
        tq = _stats([r.get("queue_wait_ms", 0) for r in tail_steps])
        if tq["count"] > 0 and tq["avg"] > 10:
            print(f"  !! Tail queue wait avg={tq['avg']:.1f}ms p90={tq['p90']:.1f}ms max={tq['max']:.1f}ms")
            print(f"     → GPU is bottleneck: head sends faster than tail can process")

    if head_push:
        gs = _stats([r.get("grpc_send_ms", 0) for r in head_push])
        if gs["count"] > 0 and gs["avg"] > 5:
            print(f"  !! gRPC send avg={gs['avg']:.1f}ms p90={gs['p90']:.1f}ms max={gs['max']:.1f}ms")
            print(f"     → Network/serialization overhead significant")

    if head_compute and tail_steps:
        hc_avg = _stats([r["head_compute_ms"] for r in head_compute])["avg"]
        tc_avg = _stats([r["compute_ms"] for r in tail_steps])["avg"]
        if hc_avg > 0 and tc_avg > 0:
            ratio = tc_avg / hc_avg
            slower = "tail" if tc_avg > hc_avg else "head"
            print(f"  !! Compute balance: head={hc_avg:.1f}ms tail={tc_avg:.1f}ms "
                  f"(tail/head={ratio:.2f}x, {slower} is slower)")
    print()


def analyze_vllm_vs_molink(vllm_dir=None, molink_dir=None):
    """Analyze vLLM profile data side-by-side with MoLink profile data."""
    if vllm_dir is None:
        vllm_dir = _latest_results_dir("vllm")
    if molink_dir is None:
        molink_dir = _latest_results_dir("molink")

    vllm_records = []
    for f in glob.glob(f"{vllm_dir}/profile_vllm_rank*.jsonl"):
        with open(f) as fh:
            for line in fh:
                try: vllm_records.append(json.loads(line))
                except Exception: pass

    molink_records = []
    for f in (glob.glob(f"{molink_dir}/profile_head_*.jsonl")
              + glob.glob(f"{molink_dir}/profile_tail_*.jsonl")):
        with open(f) as fh:
            for line in fh:
                try: molink_records.append(json.loads(line))
                except Exception: pass

    if not vllm_records:
        print("No vLLM profile records found.")
        return

    head_r = [r for r in vllm_records if r.get("is_first", False)]
    tail_r = [r for r in vllm_records if r.get("is_last", False)]

    def p(v, fmt=".1f"):
        return f"{v:{fmt}}" if v is not None else "N/A"

    PREFILL_TH = 100
    pf_head = [r for r in head_r if r.get("num_tokens", 0) > PREFILL_TH]
    dc_head = [r for r in head_r if 0 < r.get("num_tokens", 0) <= PREFILL_TH]
    pf_tail = [r for r in tail_r if r.get("num_tokens", 0) > PREFILL_TH]
    dc_tail = [r for r in tail_r if 0 < r.get("num_tokens", 0) <= PREFILL_TH]

    print()
    print("=" * 90)
    print("  vLLM Pipeline Profile Analysis (PP=2, Ray+NCCL)")
    print(f"  Records: head={len(head_r)}, tail={len(tail_r)}")
    print(f"  Prefill: head={len(pf_head)}, tail={len(pf_tail)}")
    print(f"  Decode:  head={len(dc_head)}, tail={len(dc_tail)}")
    print("=" * 90)

    print("\n--- vLLM HEAD NODE (rank 0, RTX 4090, layers 0-21) ---")
    for key in ["compute_ms", "total_ms", "num_tokens", "output_tensor_size_mb"]:
        vals = [r[key] for r in head_r if key in r]
        s = _stats_nn(vals)
        if s:
            print(f"  {key:<30} avg={p(s['avg']):>10}  p50={p(s['p50']):>10}  p90={p(s['p90']):>10}  (n={s['count']})")

    print("\n--- vLLM TAIL NODE (rank 1, RTX 3090, layers 21-end) ---")
    for key in ["compute_ms", "sample_tokens_ms", "total_ms", "num_tokens", "input_tensor_size_mb"]:
        vals = [r[key] for r in tail_r if key in r and r.get(key, 0) != 0]
        s = _stats_nn(vals)
        if s:
            print(f"  {key:<30} avg={p(s['avg']):>10}  p50={p(s['p50']):>10}  p90={p(s['p90']):>10}  (n={s['count']})")

    print("\n--- vLLM PREFILL vs DECODE ---")
    if pf_head:
        avg_tok = sum(r["num_tokens"] for r in pf_head) / len(pf_head)
        print(f"  Prefill head: avg_tokens={avg_tok:.0f}")
        for key in ["compute_ms", "output_tensor_size_mb"]:
            vals = [r[key] for r in pf_head if key in r and (key != "output_tensor_size_mb" or r.get(key, 0) > 0)]
            s = _stats_nn(vals)
            if s:
                print(f"    {key:<28} avg={p(s['avg']):>10}  p50={p(s['p50']):>10}  p90={p(s['p90']):>10}")
    if pf_tail:
        print(f"  Prefill tail:")
        for key in ["compute_ms", "input_tensor_size_mb"]:
            vals = [r[key] for r in pf_tail if key in r and (key != "input_tensor_size_mb" or r.get(key, 0) > 0)]
            s = _stats_nn(vals)
            if s:
                print(f"    {key:<28} avg={p(s['avg']):>10}  p50={p(s['p50']):>10}  p90={p(s['p90']):>10}")
    if dc_head:
        s = _stats_nn([r["compute_ms"] for r in dc_head if "compute_ms" in r])
        if s: print(f"  Decode head: compute_ms avg={s['avg']:.1f}  p50={s['p50']:.1f}")
    if dc_tail:
        s = _stats_nn([r["compute_ms"] for r in dc_tail if "compute_ms" in r])
        if s: print(f"  Decode tail: compute_ms avg={s['avg']:.1f}  p50={s['p50']:.1f}")

    # NCCL estimate
    print(f"\n--- NCCL TRANSFER TIME ESTIMATE (Bandwidth: {NCCL_BANDWIDTH_MB_S:.1f} MB/s) ---")
    vllm_pf_sizes = [r.get("output_tensor_size_mb", 0) for r in pf_head if r.get("output_tensor_size_mb", 0) > 0]
    if vllm_pf_sizes:
        avg_s = sum(vllm_pf_sizes) / len(vllm_pf_sizes)
        print(f"  Prefill tensor avg: {avg_s:.2f} MB → est NCCL: {avg_s / NCCL_BANDWIDTH_MB_S * 1000:.1f} ms")
    vllm_dc_sizes = [r.get("output_tensor_size_mb", 0) for r in dc_head if r.get("output_tensor_size_mb", 0) > 0]
    if vllm_dc_sizes:
        avg_s = sum(vllm_dc_sizes) / len(vllm_dc_sizes)
        print(f"  Decode tensor avg:  {avg_s:.4f} MB → est NCCL: {avg_s / NCCL_BANDWIDTH_MB_S * 1000:.2f} ms")

    # Side-by-side comparison
    print("\n" + "=" * 90)
    print("  SIDE-BY-SIDE: MoLink (gRPC) vs vLLM (Ray+NCCL) — PREFILL PIPELINE STEP")
    print("=" * 90)

    m_hc = [r for r in molink_records if r.get("stage") == "head_compute" and r.get("num_tokens", 0) > 100]
    m_hp = [r for r in molink_records if r.get("stage") == "head_push"]
    m_tail = [r for r in molink_records if r.get("stage") == "tail_step"]
    m_hpipe = [r for r in molink_records if r.get("stage") == "head_pipeline"]

    pf_ids = set(r["step_id"] for r in m_hc)
    m_pf_push = [r for r in m_hp if r.get("step_id", -1) in pf_ids]
    m_pf_tail = [r for r in m_tail if r.get("step_id", -1) in pf_ids]
    m_pf_pipe = [r for r in m_hpipe if r.get("step_id", -1) in pf_ids]

    vllm_est_nccl = (sum(vllm_pf_sizes) / len(vllm_pf_sizes) / NCCL_BANDWIDTH_MB_S * 1000) if vllm_pf_sizes else 0
    nccl_vals = [vllm_est_nccl] * max(len(m_pf_push), 1)

    print(f"\n  {'Stage':<40} {'MoLink (gRPC)':>14} {'vLLM (NCCL)':>14} {'Δ':>10}")
    print(f"  {'-'*40} {'-'*14} {'-'*14} {'-'*10}")

    def compare_row(name, m_vals, v_vals, fmt=".1f"):
        ms, vs = _stats_nn(m_vals), _stats_nn(v_vals)
        m_avg, v_avg = ms["avg"] if ms else 0, vs["avg"] if vs else 0
        print(f"  {name:<40} {m_avg:>13{fmt}} {v_avg:>13{fmt}} {m_avg - v_avg:>+9{fmt}}")

    compare_row("Head compute (layers 0-21)",
                [r["head_compute_ms"] for r in m_hc], [r["compute_ms"] for r in pf_head])
    compare_row("Intermediate tensor size (MB)",
                [r["tensor_size_mb"] for r in m_hc], vllm_pf_sizes, ".2f")

    m_total_send = [r.get("tensor_serialize_ms", 0) + r.get("grpc_send_ms", 0) + r.get("pickle_ms", 0) for r in m_pf_push]
    compare_row("Tensor transfer (serialize+send+deser)", m_total_send, nccl_vals)
    compare_row("  Serialization (pickle+tensor_serialize)",
                [r.get("pickle_ms", 0) + r.get("tensor_serialize_ms", 0) for r in m_pf_push], [0] * len(m_pf_push))
    compare_row("  Network transfer (gRPC or NCCL est.)",
                [r.get("grpc_send_ms", 0) for r in m_pf_push], nccl_vals)
    compare_row("  Deserialize on tail",
                [r.get("deserialize_ms", 0) for r in m_pf_tail], [0] * len(m_pf_tail))
    compare_row("Tail compute (layers 21-end)",
                [r.get("compute_ms", 0) for r in m_pf_tail],
                [r["compute_ms"] for r in pf_tail if "compute_ms" in r])
    compare_row("Tail send result back (gRPC)",
                [r.get("grpc_send_ms", 0) for r in m_pf_tail], [0] * max(len(m_pf_tail), 1))
    compare_row("Head wait for result",
                [r.get("wait_result_ms", 0) for r in m_pf_pipe], [0] * max(len(m_pf_pipe), 1))

    # Overhead summary
    print("\n" + "=" * 90)
    print("  OVERHEAD SUMMARY")
    print("=" * 90)
    m_ser = _stats_nn([r.get("pickle_ms", 0) + r.get("tensor_serialize_ms", 0) for r in m_pf_push])
    m_grpc = _stats_nn([r.get("grpc_send_ms", 0) for r in m_pf_push])
    m_deser = _stats_nn([r.get("deserialize_ms", 0) for r in m_pf_tail])
    if m_ser:
        ser_o = m_ser["avg"]
        deser_o = m_deser["avg"] if m_deser else 0
        net_o = (m_grpc["avg"] if m_grpc else 0) - vllm_est_nccl
        total_o = ser_o + deser_o + net_o
        print(f"  MoLink serialization overhead:    +{ser_o:.1f} ms/step")
        print(f"  MoLink deserialization overhead:  +{deser_o:.1f} ms/step")
        print(f"  MoLink gRPC vs NCCL overhead:     +{net_o:.1f} ms/step")
        print(f"  Total MoLink overhead:            +{total_o:.1f} ms/step")
        print(f"  vLLM NCCL est. (prefill):          {vllm_est_nccl:.1f} ms")
        if m_grpc:
            print(f"  MoLink gRPC send (prefill):        {m_grpc['avg']:.1f} ms")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  Analysis: Docker benchmark results (from run_benchmark.sh)
# ═══════════════════════════════════════════════════════════════════════════════

def _load_json(path):
    if not Path(path).exists():
        return None
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def _percentile(data, p):
    if not data:
        return 0
    s = sorted(data)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] * (c - k) + s[c] * (k - f)


def _fmt_ms(values):
    if not values:
        return "N/A"
    avg = sum(values) / len(values)
    p50 = _percentile(values, 50)
    p99 = _percentile(values, 99)
    return f"avg={avg:.1f}ms  p50={p50:.1f}ms  p99={p99:.1f}ms  n={len(values)}"


def _fmt_avg(values):
    if not values:
        return "N/A"
    return f"{sum(values) / len(values):.1f}ms"


def _extract_molink_head(metrics_data):
    if not metrics_data:
        return {}
    result = {}
    for m in metrics_data.get("service_metrics", []):
        mtype = m.get("type")
        if mtype == "head_pipeline":
            for key in ["push_intermediate_ms", "wait_result_ms", "deserialize_result_ms",
                        "total_pipeline_ms", "pickle_ms", "tensor_serialize_ms",
                        "total_serialize_ms", "grpc_send_ms", "serialize_wall_ms",
                        "result_bytes", "serialized_bytes"]:
                if key in m:
                    result.setdefault(key, []).append(m[key])
        elif mtype == "head_compute":
            for key in ["compute_ms", "num_tokens"]:
                if key in m:
                    result.setdefault(f"head_{key}", []).append(m[key])
    return result


def _extract_molink_tail(metrics_data):
    if not metrics_data:
        return {}
    result = {}
    for m in metrics_data.get("service_metrics", []):
        if m.get("type") != "worker_step":
            continue
        for key in ["queue_wait_ms", "deserialize_ms", "compute_ms", "serialize_result_ms",
                    "grpc_send_ms", "push_ms", "total_ms", "recv_bytes", "result_bytes",
                    "compute_lock_wait_ms"]:
            if key in m:
                result.setdefault(key, []).append(m[key])
    return result


def _extract_vllm_dag(metrics_data):
    if not metrics_data:
        return {}
    result = {}
    for m in metrics_data.get("metrics", []):
        mtype = m.get("type", "")
        if mtype == "vllm_dag":
            for key in ["dag_execute_ms", "ray_get_ms", "total_dag_ms", "num_tokens"]:
                if key in m:
                    result.setdefault(key, []).append(m[key])
        elif mtype == "vllm_pipeline":
            for key in ["execute_model_gap_ms", "sample_tokens_total_ms", "num_tokens"]:
                if key in m:
                    result.setdefault(key, []).append(m[key])
    return result


def _extract_vllm_worker(metrics_data):
    if not metrics_data:
        return {}
    result = {}
    for m in metrics_data.get("worker_metrics", []):
        if m.get("type") != "vllm_worker":
            continue
        prefix = "last_" if m.get("is_last_pp_rank") else "first_"
        for key in ["model_runner_ms", "sample_tokens_ms", "total_ms", "num_tokens"]:
            if key in m:
                result.setdefault(f"{prefix}{key}", []).append(m[key])
    return result


def _compare_stage(name, molink_ms, vllm_ms):
    m_avg = sum(molink_ms) / len(molink_ms) if molink_ms else 0
    v_avg = sum(vllm_ms) / len(vllm_ms) if vllm_ms else 0
    if m_avg > 0 and v_avg > 0:
        overhead = ((m_avg / v_avg) - 1) * 100
        sign = "+" if overhead > 0 else ""
        ratio = f"{m_avg / v_avg:.2f}x ({sign}{overhead:.0f}%)"
    elif m_avg > 0:
        ratio = "MoLink only"
    elif v_avg > 0:
        ratio = "vLLM only"
    else:
        ratio = "-"
    print(f"  {name:<32} {_fmt_avg(molink_ms):>20} {_fmt_avg(vllm_ms):>20} {ratio:>22}")


def analyze_bottleneck(results_dir):
    """Analyze Docker benchmark results from run_benchmark.sh."""
    results_dir = Path(results_dir)
    print("=" * 100)
    print(f"  MoLink Performance Bottleneck Analysis")
    print(f"  Results: {results_dir}")
    print("=" * 100)

    config = _load_json(results_dir / "config.json")
    if config:
        print(f"\n  Config: PP={config.get('pp_size')} TP={config.get('tp_size')} "
              f"input={config.get('input_tokens')} output={config.get('output_tokens')} "
              f"RPS={config.get('rps_values')} duration={config.get('duration_s')}s")

    molink_dirs = sorted(results_dir.glob("molink/*/rps*"))
    vllm_dirs = sorted(results_dir.glob("vllm/*/rps*"))
    # Also match molink011/molink019 dirs
    molink_dirs += sorted(results_dir.glob("molink0*/*/rps*"))

    if not molink_dirs and not vllm_dirs:
        print("\n  No benchmark results found. Run ./run_benchmark.sh first.")
        return

    # 1. Client-side comparison
    print("\n" + "=" * 100)
    print("  1. CLIENT-SIDE COMPARISON")
    print("=" * 100)
    print(f"  {'System':<12} {'Network':<20} {'RPS':<6} "
          f"{'Throughput':>14} {'TTFT avg':>12} {'TTFT p99':>12} {'Requests':>10}")
    print("  " + "-" * 88)

    for dirs, label in [(molink_dirs, "MoLink"), (vllm_dirs, "vLLM")]:
        for d in dirs:
            result = _load_json(d / "result.json")
            if not result:
                continue
            r = result.get("results", {})
            c = result.get("config", {})
            net_label = str(d.parent.name)
            rps = c.get("rps", "?")
            throughput = r.get("throughput_tokens_per_s", 0)
            ttft = r.get("ttft_s", {})
            ttft_avg = (ttft.get("avg") or 0) * 1000
            ttft_p99 = (ttft.get("p99") or 0) * 1000
            ok = r.get("successful_requests", 0)
            fail = r.get("failed_requests", 0)
            print(f"  {label:<12} {net_label:<20} {rps:<6} "
                  f"{throughput:>12.1f} t/s {ttft_avg:>10.1f}ms {ttft_p99:>10.1f}ms "
                  f"{ok:>5} OK / {fail} fail")

    # 2. MoLink head pipeline breakdown
    print("\n" + "=" * 100)
    print("  2. MoLink HEAD NODE Pipeline Breakdown")
    print("=" * 100)

    for d in molink_dirs:
        head_data = _load_json(d / "head_metrics.json")
        if not head_data:
            continue
        head = _extract_molink_head(head_data)
        if not head:
            continue
        net_label = str(d.parent.name)
        print(f"\n  [{net_label}] Head pipeline metrics ({len(head.get('total_pipeline_ms', []))} steps):")
        print(f"  {'Stage':<32} {'Timing':>60}")
        print("  " + "-" * 92)

        for key in ["head_compute_ms", "serialize_wall_ms", "  pickle_ms", "  tensor_serialize_ms",
                    "grpc_send_ms", "push_intermediate_ms", "wait_result_ms",
                    "deserialize_result_ms", "total_pipeline_ms"]:
            actual_key = key.strip()
            indent = "  " if key.startswith("  ") else ""
            print(f"  {indent}{key:<30} {_fmt_ms(head.get(actual_key, []))}")

        if head.get("serialized_bytes"):
            avg_bytes = sum(head["serialized_bytes"]) / len(head["serialized_bytes"])
            print(f"  {'  avg serialized bytes':<30} {avg_bytes / 1024 / 1024:.1f} MB")
        if head.get("result_bytes"):
            avg_bytes = sum(head["result_bytes"]) / len(head["result_bytes"])
            print(f"  {'  avg result bytes':<30} {avg_bytes / 1024:.1f} KB")

    # 3. MoLink tail breakdown
    print("\n" + "=" * 100)
    print("  3. MoLink TAIL NODE Worker Breakdown")
    print("=" * 100)

    for d in molink_dirs:
        tail_data = _load_json(d / "tail_metrics.json")
        if not tail_data:
            continue
        tail = _extract_molink_tail(tail_data)
        if not tail:
            continue
        net_label = str(d.parent.name)
        print(f"\n  [{net_label}] Tail worker metrics ({len(tail.get('total_ms', []))} steps):")
        for key in ["queue_wait_ms", "deserialize_ms", "compute_lock_wait_ms", "compute_ms",
                    "serialize_result_ms", "grpc_send_ms", "push_ms", "total_ms"]:
            print(f"  {key:<32} {_fmt_ms(tail.get(key, []))}")
        if tail.get("recv_bytes"):
            avg_bytes = sum(tail["recv_bytes"]) / len(tail["recv_bytes"])
            print(f"  {'  avg recv bytes':<30} {avg_bytes / 1024 / 1024:.1f} MB")

    # 4. vLLM Ray DAG breakdown
    print("\n" + "=" * 100)
    print("  4. vLLM Ray DAG Breakdown")
    print("=" * 100)

    for d in vllm_dirs:
        vllm_data = _load_json(d / "vllm_metrics.json")
        if not vllm_data:
            continue
        vllm_m = _extract_vllm_dag(vllm_data)
        if not vllm_m:
            continue
        net_label = str(d.parent.name)
        print(f"\n  [{net_label}] vLLM Ray DAG metrics ({len(vllm_m.get('total_dag_ms', []))} steps):")
        for key in ["execute_model_gap_ms", "dag_execute_ms", "ray_get_ms",
                    "sample_tokens_total_ms", "total_dag_ms"]:
            print(f"  {key:<32} {_fmt_ms(vllm_m.get(key, []))}")

        for wf, stage_label in [("vllm_worker_head.json", "PP Stage 0 (head)"),
                                ("vllm_worker_tail.json", "PP Stage 1 (tail)"),
                                ("vllm_worker_middle.json", "PP Stage 1 (middle)")]:
            wd = _load_json(d / wf)
            if not wd:
                continue
            wm = _extract_vllm_worker(wd)
            if not wm:
                continue
            prefix = "first_" if "Stage 0" in stage_label else "last_"
            print(f"\n    [{stage_label}] Worker metrics:")
            for wk in ["model_runner_ms", "sample_tokens_ms", "total_ms"]:
                vals = wm.get(f"{prefix}{wk}", [])
                if vals:
                    print(f"      {wk:<28} {_fmt_ms(vals)}")

    # 5. Cross-node comparison
    print("\n" + "=" * 100)
    print("  5. CROSS-NODE COMMUNICATION COMPARISON (MoLink vs vLLM)")
    print("=" * 100)

    for m_dir in molink_dirs:
        net_label = str(m_dir.parent.name)
        rps_label = str(m_dir.name)
        v_dir = results_dir / "vllm" / net_label / rps_label
        if not v_dir.exists():
            continue

        head = _extract_molink_head(_load_json(m_dir / "head_metrics.json"))
        tail = _extract_molink_tail(_load_json(m_dir / "tail_metrics.json"))
        vllm_m = _extract_vllm_dag(_load_json(v_dir / "vllm_metrics.json"))
        vllm_wh = _extract_vllm_worker(_load_json(v_dir / "vllm_worker_head.json"))
        vllm_wt = _extract_vllm_worker(_load_json(v_dir / "vllm_worker_tail.json"))

        if not head and not tail:
            continue

        print(f"\n  [{net_label} / {rps_label}]")
        print(f"  {'Stage':<32} {'MoLink':>20} {'vLLM':>20} {'Ratio':>22}")
        print("  " + "-" * 96)

        _compare_stage("Head serialize (pickle+tensor)", head.get("serialize_wall_ms", []), [])
        _compare_stage("Head gRPC send", head.get("grpc_send_ms", []), [])
        _compare_stage("Tail queue wait", tail.get("queue_wait_ms", []), [])
        _compare_stage("Tail deserialize", tail.get("deserialize_ms", []), [])
        _compare_stage("Tail compute", tail.get("compute_ms", []), vllm_wt.get("last_model_runner_ms", []))
        _compare_stage("Tail serialize result", tail.get("serialize_result_ms", []), [])
        _compare_stage("Tail gRPC send result", tail.get("grpc_send_ms", []), [])
        _compare_stage("Head wait for result", head.get("wait_result_ms", []), [])
        _compare_stage("Head deserialize result", head.get("deserialize_result_ms", []), [])
        _compare_stage("Total pipeline (head)", head.get("total_pipeline_ms", []), vllm_m.get("total_dag_ms", []))
        _compare_stage("  dag_execute", [], vllm_m.get("dag_execute_ms", []))
        _compare_stage("  ray_get", [], vllm_m.get("ray_get_ms", []))
        _compare_stage("vLLM first stage compute", [], vllm_wh.get("first_model_runner_ms", []))
        _compare_stage("vLLM last stage compute", [], vllm_wt.get("last_model_runner_ms", []))

    # 6. Bottleneck summary
    print("\n" + "=" * 100)
    print("  6. BOTTLENECK SUMMARY")
    print("=" * 100)

    for m_dir in molink_dirs:
        net_label = str(m_dir.parent.name)
        head = _extract_molink_head(_load_json(m_dir / "head_metrics.json"))
        tail = _extract_molink_tail(_load_json(m_dir / "tail_metrics.json"))
        if not head and not tail:
            continue
        total_pipeline = head.get("total_pipeline_ms", [])
        if not total_pipeline:
            continue
        total_avg = sum(total_pipeline) / len(total_pipeline)

        stages = {
            "head_serialize": head.get("serialize_wall_ms", []),
            "head_grpc_send": head.get("grpc_send_ms", []),
            "tail_queue_wait": tail.get("queue_wait_ms", []),
            "tail_deserialize": tail.get("deserialize_ms", []),
            "tail_compute": tail.get("compute_ms", []),
            "tail_serialize_result": tail.get("serialize_result_ms", []),
            "tail_grpc_send_result": tail.get("grpc_send_ms", []),
            "head_wait_result": head.get("wait_result_ms", []),
            "head_deser_result": head.get("deserialize_result_ms", []),
        }

        print(f"\n  [{net_label}] Pipeline time breakdown (avg total: {total_avg:.1f}ms):")
        stage_avgs = []
        for name, values in stages.items():
            if values:
                avg = sum(values) / len(values)
                pct = (avg / total_avg) * 100 if total_avg > 0 else 0
                stage_avgs.append((name, avg, pct))
        stage_avgs.sort(key=lambda x: -x[1])

        for name, avg, pct in stage_avgs:
            bar = "#" * int(pct / 2)
            print(f"    {name:<28} {avg:>8.1f}ms  ({pct:>5.1f}%)  {bar}")

        if stage_avgs:
            top = stage_avgs[0]
            print(f"\n  >>> TOP BOTTLENECK: {top[0]} = {top[1]:.1f}ms ({top[2]:.1f}% of pipeline)")

            suggestions = []
            if top[0] in ("head_serialize", "head_grpc_send") and top[2] > 15:
                suggestions.append("Serialization/gRPC overhead: consider shared memory or NCCL for tensor transfer")
            if top[0] == "tail_queue_wait" and top[2] > 10:
                suggestions.append("Tail queue wait: tail node bottlenecked, consider reducing serialize/compute time")
            if top[0] == "tail_deserialize" and top[2] > 10:
                suggestions.append("Tail deserialization: consider zero-copy tensor transfer or shared memory")
            if top[0] == "head_wait_result" and top[2] > 20:
                suggestions.append("Waiting for tail result: tail is slow, profile tail compute + serialization")
            if top[0] == "tail_compute" and top[2] > 50:
                suggestions.append("Tail compute is dominant: expected GPU-bound time, less room for optimization")
            if top[0] == "tail_grpc_send_result" and top[2] > 10:
                suggestions.append("Result gRPC send: consider compressing or reducing result size")

            if suggestions:
                print(f"\n  Suggestions:")
                for i, s in enumerate(suggestions, 1):
                    print(f"    {i}. {s}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "molink"

    if cmd == "analyze":
        analyze_molink(sys.argv[2] if len(sys.argv) > 2 else None)
        return

    if cmd == "bottleneck":
        if len(sys.argv) < 3:
            print("Usage: python bench_profile.py bottleneck <results_dir>")
            sys.exit(1)
        analyze_bottleneck(sys.argv[2])
        return

    if cmd == "molink":
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(RESULTS_ROOT, timestamp, "molink")
        os.makedirs(results_dir, exist_ok=True)
        log("=" * 60)
        log(" MoLink Pipeline Profiler")
        log(f" Input={INPUT_TOKENS} tok  Output=64 tok  RPS={RPS}")
        log(f" Results: {results_dir}")
        log("=" * 60)
        cleanup()
        url = await start_molink_profile()
        if not url:
            log("MoLink failed to start.")
            cleanup()
            return
        result = await run_benchmark(url, "molink", LOCAL_MODEL,
                                     f"{results_dir}/molink_profile_result.json",
                                     duration=10, output_tokens=64)
        log("Collecting profiles...")
        time.sleep(3)
        collect_molink_profiles(results_dir)
        cleanup()
        analyze_molink(results_dir)
        return

    if cmd == "vllm":
        from datetime import datetime
        # Reuse the timestamp that has molink data so both share one directory.
        # Fall back to creating a new timestamp if no molink data exists yet.
        molink_dir = _latest_results_dir("molink")
        if molink_dir:
            timestamp = os.path.basename(os.path.dirname(molink_dir))
        elif os.path.isdir(RESULTS_ROOT):
            existing = sorted(d for d in os.listdir(RESULTS_ROOT)
                              if os.path.isdir(os.path.join(RESULTS_ROOT, d)))
            timestamp = existing[-1] if existing else datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(RESULTS_ROOT, timestamp, "vllm")
        os.makedirs(results_dir, exist_ok=True)
        log("=" * 60)
        log(" vLLM Pipeline Profiler (PP=2, Ray+NCCL)")
        log(f" Input={INPUT_TOKENS}  Output=512  RPS={RPS}  Duration=30s")
        log(f" Results: {results_dir}")
        log("=" * 60)
        cleanup()
        url = await start_vllm_profile()
        if not url:
            log("vLLM failed to start.")
            cleanup()
            return
        result = await run_benchmark(url, "vllm", VLLM_COMMON_MODEL,
                                     f"{results_dir}/vllm_profile_result.json",
                                     duration=30, output_tokens=512)
        log("Collecting profiles...")
        time.sleep(3)
        collect_vllm_profiles(results_dir)
        cleanup()
        analyze_vllm_vs_molink(vllm_dir=results_dir)
        if result:
            out_path = f"{results_dir}/vllm_profile_result.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2, default=str)
        return

    print(f"Unknown command: {cmd}")
    print("Usage: python bench_profile.py [molink|vllm|analyze|bottleneck]")
    sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        cleanup()
