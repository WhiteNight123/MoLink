#!/usr/bin/env python3
"""Analyze MoLink vs vLLM performance bottlenecks from benchmark results.

Reads timing metrics collected from both systems and produces a comparative
breakdown showing where MoLink spends disproportionate time.

Usage:
    python analyze_bottleneck.py <results_dir>
    python analyze_bottleneck.py MoLink/benchmark/results/20260517_120000
"""

import json
import sys
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0
    s = sorted(data)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] * (c - k) + s[c] * (k - f)


def extract_molink_head_metrics(metrics_data: dict) -> dict[str, list[float]]:
    """Extract head pipeline timing from MoLink metrics."""
    if not metrics_data:
        return {}
    service_metrics = metrics_data.get("service_metrics", [])
    result: dict[str, list[float]] = {}

    for m in service_metrics:
        if m.get("type") != "head_pipeline":
            continue
        for key in [
            "push_intermediate_ms", "wait_result_ms",
            "deserialize_result_ms", "total_pipeline_ms",
            "pickle_ms", "tensor_serialize_ms", "total_serialize_ms",
            "grpc_send_ms", "serialize_wall_ms",
            "result_bytes", "serialized_bytes",
        ]:
            if key in m:
                result.setdefault(key, []).append(m[key])

    # Also collect head_compute metrics
    for m in service_metrics:
        if m.get("type") != "head_compute":
            continue
        for key in ["compute_ms", "num_tokens"]:
            if key in m:
                result.setdefault(f"head_{key}", []).append(m[key])

    return result


def extract_molink_tail_metrics(metrics_data: dict) -> dict[str, list[float]]:
    """Extract tail worker timing from MoLink metrics."""
    if not metrics_data:
        return {}
    service_metrics = metrics_data.get("service_metrics", [])
    result: dict[str, list[float]] = {}

    for m in service_metrics:
        if m.get("type") != "worker_step":
            continue
        for key in [
            "queue_wait_ms", "deserialize_ms", "compute_ms",
            "serialize_result_ms", "grpc_send_ms", "push_ms",
            "total_ms", "recv_bytes", "result_bytes",
            "compute_lock_wait_ms",
        ]:
            if key in m:
                result.setdefault(key, []).append(m[key])

    return result


def extract_vllm_metrics(metrics_data: dict) -> dict[str, list[float]]:
    """Extract vLLM Ray DAG timing from metrics."""
    if not metrics_data:
        return {}
    all_metrics = metrics_data.get("metrics", [])
    result: dict[str, list[float]] = {}

    for m in all_metrics:
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


def extract_vllm_worker_metrics(metrics_data: dict) -> dict[str, list[float]]:
    """Extract vLLM worker-side timing from per-container metrics."""
    if not metrics_data:
        return {}
    worker_metrics = metrics_data.get("worker_metrics", [])
    result: dict[str, list[float]] = {}

    for m in worker_metrics:
        if m.get("type") != "vllm_worker":
            continue
        for key in [
            "model_runner_ms", "sample_tokens_ms", "total_ms",
            "num_tokens",
        ]:
            if key in m:
                prefix = "last_" if m.get("is_last_pp_rank") else "first_"
                result.setdefault(f"{prefix}{key}", []).append(m[key])

    return result


def fmt_ms(values: list[float]) -> str:
    if not values:
        return "N/A"
    avg = sum(values) / len(values)
    p50 = percentile(values, 50)
    p99 = percentile(values, 99)
    return f"avg={avg:.1f}ms  p50={p50:.1f}ms  p99={p99:.1f}ms  n={len(values)}"


def fmt_avg(values: list[float]) -> str:
    if not values:
        return "N/A"
    return f"{sum(values) / len(values):.1f}ms"


def compare_stage(name: str, molink_ms: list[float], vllm_ms: list[float]):
    """Print a comparison row for a pipeline stage."""
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

    print(f"  {name:<32} {fmt_avg(molink_ms):>20} {fmt_avg(vllm_ms):>20} {ratio:>22}")


def analyze(results_dir: Path):
    print("=" * 100)
    print(f"  MoLink Performance Bottleneck Analysis")
    print(f"  Results: {results_dir}")
    print("=" * 100)

    config = load_json(results_dir / "config.json")
    if config:
        print(f"\n  Config: PP={config.get('pp_size')} TP={config.get('tp_size')} "
              f"input={config.get('input_tokens')} output={config.get('output_tokens')} "
              f"RPS={config.get('rps_values')} duration={config.get('duration_s')}s")

    # Find all subdirectories with results
    systems_found = set()
    for d in sorted(results_dir.iterdir()):
        if d.is_dir():
            systems_found.add(d.name)

    molink_dirs = sorted(results_dir.glob("molink/*/rps*"))
    vllm_dirs = sorted(results_dir.glob("vllm/*/rps*"))

    if not molink_dirs and not vllm_dirs:
        print("\n  No benchmark results found. Run ./run_benchmark.sh first.")
        return

    # ---- 1. Client-side metrics comparison ----
    print("\n" + "=" * 100)
    print("  1. CLIENT-SIDE COMPARISON (TTFT / Throughput)")
    print("=" * 100)
    print(f"  {'System':<12} {'Network':<20} {'RPS':<6} "
          f"{'Throughput':>14} {'TTFT avg':>12} {'TTFT p99':>12} {'Requests':>10}")
    print("  " + "-" * 88)

    for dirs, system_label in [(molink_dirs, "MoLink"), (vllm_dirs, "vLLM")]:
        for d in dirs:
            result = load_json(d / "result.json")
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
            print(f"  {system_label:<12} {net_label:<20} {rps:<6} "
                  f"{throughput:>12.1f} t/s {ttft_avg:>10.1f}ms {ttft_p99:>10.1f}ms "
                  f"{ok:>5} OK / {fail} fail")

    # ---- 2. MoLink pipeline breakdown ----
    print("\n" + "=" * 100)
    print("  2. MoLink HEAD NODE Pipeline Breakdown")
    print("=" * 100)

    for d in molink_dirs:
        head_metrics_data = load_json(d / "head_metrics.json")
        if not head_metrics_data:
            continue
        head = extract_molink_head_metrics(head_metrics_data)
        if not head:
            continue

        net_label = str(d.parent.name)
        print(f"\n  [{net_label}] Head pipeline metrics ({len(head.get('total_pipeline_ms', []))} steps):")
        print(f"  {'Stage':<32} {'Timing':>60}")
        print("  " + "-" * 92)

        for key in [
            "head_compute_ms",
            "serialize_wall_ms",
            "  pickle_ms",
            "  tensor_serialize_ms",
            "grpc_send_ms",
            "push_intermediate_ms",
            "wait_result_ms",
            "deserialize_result_ms",
            "total_pipeline_ms",
        ]:
            display = key
            actual_key = key.strip()
            values = head.get(actual_key, [])
            indent = "  " if key.startswith("  ") else ""
            print(f"  {indent}{display:<30} {fmt_ms(values)}")

        if head.get("serialized_bytes"):
            avg_bytes = sum(head["serialized_bytes"]) / len(head["serialized_bytes"])
            print(f"  {'  avg serialized bytes':<30} {avg_bytes / 1024 / 1024:.1f} MB")
        if head.get("result_bytes"):
            avg_bytes = sum(head["result_bytes"]) / len(head["result_bytes"])
            print(f"  {'  avg result bytes':<30} {avg_bytes / 1024:.1f} KB")

    # ---- 3. MoLink tail node breakdown ----
    print("\n" + "=" * 100)
    print("  3. MoLink TAIL NODE Worker Breakdown")
    print("=" * 100)

    for d in molink_dirs:
        tail_metrics_data = load_json(d / "tail_metrics.json")
        if not tail_metrics_data:
            continue
        tail = extract_molink_tail_metrics(tail_metrics_data)
        if not tail:
            continue

        net_label = str(d.parent.name)
        print(f"\n  [{net_label}] Tail worker metrics ({len(tail.get('total_ms', []))} steps):")
        print(f"  {'Stage':<32} {'Timing':>60}")
        print("  " + "-" * 92)

        for key in [
            "queue_wait_ms",
            "deserialize_ms",
            "compute_lock_wait_ms",
            "compute_ms",
            "serialize_result_ms",
            "grpc_send_ms",
            "push_ms",
            "total_ms",
        ]:
            values = tail.get(key, [])
            print(f"  {key:<32} {fmt_ms(values)}")

        if tail.get("recv_bytes"):
            avg_bytes = sum(tail["recv_bytes"]) / len(tail["recv_bytes"])
            print(f"  {'  avg recv bytes':<30} {avg_bytes / 1024 / 1024:.1f} MB")

    # ---- 4. vLLM Ray DAG breakdown ----
    print("\n" + "=" * 100)
    print("  4. vLLM Ray DAG Breakdown")
    print("=" * 100)

    for d in vllm_dirs:
        vllm_metrics_data = load_json(d / "vllm_metrics.json")
        if not vllm_metrics_data:
            continue
        vllm_m = extract_vllm_metrics(vllm_metrics_data)
        if not vllm_m:
            continue

        net_label = str(d.parent.name)
        print(f"\n  [{net_label}] vLLM Ray DAG metrics ({len(vllm_m.get('total_dag_ms', []))} steps):")
        print(f"  {'Stage':<32} {'Timing':>60}")
        print("  " + "-" * 92)

        for key in [
            "execute_model_gap_ms",
            "dag_execute_ms",
            "ray_get_ms",
            "sample_tokens_total_ms",
            "total_dag_ms",
        ]:
            values = vllm_m.get(key, [])
            print(f"  {key:<32} {fmt_ms(values)}")

        # Worker-side metrics
        for worker_file, stage_label in [
            ("vllm_worker_head.json", "PP Stage 0 (head)"),
            ("vllm_worker_tail.json", "PP Stage 1 (tail)"),
            ("vllm_worker_middle.json", "PP Stage 1 (middle)"),
        ]:
            worker_data = load_json(d / worker_file)
            if not worker_data:
                continue
            wm = extract_vllm_worker_metrics(worker_data)
            if not wm:
                continue
            # Determine which prefix to use based on stage
            prefix = "first_" if "Stage 0" in stage_label else "last_"
            print(f"\n    [{stage_label}] Worker metrics ({len(wm.get(f'{prefix}total_ms', []))} steps):")
            for wkey in ["model_runner_ms", "sample_tokens_ms", "total_ms"]:
                full_key = f"{prefix}{wkey}"
                values = wm.get(full_key, [])
                if values:
                    print(f"      {wkey:<28} {fmt_ms(values)}")

    # ---- 5. Head-to-head comparison ----
    print("\n" + "=" * 100)
    print("  5. CROSS-NODE COMMUNICATION COMPARISON (MoLink vs vLLM)")
    print("=" * 100)

    # Match molink and vllm dirs by network label and RPS
    for m_dir in molink_dirs:
        net_label = str(m_dir.parent.name)
        rps_label = str(m_dir.name)
        v_dir = results_dir / "vllm" / net_label / rps_label

        if not v_dir.exists():
            continue

        head_data = load_json(m_dir / "head_metrics.json")
        tail_data = load_json(m_dir / "tail_metrics.json")
        vllm_data = load_json(v_dir / "vllm_metrics.json")

        head = extract_molink_head_metrics(head_data)
        tail = extract_molink_tail_metrics(tail_data)
        vllm_m = extract_vllm_metrics(vllm_data)

        # Load vLLM worker metrics
        vllm_worker_head_data = load_json(v_dir / "vllm_worker_head.json")
        vllm_worker_tail_data = load_json(v_dir / "vllm_worker_tail.json")
        vllm_wm_head = extract_vllm_worker_metrics(vllm_worker_head_data)
        vllm_wm_tail = extract_vllm_worker_metrics(vllm_worker_tail_data)

        if not head and not tail:
            continue

        # MoLink total cross-node: serialize + gRPC_send (head) + queue_wait + deser + compute + serialize_result + gRPC_send (tail) + wait_result + deser_result
        molink_serialize = head.get("serialize_wall_ms", [])
        molink_grpc_send = head.get("grpc_send_ms", [])
        molink_wait_result = head.get("wait_result_ms", [])
        molink_deser_result = head.get("deserialize_result_ms", [])
        molink_total_pipeline = head.get("total_pipeline_ms", [])

        tail_deser = tail.get("deserialize_ms", [])
        tail_queue_wait = tail.get("queue_wait_ms", [])
        tail_compute = tail.get("compute_ms", [])
        tail_serialize_result = tail.get("serialize_result_ms", [])
        tail_grpc_send = tail.get("grpc_send_ms", [])

        vllm_dag_total = vllm_m.get("total_dag_ms", [])
        vllm_dag_execute = vllm_m.get("dag_execute_ms", [])
        vllm_ray_get = vllm_m.get("ray_get_ms", [])

        # vLLM worker compute times
        vllm_first_compute = vllm_wm_head.get("first_model_runner_ms", [])
        vllm_last_compute = vllm_wm_tail.get("last_model_runner_ms", [])
        vllm_last_sample = vllm_wm_tail.get("last_sample_tokens_ms", [])

        print(f"\n  [{net_label} / {rps_label}]")
        print(f"  {'Stage':<32} {'MoLink':>20} {'vLLM':>20} {'Ratio':>22}")
        print("  " + "-" * 96)

        compare_stage("Head serialize (pickle+tensor)", molink_serialize, [])
        compare_stage("Head gRPC send", molink_grpc_send, [])
        compare_stage("Tail queue wait", tail_queue_wait, [])
        compare_stage("Tail deserialize", tail_deser, [])
        compare_stage("Tail compute", tail_compute, vllm_last_compute)
        compare_stage("Tail serialize result", tail_serialize_result, [])
        compare_stage("Tail gRPC send result", tail_grpc_send, [])
        compare_stage("Head wait for result", molink_wait_result, [])
        compare_stage("Head deserialize result", molink_deser_result, [])
        compare_stage("Total pipeline (head)", molink_total_pipeline, vllm_dag_total)
        compare_stage("  dag_execute", [], vllm_dag_execute)
        compare_stage("  ray_get", [], vllm_ray_get)
        compare_stage("vLLM first stage compute", [], vllm_first_compute)
        compare_stage("vLLM last stage compute", [], vllm_last_compute)
        compare_stage("vLLM last stage sample_tokens", [], vllm_last_sample)

    # ---- 6. Bottleneck summary ----
    print("\n" + "=" * 100)
    print("  6. BOTTLENECK SUMMARY")
    print("=" * 100)

    for m_dir in molink_dirs:
        net_label = str(m_dir.parent.name)
        head_data = load_json(m_dir / "head_metrics.json")
        tail_data = load_json(m_dir / "tail_metrics.json")

        head = extract_molink_head_metrics(head_data)
        tail = extract_molink_tail_metrics(tail_data)

        if not head and not tail:
            continue

        total_pipeline = head.get("total_pipeline_ms", [])
        if not total_pipeline:
            continue

        total_avg = sum(total_pipeline) / len(total_pipeline)

        # Calculate percentages of total pipeline time
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

        # Sort by average time descending
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

        # Identify top bottleneck
        if stage_avgs:
            top = stage_avgs[0]
            print(f"\n  >>> TOP BOTTLENECK: {top[0]} = {top[1]:.1f}ms ({top[2]:.1f}% of pipeline)")

            # Suggest optimizations
            suggestions = []
            if top[0] in ("head_serialize", "head_grpc_send") and top[2] > 15:
                suggestions.append("Serialization/gRPC overhead: consider using shared memory or NCCL for tensor transfer")
            if top[0] == "tail_queue_wait" and top[2] > 10:
                suggestions.append("Tail queue wait: tail node is bottlenecked, consider reducing serialize/compute time")
            if top[0] == "tail_deserialize" and top[2] > 10:
                suggestions.append("Tail deserialization: consider zero-copy tensor transfer or shared memory")
            if top[0] == "head_wait_result" and top[2] > 20:
                suggestions.append("Waiting for tail result: tail is slow, profile tail compute + serialization")
            if top[0] == "tail_compute" and top[2] > 50:
                suggestions.append("Tail compute is dominant: this is expected GPU-bound time, less room for optimization")
            if top[0] == "tail_grpc_send_result" and top[2] > 10:
                suggestions.append("Result gRPC send: consider compressing or reducing result size")

            if suggestions:
                print(f"\n  Suggestions:")
                for i, s in enumerate(suggestions, 1):
                    print(f"    {i}. {s}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_bottleneck.py <results_dir>")
        print("Example: python analyze_bottleneck.py MoLink/benchmark/results/20260517_120000")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist")
        sys.exit(1)

    analyze(results_dir)
