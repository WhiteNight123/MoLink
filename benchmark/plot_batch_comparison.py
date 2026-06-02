#!/usr/bin/env python3
"""Plot batch-level timeline from instrumented MoLink/vLLM logs.

Combines head and tail logs to produce a single timeline visualization
showing compute bars, network transfers, and request events.

Usage:
    python plot_batch_comparison.py  --logdir results_batch/20260524_163412
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyArrowPatch, Rectangle

# ── Style ──────────────────────────────────────────────────────────────────

plt.style.use("seaborn-v0_8-whitegrid")
mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["font.size"] = 10
mpl.rcParams["axes.labelsize"] = 11
mpl.rcParams["axes.titlesize"] = 12
mpl.rcParams["legend.fontsize"] = 9
mpl.rcParams["figure.figsize"] = (16, 8)
mpl.rcParams["figure.dpi"] = 100

BATCH_COLORS = {
    0: "#4E79A7", 1: "#F28E2B", 2: "#E15759", 3: "#76B7B2",
    4: "#59A14F", 5: "#EDC948", 6: "#B07AA1", 7: "#FF9DA7",
    8: "#9C755F", 9: "#BAB0AC", 10: "#499894", 11: "#A25050",
}
REQUEST_COLORS = {"added": "#2E8B57", "finished": "#DC143C"}

# Resource layout
RESOURCE_LEVELS = {"Server1 Compute": 4.0, "Network Transfer": 2.5, "Server2 Compute": 1.0}
REQUEST_LEVEL = 5.0
COMPUTE_BAR_HEIGHT = 0.35
TRANSFER_BAR_HEIGHT = 0.25
ALPHA = 0.85


# ── Log parser ─────────────────────────────────────────────────────────────

# Multiprocessing log decoration prefixes added by vLLM's decorate_logs()
# e.g. "(EngineCore pid=12345) 0 1 compute starts (prefill) at ..."
# e.g. "(Worker pid=67890) 0 1 compute ends (decode) at ..."
_PREFIX_RE = re.compile(r"^\s*\([^)]+\s+pid=\d+\)\s*")


def _clean_line(line: str) -> str:
    """Strip multiprocess log decoration prefix if present."""
    return _PREFIX_RE.sub("", line).strip()


def parse_log_file(filename):
    """Parse plain-text instrumentation log. Returns (events, request_events)."""
    events = defaultdict(list)
    request_events = defaultdict(list)

    with open(filename, "r") as f:
        for raw_line in f:
            line = _clean_line(raw_line)
            if not line:
                continue

            # Request counted events
            m = re.match(r"(\d+)\s+requests\s+(finished|added)\s+at\s+([\d.]+)$", line)
            if m:
                count = int(m.group(1))
                action = m.group(2)
                ts = float(m.group(3))
                request_events[ts].append((f"{action}_count", count))
                continue

            # Per-request events
            m = re.match(r"request\s+([a-zA-Z0-9\-]+)\s+is added at\s+([\d.]+)$", line)
            if m:
                request_events[float(m.group(2))].append(("added", m.group(1)))
                continue
            m = re.match(r"request\s+([a-zA-Z0-9\-]+)\s+finished at\s+([\d.]+)$", line)
            if m:
                request_events[float(m.group(2))].append(("finished", m.group(1)))
                continue

            # Batch event: "0 1P0D compute starts (prefill) at 1234.56"
            # or legacy: "0 1 compute starts (prefill) at 1234.56"
            m = re.match(r"(\d+)(?:\s+(\d+(?:P\d+D)?))?\s+(.+?)\s+at\s+([\d.]+)$", line)
            if not m:
                continue

            batch_id = int(m.group(1))
            batch_spec = m.group(2)
            event_desc = m.group(3).strip()
            timestamp = float(m.group(4))

            # Extract stage
            stage_match = re.search(r"\(([^)]+)\)", event_desc)
            stage = stage_match.group(1) if stage_match else None

            # Parse batch spec: "3P2D" → prefill=3, decode=2; or legacy "5" → batch_size=5
            prefill_val = 0
            decode_val = 0
            batch_size_val = 0
            if batch_spec:
                xpd_match = re.match(r"(\d+)P(\d+)D", batch_spec)
                if xpd_match:
                    prefill_val = int(xpd_match.group(1))
                    decode_val = int(xpd_match.group(2))
                    batch_size_val = prefill_val + decode_val
                else:
                    batch_size_val = int(batch_spec)

            # Classify event type
            if "compute starts" in event_desc:
                etype = "compute_start"
            elif "compute ends" in event_desc:
                etype = "compute_end"
            elif "trans starts" in event_desc:
                etype = "trans_start"
            elif "trans ends" in event_desc:
                etype = "trans_end"
            elif "serialization starts" in event_desc:
                etype = "serial_start"
            elif "serialization ends" in event_desc:
                etype = "serial_end"
            elif "back to head" in event_desc:
                etype = "back_to_head"
            elif "sample starts" in event_desc:
                etype = "sample_start"
            elif "sample ends" in event_desc:
                etype = "sample_end"
            elif "recv" in event_desc:
                etype = "recv"
            else:
                continue

            events[batch_id].append((etype, timestamp, batch_size_val, stage, prefill_val, decode_val))

    return events, request_events


# ── Drawing functions ──────────────────────────────────────────────────────

def draw_compute_segments(events, resource_level, ax, norm_start, norm_end, base_time, is_top=True):
    for batch_id, batch_events in events.items():
        starts = [(ts, size, stage, prefill, decode) for etype, ts, size, stage, prefill, decode in batch_events if etype == "compute_start"]
        ends = [(ts, stage) for etype, ts, _, stage, _, _ in batch_events if etype == "compute_end"]
        starts.sort(key=lambda x: x[0])
        ends.sort(key=lambda x: x[0])

        for (start, batch_size, stage, prefill_val, decode_val), (end, _) in zip(starts, ends):
            norm_start_ts = start - base_time
            norm_end_ts = end - base_time

            if norm_end_ts > norm_start and norm_start_ts < norm_end:
                display_start = max(norm_start_ts, norm_start)
                display_end = min(norm_end_ts, norm_end)
                if display_end <= display_start:
                    continue

                duration_ms = (end - start) * 1000
                color = BATCH_COLORS.get(batch_id % len(BATCH_COLORS), "#1f77b4")

                y_bottom = resource_level - COMPUTE_BAR_HEIGHT / 2.0
                rect = Rectangle((display_start, y_bottom), display_end - display_start,
                                 COMPUTE_BAR_HEIGHT, facecolor=color, edgecolor=color,
                                 alpha=ALPHA, linewidth=0.5, zorder=2)
                if stage:
                    stage_lower = stage.lower()
                    if "prefill" in stage_lower:
                        rect.set_hatch("//")
                    elif "decode" in stage_lower:
                        rect.set_hatch("..")
                ax.add_patch(rect)

                mid_x = (display_start + display_end) / 2
                if prefill_val > 0 or decode_val > 0:
                    text = f"{prefill_val}P{decode_val}D\n{duration_ms:.0f}ms\n{stage or ''}"
                else:
                    text = f"bs={batch_size}\n{duration_ms:.0f}ms\n{stage or ''}"
                ann_y = resource_level - COMPUTE_BAR_HEIGHT / 2 - 0.2 if is_top else resource_level + COMPUTE_BAR_HEIGHT / 2 + 0.2
                ax.text(mid_x, ann_y, text, ha="center", va="center", fontsize=8,
                        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray", boxstyle="round,pad=0.2"))


def draw_transfers(s1_events, s2_events, ax, norm_start, norm_end, base_time):
    """Match trans_start in one server's log to recv/back_to_head in the other."""
    for batch_id in set(list(s1_events.keys()) + list(s2_events.keys())):
        # Server1 → Server2: trans_start (s1) → recv (s2)
        s1_trans = [ts for etype, ts, *_ in s1_events.get(batch_id, []) if etype == "trans_start"]
        s2_recv = [ts for etype, ts, *_ in s2_events.get(batch_id, []) if etype == "recv"]
        s1_trans.sort()
        s2_recv.sort()

        for start, end in zip(s1_trans, s2_recv):
            norm_s = start - base_time
            norm_e = end - base_time
            if norm_e > norm_start and norm_s < norm_end:
                display_start = max(norm_s, norm_start)
                display_end = min(norm_e, norm_end)
                if display_end <= display_start:
                    continue
                duration_ms = (end - start) * 1000
                color = BATCH_COLORS.get(batch_id % len(BATCH_COLORS), "#1f77b4")
                y_bottom = RESOURCE_LEVELS["Network Transfer"] - TRANSFER_BAR_HEIGHT / 2.0
                rect = Rectangle((display_start, y_bottom), display_end - display_start,
                                 TRANSFER_BAR_HEIGHT, facecolor=color, edgecolor=color,
                                 alpha=ALPHA, linewidth=0.5, zorder=1)
                ax.add_patch(rect)
                mid_x = (display_start + display_end) / 2
                ax.text(mid_x, y_bottom - 0.15, f"→Server2\n{duration_ms:.0f}ms",
                        ha="center", va="center", fontsize=8,
                        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray", boxstyle="round,pad=0.2"))

        # Server2 → Server1: trans_start (s2) → back_to_head (s1)
        s2_trans = [ts for etype, ts, *_ in s2_events.get(batch_id, []) if etype == "trans_start"]
        s1_back = [ts for etype, ts, *_ in s1_events.get(batch_id, []) if etype == "back_to_head"]
        s2_trans.sort()
        s1_back.sort()

        for start, end in zip(s2_trans, s1_back):
            norm_s = start - base_time
            norm_e = end - base_time
            if norm_e > norm_start and norm_s < norm_end:
                display_start = max(norm_s, norm_start)
                display_end = min(norm_e, norm_end)
                if display_end <= display_start:
                    continue
                duration_ms = (end - start) * 1000
                color = BATCH_COLORS.get(batch_id % len(BATCH_COLORS), "#1f77b4")
                y_bottom = RESOURCE_LEVELS["Network Transfer"] - TRANSFER_BAR_HEIGHT / 2.0
                rect = Rectangle((display_start, y_bottom), display_end - display_start,
                                 TRANSFER_BAR_HEIGHT, facecolor=color, edgecolor=color,
                                 alpha=ALPHA * 0.85, linewidth=0.5, zorder=1)
                ax.add_patch(rect)
                mid_x = (display_start + display_end) / 2
                ax.text(mid_x, y_bottom - 0.15, f"→Server1\n{duration_ms:.0f}ms",
                        ha="center", va="center", fontsize=8,
                        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray", boxstyle="round,pad=0.2"))


def _merge_events(request_events_dict, base_time, merge_window=0.1):
    """Merge per-request events whose timestamps fall within *merge_window* seconds."""
    # Collect all (norm_ts, event_type) pairs
    raw = []
    for ts, evts in request_events_dict.items():
        for ev in evts:
            etype = ev[0]
            if etype.endswith("_count"):
                action = etype.replace("_count", "")
                raw.append((ts - base_time, action, int(ev[1])))
            else:
                raw.append((ts - base_time, ev[0], 1))
    raw.sort()

    merged = defaultdict(lambda: defaultdict(int))  # norm_ts -> {type: count}
    bucket_ts = None
    for norm_ts, etype, count in raw:
        if bucket_ts is None or norm_ts - bucket_ts > merge_window:
            bucket_ts = norm_ts
        merged[bucket_ts][etype] += count
    return merged


def draw_request_events(request_events_dict, ax, norm_start, norm_end, base_time):
    merged = _merge_events(request_events_dict, base_time)

    for norm_ts, event_counts in merged.items():
        if not (norm_start <= norm_ts <= norm_end):
            continue
        for event_type, count in event_counts.items():
            color = REQUEST_COLORS.get(event_type, "#000000")
            event_name = "arrives" if event_type == "added" else "finishes"
            arrow = FancyArrowPatch((norm_ts, REQUEST_LEVEL),
                                   (norm_ts, RESOURCE_LEVELS["Server1 Compute"] + 0.2),
                                   arrowstyle="->", mutation_scale=15, color=color,
                                   linewidth=2, alpha=0.8)
            ax.add_patch(arrow)
            ax.text(norm_ts, REQUEST_LEVEL + 0.15, f"{count} req {event_name}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold", color=color,
                    bbox=dict(facecolor="white", alpha=0.9, edgecolor=color, boxstyle="round,pad=0.3"))


# ── Main plotting ──────────────────────────────────────────────────────────

def plot_timeline(s1_events, s2_events, request_events_dict, output_path, title, time_range=None):
    all_ts = []
    for evts in [s1_events, s2_events]:
        for batch_evts in evts.values():
            for _, ts, *_ in batch_evts:
                all_ts.append(ts)
    for ts in request_events_dict:
        all_ts.append(ts)

    if not all_ts:
        print("ERROR: No events parsed!", file=sys.stderr)
        return

    base_time = min(all_ts)

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.set_facecolor("#F8F9FA")
    ax.set_facecolor("#FFFFFF")

    # Determine time range
    if time_range is not None:
        norm_start, norm_end = time_range
    else:
        all_norm = [t - base_time for t in all_ts]
        norm_start = min(all_norm)
        norm_end = max(all_norm)
        margin = (norm_end - norm_start) * 0.05
        norm_start -= margin
        norm_end += margin

    # Draw
    draw_compute_segments(s1_events, RESOURCE_LEVELS["Server1 Compute"], ax, norm_start, norm_end, base_time, is_top=True)
    draw_compute_segments(s2_events, RESOURCE_LEVELS["Server2 Compute"], ax, norm_start, norm_end, base_time, is_top=False)
    draw_transfers(s1_events, s2_events, ax, norm_start, norm_end, base_time)
    draw_request_events(request_events_dict, ax, norm_start, norm_end, base_time)

    # Axis setup
    ax.set_yticks(list(RESOURCE_LEVELS.values()) + [REQUEST_LEVEL])
    ax.set_yticklabels(list(RESOURCE_LEVELS.keys()) + ["Request Events"],
                       fontsize=12, fontweight="bold")
    ax.set_xlabel(f"Time (seconds from {base_time:.3f})", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlim(norm_start, norm_end)
    ax.set_ylim(0.3, 5.5)
    ax.grid(True, axis="x", linestyle="--", alpha=0.6)

    for y in RESOURCE_LEVELS.values():
        ax.axhline(y=y, color="gray", alpha=0.3, linewidth=0.5)
    ax.axhline(y=REQUEST_LEVEL, color="gray", alpha=0.3, linewidth=0.5, linestyle=":")

    for spine in ax.spines.values():
        spine.set_edgecolor("#DDDDDD")

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1, top=0.92, left=0.08, right=0.95)

    # Save
    for fmt in ["pdf", "png"]:
        fig.savefig(f"{output_path}.{fmt}", bbox_inches="tight", dpi=150)
        print(f"Saved {output_path}.{fmt}")

    plt.close(fig)


def _plot_molink(logdir, outdir, time_range):
    head_log = logdir / "head.log"
    tail_log = logdir / "tail.log"
    if not head_log.exists() or not tail_log.exists():
        print(f"ERROR: Missing log files in {logdir}", file=sys.stderr)
        return

    print(f"Parsing head log: {head_log}")
    s1_events, _ = parse_log_file(str(head_log))
    print(f"Parsing tail log: {tail_log}")
    s2_events, _ = parse_log_file(str(tail_log))

    request_events = defaultdict(list)
    _, req1 = parse_log_file(str(head_log))
    for ts, evts in req1.items():
        request_events[ts].extend(evts)

    title = "MoLink Pipeline Parallelism — Batch-Level Timeline"
    if time_range:
        title += f" (t={time_range[0]:.1f}s–{time_range[1]:.1f}s)"
    plot_timeline(s1_events, s2_events, request_events, str(outdir / "molink_timeline"), title, time_range)


def _plot_vllm(logdir, outdir, time_range):
    head_log = logdir / "head.log"
    tail_log = logdir / "tail.log"
    if not head_log.exists() or not tail_log.exists():
        print(f"ERROR: Missing head.log or tail.log in {logdir}", file=sys.stderr)
        return

    print(f"Parsing vLLM head log: {head_log}")
    s1_events, req1 = parse_log_file(str(head_log))
    print(f"Parsing vLLM tail log: {tail_log}")
    s2_events, _ = parse_log_file(str(tail_log))

    request_events = defaultdict(list)
    for ts, evts in req1.items():
        request_events[ts].extend(evts)
    vllm_log = logdir / "vllm.log"
    if vllm_log.exists():
        _, req_events = parse_log_file(str(vllm_log))
        for ts, evts in req_events.items():
            request_events[ts].extend(evts)

    title = "vLLM Pipeline Parallelism — Batch-Level Timeline"
    if time_range:
        title += f" (t={time_range[0]:.1f}s–{time_range[1]:.1f}s)"
    plot_timeline(s1_events, s2_events, request_events, str(outdir / "vllm_timeline"), title, time_range)


_SYSTEM_HANDLERS = {
    "molink": _plot_molink,
    "vllm": _plot_vllm,
}


def main():
    ap = argparse.ArgumentParser(description="Plot batch-level timeline")
    ap.add_argument("--system", choices=["molink", "vllm"],
                    help="Plot a specific system. If omitted, auto-detects molink/ and vllm/ subdirectories.")
    ap.add_argument("--logdir", required=True,
                    help="Results directory. Either contains head.log/tail.log directly, "
                         "or has molink/ and vllm/ subdirectories with logs/ inside.")
    ap.add_argument("--output",
                    help="Output directory for plots. Defaults to --logdir/plots.")
    ap.add_argument("--time-range", default=None,
                    help="Plot time window in seconds from start, e.g. '1-3' plots 1s to 3s")
    args = ap.parse_args()

    time_range = None
    if args.time_range:
        parts = args.time_range.split("-", 1)
        time_range = (float(parts[0]), float(parts[1]))

    logdir = Path(args.logdir).resolve()

    if args.system:
        # Explicit single-system mode: logdir points at the log directory directly
        outdir = Path(args.output) if args.output else logdir / "plots"
        outdir.mkdir(parents=True, exist_ok=True)
        _SYSTEM_HANDLERS[args.system](logdir, outdir, time_range)
    else:
        # Auto-detect mode: logdir contains molink/ and/or vllm/ subdirectories
        outdir = Path(args.output) if args.output else logdir / "plots"
        outdir.mkdir(parents=True, exist_ok=True)
        found = False
        for name, handler in _SYSTEM_HANDLERS.items():
            sys_logdir = logdir / name / "logs"
            if sys_logdir.is_dir():
                print(f"\n=== Detected {name} logs at {sys_logdir} ===")
                handler(sys_logdir, outdir, time_range)
                found = True
        if not found:
            print(f"ERROR: No molink/logs/ or vllm/logs/ found under {logdir}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
