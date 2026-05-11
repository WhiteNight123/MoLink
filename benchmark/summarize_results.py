#!/usr/bin/env python3
"""
Summarize benchmark results into CSV tables and comparison plots.

Usage:
    python summarize_results.py /home/emnets-2/gxq/molink-measurement/MoLink/benchmark/results/20260509_234909
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 10,
    "legend.frameon": True,
    "legend.edgecolor": "0.85",
    "legend.fancybox": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.pad_inches": 0.1,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "lines.linewidth": 1.8,
    "lines.markersize": 6,
})

NET_DISPLAY = {
    "bw1gbit_delay10ms": "1 Gbps\n10 ms",
    "bw1gbit_delay20ms": "1 Gbps\n20 ms",
    "bw5gbit_delay10ms": "5 Gbps\n10 ms",
}

SYSTEM_STYLE = {
    "molink": {
        "color": "#2166AC",
        "marker": "o",
        "hatch": "",
        "label": "MoLink",
    },
    "vllm": {
        "color": "#B2182B",
        "marker": "D",
        "hatch": "///",
        "label": "vLLM",
    },
}

SUBPLOT_LABELS = ["(a)", "(b)", "(c)"]


def _add_ygrid(ax):
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    ax.set_axisbelow(True)


def load_results(results_dir: Path) -> dict:
    data: dict = {}
    for system_dir in sorted(results_dir.iterdir()):
        if not system_dir.is_dir():
            continue
        system = system_dir.name
        if system == "summary":
            continue
        data[system] = {}
        for net_dir in sorted(system_dir.iterdir()):
            if not net_dir.is_dir():
                continue
            net_label = net_dir.name
            data[system][net_label] = {}
            for rps_dir in sorted(net_dir.iterdir()):
                result_file = rps_dir / "result.json"
                if result_file.exists():
                    data[system][net_label][rps_dir.name] = json.loads(
                        result_file.read_text()
                    )
    return data


def _get_metrics(entry: dict) -> tuple[float, float, float]:
    """Return (ttft_avg_ms, tpop_avg_ms, throughput_tok_s)."""
    r = entry["results"]
    return (
        r["ttft_s"]["avg"] * 1000,
        r["tpop_s"]["avg"] * 1000,
        r["throughput_tokens_per_s"],
    )


def export_csv(data: dict, output_path: Path) -> None:
    rows = [
        "system,network,rps,throughput_tok_s,ttft_avg_ms,ttft_p50_ms,ttft_p99_ms,"
        "tpop_avg_ms,tpop_p50_ms,tpop_p99_ms,success,total_requests"
    ]

    def fmt_ms(val):
        return "N/A" if val is None else f"{val * 1000:.1f}"

    def fmt_tok(val):
        return "N/A" if val is None else f"{val:.1f}"

    for system in sorted(data.keys()):
        for net in sorted(data[system].keys()):
            for rps_label in sorted(
                data[system][net].keys(), key=lambda x: float(x.replace("rps", ""))
            ):
                entry = data[system][net][rps_label]
                r = entry["results"]
                cfg = entry["config"]
                rows.append(
                    f"{system},{net},{cfg['rps']},"
                    f"{fmt_tok(r.get('throughput_tokens_per_s'))},"
                    f"{fmt_ms(r.get('ttft_s', {}).get('avg'))},"
                    f"{fmt_ms(r.get('ttft_s', {}).get('p50'))},"
                    f"{fmt_ms(r.get('ttft_s', {}).get('p99'))},"
                    f"{fmt_ms(r.get('tpop_s', {}).get('avg'))},"
                    f"{fmt_ms(r.get('tpop_s', {}).get('p50'))},"
                    f"{fmt_ms(r.get('tpop_s', {}).get('p99'))},"
                    f"{r['successful_requests']},{cfg['total_requests']}"
                )

    output_path.write_text("\n".join(rows) + "\n")


def _rps_dir(rps: float) -> str:
    """Format RPS value to match directory naming: rps0.5, rps1, rps3 ..."""
    return f"rps{int(rps)}" if rps == int(rps) else f"rps{rps}"


def _collect_rps_series(data: dict, net: str) -> dict:
    """Collect {system: {rps: (ttft, tpop, tp)}} for a fixed network."""
    rps_set: set[float] = set()
    for sys_data in data.values():
        if net in sys_data:
            rps_set.update(float(k.replace("rps", "")) for k in sys_data[net])

    series: dict = {}
    for system in ("molink", "vllm"):
        series[system] = {}
        for rps in sorted(rps_set):
            entry = data.get(system, {}).get(net, {}).get(_rps_dir(rps))
            if entry:
                series[system][rps] = _get_metrics(entry)
    return sorted(rps_set), series


def _collect_net_series(data: dict, rps: float) -> tuple[list, dict]:
    """Collect {system: {net: (ttft, tpop, tp)}} for a fixed RPS."""
    net_set: set[str] = set()
    for sys_data in data.values():
        net_set.update(sys_data.keys())

    series: dict = {}
    for system in ("molink", "vllm"):
        series[system] = {}
        for net in sorted(net_set):
            entry = data.get(system, {}).get(net, {}).get(_rps_dir(rps))
            if entry:
                series[system][net] = _get_metrics(entry)
    return sorted(net_set), series


def plot_rps_comparison(data: dict, output_dir: Path, net: str = "bw1gbit_delay10ms") -> Path:
    rps_list, series = _collect_rps_series(data, net)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    ylabels = ["TTFT (ms)", "TPOP (ms)", "Throughput (tok/s)"]
    titles = ["Time to First Token", "Time per Output Token", "Throughput"]
    metric_idx = [0, 1, 2]

    handles, labels = [], []
    for ax, ylabel, title, midx, slbl in zip(axes, ylabels, titles, metric_idx, SUBPLOT_LABELS):
        for system in ("molink", "vllm"):
            if system not in series or not series[system]:
                continue
            sty = SYSTEM_STYLE[system]
            vals = [series[system].get(r, (None, None, None))[midx] for r in rps_list]
            line = ax.plot(
                rps_list, vals,
                marker=sty["marker"], color=sty["color"],
                label=sty["label"], linewidth=1.6, markersize=5.5,
                markeredgecolor="white", markeredgewidth=0.5,
            )
            if sty["label"] not in labels:
                handles.extend(line)
                labels.append(sty["label"])
        ax.set_xlabel("Requests per Second (RPS)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{slbl} {title}")
        ax.set_xticks(rps_list)
        _add_ygrid(ax)

    fig.tight_layout(rect=[0, 0.15, 1, 1])
    fig.legend(
        handles, labels,
        loc="lower center", ncol=2,
        bbox_to_anchor=(0.5, 0.01),
        frameon=True, edgecolor="0.85",
    )

    out_path = output_dir / f"rps_comparison_{net}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_network_comparison(data: dict, output_dir: Path, rps: float = 5) -> Path:
    net_list, series = _collect_net_series(data, rps)
    net_labels = [NET_DISPLAY.get(n, n) for n in net_list]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    ylabels = ["TTFT (ms)", "TPOP (ms)", "Throughput (tok/s)"]
    titles = ["Time to First Token", "Time per Output Token", "Throughput"]
    metric_idx = [0, 1, 2]

    x = np.arange(len(net_list))
    width = 0.3
    systems = [s for s in ("molink", "vllm") if s in series and series[s]]

    def _fmt_bar(val, midx):
        if midx == 0 and val >= 1000:
            return f"{val / 1000:.1f}k"
        if midx == 2:
            return f"{val:.0f}"
        return f"{val:.1f}"

    handles, labels = [], []
    for ax, ylabel, title, midx, slbl in zip(axes, ylabels, titles, metric_idx, SUBPLOT_LABELS):
        for i, system in enumerate(systems):
            sty = SYSTEM_STYLE[system]
            vals = [series[system].get(n, (None, None, None))[midx] for n in net_list]
            offset = (i - (len(systems) - 1) / 2) * width
            bars = ax.bar(
                x + offset, vals, width,
                color=sty["color"], alpha=0.82,
                edgecolor="0.2", linewidth=0.5,
                hatch=sty["hatch"],
            )
            if sty["label"] not in labels:
                handles.append(bars)
                labels.append(sty["label"])
            bar_labels = [_fmt_bar(v, midx) if v is not None else "" for v in vals]
            ax.bar_label(bars, labels=bar_labels, fontsize=7.5, padding=2)
        ax.set_xticks(x)
        ax.set_xticklabels(net_labels)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{slbl} {title}")
        _add_ygrid(ax)

    fig.tight_layout(rect=[0, 0.15, 1, 1])
    fig.legend(
        handles, labels,
        loc="lower center", ncol=2,
        bbox_to_anchor=(0.5, 0.01),
        frameon=True, edgecolor="0.85",
    )

    out_path = output_dir / f"network_comparison_rps{rps}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Summarize benchmark results")
    ap.add_argument("results_dir", help="Path to timestamped results directory")
    ap.add_argument("--net", default="bw1gbit_delay10ms", help="Network label for RPS comparison")
    ap.add_argument("--rps", type=float, default=5, help="RPS value for network comparison")
    args = ap.parse_args()

    rdir = Path(args.results_dir)
    if not rdir.exists():
        print(f"Directory not found: {rdir}", file=sys.stderr)
        sys.exit(1)

    data = load_results(rdir)
    if not data:
        print("No results found.", file=sys.stderr)
        sys.exit(1)

    output_dir = rdir / "summary"
    output_dir.mkdir(exist_ok=True)

    csv_path = output_dir / "results.csv"
    export_csv(data, csv_path)
    print(f"CSV saved: {csv_path}")

    fig1 = plot_rps_comparison(data, output_dir, net=args.net)
    print(f"RPS comparison: {fig1}")

    fig2 = plot_network_comparison(data, output_dir, rps=args.rps)
    print(f"Network comparison: {fig2}")


if __name__ == "__main__":
    main()
