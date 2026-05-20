# Benchmark

Performance benchmarks comparing MoLink and vLLM pipeline parallelism (PP=2) on Qwen3-14B.

## Quick Start

```bash
# Docker-based benchmark (single machine, multi-container)
python bench_docker.py                    # all systems
python bench_docker.py molink             # MoLink only
python bench_docker.py vllm               # vLLM only

# Distributed benchmark (two physical machines)
python bench_distributed.py               # all systems

# Profiling with comparative analysis
python bench_profile.py molink            # MoLink profiling
python bench_profile.py analyze DIR       # analyze existing results
```

## Scripts

| Script | Description |
|---|---|
| `bench_docker.py` | Docker-based benchmark with tc/netem network shaping. Runs MoLink v0.11, v0.19, and vLLM PP=2 in separate containers. Sweeps RPS and network conditions (bandwidth, latency). |
| `bench_distributed.py` | Cross-machine benchmark: head on local RTX 4090, tail on remote RTX 3090. MoLink uses gRPC; vLLM uses Ray. |
| `bench_profile.py` | Unified profiling with timing breakdown and bottleneck analysis. Supports both cross-host and Docker result analysis. |
| `benchmark_client.py` | HTTP load generator measuring throughput (tokens/s), TTFT, and TPOT. Supports MoLink (`/generate`) and vLLM (`/v1/completions`) APIs. |
| `compare_molink_vllm.py` | Matched A/B comparison between MoLink and vLLM with identical parameters. |
| `summarize_results.py` | Aggregates raw results into CSV tables and comparison plots. |
| `bench_utils.py` | Shared constants, SSH helpers, process management, and health checks. |

## Key Options

| Option | `bench_docker.py` | `bench_distributed.py` |
|---|---|---|
| `--pp N` | Pipeline parallel size (default: 2) | — |
| `--tp N` | Tensor parallel size (default: 1) | — |
| `--rps 3 5` | RPS values (default: 3) | — |
| `--network X` | Network condition, e.g. `1gbit,5ms` | — |
| `--duration N` | Seconds per run (default: 30) | — |
| `--gpus N,N` | GPU device IDs | — |

## Test Matrix

- **Systems**: MoLink v0.11, MoLink v0.19, vLLM PP=2
- **RPS**: 0.5, 1, 3, 5
- **Network**: none, 500mbit/5ms, 1gbit/5ms, 1gbit/10ms, 1gbit/20ms, 5gbit/10ms
- **Model**: Qwen3-14B, 1024 input / 512 output tokens
- **GPU**: RTX 4090 (local head), RTX 3090 (remote tail)

## Output

Results are stored in timestamped directories under `results/` (Docker), `results_distributed/` (distributed), and `results_profile/` (profiling). Each run produces:

- `config.json` — benchmark parameters
- `molink/` or `vllm/` — per-system logs and metrics JSON
- GPU utilization charts (where applicable)

Aggregated figures and the comprehensive report live in `final_report/`.
