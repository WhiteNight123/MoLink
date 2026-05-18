# MoLink Benchmark

对比测试 MoLink 与 vLLM 在流水线并行（PP=2）下的推理性能。

## 目录结构

```
benchmark/
├── bench_utils.py           # 公共基础设施：常量、log、run_local/remote、cleanup、health_check
├── benchmark_client.py      # HTTP 压测客户端（被各脚本调用）
│
├── bench_docker.py           # Docker 方案（单机多容器，tc/netem 网络模拟，MoLink/vLLM/v0.11 对比）
├── bench_distributed.py      # 跨机分布式测试：MoLink vs vLLM PP=2
├── bench_profile.py          # 插桩 profiling（MoLink/vLLM/analyze/bottleneck 子命令）
│
├── summarize_results.py      # 结果汇总 + CSV 导出 + 图表
└── results*/                 # 测试结果输出目录
```

## 快速开始

### Docker 容器测试（单机）

```bash
python bench_docker.py                    # MoLink + vLLM
python bench_docker.py molink             # 仅 MoLink
python bench_docker.py vllm               # 仅 vLLM
python bench_docker.py --pp 3             # PP=3
python bench_docker.py --pp 2 --tp 2      # PP=2 + TP=2
python bench_docker.py --gpus 1,2         # 指定 GPU 设备

# MoLink v0.11 vs v0.19 对比
python bench_docker.py molink011 molink019
python bench_docker.py v011               # 简写，等价于上面
python bench_docker.py molink011          # 仅 v0.11
```

### 跨机分布式测试

```bash
python bench_distributed.py molink   # MoLink: 本机 head + 远程 tail
python bench_distributed.py vllm     # vLLM PP=2 via Ray
python bench_distributed.py          # 两者都测
```

### Profiling 分析

```bash
python bench_profile.py molink           # 运行 MoLink profiling benchmark
python bench_profile.py vllm             # 运行 vLLM profiling + MoLink 对比
python bench_profile.py analyze [DIR]    # 分析已有 profile 文件
python bench_profile.py bottleneck DIR   # 分析 Docker benchmark 结果
```

## 方案对比

| | Docker 方案 | 分布式方案 |
|---|---|---|
| 脚本 | `bench_docker.py` | `bench_distributed.py` |
| 运行环境 | Docker 容器（单机） | 裸金属（跨机） |
| 网络模拟 | tc/netem 可调带宽/延迟 | 真实 1Gbps 以太网 |
| MoLink 部署 | 容器间 gRPC | 本机↔远程 gRPC |
| vLLM 部署 | 容器间 Ray PP=2 | Ray PP=2 跨机 |
| 用途 | 控制变量、多网络条件对比 | 真实跨机性能验证 |

## 前提条件

### Docker 方案

- Docker 镜像 `molink:0.2`，网络 `molink`（subnet `172.26.0.0/16`）
- 模型路径 `/gxq/Qwen3-14B` 在容器内可访问（挂载自 `/home/emnets-2/gxq`）
- 至少 2 块空闲 GPU（各 ≥16GB）
- Python 依赖（宿主机）：`aiohttp`, `transformers`

### 分布式方案

- **本机**（head）：RTX 4090，模型在 `/home/emnets-2/gxq/Qwen3-14B`
- **远程机**：RTX 3090，IP `10.130.151.15:15301`，用户 `gpu2`
  - 远程模型：`/home/gpu2/gxq/Qwen3-14B`
  - MoLink 环境：`/home/gpu2/miniconda3/envs/molink`
  - vLLM 环境：`/home/gpu2/miniconda3/envs/vllm19`
- SSH 免密登录已配置
- Python 依赖：`aiohttp`, `numpy`, `transformers`

## 配置项

### Docker 方案（`bench_docker.py` 命令行参数）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--pp` | 2 | 流水线并行度 |
| `--tp` | 1 | 张量并行度 |
| `--gpus` | auto | GPU 设备列表 |
| `--network` | `1gbit,1ms` | 带宽/延迟 |
| `--rps` | `3` | 请求速率列表 |
| `--duration` | 30 | 每轮测试时长（秒） |

### 分布式方案（编辑 `bench_utils.py` 常量）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LOCAL_IP` | `10.130.151.13` | 本机 IP |
| `REMOTE_HOST` / `REMOTE_PORT` | `10.130.151.15` / `15301` | 远程机地址 |
| `HEAD_END_LAYER` | 21 | MoLink head 截断层 |
| `RPS` | 3.0 | 请求速率 |
| `BENCH_DURATION` | 30 | 测试时长（秒） |
| `INPUT_TOKENS` / `OUTPUT_TOKENS` | 1024 / 512 | 输入/输出 token 数 |

## 输出结构

### Docker 方案

```
results/<timestamp>/
├── config.json
├── molink/
│   └── bw1gbit_delay1ms/
│       └── rps3/
│           ├── result.json
│           ├── head_metrics.json
│           └── tail_metrics.json
├── vllm/
│   └── ...
└── molink011/
    └── ...
```

### 分布式方案

```
results_distributed/
├── molink_result.json
├── vllm_result.json
└── distributed_compare.json
```

## result.json 字段

```json
{
  "config": {
    "url": "http://localhost:8080/generate",
    "type": "molink",
    "input_tokens": 1024,
    "max_tokens": 512,
    "rps": 3.0,
    "duration_s": 30,
    "total_requests": 90
  },
  "results": {
    "successful_requests": 90,
    "failed_requests": 0,
    "total_output_tokens": 46080,
    "benchmark_wall_time_s": 109.7,
    "throughput_tokens_per_s": 420.1,
    "ttft_s": { "avg": 9.7, "p50": 2.3, "p90": 50.2, "p99": 50.5 },
    "tpop_s": { "avg": 0.118, "p50": 0.122, "p90": 0.150, "p99": 0.159 },
    "request_latency_s": { ... }
  }
}
```

- **TTFT**（Time To First Token）：请求发出到收到第一个 token 的时间
- **TPOP**（Time Per Output Token）：`(总耗时 - TTFT) / (输出 token 数 - 1)`
- **throughput**：`总输出 token 数 / benchmark 墙钟时间`

## 注意事项

- 运行前用 `nvidia-smi` 确认 GPU 无其他进程占用
- Docker 方案默认 IP `172.26.0.10/11`，端口 `8080/9095`
- 分布式方案自动清理本机和远程进程
- 远程机需关闭 HTTP 代理（脚本自动处理 `unset http_proxy`）
- vLLM 跨机 PP=2 依赖 Ray + Gloo，需要两台机器的 hostname 解析到真实 IP（不能是 `127.0.1.1`），否则 Gloo 连接会失败
