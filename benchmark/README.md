# MoLink Benchmark

对比测试 MoLink 与 vLLM（Ray PP=2）在不同网络条件和请求速率下的推理性能。

## 快速开始

```bash
cd /home/emnets-2/gxq/molink-measurement/MoLink/benchmark

# 默认：gRPC 传输 + 仅 1gbit/10ms + rps=3（快速验证）
./run_benchmark.sh molink

# 全部系统（MoLink + vLLM）
./run_benchmark.sh

# 仅 vLLM
./run_benchmark.sh vllm

# 使用 NCCL 传输（性能追平 vLLM）
MOLINK_TRANSPORT=nccl ./run_benchmark.sh molink
```

## 传输模式

MoLink 支持两种跨节点流水线并行传输方式，通过 `MOLINK_TRANSPORT` 环境变量切换：

| 模式 | 环境变量 | 通信栈 | 相对吞吐 | 说明 |
|------|---------|--------|---------|------|
| **gRPC**（默认） | `MOLINK_TRANSPORT=grpc` | gRPC + Protobuf | ~67% vLLM | MoLink 原生实现，独立于 vLLM 通信栈 |
| **NCCL** | `MOLINK_TRANSPORT=nccl` | NCCL via Ray | ~100% vLLM | 复用 vLLM PP 通信，应用 MoLink 层范围 |

### gRPC 模式原理

```
Head 容器 (GPU 1, layers 0-20)          Tail 容器 (GPU 2, layers 21-end)
  execute_model()                            PushIntermediateTensors()
    → 本地计算 head 层                          → 反序列化张量
    → 序列化中间张量 (CPU)                        → 获取 _compute_lock
  sample_tokens()                              → 执行 tail 层 + 采样
    → gRPC PushIntermediateTensors ──────────→  → pickle 结果
    → 等待 output_queue                       → gRPC PushSamplerOutput ────→
    → 反序列化结果                    ←────────────────────────────────
```

每步增加 3-4ms Python/Protobuf/gRPC 开销（相对 NCCL），导致解码吞吐约为 vLLM 的 67%。

### NCCL 模式原理

启动 Ray 集群 → `molink_nccl_server.py` 注入 MoLink 层范围补丁 → 委托 vLLM 原生 PP=2。

张量通信走 NCCL（无 CPU 往返、无序列化），吞吐与 vLLM 基本一致。

## 前提条件

- Docker 镜像 `molink:0.1`，网络 `molink`（subnet `172.26.0.0/16`）
- 模型路径 `/gxq/Qwen3-14B` 在容器内可访问（挂载自 `/home/emnets-2/gxq`）
- 至少 2 块空闲 GPU（各 ≥16GB），默认使用 GPU 1 和 2
- Python 依赖（宿主机）：`aiohttp`, `transformers`

## 配置项

编辑 `run_benchmark.sh` 顶部变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `GPU_HEAD` / `GPU_TAIL` | 1 / 2 | head/tail 容器使用的宿主机 GPU 编号 |
| `NETWORK_CONDITIONS` | `1gbit,10ms` | 带宽/延迟组合，可多项：`"500mbit,10ms" "1gbit,20ms"` 等 |
| `RPS_VALUES` | `3` | 请求速率列表（requests/sec） |
| `INPUT_TOKENS` | 1024 | 输入 token 数 |
| `OUTPUT_TOKENS` | 512 | 最大输出 token 数 |
| `DURATION` | 40 | 每轮测试时长（秒） |
| `COOLDOWN` | 10 | 两轮之间冷却时间（秒） |
| `MAX_MODEL_LEN` | 4096 | 模型最大上下文长度 |
| `MOLINK_TRANSPORT` | `grpc` | MoLink 传输模式：`grpc` / `nccl` |
| `MAX_CONCURRENT_BATCHES` | 2 | MoLink 虚拟引擎数（也可通过 `MOLINK_MAX_CONCURRENT_BATCHES` 环境变量覆盖）|
| `HEALTH_TIMEOUT` | 300 | 服务启动超时（秒） |

## 输出结构

```
results/<timestamp>/
├── config.json                  ← 本次运行的元数据
├── molink/
│   └── bw1gbit_delay10ms/
│       └── rps3/
│           └── result.json      ← 详细指标
├── vllm/
│   └── ...
└── summary/                     ← summarize_results.py 生成（可选）
    ├── results.csv
    └── *.pdf / *.png
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
    "duration_s": 40,
    "total_requests": 120
  },
  "results": {
    "successful_requests": 120,
    "failed_requests": 0,
    "total_output_tokens": 61440,
    "benchmark_wall_time_s": 185.6,
    "throughput_tokens_per_s": 331.1,
    "ttft_s": {
      "avg": 29.867, "p50": 2.726, "p90": 72.3, "p99": 77.8
    },
    "tpop_s": {
      "avg": 0.181, "p50": 0.176, "p90": 0.250, "p99": 0.300
    },
    "request_latency_s": { ... }
  }
}
```

- **TTFT**（Time To First Token）：请求发出到收到第一个 token 的时间
- **TPOP**（Time Per Output Token）：`(总耗时 - TTFT) / (输出 token 数 - 1)`
- **throughput**：`总输出 token 数 / benchmark 墙钟时间`

## 注意事项

- 运行前用 `nvidia-smi` 确认 GPU 无其他进程占用
- 默认 IP `172.26.0.10/11`，端口 `8080/9095`，确保无冲突
- 脚本通过 `trap cleanup EXIT` 自动清理容器
- gRPC 模式下 tail 节点独立启动 health endpoint（`:9095`），NCCL 模式下仅 head 暴露端口
- NCCL 模式需要容器间 Ray 通信正常（默认端口 `6379`）
