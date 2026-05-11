# Benchmark

自动对比测试 MoLink 与 vLLM（Ray PP=2）在不同网络条件和请求速率下的推理性能。

## 前提条件

- Docker 镜像 `molink:0.1`、网络 `molink`（subnet 172.26.0.0/16）
- 模型路径 `/gxq/Qwen3-14B` 在容器内可访问
- 至少 2 块空闲 GPU（各 ≥16GB）
- Python 依赖：`aiohttp`, `transformers`, `matplotlib`, `numpy`

## 使用

```bash
cd /home/emnets-2/gxq/molink-measurement/MoLink/benchmark

./run_benchmark.sh               # 全部测试（molink + vllm）
./run_benchmark.sh molink        # 仅 MoLink
./run_benchmark.sh vllm          # 仅 vLLM

# 生成 CSV + 对比图（PDF/PNG）
python summarize_results.py results/<timestamp>
```

## 配置项

编辑 `run_benchmark.sh` 顶部变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `GPU_HEAD` / `GPU_TAIL` | 1 / 2 | head/tail 容器使用的 GPU 编号 |
| `NETWORK_CONDITIONS` | 500mbit/10ms, 1gbit/10ms, 5gbit/10ms, 1gbit/20ms, 1gbit/30ms | 带宽/延迟组合 |
| `RPS_VALUES` | 0.5 1 3 5 7 9 | 请求速率列表 |
| `INPUT_TOKENS` | 1024 | 输入 token 数 |
| `OUTPUT_TOKENS` | 512 | 最大输出 token 数 |
| `DURATION` | 40 | 每轮测试时长（秒） |
| `HEALTH_TIMEOUT` | 600 | 服务启动超时（秒） |

## 注意事项

- 运行前用 `nvidia-smi` 确认 GPU 无其他进程占用
- 默认 IP 172.26.0.10/11，端口 8080/9095，确保无冲突
- 脚本通过 `trap cleanup EXIT` 自动清理容器

## 输出结构

```
results/<timestamp>/
├── config.json
├── molink/
│   └── bw1gbit_delay10ms/
│       ├── rps0.5/result.json
│       └── ...
├── vllm/
│   └── ...
└── summary/          ← summarize_results.py 生成
    ├── results.csv
    ├── rps_comparison_*.pdf/png
    └── network_comparison_*.pdf/png
```

每个 `result.json` 包含：throughput（tok/s）、TTFT / TPOP（avg/p50/p90/p99）、成功率。
