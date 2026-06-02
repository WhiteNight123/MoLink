# MoLink v1 技术文档

## 1. 项目概述

MoLink v1 是一个基于 vLLM v1 构建的**跨节点流水线并行（Cross-Node Pipeline Parallelism）**推理框架。它通过 gRPC 在多台物理机器之间传输中间张量（intermediate tensors），将大型语言模型的 Transformer 层拆分到不同节点上执行，从而突破单机 GPU 显存限制，实现多机协同推理。

### 核心设计目标

- 将模型的 hidden layers 按层切分到不同物理节点
- 使用 gRPC 而非 NCCL 进行跨节点通信（因为 NCCL 要求 InfiniBand/NVLink 等特殊互联硬件）
- 对 vLLM 的侵入性尽量小，通过 monkey-patch 和继承方式集成
- 支持并发批次（multi-batch），通过 virtual engine 实现流水线重叠

### 典型部署拓扑

```
[Head Node]                         [Worker Node]
  Layer 0 ~ N/2                       Layer N/2 ~ N
  ┌──────────────┐    gRPC Push      ┌──────────────┐
  │ MolinkEngine  │ ──────────────►  │ MolinkWorker  │
  │ MolinkExecutor│ ◄──────────────  │    Node       │
  │ MolinkService │   gRPC Push      │ WorkerNode    │
  └──────────────┘   SamplerOutput   └──────────────┘
       (head)                             (tail)
```

---

## 2. 目录结构与模块职责

```
molinkv1/
├── __init__.py              # 包入口（延迟导入，避免CUDA初始化）
├── config.py                # 配置类定义
├── arg_utils.py             # CLI 参数解析
├── utils.py                 # 通用工具函数与拓扑管理
├── parallel_state.py        # 跨节点 PP 状态管理与 vLLM 补丁
├── service.py               # Head 节点 gRPC 服务实现
│
├── engine/
│   ├── engine.py            # MolinkEngine（AsyncLLM 子类，引擎入口）
│   ├── core.py              # MolinkEngineCoreProc（EngineCore 子进程补丁）
│   └── worker_node.py       # MolinkWorkerNode（轻量级 Worker 节点）
│
├── executor/
│   └── executor.py          # MolinkExecutor（MultiprocExecutor 子类）
│
├── core/
│   └── scheduler.py         # MolinkScheduler / MolinkAsyncScheduler
│
├── worker/
│   └── worker.py            # MolinkWorker（GPU Worker 子类）
│
├── entrypoints/
│   └── api_server.py        # FastAPI HTTP 服务入口
│
└── comm/
    ├── molink.proto          # gRPC 协议定义
    ├── molink_pb2.py         # Protobuf 生成代码
    ├── molink_pb2.pyi        # 类型存根
    └── molink_pb2_grpc.py   # gRPC Stub/Servicer 生成代码
```

---

## 3. 模块详细说明

### 3.1 配置层 — `config.py`

#### `MolinkConfig`

```python
@dataclass
class MolinkConfig:
    initial_peer: Optional[str] = None   # 要连接的头节点 gRPC 地址（如 "192.168.1.100:50051"）
    grpc_port: int = 0                    # gRPC 监听端口（0 = 自动选择）
    start_layer: int = 0                  # 本节点负责的起始层（含）
    end_layer: int = -1                   # 本节点负责的结束层（不含），-1 表示到最后
    max_message_size_mb: int = 200        # gRPC 消息最大尺寸（MB）
    enable_metrics: bool = False          # 是否启用通信指标收集
    max_concurrent_batches: int = 2       # 并发批次上限
```

关键属性：

| 属性 | 逻辑 |
|------|------|
| `enabled` | 当 `start_layer==0 && end_layer==-1 && initial_peer is None` 时为 `False`（即单节点模式） |
| `is_head_node` | `initial_peer` 为空时为 `True` |
| `max_message_size_bytes` | `max_message_size_mb * 1024 * 1024` |

#### `VllmConfig1`

继承 vLLM 的 `VllmConfig`，通过 Pydantic `@config` 装饰器扩展，增加 `molink_config: MolinkConfig` 字段。

#### `MolinkSchedulerConfig`

继承 `SchedulerConfig`，重写 `get_scheduler_cls()` 使其返回 `MolinkScheduler` 或 `MolinkAsyncScheduler`。

---

### 3.2 CLI 参数 — `arg_utils.py`

`MolinkEngineArgs` 继承 `AsyncEngineArgs`，新增以下命令行参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--molink-initial-peer` | str | None | 头节点 gRPC 地址，不提供则为头节点 |
| `--molink-grpc-port` | int | 0 | gRPC 端口，0 自动分配 |
| `--molink-start-layer` | int | 0 | 起始层 |
| `--molink-end-layer` | int | -1 | 结束层（-1 = 剩余全部） |
| `--molink-max-message-size-mb` | int | 200 | gRPC 消息最大 MB |
| `--molink-enable-metrics` | flag | False | 启用指标 |
| `--molink-max-concurrent-batches` | int | 2 | 并发批次数 |

---

### 3.3 工具函数 — `utils.py`

#### IP 与端口

- **`extract_ip()`**：通过 UDP socket 连接探测本机 IP（不实际发包），失败返回 `127.0.0.1`。
- **`find_free_port(start_port=50051)`**：从 `start_port` 开始递增尝试 `bind`，返回第一个可用端口。

#### gRPC 选项

- **`get_grpc_options(max_message_size_mb)`**：返回预配置的 gRPC channel/server 选项列表，包含：
  - 消息大小限制
  - HTTP/2 帧大小、窗口大小（16MB）
  - 吞吐量优化目标
  - Keepalive 参数（30s 间隔）

#### 流水线拓扑 — `PipelineTopology`

管理分布式流水线中所有节点的信息：

```python
@dataclass
class PipelineTopology:
    head_ip: str                         # 头节点地址
    start_layer: int                     # 本节点起始层
    end_layer: int                       # 本节点结束层
    node_pool: List[Dict[str, Any]]      # 所有节点列表
    node_info_dict: Dict[str, int]       # IP → start_layer 映射
```

核心方法：

| 方法 | 功能 |
|------|------|
| `add_node(ip, start_layer, end_layer)` | 添加节点，已存在则更新 |
| `remove_node(ip)` | 移除节点 |
| `get_sorted_server_list()` | 按 start_layer 排序返回服务器地址列表 |
| `get_next_server(current_ip)` | 获取当前节点的下一个节点 |
| `is_last_server(current_ip)` | 判断是否为最后一个节点 |
| `get_pp_rank(ip)` | 获取流水线并行 rank |
| `get_metadata()` | 返回拓扑元数据 `{head, server_list}` |

#### 序列化

- **`serialize_metadata(metadata)`**：`dict` → JSON bytes
- **`deserialize_metadata(data)`**：JSON bytes → `dict`

---

### 3.4 跨节点并行状态 — `parallel_state.py`

维护全局 MoLink 并行状态，并对 vLLM 内部的 `get_pp_indices` 和 PP Group 属性进行 monkey-patch。

#### 全局状态变量

```python
_MOLINK_ENABLED: bool = False
_MOLINK_START_LAYER: int = 0
_MOLINK_END_LAYER: int = -1
_MOLINK_IS_FIRST_STAGE: bool = True
_MOLINK_IS_LAST_STAGE: bool = True
```

#### 核心函数

| 函数 | 说明 |
|------|------|
| `init_molink_parallel_state(enabled, start_layer, end_layer, num_hidden_layers)` | 初始化全局 MoLink 状态，自动判定 first/last stage |
| `is_molink_enabled()` | 查询 MoLink 是否启用 |
| `get_molink_pp_indices(num_hidden_layers, pp_rank, pp_size)` | 返回本节点的层范围（覆盖 vLLM 默认的均匀分配逻辑） |
| `is_molink_first_stage()` / `is_molink_last_stage()` | 查询本节点在流水线中的位置 |
| `destroy_molink_parallel_state()` | 重置所有全局状态 |

#### vLLM 补丁机制

1. **`patch_get_pp_indices()`**：替换 `vllm.distributed.utils.get_pp_indices`，当 MoLink 启用时使用 MoLink 的层范围配置。
2. **`patch_pp_group()`**：替换 `GroupCoordinator.is_first_rank` 和 `is_last_rank` 属性，使其返回 MoLink 的 stage 信息。
3. **`apply_molink_patches()`**：一次性应用所有补丁，需在分布式初始化后、模型加载前调用。

---

### 3.5 gRPC 协议 — `comm/molink.proto`

定义了 MoLink 节点间通信的所有消息和服务。

#### 消息类型

| 消息 | 字段 | 用途 |
|------|------|------|
| `TensorEntry` | `key`, `tensor_data` | 单个张量的键值对 |
| `IntermediateTensors` | `tensors[]` | 中间张量集合 |
| `GrpcRequestData` | `scheduler_output`, `intermediate_tensors`, `grpc_metadata`, `virtual_engine`, `step_id` | 推送中间张量的请求 |
| `GrpcResponseData` | `res`, `error_message`, `output_data`, `virtual_engine` | 通用响应 |
| `SamplerOutput` | `output_data`, `virtual_engine`, `step_id` | 采样器输出（最后一阶段 → 头节点） |
| `NodeInfo` | `ip`, `start_layer`, `end_layer`, `pp_rank`, `tp_size` | 节点注册信息 |
| `PipelineTopology` | `nodes[]`, `total_layers` | 整个流水线拓扑 |
| `HealthCheckRequest` / `HealthCheckResponse` | — | 健康检查 |
| `KVCacheConfigData` | `num_gpu_blocks`, `num_cpu_blocks`, `kv_cache_config` | KV 缓存配置同步 |
| `ModelConfigData` | `vllm_config`, `pp_rank`, `pp_size` | 模型配置初始化 |

#### 服务接口 — `MolinkService`

```protobuf
service MolinkService {
    rpc JoinPipeline(NodeInfo) returns (GrpcResponseData);
    rpc GetTopology(HealthCheckRequest) returns (PipelineTopology);
    rpc PushIntermediateTensors(GrpcRequestData) returns (GrpcResponseData);
    rpc PushSamplerOutput(SamplerOutput) returns (GrpcResponseData);
    rpc ExecuteWorkerStep(GrpcTriggerRequest) returns (GrpcResponseData);
    rpc SyncKVCacheConfig(KVCacheConfigData) returns (GrpcResponseData);
    rpc InitializeModel(ModelConfigData) returns (GrpcResponseData);
    rpc HealthCheck(HealthCheckRequest) returns (HealthCheckResponse);
    rpc Shutdown(HealthCheckRequest) returns (GrpcResponseData);
}
```

所有 RPC 均为 **unary-unary**（单请求单响应）模式。

---

### 3.6 Head 节点 gRPC 服务 — `service.py`

#### `MolinkService(molink_pb2_grpc.MolinkServiceServicer)`

头节点上运行的 gRPC 服务，管理流水线拓扑和跨节点数据传输。

初始化参数：`pipeline_size`, `executor`, `head_ip`, `start_layer`, `end_layer`

##### 数据结构

```python
self.input_queue: List[asyncio.Queue]   # 按 virtual_engine 索引，接收中间张量
self.output_queue: List[asyncio.Queue]  # 按 virtual_engine 索引，接收采样结果
self.topology: PipelineTopology         # 流水线拓扑
```

##### RPC 方法

| 方法 | 功能 |
|------|------|
| `JoinPipeline` | 处理新节点加入，在响应中携带头节点 `num_gpu_blocks`（pack 为 8B uint64） |
| `GetTopology` | 返回当前拓扑中所有节点信息 |
| `PushIntermediateTensors` | 接收来自前一阶段的中间张量，放入对应 VE 的 `input_queue` |
| `PushSamplerOutput` | 接收最后一阶段的采样输出，放入对应 VE 的 `output_queue` |
| `HealthCheck` | 返回健康状态 |

##### 指标收集

使用 `threading.Lock` + `deque(maxlen=2000)` 线程安全地记录通信指标。

---

### 3.7 引擎入口 — `engine/engine.py`

#### `MolinkEngine(AsyncLLM)`

MoLink 的引擎入口，继承 vLLM v1 的 `AsyncLLM`。

##### 初始化流程

```
1. 提取 MoLink 参数（initial_peer, grpc_port, start_layer, end_layer 等）
2. 将 VllmConfig.__class__ 替换为 VllmConfig1，注入 molink_config
3. 禁用 async_scheduling（因为 NCCL PP broadcast 不适用于 gRPC）
4. 设置 worker_cls 为 "molinkv1.worker.MolinkWorker"
5. 设置环境变量 MOLINK_ENABLE_PIPELINE=1
6. 替换 SchedulerConfig 的 get_scheduler_cls 为 MolinkSchedulerConfig 版本
7. 临时 monkey-patch EngineCoreProc.run_engine_core → MolinkEngineCoreProc.run_engine_core
8. 调用 super().__init__()
9. 恢复原始 run_engine_core
```

##### `from_engine_args` 工厂方法

从 `MolinkEngineArgs` 创建引擎实例，配置 `MolinkConfig` 并传入 `MolinkExecutor` 作为 executor_class。

---

### 3.8 EngineCore 补丁 — `engine/core.py`

#### `MolinkEngineCoreProc(EngineCoreProc)`

静态方法 `run_engine_core` 替换 vLLM 原本的 `EngineCoreProc.run_engine_core`。

关键区别：
- 强制设置 `data_parallel_size = 1`（MoLink 不支持数据并行）
- 创建 `MolinkEngineCoreProc` 实例而非原始 `EngineCoreProc`
- 处理 SIGTERM/SIGINT 信号，触发优雅关闭

---

### 3.9 执行器 — `executor/executor.py`

#### `MolinkExecutor(MultiprocExecutor)`

MoLink 的核心执行器，继承 vLLM 的 `MultiprocExecutor`。

##### 初始化流程

```
1. 读取/创建 MolinkConfig
2. 初始化 MoLink 并行状态（init_molink_parallel_state）
3. 初始化 gRPC 相关字段（server, service, channel cache, event loop 等）
4. 调用 super().__init__() 初始化本地 workers
5. 调用 _init_molink() → 启动 gRPC server，非头节点则 JoinPipeline
```

##### 并发批次管理

```python
self._virtual_engine_counter: int = 0   # 轮询计数器
self._pipeline_futures: Dict[int, Future]  # VE → 异步 Future
```

每个批次分配一个 `virtual_engine` slot（`0 ~ max_concurrent_batches-1` 循环），确保不同批次的 gRPC 队列互不干扰。

##### 推理执行流程

**`execute_model(scheduler_output)`**（头节点 + 非最后阶段时）：

```
1. 分配 virtual_engine slot
2. 同步执行本地 head compute（super().execute_model）
3. 通过 collective_rpc 获取中间张量
4. 立即提交 _run_cross_node_pipeline 协程到 event loop
5. non_block 时返回已解析的 Future
```

**`sample_tokens(grammar_output)`**（头节点 + 非最后阶段时）：

```
直接返回 execute_model 中预提交的 pipeline Future
```

##### 张量序列化

两种格式：

1. **Protobuf 格式**（`_serialize_tensors`）：
   - 每个 tensor：`[ndim:4B][shape:ndim*8B][dtype_len:4B][dtype][raw_data]`
   - bfloat16 特殊处理：`view(torch.uint8)` 转换

2. **Combined 格式**（`_serialize_combined`）：
   - 将 scheduler pickle + tensors 打包为单一连续 buffer
   - 布局：`[sched_len:8B][sched_pickle][num_tensors:4B]{key_len:4B, key, data_len:8B, tensor_data}...`
   - 单次内存分配，减少内存碎片

##### 跨节点流水线 — `_run_cross_node_pipeline`

```
1. 获取 pipeline metadata（首次调用时缓存）
2. 序列化中间张量 + scheduler_output
3. gRPC PushIntermediateTensors → next_server
4. 等待 output_queue[virtual_engine].get()（超时 120s）
5. pickle 反序列化得到 ModelRunnerOutput
6. 防御性修补：补齐缺失的 request ID
```

异常处理：失败时返回空的 `ModelRunnerOutput`（所有 token 为 0），让引擎继续运行。

---

### 3.10 Worker 节点 — `engine/worker_node.py`

#### `MolinkWorkerNode`

轻量级 Worker 节点，直接在进程内创建 `MolinkWorker`（不使用 MultiprocExecutor 的多进程模式）。

##### 初始化流程

```
1. 创建 MolinkWorker（local_rank=0, rank=0, is_driver_worker=True）
2. init_device() → load_model()
3. 启动独立 event loop 线程
4. 启动 gRPC server
5. JoinPipeline → 获取头节点的 num_gpu_blocks
6. 初始化 KV cache（可能 cap 到头节点的 block 数）
7. compile_or_warm_up_model()
```

##### KV 缓存同步

Worker 节点的 KV cache 大小会被限制为不超过头节点：

```python
if num_blocks > head_num_gpu_blocks:
    num_blocks = head_num_gpu_blocks
    # 重新生成 kv_cache_config
```

这确保头节点调度器分配的 block ID 在 Worker 上有效。

#### `WorkerNodeService(molink_pb2_grpc.MolinkServiceServicer)`

Worker 节点的 gRPC 服务实现。

##### 异步工作队列

```python
self._work_queue: asyncio.Queue     # 工作项队列
self._compute_lock: asyncio.Lock    # GPU 计算互斥锁
```

`PushIntermediateTensors` 不直接执行计算，而是将工作项入队后立即返回，避免阻塞头节点的 gRPC 调用。

##### 预取流水线（Prefetch Pipelining）

`_process_work_queue` 实现了 CPU-GPU 流水线重叠：

```
deserialize(N) ──► compute(N) ──► push(N)
                   deserialize(N+1) ──► ...
```

在 GPU 执行第 N 批时，同时在后台反序列化第 N+1 批的数据。

##### 恢复与容错

当头节点与尾节点状态不同步（如 gRPC 响应丢失）时：

1. **`_synthesize_recovery_output`**：生成 EOS token 的空输出，让调度器立即完成孤儿请求
2. **`_ensure_request_states`**：为 Worker 缺失的请求创建 stub 状态
3. **`_heal_missing_requests`**：移除 Worker 无法处理的请求，避免崩溃

---

### 3.11 GPU Worker — `worker/worker.py`

#### `MolinkWorker(Worker)`

继承 vLLM 的 GPU Worker，增加中间张量存储和 MoLink 分布式补丁。

```python
self._molink_intermediate_tensors: deque[IntermediateTensors]  # 中间张量队列
```

##### 核心方法

| 方法 | 功能 |
|------|------|
| `_molink_set_intermediate_tensors(tensors)` | 存储中间张量（入队） |
| `_molink_get_intermediate_tensors()` | 取出中间张量（出队） |
| `init_device()` | 临时 patch `init_worker_distributed_environment`，在分布式初始化后应用 MoLink patches |
| `execute_model(scheduler_output)` | 执行模型推理，非 first stage 时从本地存储获取中间张量作为输入 |

##### 执行逻辑

```
if forward_pass and not is_first_rank:
    intermediate_tensors = self._molink_get_intermediate_tensors()

output = model_runner.execute_model(scheduler_output, intermediate_tensors)

if output is IntermediateTensors and not is_last_rank:
    self._molink_set_intermediate_tensors(output)
    return None  # 中间结果，不采样
```

---

### 3.12 调度器 — `core/scheduler.py`

#### `MolinkScheduler(Scheduler)`

继承 vLLM v1 的 `Scheduler`，增加 `use_pp` 属性（根据 MoLink 是否启用和 `pipeline_parallel_size` 判断）。

#### `MolinkAsyncScheduler(AsyncScheduler)`

异步版本，同样增加 `use_pp` 属性。

注意：当前 MoLink 禁用了 `async_scheduling`（在 `MolinkEngine.__init__` 中强制设为 `False`），因此实际使用的是 `MolinkScheduler`。

---

### 3.13 API 服务入口 — `entrypoints/api_server.py`

基于 FastAPI 的 HTTP 服务入口，提供推理和监控接口。

#### 端点

| 端点 | 方法 | 功能 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/molink_metrics` | GET | 获取通信层指标 |
| `/molink_metrics/reset` | POST | 重置通信指标 |
| `/generate` | POST | 文本生成（支持 stream） |

#### 引擎自动检测

`init_app` 根据参数自动选择引擎模式：

```python
if has_peer:
    # Worker 节点模式 → MolinkWorkerNode
elif has_layer_split:
    # Head 节点模式 → MolinkEngine
else:
    # 单节点模式 → 原始 AsyncLLM（零 MoLink 开销）
```

---

## 4. 数据流详解

### 4.1 推理请求完整流程

以 2 节点（Head: layers 0~N/2, Worker: layers N/2~N）为例：

```
[Client HTTP POST /generate]
         │
         ▼
    AsyncLLM.generate()
         │
         ▼
    EngineCore busy loop
         │
         ▼
    MolinkScheduler.schedule()  ──►  SchedulerOutput
         │
         ▼
    MolinkExecutor.execute_model()
         │
    ┌────┴─────┐
    │ Head GPU │  layers 0~N/2 forward pass
    │ compute  │  → IntermediateTensors
    └────┬─────┘
         │ collective_rpc("_molink_get_intermediate_tensors")
         │
         ▼
    _run_cross_node_pipeline (async, in event loop thread)
         │
    ┌────┴──────────────┐
    │ gRPC Push          │  serialize tensors + scheduler_output
    │ IntermediateTensors│  → WorkerNodeService.PushIntermediateTensors
    └────┬──────────────┘
         │
    ┌────┴─────┐
    │Worker GPU│  layers N/2~N forward pass
    │ compute  │  → sample_tokens → ModelRunnerOutput
    └────┬─────┘
         │
    ┌────┴──────────────┐
    │ gRPC Push          │  pickle(ModelRunnerOutput)
    │ SamplerOutput      │  → MolinkService.PushSamplerOutput
    └────┬──────────────┘
         │
         ▼
    output_queue[ve].get()  →  pickle.loads()
         │
         ▼
    MolinkExecutor.sample_tokens() → 返回 Future 结果
         │
         ▼
    EngineCore → AsyncLLM → HTTP Response
```

### 4.2 并发批次时序图（2 个 VE）

```
时间 ──────────────────────────────────────────►

Head:  [compute VE0][compute VE1][compute VE0][compute VE1]
          │              │              │            │
gRPC:     ├─push VE0─────┤─push VE1─────┤─push VE0──┤─push VE1
          │              │              │            │
Worker:   │  [compute VE0]│  [compute VE1]│ [compute]│ [compute]
          │              │              │            │
gRPC:     │───result VE0─┤──result VE1──┤──result───┤
          ▼              ▼              ▼            ▼
```

Head 的第 N+1 批 GPU 计算与 Worker 的第 N 批 GPU 计算并行执行。

### 4.3 张量传输格式

**Combined 格式**（当前主要使用）：

```
┌──────────────┬──────────────────┬──────────┬──────────────────────┐
│ sched_len(8B)│ scheduler_pickle │ num_t(4B)│ tensor entries...     │
└──────────────┴──────────────────┴──────────┴──────────────────────┘

单个 tensor entry:
┌─────────┬─────────┬────────┬──────────────┬──────────────┐
│key_len  │ key     │td_len  │ tensor_data  │              │
│ (4B)    │(key_len)│ (8B)   │[ndim|shape|  │dtype|raw]    │
└─────────┴─────────┴────────┴──────────────┴──────────────┘
```

---

## 5. 关键设计决策

### 5.1 为什么用 gRPC 而非 NCCL

- NCCL 需要 GPU Direct RDMA 或 InfiniBand 等特殊硬件
- MoLink 面向的是通过普通以太网连接的消费级 GPU 场景
- gRPC 提供 TCP 层面的可靠传输，配合大帧/大窗口优化吞吐

### 5.2 为什么 Worker 节点不用 MultiprocExecutor

- Worker 节点只有单卡（`local_rank=0, rank=0`），不需要多进程管理
- 直接在进程内创建 Worker 和 ModelRunner，避免共享内存队列的开销
- gRPC handler 直接调用 ModelRunner，路径更短

### 5.3 为什么禁用 async_scheduling

vLLM 的 async scheduling 将 sampled tokens 存储在 GPU 上，通过 NCCL PP broadcast 传递。MoLink 使用 gRPC 传输，无法利用这个机制，因此必须关闭。

### 5.4 虚拟引擎（Virtual Engine）机制

通过 `virtual_engine` 索引实现多批次并发：
- 每个批次分配唯一的 VE slot（循环 0~max_concurrent_batches-1）
- 每条 gRPC 消息和 asyncio Queue 都按 VE 分离
- 避免不同批次之间在 gRPC 错误/超时时互相污染

---

## 6. 部署指南

### 6.1 Head 节点启动

```bash
python -m molinkv1.entrypoints.api_server \
    --model <model_path> \
    --molink-start-layer 0 \
    --molink-end-layer 16 \
    --molink-grpc-port 50051 \
    --molink-max-concurrent-batches 2 \
    --port 8000
```

启动后日志中会显示 gRPC 地址，例如 `192.168.1.100:50051`。

### 6.2 Worker 节点启动

```bash
python -m molinkv1.entrypoints.api_server \
    --model <model_path> \
    --molink-initial-peer 192.168.1.100:50051 \
    --molink-start-layer 16 \
    --molink-end-layer -1 \
    --molink-grpc-port 50051 \
    --port 8001
```

Worker 会自动连接 Head 节点加入流水线。

### 6.3 单节点模式

不提供任何 `--molink-*` 参数时，回退为原始 vLLM，无 MoLink 开销。

### 6.4 多节点流水线（>2 节点）

当前实现主要针对 2 节点场景（PP=2），但拓扑支持任意数量节点。每个中间节点接收前一个节点的输出，执行自己负责的层，然后传递给下一个节点。最后一个节点的采样结果通过 `PushSamplerOutput` 返回给 Head。

---

## 7. 监控与调试

### 7.1 通信指标

通过 `GET /molink_metrics` 获取，包含：

- `service_metrics`：每次 gRPC 通信的详细时序
  - `head_compute`：头节点 GPU 计算时间
  - `receive_intermediate` / `receive_sampler`：接收数据耗时
  - `worker_step`：Worker 节点各阶段耗时（反序列化、计算、序列化、gRPC 发送）
  - `tail_recv`：Worker 接收数据的耗时
- `delivery_metrics`：预留字段
- `node`：本节点地址
- `is_head`：是否为头节点

### 7.2 调试打印

代码中保留了 `print(..., flush=True)` 语句，输出每个批次的：
- 计算开始/结束时间戳
- 数据传输开始时间戳
- 接收/返回时间戳

格式：`{virtual_engine} {num_reqs} {event} at {timestamp}`

---

## 8. 容错机制

### 8.1 Pipeline 超时

`_run_cross_node_pipeline` 对 tail 结果等待设置 120s 超时，超时后返回空输出（token ID = 0），避免引擎挂死。

### 8.2 Head-Tail 状态不同步

当 gRPC 响应丢失导致 Head 和 Tail 状态不一致时：
- `_synthesize_recovery_output`：生成 EOS 输出完成孤儿请求
- `_ensure_request_states`：为缺失请求创建 stub 状态
- `_heal_missing_requests`：移除无法处理的请求

### 8.3 KV Cache 一致性

Worker 节点的 KV cache block 数量被限制为不超过 Head 节点，确保调度器分配的 block ID 在所有节点上有效。Head 通过 `JoinPipeline` 响应将自己的 `num_gpu_blocks` 传递给 Worker。

---

## 9. 依赖关系

### 外部依赖

| 库 | 用途 |
|---|---|
| `vllm` | 基础推理引擎 |
| `grpcio` / `grpcio-tools` | gRPC 通信 |
| `protobuf` | Protocol Buffers |
| `torch` | PyTorch 张量操作 |
| `numpy` | 张量序列化辅助 |
| `cloudpickle` | 对象序列化 |
| `fastapi` | HTTP API |

### 模块依赖图

```
api_server.py ──► engine.py ──► executor.py ──► service.py ──► comm/molink_pb2*
                     │               │              │
                     │               ├─► parallel_state.py
                     │               ├─► utils.py
                     │               └─► worker/worker.py
                     │
                     ├─► core/scheduler.py
                     └─► engine/core.py

worker_node.py ──► worker/worker.py ──► parallel_state.py
      │                │
      ├─► service.py   └─► utils.py
      ├─► comm/molink_pb2*
      └─► utils.py
```

---

## 10. 局限性与注意事项

1. **仅支持 PP=2 优化路径**：多节点中间阶段的数据传递路径（非 last stage 的 Worker → Worker）虽已实现，但未经充分测试。
2. **不支持 async_scheduling**：强制关闭，因为其依赖 NCCL PP broadcast。
3. **不支持 Data Parallel**：`MolinkEngineCoreProc` 强制设置 `data_parallel_size=1`。
4. **串行 GPU 计算**：Worker 节点通过 `asyncio.Lock` 保证同一时间只有一个前向传播在 GPU 上执行。
5. **gRPC 消息大小**：大模型/大 batch 的中间张量可能超过默认 200MB 限制，需通过 `--molink-max-message-size-mb` 调整。
6. **Worker 节点无 HTTP 推理接口**：Worker 节点的 API server 仅用于指标查询，不直接处理推理请求。
