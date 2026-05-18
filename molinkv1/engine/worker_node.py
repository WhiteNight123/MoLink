"""
MoLink Worker Node — lightweight node that directly owns a model runner.

Unlike the head node (which uses MultiprocExecutor with separate worker processes),
the worker node initializes a single MolinkWorker in-process and calls its model
runner directly from the gRPC handler.  No shared-memory message queues, no
collective_rpc.
"""

import asyncio
import collections
import pickle
import struct
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

import cloudpickle
import grpc.aio as aio
import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import ModelRunnerOutput

from molinkv1.comm import molink_pb2, molink_pb2_grpc
from molinkv1.parallel_state import (
    init_molink_parallel_state,
    apply_molink_patches,
    is_molink_last_stage,
)
from molinkv1.utils import (
    extract_ip,
    find_free_port,
    get_grpc_options,
    PipelineTopology,
    serialize_metadata,
    deserialize_metadata,
)
from molinkv1.worker.worker import MolinkWorker

logger = init_logger(__name__)


def _get_profiler():
    from molinkv1.profiler import get_profiler
    return get_profiler()


# ---------------------------------------------------------------------------
# Recovery output synthesiser (handles head-tail desync)
# ---------------------------------------------------------------------------

def _synthesize_recovery_output(scheduler_output: Any) -> "ModelRunnerOutput":
    """Build a ModelRunnerOutput that finishes orphaned requests whose state
    the tail no longer has (lost gRPC response on a prior step).  Uses EOS
    so the scheduler finishes the requests immediately."""
    req_ids = list(scheduler_output.num_scheduled_tokens.keys())
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
        sampled_token_ids=[[0] for _ in req_ids],
    )


def _ensure_request_states(worker: Any, scheduler_output: Any) -> None:
    """Create stub CachedRequestState for requests the tail doesn't know.

    When a gRPC response is lost the head may keep scheduling a request
    whose state the tail has already removed (or never created).  We
    pre-populate a minimal state so _update_states can find it.
    """
    cr = getattr(scheduler_output, "scheduled_cached_reqs", None)
    if cr is None or not cr.req_ids:
        return

    existing = worker.model_runner.requests
    missing = [req_id for req_id in cr.req_ids if req_id not in existing]
    if not missing:
        return

    from vllm.v1.worker.gpu_input_batch import CachedRequestState

    for req_id in missing:
        # Find the index of this request in the cached list.
        try:
            idx = cr.req_ids.index(req_id)
        except ValueError:
            continue

        num_computed = cr.num_computed_tokens[idx] if idx < len(cr.num_computed_tokens) else 0
        num_output = cr.num_output_tokens[idx] if idx < len(cr.num_output_tokens) else 0
        block_ids = None
        if cr.new_block_ids is not None and idx < len(cr.new_block_ids):
            block_ids = cr.new_block_ids[idx] or ()

        state = CachedRequestState(
            req_id=req_id,
            prompt_token_ids=[0] * max(num_computed - num_output, 0),
            mm_features=[],
            sampling_params=None,
            generator=None,
            block_ids=block_ids or (),
            num_computed_tokens=num_computed,
            output_token_ids=[0] * num_output,
        )
        existing[req_id] = state
        logger.warning(
            "[MoLink][TAIL] Created stub state for missing request %s "
            "(num_computed=%d, num_output=%d)",
            req_id, num_computed, num_output,
        )

def _heal_missing_requests(worker: Any, scheduler_output: Any) -> list[str]:
    """Remove cached requests the tail no longer has state for.

    Returns the list of request IDs that were removed so the caller
    can patch dummy tokens into the ModelRunnerOutput later.
    """
    cr = getattr(scheduler_output, "scheduled_cached_reqs", None)
    if cr is None or not cr.req_ids:
        return []

    existing = worker.model_runner.requests
    removed_ids: list[str] = []

    # Walk backwards so indices stay valid during popping.
    for i in range(len(cr.req_ids) - 1, -1, -1):
        rid = cr.req_ids[i]
        if rid not in existing:
            removed_ids.append(rid)
            cr.req_ids.pop(i)
            if cr.new_block_ids is not None:
                cr.new_block_ids.pop(i)
            cr.num_computed_tokens.pop(i)
            if cr.new_token_ids:
                cr.new_token_ids.pop(i)
            cr.num_output_tokens.pop(i)
            logger.warning(
                "[MoLink][TAIL] Hiding orphaned request %s (head-tail desync).",
                rid,
            )

    # Restore original order (the removed_ids were appended in reverse).
    removed_ids.reverse()
    return removed_ids

# ---------------------------------------------------------------------------
# Tensor serialization helpers (same wire format as executor.py)
# ---------------------------------------------------------------------------

def _serialize_tensors(tensors_cpu: dict[str, torch.Tensor]) -> molink_pb2.IntermediateTensors:
    """Serialize tensors using a flat binary layout with pre-sized buffers."""
    grpc_tensors = molink_pb2.IntermediateTensors()
    for key, tensor in tensors_cpu.items():
        shape = tensor.shape
        ndim = len(shape)
        dtype_str = str(tensor.dtype)
        dtype_bytes = dtype_str.encode("ascii")

        if tensor.dtype == torch.bfloat16:
            t_bf = tensor.contiguous()
            if t_bf.dim() == 0:
                raw = t_bf.unsqueeze(0).view(torch.uint8).numpy().tobytes()
            else:
                raw = t_bf.view(torch.uint8).numpy().tobytes()
        else:
            raw = tensor.numpy().tobytes()

        header_size = 4 + ndim * 8 + 4 + len(dtype_bytes)
        buf = bytearray(header_size + len(raw))
        off = 0
        struct.pack_into("<I", buf, off, ndim); off += 4
        for dim in shape:
            struct.pack_into("<Q", buf, off, dim); off += 8
        struct.pack_into("<I", buf, off, len(dtype_bytes)); off += 4
        buf[off:off + len(dtype_bytes)] = dtype_bytes; off += len(dtype_bytes)
        buf[off:off + len(raw)] = raw

        grpc_tensors.tensors.append(
            molink_pb2.TensorEntry(key=key, tensor_data=bytes(buf))
        )
    return grpc_tensors


def _deserialize_tensors(tensor_bytes: Dict[str, bytes]) -> IntermediateTensors:
    """Deserialize tensors from the flat binary layout."""
    tensors = {}
    for key, data in tensor_bytes.items():
        offset = 0
        (ndim,) = struct.unpack_from("<I", data, offset); offset += 4
        shape = []
        for _ in range(ndim):
            (dim,) = struct.unpack_from("<Q", data, offset); shape.append(dim); offset += 8
        (dtype_len,) = struct.unpack_from("<I", data, offset); offset += 4
        dtype_name = data[offset:offset + dtype_len].decode("ascii"); offset += dtype_len
        raw = data[offset:]
        if dtype_name == "torch.bfloat16":
            n_elements = 1
            for d in shape:
                n_elements *= d
            tensor = (
                torch.frombuffer(bytearray(raw), dtype=torch.uint8)
                .reshape(n_elements, 2)
                .view(torch.bfloat16)
                .reshape(tuple(shape))
                .to("cuda")
            )
        else:
            np_array = np.frombuffer(
                raw, dtype=np.dtype(dtype_name.replace("torch.", ""))
            ).reshape(tuple(shape))
            tensor = torch.from_numpy(np_array).to("cuda")
        tensors[key] = tensor
    return IntermediateTensors(tensors=tensors)


def _deserialize_combined(data: bytes) -> tuple[bytes, Dict[str, bytes]]:
    """Deserialize the combined flat buffer from _serialize_combined.

    Returns (scheduler_pickle_bytes, {tensor_key: tensor_bytes}).

    Layout:
      [scheduler_len: 8B][scheduler_pickle]
      [num_tensors: 4B]
      For each tensor:
        [key_len: 4B][key][tensor_data_len: 8B][tensor_data]
    """
    offset = 0
    (scheduler_len,) = struct.unpack_from("<Q", data, offset); offset += 8
    scheduler_bytes = data[offset:offset + scheduler_len]; offset += scheduler_len

    (num_tensors,) = struct.unpack_from("<I", data, offset); offset += 4
    tensor_bytes: Dict[str, bytes] = {}
    for _ in range(num_tensors):
        (key_len,) = struct.unpack_from("<I", data, offset); offset += 4
        key = data[offset:offset + key_len].decode("ascii"); offset += key_len
        (tensor_data_len,) = struct.unpack_from("<Q", data, offset); offset += 8
        tensor_data = data[offset:offset + tensor_data_len]; offset += tensor_data_len
        tensor_bytes[key] = tensor_data

    return scheduler_bytes, tensor_bytes


# ---------------------------------------------------------------------------
# gRPC Service for the worker node
# ---------------------------------------------------------------------------

class WorkerNodeService(molink_pb2_grpc.MolinkServiceServicer):
    """gRPC service that drives the local model runner directly."""

    def __init__(
        self,
        worker: MolinkWorker,
        executor_pool: ThreadPoolExecutor,
        head_ip: str,
        start_layer: int,
        end_layer: int,
        max_message_size_mb: int = 200,
    ):
        self.worker = worker
        self._pool = executor_pool
        self.max_message_size_mb = max_message_size_mb
        self.topology = PipelineTopology(head_ip, start_layer, end_layer)

        # Stub cache for sending results back to other nodes.
        self._stub_cache: Dict[str, molink_pb2_grpc.MolinkServiceStub] = {}
        self._channel_cache: Dict[str, aio.Channel] = {}

        # Thread-safe metrics
        self._metrics_lock = threading.Lock()
        self._metrics_deque = collections.deque(maxlen=2000)
        self._metrics_enabled = False

        # Serialize GPU compute (model runner is not concurrent-safe).
        self._compute_lock = asyncio.Lock()

        # Async work queue: PushIntermediateTensors enqueues here and returns
        # immediately so the head's gRPC call is not blocked by tail compute.
        self._work_queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None

        # Cached pipeline metadata (avoid per-step JSON deserialize).
        self._cached_server_list: Optional[list] = None
        self._cached_head_server: Optional[str] = None

    def _record_metric(self, metric: dict):
        if not self._metrics_enabled:
            return
        with self._metrics_lock:
            self._metrics_deque.append(metric)

    def get_metrics(self) -> list[dict]:
        with self._metrics_lock:
            return list(self._metrics_deque)

    def reset_metrics(self):
        with self._metrics_lock:
            self._metrics_deque.clear()

    def start_background_worker(self):
        """Launch the background queue processor.  Must be called from the event loop."""
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._process_work_queue())

    async def _process_work_queue(self):
        """Process items from the work queue sequentially (GPU is not concurrent-safe)."""
        while True:
            work_item = await self._work_queue.get()
            try:
                await self._process_work_item(work_item)
            except Exception as e:
                logger.error(
                    "[MoLink][TAIL] Background worker failed for VE %s: %s",
                    work_item.get("virtual_engine", "?"), e,
                )
                traceback.print_exc()

    async def _process_work_item(self, work_item: dict):
        """Execute the full tail pipeline for one micro-batch (deser → compute → push result)."""
        t_total_start = time.perf_counter()
        virtual_engine = work_item["virtual_engine"]
        step_id = work_item.get("step_id", -1)
        loop = asyncio.get_running_loop()
        recv_bytes = 0

        # Queue wait time (from enqueue to processing start)
        enqueue_time = work_item.get("enqueue_time", t_total_start)
        queue_wait_ms = (t_total_start - enqueue_time) * 1000 if enqueue_time else 0

        # Deserialize (CPU-bound → thread pool)
        t_deser_start = time.perf_counter()
        if "combined_data" in work_item:
            # New combined format
            def _deser_combined():
                sched_bytes, tensor_bytes = _deserialize_combined(work_item["combined_data"])
                return _deserialize_tensors(tensor_bytes), pickle.loads(sched_bytes), sum(len(v) for v in tensor_bytes.values())

            intermediate_tensors, scheduler_output, recv_bytes = await loop.run_in_executor(
                None,
                _deser_combined,
            )
        else:
            # Legacy format
            intermediate_tensors_bytes = work_item["intermediate_tensors_bytes"]
            scheduler_output_bytes = work_item["scheduler_output_bytes"]
            recv_bytes = sum(len(v) for v in intermediate_tensors_bytes.values())

            intermediate_tensors, scheduler_output = await loop.run_in_executor(
                None,
                lambda: (
                    _deserialize_tensors(intermediate_tensors_bytes),
                    pickle.loads(scheduler_output_bytes),
                ),
            )
        t_deser_end = time.perf_counter()

        # Compute (serialized — model runner is not concurrent-safe)
        t_compute_start = time.perf_counter()
        compute_lock_wait_ms = 0
        try:
            t_lock_acquire = time.perf_counter()
            async with self._compute_lock:
                compute_lock_wait_ms = (time.perf_counter() - t_lock_acquire) * 1000
                output = await self._run_step(scheduler_output, intermediate_tensors)
        except Exception as e:
            logger.warning(
                "[MoLink][TAIL] _run_step failed (VE %s): %s. "
                "Synthesizing recovery output.",
                virtual_engine, e,
            )
            output = await asyncio.get_running_loop().run_in_executor(
                None,
                _synthesize_recovery_output,
                scheduler_output,
            )
            if output is None:
                raise
        t_compute_end = time.perf_counter()

        # Route the result (uses cached topology — same for every step)
        server_list = self._cached_server_list
        my_address = f"{self._ip}:{self._grpc_port}"
        try:
            my_idx = server_list.index(my_address)
        except ValueError:
            my_idx = len(server_list) - 1

        is_last_stage = (my_idx == len(server_list) - 1)

        # Push result
        t_push_start = time.perf_counter()
        t_serialize_result_ms = 0
        t_grpc_send_ms = 0
        if is_last_stage:
            t_ser_start = time.perf_counter()
            output_bytes = await loop.run_in_executor(
                None,
                pickle.dumps,
                output,
                pickle.HIGHEST_PROTOCOL,
            )
            t_ser_end = time.perf_counter()
            t_serialize_result_ms = (t_ser_end - t_ser_start) * 1000

            t_grpc_start = time.perf_counter()
            await self._push_sampler_output(output_bytes, virtual_engine, self._cached_head_server, step_id)
            t_grpc_end = time.perf_counter()
            t_grpc_send_ms = (t_grpc_end - t_grpc_start) * 1000
        else:
            next_server = server_list[my_idx + 1]
            tensors = output.tensors if isinstance(output, IntermediateTensors) else {"hidden_states": output}
            cached_meta = {"head": self._cached_head_server, "server_list": self._cached_server_list}
            await self._push_intermediate_tensors(
                tensors, scheduler_output_bytes, cached_meta, virtual_engine, next_server
            )
        t_push_end = time.perf_counter()

        self._record_metric({
            "type": "worker_step",
            "queue_wait_ms": queue_wait_ms,
            "deserialize_ms": (t_deser_end - t_deser_start) * 1000,
            "compute_lock_wait_ms": compute_lock_wait_ms,
            "compute_ms": (t_compute_end - t_compute_start) * 1000,
            "serialize_result_ms": t_serialize_result_ms,
            "grpc_send_ms": t_grpc_send_ms,
            "push_ms": (t_push_end - t_push_start) * 1000,
            "total_ms": (time.perf_counter() - t_total_start) * 1000,
            "is_last_stage": is_last_stage,
            "recv_bytes": recv_bytes,
            "result_bytes": len(output_bytes) if is_last_stage else 0,
            "virtual_engine": virtual_engine,
            "timestamp": time.time(),
        })

        profiler = _get_profiler()
        if profiler is not None:
            profiler.record("tail_step", {
                "queue_wait_ms": queue_wait_ms,
                "deserialize_ms": (t_deser_end - t_deser_start) * 1000,
                "compute_lock_wait_ms": compute_lock_wait_ms,
                "compute_ms": (t_compute_end - t_compute_start) * 1000,
                "serialize_result_ms": t_serialize_result_ms,
                "grpc_send_ms": t_grpc_send_ms,
                "total_ms": (time.perf_counter() - t_total_start) * 1000,
                "recv_bytes": recv_bytes,
                "result_bytes": len(output_bytes) if is_last_stage else 0,
            }, step_id=step_id)

    def _get_stub(self, address: str) -> molink_pb2_grpc.MolinkServiceStub:
        if address not in self._stub_cache:
            channel = aio.insecure_channel(address, options=get_grpc_options(self.max_message_size_mb))
            self._channel_cache[address] = channel
            self._stub_cache[address] = molink_pb2_grpc.MolinkServiceStub(channel)
        return self._stub_cache[address]

    # -- topology -----------------------------------------------------------

    async def JoinPipeline(self, request, context):
        self.topology.add_node(request.ip, request.start_layer, request.end_layer)
        logger.info(f"Node {request.ip} joined pipeline (layers {request.start_layer}-{request.end_layer})")
        return molink_pb2.GrpcResponseData(res=1)

    async def GetTopology(self, request, context):
        nodes = [molink_pb2.NodeInfo(ip=n["ip"], start_layer=n["start_layer"], end_layer=n["end_layer"])
                 for n in self.topology.node_pool]
        return molink_pb2.PipelineTopology(nodes=nodes)

    # -- data handlers ------------------------------------------------------

    async def PushIntermediateTensors(self, request, context):
        """Enqueue work and return immediately so the head is not blocked by tail compute."""
        # Cache pipeline metadata on first call (same for every step).
        if self._cached_server_list is None:
            grpc_metadata = deserialize_metadata(request.grpc_metadata)
            self._cached_server_list = grpc_metadata.get("server_list", [])
            self._cached_head_server = grpc_metadata.get("head")

        # Parse from combined format (scheduler_output contains everything)
        # or legacy format (intermediate_tensors has per-tensor entries).
        if len(request.intermediate_tensors.tensors) == 0:
            # New combined format — parse lazily in background worker
            work_item = {
                "virtual_engine": request.virtual_engine,
                "combined_data": request.scheduler_output,
                "enqueue_time": time.perf_counter(),
                "step_id": request.step_id,
            }
        else:
            # Legacy format
            intermediate_tensors_bytes = {}
            for entry in request.intermediate_tensors.tensors:
                intermediate_tensors_bytes[entry.key] = entry.tensor_data
            work_item = {
                "virtual_engine": request.virtual_engine,
                "intermediate_tensors_bytes": intermediate_tensors_bytes,
                "scheduler_output_bytes": request.scheduler_output,
                "enqueue_time": time.perf_counter(),
                "step_id": request.step_id,
            }
        await self._work_queue.put(work_item)
        return molink_pb2.GrpcResponseData(res=1)

    async def PushSamplerOutput(self, request, context):
        # Worker nodes don't normally receive sampler output, but keep for compatibility.
        return molink_pb2.GrpcResponseData(res=1)

    async def ExecuteWorkerStep(self, request, context):
        # Not used in the new architecture — data arrives via PushIntermediateTensors.
        return molink_pb2.GrpcResponseData(res=1)

    async def HealthCheck(self, request, context):
        return molink_pb2.HealthCheckResponse(status="healthy")

    # -- internal -----------------------------------------------------------

    async def _run_step(self, scheduler_output, intermediate_tensors):
        """Run execute_model + sample_tokens on the local worker.

        If the model runner is missing state for a cached request
        (head-tail desync after a lost gRPC response), we synthesize a
        recovery output so the head can advance the request cleanly
        instead of crashing the entire pipeline.
        """
        loop = asyncio.get_running_loop()
        self.worker._molink_set_intermediate_tensors(intermediate_tensors)

        try:
            output = await loop.run_in_executor(
                None, self.worker.execute_model, scheduler_output
            )
        except (KeyError, IndexError, ValueError, RuntimeError) as e:
            logger.warning(
                "[MoLink][TAIL] _run_step failed (VE %s): %s. "
                "Synthesizing recovery output.",
                getattr(scheduler_output, "virtual_engine", 0), e,
            )
            output = await loop.run_in_executor(
                None,
                _synthesize_recovery_output,
                scheduler_output,
            )
            if output is not None:
                return output
            raise

        if output is None:
            stored = self.worker._molink_get_intermediate_tensors()
            if stored is not None:
                output = stored
            else:
                output = await loop.run_in_executor(
                    None, self.worker.sample_tokens, None
                )

        from vllm.v1.outputs import AsyncModelRunnerOutput
        if isinstance(output, AsyncModelRunnerOutput):
            output = await loop.run_in_executor(None, output.get_output)

        return output

    async def _push_sampler_output(self, output_bytes, virtual_engine, head_server, step_id=-1):
        request = molink_pb2.SamplerOutput(
            output_data=output_bytes, virtual_engine=virtual_engine, step_id=step_id
        )
        stub = self._get_stub(head_server)
        await stub.PushSamplerOutput(request)

    async def _push_intermediate_tensors(self, tensors, scheduler_output_bytes,
                                          grpc_metadata, virtual_engine, next_server):
        tensors_cpu = {k: v.to("cpu") for k, v in tensors.items()}
        grpc_tensors = _serialize_tensors(tensors_cpu)
        request = molink_pb2.GrpcRequestData(
            scheduler_output=scheduler_output_bytes,
            intermediate_tensors=grpc_tensors,
            grpc_metadata=serialize_metadata(grpc_metadata),
            virtual_engine=virtual_engine,
        )
        stub = self._get_stub(next_server)
        await stub.PushIntermediateTensors(request)


# ---------------------------------------------------------------------------
# Worker Node — top-level orchestrator
# ---------------------------------------------------------------------------

class MolinkWorkerNode:
    """Lightweight worker node that directly owns a model runner.

    Initialisation sequence:
    1. Create MolinkWorker (in-process, no separate worker process)
    2. init_device → load_model → determine_available_memory →
       initialize_from_config → compile_or_warm_up_model
    3. Start gRPC server
    4. Join pipeline (connect to head node)
    """

    def __init__(self, vllm_config: VllmConfig):
        from molinkv1.config import MolinkConfig
        self.molink_config: MolinkConfig = getattr(vllm_config, "molink_config", None)
        self.vllm_config = vllm_config
        self.grpc_server: Optional[aio.Server] = None
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._pool = ThreadPoolExecutor(max_workers=16)

        # Initialize MoLink parallel state.
        start_layer, end_layer = self.molink_config.get_serving_layers()
        init_molink_parallel_state(
            enabled=True, start_layer=start_layer, end_layer=end_layer,
        )

        # ---- Create worker directly (no MultiprocExecutor) ----
        dist_port = find_free_port(start_port=29500)
        self.worker = MolinkWorker(
            vllm_config=vllm_config,
            local_rank=0,
            rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{dist_port}",
            is_driver_worker=True,
        )

        # Initialize CUDA and load model.
        from vllm.config import set_current_vllm_config
        with set_current_vllm_config(vllm_config):
            self.worker.init_device()
            self.worker.load_model()

        # ---- Start gRPC server first so we can receive head's num_gpu_blocks ----
        self.ip = extract_ip()
        self.grpc_port = find_free_port(
            start_port=self.molink_config.grpc_port if self.molink_config.grpc_port > 0 else 50051
        )
        self.grpc_address = f"{self.ip}:{self.grpc_port}"

        self._start_event_loop_thread()
        future = asyncio.run_coroutine_threadsafe(
            self._start_grpc_server(start_layer, end_layer), self._event_loop
        )
        future.result(timeout=30)

        # ---- Join pipeline and get head's num_gpu_blocks ----
        head_num_gpu_blocks = None
        future = asyncio.run_coroutine_threadsafe(
            self._join_pipeline(), self._event_loop
        )
        try:
            head_num_gpu_blocks = future.result(timeout=30)
        except Exception as e:
            logger.error(f"Failed to join pipeline: {e}")

        # ---- Initialize KV cache (possibly capped to head's num_gpu_blocks) ----
        with set_current_vllm_config(vllm_config):
            self._init_kv_cache(head_num_gpu_blocks=head_num_gpu_blocks)
            self.worker.compile_or_warm_up_model()

        logger.info("MolinkWorkerNode: model runner ready")

        logger.info(
            f"MolinkWorkerNode initialized at {self.grpc_address}, "
            f"layers {start_layer}-{end_layer}"
        )

    # -- KV cache -----------------------------------------------------------

    def _init_kv_cache(self, head_num_gpu_blocks: Optional[int] = None):
        """Profile memory, compute KV cache config, allocate and initialize.

        Args:
            head_num_gpu_blocks: The head node's num_gpu_blocks.  If provided,
                the worker caps its own num_gpu_blocks to this value so that
                the head scheduler never allocates blocks the worker doesn't
                have.
        """
        from vllm.v1.core.kv_cache_utils import (
            get_kv_cache_configs,
            generate_scheduler_kv_cache_config,
        )

        available_memory = self.worker.determine_available_memory()
        if isinstance(available_memory, list):
            available_memory = available_memory[0]

        kv_cache_specs = self.worker.get_kv_cache_spec()
        if isinstance(kv_cache_specs, list):
            kv_cache_specs = kv_cache_specs[0]

        kv_cache_configs = get_kv_cache_configs(
            self.vllm_config, [kv_cache_specs], [available_memory]
        )

        scheduler_kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)
        num_blocks = scheduler_kv_cache_config.num_blocks

        # Cap to the head node's value so the scheduler's block IDs are
        # valid on this worker.
        if head_num_gpu_blocks is not None and head_num_gpu_blocks > 0:
            if num_blocks > head_num_gpu_blocks:
                logger.info(
                    f"Worker num_gpu_blocks ({num_blocks}) > head "
                    f"({head_num_gpu_blocks}). Capping to head's value."
                )
                num_blocks = head_num_gpu_blocks
                self.vllm_config.cache_config.num_gpu_blocks = num_blocks
                # Regenerate config with capped blocks.
                self.vllm_config.cache_config.num_gpu_blocks_override = num_blocks
                kv_cache_configs = get_kv_cache_configs(
                    self.vllm_config, [kv_cache_specs], [available_memory]
                )
                scheduler_kv_cache_config = generate_scheduler_kv_cache_config(
                    kv_cache_configs
                )
                num_blocks = scheduler_kv_cache_config.num_blocks
            else:
                logger.info(
                    f"Worker num_gpu_blocks ({num_blocks}) <= head "
                    f"({head_num_gpu_blocks}). No capping needed."
                )
                self.vllm_config.cache_config.num_gpu_blocks = num_blocks
        else:
            self.vllm_config.cache_config.num_gpu_blocks = num_blocks

        kv_cache_groups = scheduler_kv_cache_config.kv_cache_groups
        if kv_cache_groups:
            self.vllm_config.cache_config.block_size = min(
                g.kv_cache_spec.block_size for g in kv_cache_groups
            )

        self.worker.initialize_from_config(kv_cache_configs[0])

        logger.info(
            f"KV cache initialized: num_gpu_blocks={num_blocks}"
        )

    # -- Event loop ---------------------------------------------------------

    def _start_event_loop_thread(self):
        loop_ready = threading.Event()

        def run_loop():
            self._event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._event_loop)
            loop_ready.set()
            self._event_loop.run_forever()

        self._loop_thread = threading.Thread(target=run_loop, daemon=True, name="MolinkWorkerEventLoop")
        self._loop_thread.start()
        loop_ready.wait(timeout=10)

    async def _start_grpc_server(self, start_layer, end_layer):
        config = self.molink_config
        self.grpc_server = aio.server(self._pool, options=get_grpc_options(config.max_message_size_mb))

        service = WorkerNodeService(
            worker=self.worker,
            executor_pool=self._pool,
            head_ip=f"{self.ip}:{self.grpc_port}",
            start_layer=start_layer,
            end_layer=end_layer,
        )
        service._ip = self.ip
        service._grpc_port = self.grpc_port
        service._metrics_enabled = config.enable_metrics
        if config.enable_metrics:
            from molinkv1.profiler import init_profiler
            import os
            profiler_dir = os.environ.get("MOLINK_PROFILER_DIR", "/tmp/molink_profile")
            init_profiler(profiler_dir, f"tail_{self.grpc_port}")
            self._shutdown_event = threading.Event()
            self._metrics_flush_thread = threading.Thread(
                target=self._metrics_flush_loop, daemon=True, name="MolinkTailMetricsFlush"
            )
            self._metrics_flush_thread.start()
        service.start_background_worker()
        self.service = service

        molink_pb2_grpc.add_MolinkServiceServicer_to_server(service, self.grpc_server)
        self.grpc_server.add_insecure_port(f"[::]:{self.grpc_port}")
        await self.grpc_server.start()
        logger.info(f"Worker gRPC server started on port {self.grpc_port}")

    async def _join_pipeline(self) -> Optional[int]:
        """Join the pipeline and return the head node's num_gpu_blocks."""
        config = self.molink_config
        channel = None
        try:
            channel = aio.insecure_channel(
                config.initial_peer,
                options=get_grpc_options(config.max_message_size_mb),
            )
            stub = molink_pb2_grpc.MolinkServiceStub(channel)
            start_layer, end_layer = config.get_serving_layers()
            node_info = molink_pb2.NodeInfo(
                ip=self.grpc_address, start_layer=start_layer, end_layer=end_layer
            )
            response = await stub.JoinPipeline(node_info)
            if response.res == 1:
                logger.info(f"Successfully joined pipeline at {config.initial_peer}")
            else:
                logger.error(f"Failed to join pipeline: {response.error_message}")

            # Extract head's num_gpu_blocks from the response.
            head_num_gpu_blocks = None
            if response.output_data:
                import struct as _struct
                (head_num_gpu_blocks,) = _struct.unpack(
                    "<Q", response.output_data
                )
            return head_num_gpu_blocks
        except Exception as e:
            logger.error(f"Error joining pipeline: {e}")
            traceback.print_exc()
            raise
        finally:
            if channel is not None:
                await channel.close()

    # -- Lifecycle ----------------------------------------------------------

    def shutdown(self):
        from molinkv1.parallel_state import destroy_molink_parallel_state
        destroy_molink_parallel_state()

        if hasattr(self, '_shutdown_event'):
            self._shutdown_event.set()
        if hasattr(self, '_metrics_flush_thread') and self._metrics_flush_thread.is_alive():
            self._flush_metrics_to_file()
            self._metrics_flush_thread.join(timeout=3)

        if self._event_loop and self._event_loop.is_running():
            future = asyncio.run_coroutine_threadsafe(self._async_shutdown(), self._event_loop)
            try:
                future.result(timeout=10)
            except Exception:
                pass
            self._event_loop.call_soon_threadsafe(self._event_loop.stop)

        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=5)

        self._pool.shutdown(wait=False)
        logger.info("MolinkWorkerNode shutdown complete")

    async def _async_shutdown(self):
        if self.grpc_server:
            await self.grpc_server.stop(grace=5)
        for channel in self.service._channel_cache.values():
            await channel.close()

    # -- API surface for api_server.py compat --------------------------------

    def get_communication_metrics(self) -> dict:
        service_metrics = []
        if hasattr(self, 'service') and self.service is not None:
            service_metrics = self.service.get_metrics()
        return {
            "service_metrics": service_metrics,
            "delivery_metrics": [],
            "node": self.grpc_address,
            "is_head": False,
        }

    def reset_communication_metrics(self):
        if hasattr(self, 'service') and self.service is not None:
            self.service.reset_metrics()

    def _flush_metrics_to_file(self):
        import json
        import tempfile
        data = self.get_communication_metrics()
        path = os.path.join(tempfile.gettempdir(), f"molink_metrics_{self.grpc_port}.json")
        try:
            tmp_path = path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(data, f)
            os.replace(tmp_path, path)
        except Exception:
            pass

    def _metrics_flush_loop(self):
        while not self._shutdown_event.is_set():
            self._shutdown_event.wait(timeout=2.0)
            try:
                self._flush_metrics_to_file()
            except Exception:
                pass