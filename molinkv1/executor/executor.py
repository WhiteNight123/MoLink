"""
MoLink Executor for cross-node pipeline parallelism in vLLM v1.

This executor enables distributed pipeline parallelism across multiple physical
nodes using gRPC for communication. It extends the MultiprocExecutor to handle
cross-node tensor transfer and synchronization.
"""

import asyncio
import os
import pickle
import struct
import threading
import time
import traceback
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import cloudpickle
import grpc.aio as aio
import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.v1.outputs import ModelRunnerOutput

from molinkv1.profiler import get_profiler
from molinkv1.service import MolinkService
from molinkv1.utils import (
    extract_ip,
    find_free_port,
    get_grpc_options,
    serialize_metadata,
)
from molinkv1.parallel_state import (
    init_molink_parallel_state,
    destroy_molink_parallel_state,
    is_molink_last_stage,
)
from molinkv1.comm import molink_pb2, molink_pb2_grpc

if TYPE_CHECKING:
    from molinkv1.config import MolinkConfig
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


def _serialize_tensors(tensors_cpu: Dict[str, torch.Tensor]) -> molink_pb2.IntermediateTensors:
    """Serialize a dict of CPU tensors into a protobuf IntermediateTensors message.

    Uses a flat binary layout per tensor to minimize Python overhead:
    [ndim(4B)][shape(ndim*8B)][dtype_len(4B)][dtype][raw_data]
    """
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

        # Pre-compute header size and use single allocation
        header_size = 4 + ndim * 8 + 4 + len(dtype_bytes)
        buf = bytearray(header_size + len(raw))
        off = 0

        # Shape
        struct.pack_into("<I", buf, off, ndim); off += 4
        for dim in shape:
            struct.pack_into("<Q", buf, off, dim); off += 8

        # Dtype
        struct.pack_into("<I", buf, off, len(dtype_bytes)); off += 4
        buf[off:off + len(dtype_bytes)] = dtype_bytes; off += len(dtype_bytes)

        # Raw data
        buf[off:off + len(raw)] = raw

        grpc_tensors.tensors.append(
            molink_pb2.TensorEntry(key=key, tensor_data=bytes(buf))
        )
    return grpc_tensors


def _serialize_combined(scheduler_pickle: bytes, tensors: Dict[str, torch.Tensor]) -> bytes:
    """Serialize scheduler output + tensors into a single flat bytes buffer.

    Layout:
      [scheduler_len: 8B][scheduler_pickle]
      [num_tensors: 4B]
      For each tensor:
        [key_len: 4B][key][tensor_data_len: 8B][tensor_data]
      where tensor_data = [ndim:4B][shape:ndim*8B][dtype_len:4B][dtype][raw]

    Pre-computes total size and allocates once to minimize memory churn.
    """
    # Phase 1: prepare raw data + compute total size.
    entries: list[tuple[bytes, bytes]] = []  # [(key_bytes, tensor_data), ...]

    sched_header = struct.pack("<Q", len(scheduler_pickle))
    num_tensors_header = struct.pack("<I", len(tensors))
    total = len(sched_header) + len(scheduler_pickle) + len(num_tensors_header)

    for key, tensor in tensors.items():
        tensor_cpu = tensor.detach().cpu()
        shape = tensor_cpu.shape
        ndim = len(shape)
        dtype_str = str(tensor_cpu.dtype)
        dtype_bytes = dtype_str.encode("ascii")

        if tensor_cpu.dtype == torch.bfloat16:
            t_bf = tensor_cpu.contiguous()
            raw = t_bf.view(torch.uint8).numpy().tobytes() if t_bf.dim() != 0 \
                else t_bf.unsqueeze(0).view(torch.uint8).numpy().tobytes()
        else:
            raw = tensor_cpu.numpy().tobytes()

        header_size = 4 + ndim * 8 + 4 + len(dtype_bytes)
        td_len = header_size + len(raw)
        key_bytes = key.encode("ascii")
        total += 4 + len(key_bytes) + 8 + td_len

        tensor_data = bytearray(td_len)
        off = 0
        struct.pack_into("<I", tensor_data, off, ndim); off += 4
        for dim in shape:
            struct.pack_into("<Q", tensor_data, off, dim); off += 8
        struct.pack_into("<I", tensor_data, off, len(dtype_bytes)); off += 4
        tensor_data[off:off + len(dtype_bytes)] = dtype_bytes; off += len(dtype_bytes)
        tensor_data[off:off + len(raw)] = raw

        entries.append((key_bytes, bytes(tensor_data)))

    # Phase 2: single allocation + copy.
    buf = bytearray(total)
    off = 0
    buf[off:off + len(sched_header)] = sched_header; off += len(sched_header)
    buf[off:off + len(scheduler_pickle)] = scheduler_pickle; off += len(scheduler_pickle)
    buf[off:off + len(num_tensors_header)] = num_tensors_header; off += len(num_tensors_header)

    for key_bytes, tensor_data in entries:
        struct.pack_into("<I", buf, off, len(key_bytes)); off += 4
        buf[off:off + len(key_bytes)] = key_bytes; off += len(key_bytes)
        struct.pack_into("<Q", buf, off, len(tensor_data)); off += 8
        buf[off:off + len(tensor_data)] = tensor_data; off += len(tensor_data)

    return bytes(buf)


class MolinkExecutor(MultiprocExecutor):
    """Executor for cross-node pipeline parallelism using gRPC."""

    supports_pp: bool = True

    @property
    def max_concurrent_batches(self) -> int:
        config = self.molink_config
        result = config.max_concurrent_batches if config is not None else 1
        if (config is not None and config.is_head_node
                and config.get_serving_layers()[1] != -1):
            result = max(2, result)
        return result

    def __init__(self, vllm_config: VllmConfig, monitor_workers: bool = True):
        self.molink_config: "MolinkConfig" = getattr(
            vllm_config, "molink_config", None
        )

        if self.molink_config is None:
            from molinkv1.config import MolinkConfig
            self.molink_config = MolinkConfig()

        # Initialize MoLink parallel state BEFORE parent initialization
        start_layer, end_layer = self.molink_config.get_serving_layers()
        init_molink_parallel_state(
            enabled=True,
            start_layer=start_layer,
            end_layer=end_layer,
        )

        # gRPC server and service
        self.grpc_server: Optional[aio.Server] = None
        self.molink_service: Optional[MolinkService] = None

        # Node information
        self.ip: Optional[str] = None
        self.grpc_port: Optional[int] = None
        self.grpc_address: Optional[str] = None

        # Stub cache for connecting to other nodes
        self._channel_cache: Dict[str, aio.Channel] = {}

        # Event loop for asyncio in separate thread
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()

        # Thread pool for gRPC calls
        self._executor_pool = ThreadPoolExecutor(max_workers=16)

        # Per-VE pipeline futures: execute_model submits the cross-node
        # pipeline to the event loop immediately after head compute, so
        # gRPC serialization/transfer overlaps with subsequent batches.
        # sample_tokens retrieves the pre-submitted Future via _last_submitted_ve.
        self._pipeline_futures: Dict[int, Future] = {}
        self._last_submitted_ve: int = 0

        # Cached pipeline metadata (avoid per-step serialization).
        self._molink_server_list: list | None = None
        self._molink_grpc_metadata_bytes: bytes | None = None

        # Virtual engine counter for multiplexing concurrent batches.
        # Each concurrent batch gets a distinct slot so that cross-batch
        # interference cannot happen even under gRPC errors or timeouts.
        self._virtual_engine_counter: int = 0

        # Pipeline step counter for profiling correlation.
        self._step_id_counter: int = 0

        # Initialize parent executor
        super().__init__(vllm_config, monitor_workers=monitor_workers)

    def _init_executor(self) -> None:
        """Initialize the executor with MoLink support."""
        # First initialize the parent executor (creates local workers)
        super()._init_executor()

        # Initialize MoLink components
        self._init_molink()

    def _is_molink_last_stage(self) -> bool:
        return is_molink_last_stage()

    def _init_molink(self) -> None:
        """Initialize MoLink gRPC server and services."""
        config = self.molink_config

        # Get node IP
        self.ip = extract_ip()

        # Find available gRPC port
        self.grpc_port = find_free_port(
            start_port=config.grpc_port if config.grpc_port > 0 else 50051
        )

        self.grpc_address = f"{self.ip}:{self.grpc_port}"
        logger.info(f"MoLink gRPC server starting at {self.grpc_address}")
        logger.info(
            "DISTRIBUTED SERVICE INFO: If this is the first node of the swarm, "
            f"you can copy the GRPC INFO ({self.grpc_address}) as the initial peer of following nodes"
        )

        # Get layer range
        start_layer, end_layer = config.get_serving_layers()

        # Start event loop in a separate thread
        self._start_event_loop_thread()

        # Schedule gRPC server start in the event loop
        future = asyncio.run_coroutine_threadsafe(
            self._init_grpc_server(start_layer, end_layer), self._event_loop
        )
        future.result(timeout=30)

        # If not head node, join the pipeline
        if not config.is_head_node:
            future = asyncio.run_coroutine_threadsafe(
                self._join_pipeline(), self._event_loop
            )
            try:
                future.result(timeout=30)
            except Exception as e:
                logger.error(f"Failed to join pipeline: {e}")

        logger.info(
            f"MoLink executor initialized. "
            f"Head node: {config.is_head_node}, "
            f"Serving layers: {start_layer}-{end_layer}"
        )

        # Start periodic metrics flush to temp file (read by api_server).
        if config.enable_metrics:
            from molinkv1.profiler import init_profiler
            profiler_dir = os.environ.get("MOLINK_PROFILER_DIR", "/tmp/molink_profile")
            init_profiler(profiler_dir, f"head_{self.grpc_port}")
            self._metrics_flush_thread = threading.Thread(
                target=self._metrics_flush_loop, daemon=True, name="MolinkMetricsFlush"
            )
            self._metrics_flush_thread.start()

    def _start_event_loop_thread(self) -> None:
        """Start a thread with an event loop for asyncio operations."""
        loop_ready = threading.Event()

        def run_loop():
            self._event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._event_loop)
            loop_ready.set()
            self._event_loop.run_forever()

        self._loop_thread = threading.Thread(
            target=run_loop, daemon=True, name="MolinkEventLoop"
        )
        self._loop_thread.start()

        # Wait for event loop to be ready
        loop_ready.wait(timeout=10)

    async def _init_grpc_server(self, start_layer: int, end_layer: int) -> None:
        """Initialize and start the gRPC server."""
        config = self.molink_config

        self.grpc_server = aio.server(
            self._executor_pool, options=get_grpc_options(config.max_message_size_mb)
        )

        max_batch_num = max(config.max_concurrent_batches, 10)
        self.molink_service = MolinkService(
            pipeline_size=max_batch_num,
            executor=self,
            head_ip=self.grpc_address,
            start_layer=start_layer,
            end_layer=end_layer,
        )
        self.molink_service._metrics_enabled = config.enable_metrics

        molink_pb2_grpc.add_MolinkServiceServicer_to_server(
            self.molink_service, self.grpc_server
        )

        self.grpc_server.add_insecure_port(f"[::]:{self.grpc_port}")

        await self.grpc_server.start()
        logger.info(f"MoLink gRPC server started on port {self.grpc_port}")

    async def _join_pipeline(self) -> None:
        """Join an existing pipeline as a worker node."""
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

        except Exception as e:
            logger.error(f"Error joining pipeline: {e}")
            traceback.print_exc()
            raise
        finally:
            if channel is not None:
                await channel.close()

    async def _push_intermediate_tensors(
        self,
        tensors: Dict[str, torch.Tensor],
        scheduler_output: "SchedulerOutput",
        grpc_metadata_bytes: bytes,
        virtual_engine: int,
        next_server: str,
        step_id: int = -1,
    ) -> dict:
        """Serialize and send intermediate tensors + scheduler_output to the next stage.

        All CPU-bound serialization (pickle + tensor copies) runs in a single
        thread-pool call; the gRPC call is awaited to detect failures early.

        Returns timing dict for instrumentation.
        """
        timings = {}
        loop = asyncio.get_running_loop()

        def _prepare_request():
            t0 = time.perf_counter()
            sched_bytes = pickle.dumps(scheduler_output, pickle.HIGHEST_PROTOCOL)
            t_pickle = time.perf_counter()
            combined = _serialize_combined(sched_bytes, tensors)
            t_serialize = time.perf_counter()
            timings["pickle_ms"] = (t_pickle - t0) * 1000
            timings["tensor_serialize_ms"] = (t_serialize - t_pickle) * 1000
            timings["total_serialize_ms"] = (t_serialize - t0) * 1000
            timings["serialized_bytes"] = len(combined)
            return molink_pb2.GrpcRequestData(
                scheduler_output=combined,
                grpc_metadata=grpc_metadata_bytes,
                virtual_engine=virtual_engine,
                step_id=step_id,
            )

        t_ser_start = time.perf_counter()
        request = await loop.run_in_executor(self._executor_pool, _prepare_request)
        t_ser_end = time.perf_counter()

        t_grpc_start = time.perf_counter()
        stub = self._get_stub(next_server)
        await stub.PushIntermediateTensors(request)
        t_grpc_end = time.perf_counter()

        timings["serialize_wall_ms"] = (t_ser_end - t_ser_start) * 1000
        timings["grpc_send_ms"] = (t_grpc_end - t_grpc_start) * 1000
        profiler = get_profiler()
        if profiler is not None:
            profiler.record("head_push", timings, step_id=step_id)
        return timings

    def _get_stub(self, address: str) -> molink_pb2_grpc.MolinkServiceStub:
        if address not in self._channel_cache:
            channel = aio.insecure_channel(
                address,
                options=get_grpc_options(self.molink_config.max_message_size_mb),
            )
            self._channel_cache[address] = channel
            return molink_pb2_grpc.MolinkServiceStub(channel)
        return molink_pb2_grpc.MolinkServiceStub(self._channel_cache[address])

    async def _push_sampler_output(
        self,
        output_bytes: bytes,
        virtual_engine: int,
        head_server: str,
    ) -> None:
        """Send sampler output to the head node via gRPC."""
        request = molink_pb2.SamplerOutput(
            output_data=output_bytes, virtual_engine=virtual_engine
        )
        stub = self._get_stub(head_server)
        await stub.PushSamplerOutput(request)

    def execute_model(
        self, scheduler_output: "SchedulerOutput", non_block: bool = False
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        """Execute the model on local workers.

        For the head node, runs head compute synchronously (even when
        non_block=True) so intermediate tensors are stored before
        sample_tokens runs the cross-node pipeline.  The result (None)
        is wrapped in a resolved Future when non_block=True.
        """
        if self.molink_config.is_head_node and not self._is_molink_last_stage():
            if scheduler_output.total_num_scheduled_tokens > 0:
                t_start = time.perf_counter()
                step_id = self._step_id_counter
                self._step_id_counter += 1
                # Assign a distinct virtual engine slot to this batch so
                # concurrent batches do not share the same gRPC queue and
                # cannot contaminate each other under errors or timeouts.
                max_ve = self.max_concurrent_batches
                scheduler_output.virtual_engine = self._virtual_engine_counter
                self._virtual_engine_counter = (self._virtual_engine_counter + 1) % max_ve
                scheduler_output._molink_step_id = step_id
                # Always run head compute synchronously so intermediate
                # tensors are ready before sample_tokens.
                result = super().execute_model(scheduler_output, non_block=False)
                t_after_compute = time.perf_counter()
                # Retrieve intermediate tensors immediately in the engine
                # thread to avoid RPC races with _do_pipeline coroutines.
                tensors_result = MultiprocExecutor.collective_rpc(
                    self, "_molink_get_intermediate_tensors")
                intermediate = (tensors_result[0] if isinstance(tensors_result, list)
                               else tensors_result)
                # Submit cross-node pipeline to event loop IMMEDIATELY so
                # gRPC serialization/transfer starts before sample_tokens
                # is even called.  This lets head GPU compute for batch
                # N+1 overlap with tail GPU compute for batch N.
                ve = scheduler_output.virtual_engine
                self._last_submitted_ve = ve
                self._pipeline_futures[ve] = asyncio.run_coroutine_threadsafe(
                    self._run_cross_node_pipeline(scheduler_output, intermediate),
                    self._event_loop,
                )
                head_compute_ms = (t_after_compute - t_start) * 1000
                head_total_ms = (time.perf_counter() - t_start) * 1000
                self.molink_service._record_metric({
                    "type": "head_compute",
                    "compute_ms": head_total_ms,
                    "num_tokens": scheduler_output.total_num_scheduled_tokens,
                    "timestamp": time.time(),
                })
                profiler = get_profiler()
                if profiler is not None:
                    tensor_size_mb = 0
                    if intermediate is not None:
                        for t in intermediate.tensors.values():
                            tensor_size_mb += t.element_size() * t.nelement()
                        tensor_size_mb /= 1024 * 1024
                    profiler.record("head_compute", {
                        "head_compute_ms": head_compute_ms,
                        "head_total_ms": head_total_ms,
                        "num_tokens": scheduler_output.total_num_scheduled_tokens,
                        "tensor_size_mb": round(tensor_size_mb, 2),
                        "virtual_engine": scheduler_output.virtual_engine,
                    }, step_id=step_id)
                # When engine core requested non_block, wrap the result
                # in an already-resolved Future.
                if non_block:
                    f: Future = Future()
                    f.set_result(result)
                    return f
                return result

        return super().execute_model(scheduler_output, non_block)

    def sample_tokens(
        self, grammar_output: Any, non_block: bool = False
    ) -> ModelRunnerOutput | Future[ModelRunnerOutput]:
        """Sample tokens or orchestrate cross-node pipeline."""
        if not (self.molink_config.is_head_node and not self._is_molink_last_stage()):
            return super().sample_tokens(grammar_output, non_block)

        return self._sample_tokens_distributed(non_block)

    async def _run_cross_node_pipeline(
        self, scheduler_output, intermediate_tensors
    ) -> ModelRunnerOutput:
        """Run the full cross-node pipeline: serialize → send → wait → receive.

        This is submitted to the event loop immediately after head compute
        (in execute_model), so gRPC work starts before sample_tokens is called.
        """
        t_total_start = time.perf_counter()

        try:
            return await self._run_cross_node_pipeline_inner(
                scheduler_output, intermediate_tensors, t_total_start
            )
        except Exception as e:
            # Drain any stale result the tail may have pushed.
            ve = getattr(scheduler_output, "virtual_engine", 0)
            try:
                self.molink_service.output_queue[ve].get_nowait()
            except asyncio.QueueEmpty:
                pass
            # Collect ALL request IDs the scheduler expects.
            req_ids_set: set = set()
            req_ids_set.update(scheduler_output.num_scheduled_tokens.keys())
            for nr in getattr(scheduler_output, "scheduled_new_reqs", []) or []:
                req_ids_set.add(nr.req_id)
            cr = getattr(scheduler_output, "scheduled_cached_reqs", None)
            if cr is not None and cr.req_ids:
                req_ids_set.update(cr.req_ids)
            req_ids = list(req_ids_set)
            logger.error(
                "[MoLink][PIPELINE] gRPC pipeline failed for VE %s (%d reqs): %s. "
                "Returning empty output so engine survives.",
                ve, len(req_ids), e
            )
            from vllm.v1.outputs import ModelRunnerOutput as MRO
            return MRO(
                req_ids=req_ids,
                req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
                sampled_token_ids=[[0] for _ in req_ids],
            )

    async def _run_cross_node_pipeline_inner(
        self, scheduler_output, intermediate_tensors, t_total_start: float,
    ) -> ModelRunnerOutput:
        if intermediate_tensors is None:
            logger.error(
                "[MoLink][PIPELINE] intermediate_tensors is None - "
                "model runner may not have produced them. "
                "Falling back to local sample_tokens."
            )
            future = MultiprocExecutor.sample_tokens(
                self, None, non_block=True
            )
            return await asyncio.get_running_loop().run_in_executor(
                None, future.result
            )

        virtual_engine = getattr(scheduler_output, "virtual_engine", 0)
        step_id = getattr(scheduler_output, "_molink_step_id", -1)

        # Get pipeline metadata (cached after first call).
        server_list = self._molink_server_list
        if server_list is None:
            grpc_metadata = self.molink_service.topology.get_metadata()
            server_list = grpc_metadata.get("server_list", [])
            self._molink_server_list = server_list
            self._molink_grpc_metadata_bytes = serialize_metadata(grpc_metadata)

        if len(server_list) < 2:
            logger.error(
                f"[MoLink][PIPELINE] Not enough servers in topology: "
                f"{server_list}. Falling back to local sample_tokens."
            )
            future = MultiprocExecutor.sample_tokens(
                self, None, non_block=True
            )
            return await asyncio.get_running_loop().run_in_executor(
                None, future.result
            )

        # Serialize and send in one thread-pool call.
        loop = asyncio.get_running_loop()
        next_server = server_list[1]
        t_push_start = time.perf_counter()
        push_timings = await self._push_intermediate_tensors(
            intermediate_tensors.tensors,
            scheduler_output,
            self._molink_grpc_metadata_bytes,
            virtual_engine,
            next_server,
            step_id=step_id,
        )
        t_push_end = time.perf_counter()

        # Wait for final result from output_queue.
        t_wait_start = time.perf_counter()
        try:
            output_bytes = await asyncio.wait_for(
                self.molink_service.output_queue[virtual_engine].get(),
                timeout=120.0,
            )
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"[MoLink][VE{virtual_engine}] Timed out after 120s "
                f"waiting for tail result — tail may have crashed."
            )
        t_wait_end = time.perf_counter()

        t_deser_start = time.perf_counter()
        result = await loop.run_in_executor(None, pickle.loads, output_bytes)
        t_deser_end = time.perf_counter()

        # Defensive: pad result if it's missing requests the scheduler expects.
        expected = list(scheduler_output.num_scheduled_tokens.keys())
        missing = [r for r in expected if r not in result.req_id_to_index]
        if missing:
            logger.warning(
                "[MoLink][VE%d] Tail output missing %d reqs (out of %d). "
                "Patching result.",
                virtual_engine, len(missing), len(expected),
            )
            all_ids = list(dict.fromkeys(expected + result.req_ids))
            result.req_ids = all_ids
            result.req_id_to_index = {rid: i for i, rid in enumerate(all_ids)}
            while len(result.sampled_token_ids) < len(all_ids):
                result.sampled_token_ids.append([0])

        metric = {
            "type": "head_pipeline",
            "push_intermediate_ms": (t_push_end - t_push_start) * 1000,
            "wait_result_ms": (t_wait_end - t_wait_start) * 1000,
            "deserialize_result_ms": (t_deser_end - t_deser_start) * 1000,
            "result_bytes": len(output_bytes),
            "total_pipeline_ms": (time.perf_counter() - t_total_start) * 1000,
            "virtual_engine": virtual_engine,
            "num_servers": len(server_list),
            "timestamp": time.time(),
        }
        metric.update(push_timings)
        self.molink_service._record_metric(metric)

        profiler = get_profiler()
        if profiler is not None:
            profiler.record("head_pipeline", {
                "total_pipeline_ms": metric["total_pipeline_ms"],
                "wait_result_ms": metric["wait_result_ms"],
                "deserialize_result_ms": metric["deserialize_result_ms"],
                "result_bytes": metric["result_bytes"],
            }, step_id=step_id, virtual_engine=virtual_engine)

        return result

    def _sample_tokens_distributed(
        self, non_block: bool
    ) -> ModelRunnerOutput | Future[ModelRunnerOutput]:
        """Retrieve the pre-submitted pipeline Future.

        execute_model already submitted _run_cross_node_pipeline to the
        event loop; we just need to wait for (or return) its Future.
        """
        ve = self._last_submitted_ve
        future = self._pipeline_futures.pop(ve, None)
        if future is None:
            raise RuntimeError(
                f"[MoLink] sample_tokens called without pending pipeline "
                f"future for VE {ve}"
            )
        if non_block:
            return future
        else:
            return future.result()

    def shutdown(self) -> None:
        """Shutdown the executor and clean up resources."""
        destroy_molink_parallel_state()

        # Stop gRPC server and close channels in event loop
        if self._event_loop and self._event_loop.is_running():
            future = asyncio.run_coroutine_threadsafe(
                self._async_shutdown(), self._event_loop
            )
            try:
                future.result(timeout=10)
            except Exception as e:
                logger.warning(f"Error during async shutdown: {e}")

        # Stop event loop
        if self._event_loop and self._event_loop.is_running():
            self._event_loop.call_soon_threadsafe(self._event_loop.stop)

        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=5)

        self._executor_pool.shutdown(wait=False)

        super().shutdown()
        logger.info("MoLink executor shutdown complete")

    def get_communication_metrics(self) -> dict:
        return {
            "service_metrics": self.molink_service.get_metrics() if self.molink_service else [],
            "node": self.grpc_address,
            "is_head": self.molink_config.is_head_node,
        }

    def reset_communication_metrics(self):
        if self.molink_service:
            self.molink_service.reset_metrics()

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

    async def _async_shutdown(self) -> None:
        if self.grpc_server:
            await self.grpc_server.stop(grace=5)
        for channel in self._channel_cache.values():
            await channel.close()
