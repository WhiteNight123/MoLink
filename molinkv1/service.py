"""
MoLink gRPC service implementation for cross-node pipeline parallelism.
"""

import asyncio
import struct as _struct
import threading
import time
import traceback
from collections import deque
from typing import TYPE_CHECKING

from vllm.logger import init_logger

from molinkv1.comm import molink_pb2, molink_pb2_grpc
from molinkv1.utils import PipelineTopology, deserialize_metadata

if TYPE_CHECKING:
    from .executor import MolinkExecutor

logger = init_logger(__name__)


class MolinkService(molink_pb2_grpc.MolinkServiceServicer):
    """gRPC service implementation for MoLink cross-node pipeline parallelism.

    This service handles:
    - Pipeline topology management (joining nodes)
    - Intermediate tensor transfer between pipeline stages
    - Sampler output collection at head node
    - Worker step execution triggers
    """

    def __init__(
        self,
        pipeline_size: int,
        executor: "MolinkExecutor",
        head_ip: str,
        start_layer: int,
        end_layer: int,
    ):
        """Initialize the MoLink service.

        Args:
            pipeline_size: Maximum number of concurrent batches/virtual engines.
            executor: The executor that owns this service.
            head_ip: The IP:port of this node.
            start_layer: First layer this node handles.
            end_layer: Last layer this node handles.
        """
        self.executor = executor
        self.pipeline_size = pipeline_size

        # Thread-safe metrics store (no pp_lock — micro-batches overlap freely)
        self._metrics_lock = threading.Lock()
        self._metrics_deque: deque = deque(maxlen=2000)
        self._metrics_enabled = False

        # Queues for inter-stage communication
        # input_queue: receives (scheduler_output, intermediate_tensors, grpc_metadata)
        # output_queue: receives final ModelRunnerOutput
        self.input_queue = [asyncio.Queue() for _ in range(pipeline_size)]
        self.output_queue = [asyncio.Queue() for _ in range(pipeline_size)]

        # Pipeline topology
        self.topology = PipelineTopology(head_ip, start_layer, end_layer)

        logger.info(f"MoLink service initialized for node {head_ip}")

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

    async def JoinPipeline(
        self, request: molink_pb2.NodeInfo, context
    ) -> molink_pb2.GrpcResponseData:
        """Handle a new node joining the pipeline.

        Sends the head node's num_gpu_blocks in the response so the worker
        node can synchronise its KV cache size with the scheduler.

        Args:
            request: NodeInfo containing the joining node's information.
            context: gRPC context.

        Returns:
            GrpcResponseData indicating success or failure.
        """
        try:
            node_ip = request.ip
            start_layer = request.start_layer
            end_layer = request.end_layer

            self.topology.add_node(node_ip, start_layer, end_layer)

            logger.info(
                f"Node {node_ip} joined pipeline " f"(layers {start_layer}-{end_layer})"
            )

            # Send head node's num_gpu_blocks so the worker can cap its own.
            head_num_blocks = getattr(
                self.executor.vllm_config.cache_config, "num_gpu_blocks", 0
            )
            if head_num_blocks is None:
                head_num_blocks = 0

            return molink_pb2.GrpcResponseData(
                res=1,
                output_data=_struct.pack("<Q", head_num_blocks),
            )

        except Exception as e:
            logger.error(f"Error in JoinPipeline: {e}")
            traceback.print_exc()
            return molink_pb2.GrpcResponseData(res=0, error_message=str(e))

    async def GetTopology(
        self, request: molink_pb2.HealthCheckRequest, context
    ) -> molink_pb2.PipelineTopology:
        """Get the current pipeline topology.

        Args:
            request: Health check request (empty).
            context: gRPC context.

        Returns:
            PipelineTopology containing all nodes in the pipeline.
        """
        nodes = []
        for node in self.topology.node_pool:
            nodes.append(
                molink_pb2.NodeInfo(
                    ip=node["ip"],
                    start_layer=node["start_layer"],
                    end_layer=node["end_layer"],
                )
            )

        return molink_pb2.PipelineTopology(nodes=nodes)

    async def PushIntermediateTensors(
        self, request: molink_pb2.GrpcRequestData, context
    ) -> molink_pb2.GrpcResponseData:
        """Receive intermediate tensors from the previous pipeline stage."""
        t_start = time.perf_counter()
        try:
            virtual_engine = request.virtual_engine

            scheduler_output_bytes = request.scheduler_output

            intermediate_tensors_bytes = {}
            total_bytes = 0
            for entry in request.intermediate_tensors.tensors:
                key = entry.key
                byte_data = entry.tensor_data
                intermediate_tensors_bytes[key] = byte_data
                total_bytes += len(byte_data)

            grpc_metadata = deserialize_metadata(request.grpc_metadata)

            await self.input_queue[virtual_engine].put(
                (
                    scheduler_output_bytes,
                    intermediate_tensors_bytes,
                    grpc_metadata,
                )
            )

            self._record_metric({
                "type": "receive_intermediate",
                "bytes": total_bytes,
                "duration_ms": (time.perf_counter() - t_start) * 1000,
                "timestamp": time.time(),
            })

            return molink_pb2.GrpcResponseData(res=1)

        except Exception as e:
            logger.error(f"[MoLink][SERVICE] Error in PushIntermediateTensors: {e}")
            traceback.print_exc()
            return molink_pb2.GrpcResponseData(res=0, error_message=str(e))

    async def PushSamplerOutput(
        self, request: molink_pb2.SamplerOutput, context
    ) -> molink_pb2.GrpcResponseData:
        """Receive sampler output from the last pipeline stage."""
        try:
            virtual_engine = request.virtual_engine
            output_bytes = request.output_data

            print(f"{virtual_engine} 0 back to head at {time.time()}", flush=True)
            await self.output_queue[virtual_engine].put(output_bytes)

            self._record_metric({
                "type": "receive_sampler",
                "bytes": len(output_bytes),
                "timestamp": time.time(),
            })

            return molink_pb2.GrpcResponseData(res=1)

        except Exception as e:
            logger.error(f"[MoLink][SERVICE] Error in PushSamplerOutput: {e}")
            traceback.print_exc()
            return molink_pb2.GrpcResponseData(res=0, error_message=str(e))

    async def HealthCheck(
        self, request: molink_pb2.HealthCheckRequest, context
    ) -> molink_pb2.HealthCheckResponse:
        """Health check endpoint.

        Args:
            request: HealthCheckRequest (empty).
            context: gRPC context.

        Returns:
            HealthCheckResponse indicating the service is healthy.
        """
        return molink_pb2.HealthCheckResponse(status="healthy")
