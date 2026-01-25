from dataclasses import asdict, fields
from vllm.config import VllmConfig
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext
from vllm.v1.metrics.loggers import StatLoggerFactory
from molinkv1.executor import MolinkExecutor
from molinkv1.config import MolinkSchedulerConfig
from molinkv1.config import MolinkConfig, VllmConfig1
from molinkv1.arg_utils import MolinkEngineArgs

logger = init_logger(__name__)


class MolinkEngine(AsyncLLM):

    def __init__(self, *args, **kwargs) -> None:
        config = kwargs.get("vllm_config")
        molink_enabled = kwargs.get("molink_enabled")
        molink_initial_peer = kwargs.get("molink_initial_peer")
        molink_grpc_port = kwargs.get("molink_grpc_port")
        molink_start_layer = kwargs.get("molink_start_layer")
        molink_end_layer = kwargs.get("molink_end_layer")
        config.__class__ = VllmConfig1
        molink_config = MolinkConfig(
            molink_enabled,
            molink_initial_peer,
            molink_grpc_port,
            molink_start_layer,
            molink_end_layer,
        )
        config._update_attr(molink_config)

        # Set worker class to MolinkWorker
        config.parallel_config.worker_cls = "molinkv1.worker.MolinkWorker"

        # Replace scheduler config with MolinkSchedulerConfig, preserving fields
        try:
            sched = config.scheduler_config
            if not isinstance(sched, MolinkSchedulerConfig):
                try:
                    sched_kwargs = asdict(sched)
                except TypeError:
                    # Fallback: copy overlapping fields
                    sched_kwargs = {
                        f.name: getattr(sched, f.name)
                        for f in fields(MolinkSchedulerConfig)
                        if hasattr(sched, f.name)
                    }
                config.scheduler_config = MolinkSchedulerConfig(**sched_kwargs)
        except Exception:
            # As a last resort, coerce the class
            try:
                config.scheduler_config.__class__ = MolinkSchedulerConfig
            except Exception:
                pass

        del kwargs["molink_enabled"]
        del kwargs["molink_initial_peer"]
        del kwargs["molink_grpc_port"]
        del kwargs["molink_start_layer"]
        del kwargs["molink_end_layer"]
        super().__init__(*args, **kwargs)

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
        enable_log_requests: bool = False,
        aggregate_engine_logging: bool = False,
        disable_log_stats: bool = False,
        client_addresses: dict[str, str] | None = None,
        client_count: int = 1,
        client_index: int = 0,
        disable_log_requests: bool = True,  # Deprecated, will be removed
        molink_enabled: bool = False,
        molink_initial_peer: str | None = None,
        molink_grpc_port: int = 0,
        molink_start_layer: int = 0,
        molink_end_layer: int = -1,
        **kwargs,
    ) -> "MolinkEngine":
        """Create a MolinkEngine from VllmConfig.
        
        This override accepts MoLink-specific parameters in addition to
        standard AsyncLLM parameters.
        """
        executor_class = MolinkExecutor
        return cls(
            vllm_config=vllm_config,
            executor_class=executor_class,
            start_engine_loop=start_engine_loop,
            stat_loggers=stat_loggers,
            log_requests=enable_log_requests,
            log_stats=not disable_log_stats,
            aggregate_engine_logging=aggregate_engine_logging,
            usage_context=usage_context,
            client_addresses=client_addresses,
            client_count=client_count,
            client_index=client_index,
            molink_enabled=molink_enabled,
            molink_initial_peer=molink_initial_peer,
            molink_grpc_port=molink_grpc_port,
            molink_start_layer=molink_start_layer,
            molink_end_layer=molink_end_layer,
            **kwargs,
        )

    @classmethod
    def from_engine_args(
        cls,
        engine_args: MolinkEngineArgs,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
    ) -> "AsyncLLM":
        """Create an AsyncLLM from the EngineArgs."""
        # Create the engine configs.
        vllm_config = engine_args.create_engine_config(usage_context)
        executor_class = MolinkExecutor
        # Create the AsyncLLM.
        return cls(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_requests=engine_args.enable_log_requests,
            log_stats=not engine_args.disable_log_stats,
            start_engine_loop=start_engine_loop,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            molink_enabled=engine_args.molink_enabled,
            molink_initial_peer=engine_args.molink_initial_peer,
            molink_grpc_port=engine_args.molink_grpc_port,
            molink_start_layer=engine_args.molink_start_layer,
            molink_end_layer=engine_args.molink_end_layer,
        )
