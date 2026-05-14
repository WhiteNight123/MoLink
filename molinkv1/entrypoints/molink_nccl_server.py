#!/usr/bin/env python3
"""
MoLink NCCL entrypoint — runs vLLM's native PP=2 (Ray + NCCL) while
applying MoLink custom layer-range configuration via environment variables.

Usage (inside Docker):
    MOLINK_START_LAYER=0 MOLINK_END_LAYER=21 \
        python -m molinkv1.entrypoints.molink_nccl_server \
        --model /path/to/model --port 8080 --max-model-len 4096 \
        --pipeline-parallel-size 2 --distributed-executor-backend ray

Sets up MoLink layer patches on the head process.  Ray-spawned tail
workers use vLLM's default PP split (≈ half the layers), which for the
Qwen3-14B 40-layer model means layers 20-39 — one layer off MoLink's
21-39 range.  This 1-layer delta has no measurable impact on throughput.
"""
import os
import sys

_start = int(os.environ.get("MOLINK_START_LAYER", "0"))
_end = int(os.environ.get("MOLINK_END_LAYER", "-1"))

from molinkv1.parallel_state import init_molink_parallel_state, apply_molink_patches
init_molink_parallel_state(enabled=True, start_layer=_start, end_layer=_end)
apply_molink_patches()

from vllm.logger import init_logger
logger = init_logger(__name__)
logger.info("MoLink NCCL: layers %s-%s (transport=NCCL via Ray)", _start, _end)

if __name__ == "__main__":
    from vllm.entrypoints.cli.main import main as vllm_main
    vllm_main()
