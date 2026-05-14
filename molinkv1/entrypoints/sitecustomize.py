"""
sitecustomize.py — auto-applied by Python at startup when placed in site-packages.

When MOLINK_NCCL=1 is set, this patches vLLM's worker init so MoLink custom
layer ranges are respected by every Python process (including Ray workers).
"""
import os

if os.environ.get("MOLINK_NCCL") == "1":
    import sys
    import importlib
    from importlib.abc import MetaPathFinder

    # --- Import hook: patch gpu_worker after it's loaded ---
    class _MolinkPatcher(MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != "vllm.v1.worker.gpu_worker":
                return None
            import importlib._bootstrap
            spec = importlib._bootstrap._find_spec(fullname, path, target)
            if spec is None:
                return None
            original_exec = spec.loader.exec_module

            def _exec_module(module):
                original_exec(module)
                _orig = module.init_worker_distributed_environment

                def _patched(vllm_config, rank, dist_init_method=None,
                             local_rank=-1, backend="nccl"):
                    _orig(vllm_config, rank, dist_init_method, local_rank, backend)
                    start = int(os.environ.get("MOLINK_START_LAYER", "0"))
                    end = int(os.environ.get("MOLINK_END_LAYER", "-1"))
                    from molinkv1.parallel_state import (
                        init_molink_parallel_state,
                        apply_molink_patches,
                    )
                    num_layers = getattr(
                        getattr(vllm_config, 'model_config', None),
                        'get_num_layers', lambda: None,
                    )()
                    init_molink_parallel_state(
                        enabled=True, start_layer=start, end_layer=end,
                        num_hidden_layers=num_layers,
                    )
                    apply_molink_patches()

                module.init_worker_distributed_environment = _patched

            spec.loader.exec_module = _exec_module
            return spec

    sys.meta_path.insert(0, _MolinkPatcher())

    # --- Apply immediately for the current process ---
    _start = int(os.environ.get("MOLINK_START_LAYER", "0"))
    _end = int(os.environ.get("MOLINK_END_LAYER", "-1"))
    from molinkv1.parallel_state import init_molink_parallel_state, apply_molink_patches
    init_molink_parallel_state(enabled=True, start_layer=_start, end_layer=_end)
    apply_molink_patches()
