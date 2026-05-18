"""Lightweight pipeline profiler for MoLink.

Writes per-step timing data to a JSONL file for offline analysis.
Enabled via --molink-enable-metrics flag or MOLINK_PROFILER_DIR env var.
"""

import json
import os
import threading
import time
from pathlib import Path


class PipelineProfiler:
    _instance = None
    _lock = threading.Lock()

    def __init__(self, output_dir: str, node_name: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.node_name = node_name
        self._file_lock = threading.Lock()
        ts = int(time.time())
        self._path = self.output_dir / f"profile_{node_name}_{ts}.jsonl"
        self._fh = open(self._path, "a")
        self._step_counter = 0
        self._step_lock = threading.Lock()

    def next_step_id(self) -> int:
        with self._step_lock:
            sid = self._step_counter
            self._step_counter += 1
            return sid

    def record(self, stage: str, timings: dict, step_id: int = -1, **extra):
        entry = {
            "ts": time.time(),
            "node": self.node_name,
            "step_id": step_id,
            "stage": stage,
            **timings,
            **extra,
        }
        line = json.dumps(entry, default=str)
        with self._file_lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self):
        with self._file_lock:
            self._fh.close()

    @property
    def path(self) -> str:
        return str(self._path)


# Module-level helpers

_profiler: PipelineProfiler | None = None


def init_profiler(output_dir: str, node_name: str) -> PipelineProfiler:
    global _profiler
    _profiler = PipelineProfiler(output_dir, node_name)
    return _profiler


def get_profiler() -> PipelineProfiler | None:
    return _profiler


def record(stage: str, timings: dict, step_id: int = -1, **extra):
    if _profiler is not None:
        _profiler.record(stage, timings, step_id, **extra)
