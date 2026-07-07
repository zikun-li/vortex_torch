"""Tiny opt-in CUDA-event profiler for the GT decode path (debug only).

Disabled by default and **zero-cost when off** (``gtprof`` returns a shared
no-op context manager). Enable by setting the env var ``GT_DECODE_PROFILE`` OR
creating the sentinel file ``logs/decode_kernel/GTPROF_ON`` before booting the
engine — the sentinel is the reliable switch because sglang's scheduler runs in
a spawned subprocess whose env/`/tmp`/mount view may not match the launcher's.

When enabled it accumulates per-stage GPU time across all decode layer-steps and,
because the scheduler's filesystem writes may land in a private namespace,
periodically echoes the running totals to **stderr** (``[GTPROF] ...``, captured
in the engine log) in addition to writing ``GT_DECODE_PROFILE_OUT``.

Each region syncs on exit to read its own elapsed time; the decode stages are
data-dependent (score -> top-k -> attention) so they don't overlap anyway, and
the profiling run only cares about the relative split.
"""

from __future__ import annotations

import atexit
import json
import os
import sys

import torch

# Sentinel path is resolved against the engine's CWD (the repo root the launcher
# runs from), which the scheduler subprocess inherits; override either with an env
# var. Kept relative so nothing machine-specific is baked in.
_SENTINEL = os.environ.get("GT_DECODE_PROFILE_SENTINEL", "logs/decode_kernel/GTPROF_ON")
_ENABLED = bool(os.environ.get("GT_DECODE_PROFILE")) or os.path.exists(_SENTINEL)
_OUT = os.environ.get("GT_DECODE_PROFILE_OUT", "logs/decode_kernel/gt_decode_profile.json")
_acc: dict[str, list] = {}          # stage -> [total_ms, count]
_since_dump = 0
_DUMP_EVERY = 200


class _NoOp:
    __slots__ = ()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_NOOP = _NoOp()


class _Region:
    __slots__ = ("stage", "s", "e")

    def __init__(self, stage: str):
        self.stage = stage

    def __enter__(self):
        self.s = torch.cuda.Event(enable_timing=True)
        self.e = torch.cuda.Event(enable_timing=True)
        self.s.record()
        return self

    def __exit__(self, *exc):
        self.e.record()
        self.e.synchronize()
        ms = self.s.elapsed_time(self.e)
        rec = _acc.setdefault(self.stage, [0.0, 0])
        rec[0] += ms
        rec[1] += 1
        global _since_dump
        _since_dump += 1
        if _since_dump >= _DUMP_EVERY:   # periodic dump survives an unclean exit
            _since_dump = 0
            _dump()
        return False


def gtprof(stage: str):
    """Context manager timing ``stage``; a no-op (no CUDA events) when disabled."""
    return _Region(stage) if _ENABLED else _NOOP


def _dump():
    if not _acc:
        return
    out = {s: {"total_ms": v[0], "count": v[1],
               "avg_ms": (v[0] / v[1] if v[1] else 0.0)}
           for s, v in _acc.items()}
    print(f"[GTPROF] {json.dumps(out)}", file=sys.stderr, flush=True)
    try:
        with open(_OUT, "w") as f:
            json.dump(out, f, indent=2)
    except OSError:
        pass


if _ENABLED:
    atexit.register(_dump)
