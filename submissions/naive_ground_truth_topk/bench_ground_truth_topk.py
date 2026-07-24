"""Benchmark wrapper around the `naive_ground_truth_topk` submission.

This module is used as the ``vortex_module_path`` for the per-layer indexer
overhead microbenchmark. It does two things at import time (inside the sglang
scheduler subprocess):

  1. Loads and registers the *unmodified* ground-truth flow from
     ``ground_truth_topk.py`` (so ``vortex_module_name`` resolves normally).
  2. If ``VORTEX_TIME_INDEXER=1``, installs a class-level monkeypatch on
     ``VortexFlashInferBackend`` that wraps the compiled indexer's ``forward``
     (decode) and ``compiled_indexer_prefill.forward`` (prefill) with
     CUDA-event timing, tagging each sample with the transformer ``layer_id``
     and regime, and appending one JSON record per call to the file named by
     ``VORTEX_TIME_OUT``.

The measured quantity is the *indexer only* — i.e. the sparse-routing kernel
the flow adds per layer — NOT the subsequent sparse/dense attention compute.
"""

import os
import json
import importlib.util

# --- 1. Load + register the real flow (runs its @register decorator) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ORIG_PATH = os.path.join(_HERE, "ground_truth_topk.py")
_spec = importlib.util.spec_from_file_location("naive_ground_truth_topk_orig", _ORIG_PATH)
_orig_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_orig_mod)
# Expose the class in this module's namespace too, in case the loader resolves
# vortex_module_name by attribute rather than the global registry.
NaiveGroundTruthTopK = _orig_mod.NaiveGroundTruthTopK


# --- 2. Env-gated per-layer indexer timing patch ---
def _install_probe():
    import torch
    from vortex_torch.engine.sgl.attention_backend import flashinfer as _F

    Backend = _F.VortexFlashInferBackend
    if getattr(Backend, "_probe_installed", False):
        return

    out_path = os.environ.get("VORTEX_TIME_OUT")
    # Shared mutable state: which layer is currently executing, a per-regime
    # tile counter (each regime fires the indexer once per query-tile), and a
    # monotonic sequence id for warmup discarding.
    state = {"layer": -1, "tile": {"decode": 0, "prefill": 0, "prefill_select": 0},
             "seq": 0}

    def _emit(regime, layer, ms, tile, seq):
        rec = {
            "regime": regime,
            "layer": int(layer),
            "ms": float(ms),
            "tile": int(tile),
            "seq": int(seq),
        }
        line = json.dumps(rec)
        if out_path:
            with open(out_path, "a") as fh:
                fh.write(line + "\n")
        print("PROBE " + line, flush=True)

    def _wrap_indexer(fn, regime):
        def wrapped(*args, **kwargs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = fn(*args, **kwargs)
            end.record()
            end.synchronize()
            state["seq"] += 1
            _emit(regime, state["layer"], start.elapsed_time(end),
                  state["tile"][regime], state["seq"])
            state["tile"][regime] += 1
            return out
        return wrapped

    _orig_decode = Backend.forward_decode

    def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True):
        if layer.layer_id not in self.layers_skip:
            if not getattr(self, "_probe_dec", False):
                self.compiled_indexer.forward = _wrap_indexer(
                    self.compiled_indexer.forward, "decode")
                self._probe_dec = True
            state["layer"] = layer.layer_id
            state["tile"]["decode"] = 0
        return _orig_decode(self, q, k, v, layer, forward_batch, save_kv_cache)

    Backend.forward_decode = forward_decode

    _orig_sparse = Backend._forward_extend_sparse

    # Also time the prefill top-k selection (module-level ``select_prefill_fast``,
    # called by _forward_extend_sparse after scoring). The decode indexer figure
    # already fuses score + select, so timing prefill scoring alone would
    # understate it — this keeps the prefill/decode overheads comparable.
    _F.select_prefill_fast = _wrap_indexer(_F.select_prefill_fast, "prefill_select")

    def _forward_extend_sparse(self, q, k, v, layer, forward_batch, cache_loc,
                               logits_soft_cap):
        if not getattr(self, "_probe_pre", False):
            self.compiled_indexer_prefill.forward = _wrap_indexer(
                self.compiled_indexer_prefill.forward, "prefill")
            self._probe_pre = True
        state["layer"] = layer.layer_id
        state["tile"]["prefill"] = 0
        state["tile"]["prefill_select"] = 0
        return _orig_sparse(self, q, k, v, layer, forward_batch, cache_loc,
                            logits_soft_cap)

    Backend._forward_extend_sparse = _forward_extend_sparse
    Backend._probe_installed = True
    print("PROBE installed (VORTEX_TIME_INDEXER=1)", flush=True)


if os.environ.get("VORTEX_TIME_INDEXER") == "1":
    _install_probe()
