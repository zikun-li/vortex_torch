from __future__ import annotations

import math
import os

import pytest
import torch

from vortex_torch.engine.sgl import PrefillPatchTrace, get_engine


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.environ.get("VORTEX_RUN_GPU_TESTS") != "1",
        reason="set VORTEX_RUN_GPU_TESTS=1 to run real-model tests",
    ),
]


def _engine(*, sparse_prefill: bool, patch: dict):
    return get_engine(
        model_path="Qwen/Qwen3-0.6B",
        vortex_module_path=(
            "submissions/ground_truth_kernel_topk/ground_truth_kernel_topk.py"
        ),
        vortex_module_name="ground_truth_kernel_topk_sub",
        vortex_block_size=16,
        vortex_topk_val=29,
        vortex_topk_ratio=0.0,
        vortex_block_reserved_bos=1,
        vortex_block_reserved_eos=2,
        vortex_layers_skip=[],
        vortex_sparse_prefill=sparse_prefill,
        vortex_prefill_patch=patch,
        context_length=1024,
        vortex_max_seq_lens=1024,
        max_prefill_tokens=1024,
        chunked_prefill_size=-1,
        disable_radix_cache=True,
        disable_cuda_graph=True,
        max_running_requests=1,
        mem_fraction_static=0.6,
    )


def _run_once(engine) -> float:
    result = engine.generate(
        input_ids=[100] * 600,
        sampling_params={"temperature": 0.0, "max_new_tokens": 1},
        return_logprob=True,
        token_ids_logprob=[200],
    )
    return float(result["meta_info"]["output_token_logprobs"][0][0])


def _capture(*, sparse_prefill: bool, patch: dict) -> float:
    engine = _engine(sparse_prefill=sparse_prefill, patch=patch)
    try:
        return _run_once(engine)
    finally:
        engine.shutdown()


def test_gt_prefill_patch_capture_and_component_matrix(tmp_path):
    dense_dir = tmp_path / "dense"
    sparse_dir = tmp_path / "sparse"
    layers = [0, 1]

    dense_logp = _capture(
        sparse_prefill=False,
        patch={
            "mode": "capture_dense",
            "output_dir": str(dense_dir),
            "layers": layers,
        },
    )
    sparse_logp = _capture(
        sparse_prefill=True,
        patch={
            "mode": "capture_sparse",
            "output_dir": str(sparse_dir),
            "layers": layers,
        },
    )
    assert math.isfinite(dense_logp)
    assert math.isfinite(sparse_logp)
    dense = PrefillPatchTrace.open(dense_dir)
    sparse = PrefillPatchTrace.open(sparse_dir)
    assert dense.layers == layers
    assert sparse.layers == layers

    for components in ("q", "kv", "qkv"):
        for routing in ("frozen", "recompute"):
            out_dir = tmp_path / f"applied_{components}_{routing}"
            logp = _capture(
                sparse_prefill=True,
                patch={
                    "mode": "apply",
                    "output_dir": str(out_dir),
                    "dense_trace_dir": str(dense_dir),
                    "sparse_trace_dir": str(sparse_dir),
                    "components": components,
                    "routing": routing,
                    "layers": layers,
                },
            )
            assert math.isfinite(logp)
            applied = PrefillPatchTrace.open(out_dir)
            assert applied.layers == layers
            assert applied.manifest["patch"]["components"] == components
            assert applied.manifest["patch"]["routing"] == routing
            for layer_id in layers:
                layer = applied.load_layer(layer_id)
                assert layer["scores_target"].isfinite().all()
                if routing == "frozen":
                    source = sparse.load_layer(layer_id)
                    assert torch.equal(layer["kv_indptr"], source["kv_indptr"])
                    assert torch.equal(layer["block_ids"], source["block_ids"])
