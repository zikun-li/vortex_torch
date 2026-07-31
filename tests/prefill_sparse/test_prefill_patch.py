from __future__ import annotations

import json

import pytest
import torch

from vortex_torch.engine.sgl.config import PrefillPatchConfig, VortexConfig
from vortex_torch.engine.sgl.prefill_patch import (
    PrefillPatchTrace,
    _TraceWriter,
    extract_tail_selection,
    validate_trace_pair,
)


def _metadata():
    return {
        "model": {"path": "test", "revision": None, "config_sha256": "abc"},
        "geometry": {
            "num_layers": 2,
            "num_q_heads": 4,
            "num_kv_heads": 2,
            "head_dim": 8,
            "block_size": 4,
            "model_dtype": "torch.bfloat16",
            "kv_cache_dtype": "torch.bfloat16",
            "tp_size": 1,
            "pp_size": 1,
        },
        "routing_config": {
            "topk_val": 2,
            "topk_ratio": 0.0,
            "reserved_bos": 1,
            "reserved_eos": 1,
        },
    }


def test_prefill_patch_config_from_flat():
    cfg = VortexConfig.from_flat(
        {
            "vortex_prefill_patch": {
                "mode": "apply",
                "output_dir": "/tmp/out",
                "dense_trace_dir": "/tmp/dense",
                "sparse_trace_dir": "/tmp/sparse",
                "components": "kv",
                "routing": "recompute",
                "layers": [1, 3],
            }
        }
    )
    assert isinstance(cfg.prefill_patch, PrefillPatchConfig)
    assert cfg.prefill_patch.components == "kv"
    assert cfg.prefill_patch.routing == "recompute"
    assert cfg.prefill_patch.layers == [1, 3]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "bad", "output_dir": "/tmp/x"},
        {"mode": "apply", "output_dir": "/tmp/x"},
        {"mode": "capture_dense", "output_dir": "", "layers": []},
        {"mode": "capture_sparse", "output_dir": "/tmp/x", "layers": [1, 1]},
    ],
)
def test_prefill_patch_config_rejects_invalid_values(kwargs):
    with pytest.raises((TypeError, ValueError)):
        PrefillPatchConfig(**kwargs)


def test_extract_tail_selection_head_major():
    # H=2, T=5. Each row owns [row, row+100]. Extract local row 4 from each head.
    lengths = torch.full((10,), 2, dtype=torch.int32)
    ptr = torch.cat([torch.zeros(1, dtype=torch.int32), lengths.cumsum(0)])
    ids = torch.tensor(
        [value for row in range(10) for value in (row, row + 100)],
        dtype=torch.int32,
    )
    out_ptr, out_ids, tail_start = extract_tail_selection(
        ptr,
        ids,
        num_kv_heads=2,
        tile_len=5,
        tile_offset=16,
        seq_len=21,
        block_size=4,
    )
    assert tail_start == 20
    assert out_ptr.tolist() == [0, 2, 4]
    assert out_ids.tolist() == [4, 104, 9, 109]


def test_trace_round_trip_and_pair_validation(tmp_path):
    ids = torch.tensor([3, 4, 5, 6, 7], dtype=torch.int64)
    dense_dir = tmp_path / "dense"
    sparse_dir = tmp_path / "sparse"
    dense_cfg = PrefillPatchConfig(mode="capture_dense", output_dir=str(dense_dir))
    sparse_cfg = PrefillPatchConfig(mode="capture_sparse", output_dir=str(sparse_dir))

    dense_writer = _TraceWriter(
        dense_dir,
        kind="dense",
        base_manifest=_metadata(),
        layers=[0],
        patch_config=dense_cfg,
    )
    dense_writer.set_input(ids)
    dense_writer.write_layer(
        0,
        {
            "q_target": torch.ones(4, 8, dtype=torch.bfloat16),
            "k_prefix": torch.ones(5, 2, 8, dtype=torch.bfloat16),
            "v_prefix": torch.zeros(5, 2, 8, dtype=torch.bfloat16),
        },
    )

    sparse_writer = _TraceWriter(
        sparse_dir,
        kind="sparse",
        base_manifest=_metadata(),
        layers=[0],
        patch_config=sparse_cfg,
    )
    sparse_writer.set_input(ids)
    sparse_writer.write_layer(
        0,
        {
            "kv_indptr": torch.tensor([0, 2, 4], dtype=torch.int32),
            "block_ids": torch.tensor([0, 1, 0, 1], dtype=torch.int32),
            "scores_target": torch.ones(2, 2),
            "tail_start": torch.tensor(4),
            "tail_len": torch.tensor(1),
        },
    )

    dense = PrefillPatchTrace.open(dense_dir)
    sparse = PrefillPatchTrace.open(sparse_dir)
    assert dense.load_layer(0)["q_target"].dtype == torch.bfloat16
    validate_trace_pair(
        dense,
        sparse,
        input_ids=ids,
        current_metadata=_metadata(),
        layers=[0],
    )

    with pytest.raises(ValueError, match="input tokens"):
        validate_trace_pair(
            dense,
            sparse,
            input_ids=torch.tensor([3, 4, 9]),
            current_metadata=_metadata(),
            layers=[0],
        )


def test_incomplete_trace_is_rejected(tmp_path):
    cfg = PrefillPatchConfig(mode="capture_dense", output_dir=str(tmp_path / "x"))
    _TraceWriter(
        tmp_path / "x",
        kind="dense",
        base_manifest=_metadata(),
        layers=[0, 1],
        patch_config=cfg,
    )
    manifest = json.loads((tmp_path / "x" / "manifest.json").read_text())
    assert manifest["complete"] is False
    with pytest.raises(ValueError, match="incomplete"):
        PrefillPatchTrace.open(tmp_path / "x")
