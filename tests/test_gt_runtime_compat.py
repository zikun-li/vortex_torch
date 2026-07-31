import os
import tomllib
from pathlib import Path

import flashinfer.topk
import pytest
import torch

from gt_score_kernels.group_score_topk import assemble
from vortex_torch.version import __version__


def test_module_version_matches_package_version():
    with (Path(__file__).parents[1] / "pyproject.toml").open("rb") as f:
        package_version = tomllib.load(f)["project"]["version"]
    assert __version__ == package_version


def _gt_prefill_select(*args, **kwargs):
    os.environ.setdefault("SGLANG_ENABLE_TORCH_COMPILE", "0")
    from vortex_torch.engine.sgl.attention_backend.gt_runtime import gt_prefill_select

    return gt_prefill_select(*args, **kwargs)


def _patch_assemble(monkeypatch):
    def fake_assemble(out_mid, n_blocks, k_take, **kwargs):
        del n_blocks, k_take, kwargs
        return out_mid.reshape(-1), torch.tensor([0, out_mid.numel()]), None

    monkeypatch.setattr(assemble, "assemble_block_ids", fake_assemble)


def _scores():
    return torch.tensor([[[0.1, 0.4, 0.3, 0.2]]], dtype=torch.float32)


def test_prefill_select_omits_unsupported_false_deterministic_kwarg(monkeypatch):
    _patch_assemble(monkeypatch)
    called = False

    def legacy_topk(input, offsets, lengths, k):
        nonlocal called
        del input, offsets, lengths
        called = True
        return torch.zeros((1, k), dtype=torch.int32)

    monkeypatch.setattr(flashinfer.topk, "top_k_ragged_transform", legacy_topk)
    _gt_prefill_select(
        _scores(), block_size=1, q_offset=3, topk_val=1,
        reserved_bos=1, reserved_eos=1, deterministic=False,
    )
    assert called


def test_prefill_select_rejects_unsupported_enabled_determinism(monkeypatch):
    _patch_assemble(monkeypatch)

    def legacy_topk(input, offsets, lengths, k):
        del input, offsets, lengths
        return torch.zeros((1, k), dtype=torch.int32)

    monkeypatch.setattr(flashinfer.topk, "top_k_ragged_transform", legacy_topk)
    with pytest.raises(RuntimeError, match="requires a FlashInfer"):
        _gt_prefill_select(
            _scores(), block_size=1, q_offset=3, topk_val=1,
            reserved_bos=1, reserved_eos=1, deterministic=True,
        )


def test_prefill_select_forwards_enabled_deterministic_kwarg(monkeypatch):
    _patch_assemble(monkeypatch)
    seen = None

    def modern_topk(input, offsets, lengths, k, deterministic=False):
        nonlocal seen
        del input, offsets, lengths
        seen = deterministic
        return torch.zeros((1, k), dtype=torch.int32)

    monkeypatch.setattr(flashinfer.topk, "top_k_ragged_transform", modern_topk)
    _gt_prefill_select(
        _scores(), block_size=1, q_offset=3, topk_val=1,
        reserved_bos=1, reserved_eos=1, deterministic=True,
    )
    assert seen is True


def test_runtime_module_path_falls_back_to_packaged_submission(monkeypatch, tmp_path):
    from vortex_torch.engine.sgl.api import _resolve_runtime_module_path

    monkeypatch.chdir(tmp_path)
    resolved = Path(_resolve_runtime_module_path("submissions/example_block_sparse_attention.py"))
    assert resolved.is_absolute()
    assert resolved.is_file()
    assert resolved.name == "example_block_sparse_attention.py"
