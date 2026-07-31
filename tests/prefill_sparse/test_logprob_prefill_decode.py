from __future__ import annotations

import os

import pytest
import torch

import sglang as sgl
from vortex_torch.engine.sgl import get_engine


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.environ.get("VORTEX_RUN_GPU_TESTS") != "1",
        reason="set VORTEX_RUN_GPU_TESTS=1 to run real-model tests",
    ),
]

LOCAL_LENGTHS = [495, 496, 511, 512, 513, 527, 528, 769, 1209]
LOCAL_MAX_ABS = 0.06
ROLLOUT_TOKENS = 128
ROLLOUT_MEAN_MARGIN = 0.015
ROLLOUT_P95_MARGIN = 0.04


def _model_names() -> list[str]:
    value = os.environ.get("VORTEX_PARITY_MODELS", "Qwen/Qwen3-4B,Qwen/Qwen3-32B")
    return [x.strip() for x in value.split(",") if x.strip()]


def _memory_fraction(model: str) -> float:
    return 0.8 if "32B" in model else 0.7


def _sparse_engine(model: str):
    return get_engine(
        model_path=model,
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
        vortex_sparse_prefill=True,
        context_length=2048,
        vortex_max_seq_lens=2048,
        max_prefill_tokens=2048,
        chunked_prefill_size=-1,
        disable_radix_cache=True,
        disable_cuda_graph=True,
        max_running_requests=1,
        mem_fraction_static=_memory_fraction(model),
    )


def _dense_engine(model: str):
    return sgl.Engine(
        model_path=model,
        attention_backend="flashinfer",
        page_size=16,
        dtype="bfloat16",
        kv_cache_dtype="auto",
        context_length=2048,
        max_prefill_tokens=2048,
        chunked_prefill_size=-1,
        disable_radix_cache=True,
        disable_cuda_graph=True,
        disable_overlap_schedule=True,
        max_running_requests=1,
        tp_size=1,
        mem_fraction_static=_memory_fraction(model),
    )


def _decode_and_rescore(
    engine,
    prompt: str | list[int],
    *,
    new_tokens: int,
    seed: int,
    temperature: float,
):
    prompt_kwargs = (
        {"prompt": prompt} if isinstance(prompt, str) else {"input_ids": prompt}
    )
    generated = engine.generate(
        **prompt_kwargs,
        sampling_params={
            "temperature": temperature,
            "top_p": 1.0,
            "top_k": -1,
            "sampling_seed": seed,
            "max_new_tokens": new_tokens,
            "ignore_eos": True,
        },
        return_logprob=True,
        logprob_start_len=0,
    )
    meta = generated["meta_info"]
    prompt_tokens = [x[1] for x in meta["input_token_logprobs"]]
    output = meta["output_token_logprobs"]
    output_tokens = [x[1] for x in output]
    decode_logp = torch.tensor([x[0] for x in output], dtype=torch.float64)

    rescored = engine.generate(
        input_ids=prompt_tokens + output_tokens,
        sampling_params={"temperature": 0.0, "max_new_tokens": 0},
        return_logprob=True,
        logprob_start_len=0,
    )
    scored_output = rescored["meta_info"]["input_token_logprobs"][len(prompt_tokens) :]
    assert [x[1] for x in scored_output] == output_tokens
    prefill_logp = torch.tensor([x[0] for x in scored_output], dtype=torch.float64)
    return decode_logp, prefill_logp


def _local_boundary_max(engine) -> tuple[float, list[tuple[int, float]]]:
    rows = []
    for length in LOCAL_LENGTHS:
        decode, prefill = _decode_and_rescore(
            engine, [100] * length, new_tokens=2, seed=0, temperature=0.0
        )
        # Token 0 is sampled from prompt-prefill logits. Token 1 is the first
        # token whose logits were produced by an actual decode attention step.
        diff = float((decode[1] - prefill[1]).abs())
        rows.append((length, diff))
    return max(diff for _, diff in rows), rows


def _rollout_stats(engine) -> dict[str, float]:
    prompt = (
        "The quick brown fox jumps over the lazy dog and then carefully returns home. "
        * 80
        + "\nContinue this passage with a detailed argument:"
    )
    decode, prefill = _decode_and_rescore(
        engine,
        prompt,
        new_tokens=ROLLOUT_TOKENS,
        seed=1234,
        temperature=1.0,
    )
    diff = (decode[1:] - prefill[1:]).abs()
    return {
        "mean": float(diff.mean()),
        "p95": float(torch.quantile(diff, 0.95)),
        "max": float(diff.max()),
    }


@pytest.mark.parametrize("model", _model_names())
def test_gt_sparse_prefill_decode_logprob_parity(model, capsys):
    sparse = _sparse_engine(model)
    try:
        local_max, local_rows = _local_boundary_max(sparse)
        sparse_rollout = _rollout_stats(sparse)
    finally:
        sparse.shutdown()

    dense = _dense_engine(model)
    try:
        dense_rollout = _rollout_stats(dense)
    finally:
        dense.shutdown()

    with capsys.disabled():
        print(
            f"\n[{model}] local={local_rows} sparse_rollout={sparse_rollout} "
            f"dense_rollout={dense_rollout}"
        )

    assert local_max <= LOCAL_MAX_ABS
    assert sparse_rollout["mean"] <= dense_rollout["mean"] + ROLLOUT_MEAN_MARGIN
    assert sparse_rollout["p95"] <= dense_rollout["p95"] + ROLLOUT_P95_MARGIN
