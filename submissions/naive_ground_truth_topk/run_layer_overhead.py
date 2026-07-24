"""Driver: measure per-layer INDEXER overhead of naive_ground_truth_topk at 4k
context (Qwen3-1.7B), for both prefill and decode.

Boots the real vortex engine (sparse-prefill enabled so the prefill indexer
fires), runs a single 4096-token request generating a handful of tokens, and
collects CUDA-event timings emitted by bench_ground_truth_topk.py's probe.

Usage:
    CUDA_VISIBLE_DEVICES=<gpu> python submissions/naive_ground_truth_topk/run_layer_overhead.py

Env knobs:
    PROMPT_LEN     (default 4096)  prefill length in tokens
    MAX_NEW_TOKENS (default 24)    decode steps to sample
    WARMUP_STEPS   (default 3)     leading decode steps discarded per layer
"""

import os
import sys
import json
import statistics
import tempfile

CONFIG = "submissions/naive_ground_truth_topk/bench_ground_truth_topk.json"
PROMPT_LEN = int(os.environ.get("PROMPT_LEN", "4096"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "24"))
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", "3"))


def main():
    timing_path = tempfile.mktemp(prefix="vortex_indexer_", suffix=".jsonl")
    # Must be set BEFORE the engine spawns its scheduler subprocess so the
    # env propagates across `spawn`.
    os.environ["VORTEX_TIME_INDEXER"] = "1"
    os.environ["VORTEX_TIME_OUT"] = timing_path
    open(timing_path, "w").close()

    import torch
    import vortex_torch  # noqa: F401  (registers backends / server args)
    from vortex_torch.engine.sgl import get_engine_from_json

    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"[driver] device={dev_name}  config={CONFIG}", flush=True)
    print(f"[driver] prompt_len={PROMPT_LEN}  max_new_tokens={MAX_NEW_TOKENS}", flush=True)

    llm = get_engine_from_json(CONFIG)
    try:
        # Exactly PROMPT_LEN tokens of prefill; ids kept well within vocab.
        input_ids = [(i % 30000) + 10 for i in range(PROMPT_LEN)]
        out = llm.generate(
            input_ids=input_ids,
            sampling_params={"temperature": 0.0, "max_new_tokens": MAX_NEW_TOKENS},
        )
        meta = out.get("meta_info", {}) if isinstance(out, dict) else {}
        print(f"[driver] generated tokens: {meta.get('completion_tokens', '?')}", flush=True)
    finally:
        try:
            llm.shutdown()
        except Exception:
            pass

    records = []
    with open(timing_path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    report(records, dev_name)


def _summ(xs):
    xs = sorted(xs)
    return {
        "n": len(xs),
        "min": min(xs),
        "median": statistics.median(xs),
        "mean": statistics.fmean(xs),
        "max": max(xs),
    }


def report(records, dev_name):
    dec = [r for r in records if r["regime"] == "decode"]
    pre = [r for r in records if r["regime"] == "prefill"]

    print("\n" + "=" * 74)
    print(f"PER-LAYER INDEXER OVERHEAD  (Qwen3-1.7B, {dev_name})")
    print(f"prompt_len={PROMPT_LEN}  block_size=16  topk_val=64  layers_skip=[]")
    print("=" * 74)

    if not records:
        print("NO TIMING RECORDS COLLECTED — check the child log for errors.")
        return

    # ---- Decode: per-layer, drop leading warmup steps ----
    layers = sorted({r["layer"] for r in dec})
    print(f"\nDECODE  (1 query token, {PROMPT_LEN} in KV cache; "
          f"per layer, first {WARMUP_STEPS} steps dropped as warmup)")
    per_layer_med = []
    for L in layers:
        samples = [r["ms"] for r in dec if r["layer"] == L]
        samples = samples[WARMUP_STEPS:] if len(samples) > WARMUP_STEPS else samples
        if not samples:
            continue
        s = _summ(samples)
        per_layer_med.append(s["median"])
    if per_layer_med:
        print(f"  layers measured        : {len(per_layer_med)}")
        print(f"  per-layer median       : "
              f"{statistics.fmean(per_layer_med):.4f} ms "
              f"(min {min(per_layer_med):.4f}, max {max(per_layer_med):.4f})")
        print(f"  indexer / token (28 L) : "
              f"{sum(per_layer_med):.4f} ms  (sum of per-layer medians)")
        alld = [r["ms"] for r in dec if r["layer"] in layers]
        print(f"  all-sample summary     : {_summ(alld)}")

    # ---- Prefill: per query-tile position (each tile does different work) ----
    print(f"\nPREFILL  (single {PROMPT_LEN}-token prompt; indexer fires once "
          f"per query-tile)")
    # Layer count for the whole-prompt estimate must come from the PREFILL
    # records: decode may be empty (MAX_NEW_TOKENS=1 or early EOS), which would
    # otherwise zero out the prefill total.
    pre_layers = sorted({r["layer"] for r in pre})
    sel = [r for r in records if r["regime"] == "prefill_select"]
    tiles = sorted({r["tile"] for r in pre})
    per_tile_med = []
    for t in tiles:
        # Across the 28 layers, drop the very first occurrence (JIT autotune).
        samples = [r["ms"] for r in sorted(
            (x for x in pre if x["tile"] == t), key=lambda z: z["seq"])]
        warm = samples[1:] if len(samples) > 1 else samples
        s = _summ(warm)
        per_tile_med.append(s["median"])
        print(f"  tile {t:>2}: median {s['median']:.4f} ms  "
              f"(min {s['min']:.4f}, max {s['max']:.4f}, n={s['n']} layers)")
    # Prefill selection (top-k) per tile, so the prefill indexer total includes
    # the same score+select stages the decode indexer figure already does.
    sel_tile_med = []
    for t in sorted({r["tile"] for r in sel}):
        samples = [r["ms"] for r in sorted(
            (x for x in sel if x["tile"] == t), key=lambda z: z["seq"])]
        warm = samples[1:] if len(samples) > 1 else samples
        sel_tile_med.append(_summ(warm)["median"])
    if per_tile_med:
        score_total = sum(per_tile_med)
        select_total = sum(sel_tile_med)
        combined = score_total + select_total
        print(f"  per-layer scoring      : {score_total:.4f} ms  "
              f"(sum over {len(tiles)} tiles, medians)")
        print(f"  per-layer selection    : {select_total:.4f} ms  "
              f"(top-k; sum over tiles, medians)")
        print(f"  per-layer prefill total: {combined:.4f} ms  "
              f"(scoring + selection, comparable to the decode indexer figure)")
        print(f"  whole-prompt prefill   : {combined * len(pre_layers):.4f} ms  "
              f"(x{len(pre_layers)} layers)")

    print("\n(raw JSONL retained at $VORTEX_TIME_OUT)")


if __name__ == "__main__":
    sys.exit(main())
