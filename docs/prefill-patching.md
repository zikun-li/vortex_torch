# GT sparse-prefill patching

Vortex exposes an analysis-only, three-run workflow for measuring a target
token under dense Q/K/V states and GT sparse routing. The input must be the
single fresh prefix ending immediately before the token whose probability is
requested. Use SGLang's `token_ids_logprob` to request that candidate's logp.

The current implementation requires BF16, `tp_size=pp_size=1`, GT top-k,
`topk_ratio=0`, `chunked_prefill_size=-1`, and `disable_radix_cache=True`.
It recomputes the final aligned query block but replaces only its last output
row.

## 1. Capture dense attention inputs

Initialize the GT Vortex engine with sparse prefill disabled:

```python
vortex_sparse_prefill=False,
vortex_prefill_patch={
    "mode": "capture_dense",
    "output_dir": "/path/to/trace/dense",
    "layers": None,  # all non-skipped layers
}
```

This stores the post-QK-normalization/RoPE target Q and the full logical-prefix
K/V for each layer.

## 2. Capture sparse routing

Run the identical input IDs through GT sparse prefill:

```python
vortex_sparse_prefill=True,
vortex_prefill_patch={
    "mode": "capture_sparse",
    "output_dir": "/path/to/trace/sparse",
}
```

This stores the final query block's head-major CSR and all GT candidate scores
for the target row.

## 3. Apply an intervention

```python
vortex_sparse_prefill=True,
vortex_prefill_patch={
    "mode": "apply",
    "output_dir": "/path/to/trace/applied-qkv-frozen",
    "dense_trace_dir": "/path/to/trace/dense",
    "sparse_trace_dir": "/path/to/trace/sparse",
    "components": "qkv",       # "q", "kv", or "qkv"
    "routing": "frozen",       # "frozen" or "recompute"
    "layers": None,
}
```

Frozen routing replays the captured sparse CSR. Recomputed routing scores the
patched Q/K with the same GT selector. Both modes write the IDs and scores used
by the applied run.

Artifacts are versioned JSON manifests plus per-layer BF16 safetensors. Read
them with:

```python
from vortex_torch.engine.sgl import PrefillPatchTrace

trace = PrefillPatchTrace.open("/path/to/trace/sparse")
layer = trace.load_layer(0)
```

Vortex validates the model configuration, exact input-token hash, tensor
geometry, routing configuration, and available layers before applying a trace.
