"""Independent vortex configuration object.

All vortex hyper-parameters live here, in one dataclass owned by vortex_torch,
instead of as ~18 scattered ``vortex_*`` fields on sglang's ``ServerArgs``.
``ServerArgs`` keeps a single ``vortex: Optional[VortexConfig]`` field (the
spawn-safe channel: sglang pickles ``ServerArgs`` to its worker), plus a small
backward-compatible ``__getattr__`` shim so the many existing
``server_args.vortex_*`` / ``server_args.enable_vortex_sparsity`` read sites keep
working unchanged.

Two entry points populate it:
  * Python: ``sgl.Engine(vortex_topk_val=..., enable_vortex_sparsity=True, ...)``
    still works — :func:`install_serverargs_adapter` folds those flat kwargs into
    a ``VortexConfig`` at the ``ServerArgs`` boundary.
  * Explicit: ``sgl.Engine(vortex=VortexConfig(topk_val=..., ...))``.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, List, Literal, Optional, Tuple


@dataclass
class PrefillPatchConfig:
    """Analysis-only capture/patch configuration for one fresh prefill request.

    ``capture_dense`` writes post-QK-norm/RoPE Q/K and projected V.
    ``capture_sparse`` writes the GT selection for the final query block.
    ``apply`` reads both traces and replaces only the final query row.
    """

    mode: Literal["capture_dense", "capture_sparse", "apply"]
    output_dir: str
    dense_trace_dir: Optional[str] = None
    sparse_trace_dir: Optional[str] = None
    components: Literal["q", "kv", "qkv"] = "qkv"
    routing: Literal["frozen", "recompute"] = "frozen"
    layers: Optional[List[int]] = None

    def __post_init__(self) -> None:
        if self.mode not in {"capture_dense", "capture_sparse", "apply"}:
            raise ValueError(f"unknown prefill patch mode: {self.mode!r}")
        if not isinstance(self.output_dir, str) or not self.output_dir:
            raise ValueError("prefill patch output_dir must be a non-empty string")
        if self.components not in {"q", "kv", "qkv"}:
            raise ValueError(f"unknown prefill patch components: {self.components!r}")
        if self.routing not in {"frozen", "recompute"}:
            raise ValueError(f"unknown prefill patch routing: {self.routing!r}")
        if self.mode == "apply":
            if not self.dense_trace_dir or not self.sparse_trace_dir:
                raise ValueError(
                    "prefill patch apply mode requires dense_trace_dir and "
                    "sparse_trace_dir"
                )
        if self.layers is not None:
            if not isinstance(self.layers, list) or any(
                isinstance(x, bool) or not isinstance(x, int) or x < 0
                for x in self.layers
            ):
                raise ValueError("prefill patch layers must be a list of non-negative ints")
            if len(set(self.layers)) != len(self.layers):
                raise ValueError("prefill patch layers must not contain duplicates")

    @classmethod
    def from_value(cls, value: Any) -> "PrefillPatchConfig":
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**value)
        raise TypeError(
            "vortex_prefill_patch must be a PrefillPatchConfig or mapping, "
            f"got {type(value).__name__}"
        )


@dataclass
class VortexConfig:
    """All vortex sparse-attention hyper-parameters (defaults mirror the former
    ``ServerArgs.vortex_*`` defaults exactly, so behaviour is unchanged)."""

    topk_val: int = 30
    max_topk_val: Optional[int] = None
    layers_skip: Optional[List[int]] = None
    block_reserved_bos: int = 1
    block_reserved_eos: int = 1
    max_seq_lens: int = -1
    workload_chunk_size: int = 32
    dtype: str = "bfloat16"
    module_path: Optional[str] = None
    module_name: Optional[str] = None
    block_size: int = 16
    topk_ratio: float = 0.0
    compilation_cache_dir: Optional[str] = None
    schedule_policy: Optional[str] = None
    attention_backend: str = "flashinfer"
    impl_backend: str = "triton"
    use_tensor_core: bool = False
    # Opt-in GQA sparse-prefill path (flashinfer VariableBlockSparseAttention).
    # Default off preserves the historical dense-prefill behaviour exactly.
    sparse_prefill: bool = False
    # Deterministic GT top-k: propagate deterministic=True to flashinfer's
    # top_k_ragged_transform for single-request bit-exact block selection (pair with
    # FLASHINFER_TOPK_ALGO=filtered). Default off preserves prior behaviour.
    deterministic_topk: bool = False
    # Analysis-only target-row capture/patch support. None has zero runtime cost.
    prefill_patch: Optional[PrefillPatchConfig] = None

    def __post_init__(self) -> None:
        if self.prefill_patch is not None:
            self.prefill_patch = PrefillPatchConfig.from_value(self.prefill_patch)

    @classmethod
    def from_flat(cls, flat: Dict[str, Any]) -> "VortexConfig":
        """Build from a dict of ``vortex_<name>`` keys (prefix stripped)."""
        names = {f.name for f in fields(cls)}
        kw = {}
        for k, v in flat.items():
            key = k[len("vortex_"):] if k.startswith("vortex_") else k
            if key in names:
                if key == "prefill_patch" and v is not None:
                    v = PrefillPatchConfig.from_value(v)
                kw[key] = v
        return cls(**kw)


# The legacy ``vortex_<name>`` -> default map, used by the ServerArgs shim when
# vortex is disabled (so a stray read returns the historical default). Kept in
# sync with the dataclass defaults above; duplicated into server_args.py as a
# plain literal to avoid sglang importing vortex_torch.
def legacy_defaults() -> Dict[str, Any]:
    return {f.name: f.default for f in fields(VortexConfig)}


def split_flat_kwargs(kwargs: Dict[str, Any]) -> Tuple[Optional[VortexConfig], Dict[str, Any]]:
    """Pop ``enable_vortex_sparsity`` + ``vortex_*`` from ``kwargs``.

    Returns ``(config_or_None, remaining_kwargs)``. The config is built iff
    ``enable_vortex_sparsity`` is truthy; otherwise the vortex_* keys are simply
    dropped (vortex stays off).
    """
    enabled = bool(kwargs.pop("enable_vortex_sparsity", False))
    flat = {k: kwargs.pop(k) for k in list(kwargs) if k.startswith("vortex_") and k != "vortex"}
    cfg = VortexConfig.from_flat(flat) if enabled else None
    return cfg, kwargs


def install_serverargs_adapter() -> bool:
    """Wrap ``ServerArgs.__init__`` so flat ``vortex_*`` kwargs fold into the
    single ``vortex`` field. Idempotent; parent-process only (the spawned worker
    unpickles ``ServerArgs`` and never re-runs ``__init__``). Returns False if
    sglang is unavailable.
    """
    try:
        from sglang.srt.server_args import ServerArgs
    except Exception:
        return False
    if getattr(ServerArgs, "_vortex_adapter_installed", False):
        return True

    _orig_init = ServerArgs.__init__

    def __init__(self, *args, **kwargs):
        v = kwargs.get("vortex")
        if isinstance(v, VortexConfig):
            # Explicit object wins; drop any stray flat vortex_* / enable flag.
            for k in [k for k in kwargs if k.startswith("vortex_")]:
                kwargs.pop(k)
            kwargs.pop("enable_vortex_sparsity", None)
        elif isinstance(v, str):
            # CLI path: --vortex-config '<json>' arrives as a JSON string.
            import json
            kwargs["vortex"] = VortexConfig.from_flat(json.loads(v))
        else:
            # Python path: fold flat vortex_* kwargs (gated by enable flag).
            cfg, kwargs = split_flat_kwargs(kwargs)
            kwargs["vortex"] = cfg
        _orig_init(self, *args, **kwargs)

    ServerArgs.__init__ = __init__
    ServerArgs._vortex_adapter_installed = True
    return True


def cfg(model_runner_or_server_args) -> Optional[VortexConfig]:
    """Accessor: return the VortexConfig from a ModelRunner or ServerArgs."""
    sa = getattr(model_runner_or_server_args, "server_args", model_runner_or_server_args)
    return getattr(sa, "vortex", None)
