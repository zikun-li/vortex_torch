"""Versioned artifacts and runtime state for GT sparse-prefill interventions."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
from safetensors.torch import load_file, save_file

from .config import PrefillPatchConfig


TRACE_VERSION = 1


def _json_hash(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def token_ids_hash(input_ids: torch.Tensor) -> str:
    ids = input_ids.detach().to(device="cpu", dtype=torch.int64).contiguous()
    return hashlib.sha256(ids.numpy().tobytes()).hexdigest()


def _atomic_json(path: Path, value: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class PrefillPatchTrace:
    """Reader for a completed dense, sparse-selection, or applied trace."""

    def __init__(self, root: Path, manifest: Dict[str, Any]):
        self.root = root
        self.manifest = manifest

    @classmethod
    def open(cls, path: str | Path) -> "PrefillPatchTrace":
        root = Path(path).expanduser().resolve()
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"prefill patch trace has no manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format_version") != TRACE_VERSION:
            raise ValueError(
                "unsupported prefill patch trace format: "
                f"{manifest.get('format_version')!r}"
            )
        if manifest.get("complete") is not True:
            raise ValueError(f"prefill patch trace is incomplete: {root}")
        if manifest.get("kind") not in {"dense", "sparse", "applied"}:
            raise ValueError(f"invalid prefill patch trace kind in {manifest_path}")
        return cls(root, manifest)

    @property
    def kind(self) -> str:
        return str(self.manifest["kind"])

    @property
    def layers(self) -> list[int]:
        return [int(x) for x in self.manifest["layers"]]

    def load_layer(
        self, layer_id: int, *, device: torch.device | str = "cpu"
    ) -> Dict[str, torch.Tensor]:
        rel = self.manifest.get("layer_files", {}).get(str(layer_id))
        if rel is None:
            raise KeyError(f"layer {layer_id} is not present in {self.root}")
        tensors = load_file(str(self.root / rel), device="cpu")
        if str(device) != "cpu":
            tensors = {k: v.to(device=device) for k, v in tensors.items()}
        return tensors


class _TraceWriter:
    def __init__(
        self,
        path: str | Path,
        *,
        kind: str,
        base_manifest: Dict[str, Any],
        layers: Iterable[int],
        patch_config: PrefillPatchConfig,
    ) -> None:
        self.root = Path(path).expanduser().resolve()
        if self.root.exists() and any(self.root.iterdir()):
            raise ValueError(
                f"prefill patch output directory is not empty: {self.root}"
            )
        self.layers_dir = self.root / "layers"
        self.layers_dir.mkdir(parents=True, exist_ok=True)
        self.expected = tuple(sorted(int(x) for x in layers))
        if not self.expected:
            raise ValueError("prefill patch trace has no selected layers")
        self.written: set[int] = set()
        self.manifest: Dict[str, Any] = {
            "format_version": TRACE_VERSION,
            "complete": False,
            "kind": kind,
            "layers": list(self.expected),
            "layer_files": {},
            "patch": {
                "mode": patch_config.mode,
                "components": patch_config.components,
                "routing": patch_config.routing,
            },
            **base_manifest,
        }
        _atomic_json(self.root / "manifest.json", self.manifest)

    def set_input(self, input_ids: torch.Tensor) -> None:
        input_meta = {
            "seq_len": int(input_ids.numel()),
            "query_position": int(input_ids.numel()) - 1,
            "token_ids_sha256": token_ids_hash(input_ids),
        }
        existing = self.manifest.get("input")
        if existing is not None:
            if existing != input_meta:
                raise ValueError(
                    "prefill patch trace received more than one input request"
                )
            return
        self.manifest["input"] = input_meta
        _atomic_json(self.root / "manifest.json", self.manifest)

    def write_layer(self, layer_id: int, tensors: Dict[str, torch.Tensor]) -> None:
        if layer_id not in self.expected:
            raise ValueError(f"unexpected layer {layer_id} for prefill patch trace")
        if layer_id in self.written:
            raise ValueError(f"layer {layer_id} was written more than once")
        cpu = {
            name: tensor.detach().to("cpu").contiguous()
            for name, tensor in tensors.items()
        }
        rel = f"layers/layer_{layer_id:03d}.safetensors"
        save_file(cpu, str(self.root / rel))
        self.written.add(layer_id)
        self.manifest["layer_files"][str(layer_id)] = rel
        if self.written == set(self.expected):
            self.manifest["complete"] = True
        _atomic_json(self.root / "manifest.json", self.manifest)


def build_trace_metadata(model_runner, *, block_size: int) -> Dict[str, Any]:
    model_config = model_runner.model_config
    hf_config = getattr(model_config, "hf_config", None)
    if hf_config is not None and hasattr(hf_config, "to_dict"):
        model_dict = hf_config.to_dict()
    else:
        model_dict = {
            "num_hidden_layers": model_config.num_hidden_layers,
            "num_attention_heads": model_config.num_attention_heads,
            "num_key_value_heads": model_config.num_key_value_heads,
            "head_dim": model_config.head_dim,
        }
    sa = model_runner.server_args
    return {
        "model": {
            "path": str(sa.model_path),
            "revision": getattr(sa, "revision", None),
            "config_sha256": _json_hash(model_dict),
        },
        "geometry": {
            "num_layers": int(model_config.num_hidden_layers),
            "num_q_heads": int(model_config.num_attention_heads),
            "num_kv_heads": int(model_config.num_key_value_heads),
            "head_dim": int(model_config.head_dim),
            "block_size": int(block_size),
            "model_dtype": str(model_runner.dtype),
            "kv_cache_dtype": str(model_runner.kv_cache_dtype),
            "tp_size": int(sa.tp_size),
            "pp_size": int(sa.pp_size),
        },
        "routing_config": {
            "topk_val": int(sa.vortex_topk_val),
            "topk_ratio": float(sa.vortex_topk_ratio),
            "reserved_bos": int(sa.vortex_block_reserved_bos),
            "reserved_eos": int(sa.vortex_block_reserved_eos),
        },
    }


def extract_tail_selection(
    kv_indptr: torch.Tensor,
    block_ids: torch.Tensor,
    *,
    num_kv_heads: int,
    tile_len: int,
    tile_offset: int,
    seq_len: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Extract the final block's rows from a head-major per-tile CSR."""

    tail_start = ((seq_len - 1) // block_size) * block_size
    local_start = tail_start - tile_offset
    tail_len = seq_len - tail_start
    if local_start < 0 or local_start + tail_len > tile_len:
        raise ValueError("final query block is not contained in the current GT tile")

    pieces = []
    lengths = []
    for head in range(num_kv_heads):
        row0 = head * tile_len + local_start
        row_ptr = (
            kv_indptr[row0 : row0 + tail_len + 1]
            .detach()
            .to(device="cpu", dtype=torch.int64)
        )
        start, end = int(row_ptr[0]), int(row_ptr[-1])
        pieces.append(block_ids[start:end].detach().to(device="cpu", dtype=torch.int32))
        lengths.extend((row_ptr[1:] - row_ptr[:-1]).tolist())
    out_ptr = torch.zeros(num_kv_heads * tail_len + 1, dtype=torch.int32, device="cpu")
    if lengths:
        out_ptr[1:] = torch.tensor(lengths, dtype=torch.int32).cumsum(0)
    out_ids = torch.cat(pieces) if pieces else torch.empty(0, dtype=torch.int32)
    return out_ptr, out_ids, tail_start


def validate_trace_pair(
    dense: PrefillPatchTrace,
    sparse: PrefillPatchTrace,
    *,
    input_ids: torch.Tensor,
    current_metadata: Dict[str, Any],
    layers: Iterable[int],
) -> None:
    if dense.kind != "dense" or sparse.kind != "sparse":
        raise ValueError("apply mode requires one dense and one sparse trace")
    expected_input = {
        "seq_len": int(input_ids.numel()),
        "query_position": int(input_ids.numel()) - 1,
        "token_ids_sha256": token_ids_hash(input_ids),
    }
    for name, trace in (("dense", dense), ("sparse", sparse)):
        if trace.manifest.get("input") != expected_input:
            raise ValueError(f"{name} trace input tokens do not match this request")
        for key in ("model", "geometry"):
            if trace.manifest.get(key) != current_metadata.get(key):
                raise ValueError(f"{name} trace {key} does not match this engine")
    if sparse.manifest.get("routing_config") != current_metadata.get("routing_config"):
        raise ValueError(
            "sparse trace routing configuration does not match this engine"
        )
    required = set(int(x) for x in layers)
    if not required.issubset(set(dense.layers)):
        raise ValueError("dense trace is missing requested patch layers")
    if not required.issubset(set(sparse.layers)):
        raise ValueError("sparse trace is missing requested patch layers")


class PrefillPatchRuntime:
    """Spawn-local trace state. Attention math remains in the backend."""

    def __init__(self, model_runner, *, block_size: int, layers_skip: Iterable[int]):
        config = model_runner.server_args.vortex_prefill_patch
        self.config: Optional[PrefillPatchConfig] = config
        if config is None:
            self.layers: tuple[int, ...] = ()
            self.writer = None
            self.dense = None
            self.sparse = None
            return

        sa = model_runner.server_args
        if sa.tp_size != 1 or sa.pp_size != 1:
            raise ValueError(
                "prefill patch analysis currently requires tp_size=pp_size=1"
            )
        if (
            model_runner.dtype != torch.bfloat16
            or model_runner.kv_cache_dtype != torch.bfloat16
        ):
            raise ValueError(
                "prefill patch analysis currently requires BF16 model and KV"
            )
        if (
            getattr(sa, "chunked_prefill_size", None) != -1
            or not sa.disable_radix_cache
        ):
            raise ValueError(
                "prefill patch analysis requires chunked_prefill_size=-1 and "
                "disable_radix_cache=True"
            )
        if float(sa.vortex_topk_ratio) != 0.0:
            raise ValueError("prefill patch analysis currently requires topk_ratio=0")

        skipped = set(int(x) for x in layers_skip)
        all_layers = set(range(int(model_runner.model_config.num_hidden_layers)))
        if config.layers is None:
            selected = sorted(all_layers - skipped)
        else:
            selected = sorted(config.layers)
            invalid = set(selected) - all_layers
            if invalid:
                raise ValueError(f"invalid prefill patch layers: {sorted(invalid)}")
            if config.mode != "capture_dense" and set(selected) & skipped:
                raise ValueError("cannot patch a layer listed in vortex_layers_skip")
        self.layers = tuple(selected)
        self.metadata = build_trace_metadata(model_runner, block_size=block_size)
        kind = {
            "capture_dense": "dense",
            "capture_sparse": "sparse",
            "apply": "applied",
        }[config.mode]
        self.writer = _TraceWriter(
            config.output_dir,
            kind=kind,
            base_manifest=self.metadata,
            layers=self.layers,
            patch_config=config,
        )
        self.dense = (
            PrefillPatchTrace.open(config.dense_trace_dir)
            if config.mode == "apply"
            else None
        )
        self.sparse = (
            PrefillPatchTrace.open(config.sparse_trace_dir)
            if config.mode == "apply"
            else None
        )
        self._validated_input = False

    @property
    def enabled(self) -> bool:
        return self.config is not None

    def wants_layer(self, layer_id: int) -> bool:
        return layer_id in self.layers

    def set_input(self, input_ids: torch.Tensor, *, batch_size: int) -> None:
        if not self.enabled:
            return
        if batch_size != 1:
            raise ValueError("prefill patch analysis requires exactly one request")
        assert self.writer is not None
        self.writer.set_input(input_ids)
        if self.config.mode == "apply" and not self._validated_input:
            assert self.dense is not None and self.sparse is not None
            validate_trace_pair(
                self.dense,
                self.sparse,
                input_ids=input_ids,
                current_metadata=self.metadata,
                layers=self.layers,
            )
            self._validated_input = True

    def capture_dense(
        self,
        layer_id: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        if not self.enabled or self.config.mode != "capture_dense":
            return
        if not self.wants_layer(layer_id):
            return
        assert self.writer is not None
        self.writer.write_layer(
            layer_id,
            {"q_target": q[-1], "k_prefix": k, "v_prefix": v},
        )

    def capture_sparse(
        self,
        layer_id: int,
        *,
        kv_indptr: torch.Tensor,
        block_ids: torch.Tensor,
        scores: torch.Tensor,
        tile_len: int,
        tile_offset: int,
        seq_len: int,
        block_size: int,
        num_kv_heads: int,
    ) -> None:
        if not self.enabled or self.config.mode != "capture_sparse":
            return
        if not self.wants_layer(layer_id):
            return
        out_ptr, out_ids, tail_start = extract_tail_selection(
            kv_indptr,
            block_ids,
            num_kv_heads=num_kv_heads,
            tile_len=tile_len,
            tile_offset=tile_offset,
            seq_len=seq_len,
            block_size=block_size,
        )
        assert self.writer is not None
        self.writer.write_layer(
            layer_id,
            {
                "kv_indptr": out_ptr,
                "block_ids": out_ids,
                "scores_target": scores[-1],
                "tail_start": torch.tensor(tail_start, dtype=torch.int64),
                "tail_len": torch.tensor(seq_len - tail_start, dtype=torch.int64),
            },
        )

    def load_apply_layer(
        self, layer_id: int, *, device: torch.device | str
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if not self.enabled or self.config.mode != "apply":
            raise RuntimeError("load_apply_layer called outside apply mode")
        assert self.dense is not None and self.sparse is not None
        return (
            self.dense.load_layer(layer_id, device=device),
            self.sparse.load_layer(layer_id, device=device),
        )

    def record_applied(
        self,
        layer_id: int,
        *,
        kv_indptr: torch.Tensor,
        block_ids: torch.Tensor,
        scores_target: torch.Tensor,
        tail_start: int,
        tail_len: int,
    ) -> None:
        assert self.writer is not None
        self.writer.write_layer(
            layer_id,
            {
                "kv_indptr": kv_indptr,
                "block_ids": block_ids,
                "scores_target": scores_target,
                "tail_start": torch.tensor(tail_start, dtype=torch.int64),
                "tail_len": torch.tensor(tail_len, dtype=torch.int64),
            },
        )
