from __future__ import annotations

from typing import Any, Final, Optional, Union
import uuid

import torch

from ..abs import ContextBase
from ..utils import UNSET, Mode, resolve_dtype
from .metadata import MetaData


class Context(ContextBase):
    """Static, single-instance indexer context.

    Holds **only** configuration that's fixed for the lifetime of the
    compiled indexer — shapes, page/block sizes, head counts, allocation
    budgets, codegen scratch (op/tensor lists), backend identity, …

    Per-forward-batch buffers (winfo_*, dense/sparse_kv_indptr+indices,
    dense/sparse_seqlens, dense/sparse_block_tables, kv_last_page_len)
    and ``batch_size`` live on a separate :class:`MetaData` object,
    pre-allocated at attention-backend ``__init__`` and exposed as
    ``ctx.metadata``. Codegen emits ``ctx.metadata.<field>`` for any
    value that varies between forward batches.

    Build pattern (from each attention backend's ``_compile``):
        ctx = Context()
        ctx.create(self, model_runner)              # static fields
        ctx.metadata = MetaData.preallocate(ctx, device=...)
    """

    __slots__ = ContextBase.__slots__ + (
        # ---- per-forward-batch state (pre-allocated MetaData) ----
        "metadata",
        # ---- batch budget ----
        "max_bs",
        # ---- active attention backend ----
        # "flashinfer" (CSR indices, default) or "trtllm" (2D block_tables).
        # Set by Context.create() from server_args.vortex_attention_backend;
        # consumed by the indexer compiler's IndexerBackend traits.
        "vortex_attention_backend",
        # ---- workload-scheduler shape ----
        "max_num_workloads",
        "workload_chunk_size",
        # ---- head / shape ----
        "group_size", "num_kv_heads", "num_qo_heads", "head_dim",
        # ---- hardware / paging ----
        "num_sms", "page_size", "max_num_pages", "max_num_pages_per_request",
        "block_size", "max_num_blocks", "max_num_blocks_per_request",
        "num_blocks_per_page", "num_pages_per_workload",
        # ---- topk / reserved-slot policy ----
        "topk_val", "topk_ratio", "block_reserved_bos", "block_reserved_eos",
        "max_topk_val",
        # ---- auxiliary accounting ----
        "_aux_total_bytes", "_aux_total_flops",
        # ---- codegen / graph scratch ----
        "tensor_list", "op_list", "output_tensor_to_op_list",
        "op_to_input_tensor_list", "op_to_output_tensor_list",
        "side_effect_op_ids",
        "sparse_attention_name", "impl_backend", "tensor_id_to_tensor_name_map",
        "query_arg_names",
        "compilation_header_lines", "auxilary_func_def_lines",
        "compilation_cache_dir",
        # ---- tensor-core (bf16-compute) codegen toggle ----
        # When True, the triton W-kernel keeps compute blocks in bf16
        # (loads cast to bf16, accumulations promote to fp32) and emits
        # ``tl.dot`` for MMA-friendly GeMMs. Triton-impl only.
        "use_tensor_core",
        # ---- sparse-prefill compile marker ----
        # True only for the *prefill* indexer compile (``ctx_prefill``): the
        # leading BATCHED axis is (sum S_q)*num_kv_heads (one row per
        # (query-token, kv-head)) rather than decode's B*num_kv_heads, budgets
        # are re-derived for the causal score axis, and the terminal ``topK``
        # lowers to an "emit scores" path (writes per-(row, block) RAGGED scores
        # into ``o`` instead of selecting — the standalone Phase-3 kernel does
        # the causal top-k + mask afterward). Default False = decode, unchanged.
        "sparse_prefill",
        # ---- ground-truth op runtime side-channel ----
        # A plain dict the engine (``_forward_extend_sparse``) fills per query
        # tile with the raw per-tile K, query offset, scale, and tile length,
        # which the GTGroupScore/GTTopK custom-kernel launchers read from (they
        # need contiguous K + geometry the paged ``cache`` view and the planner
        # metadata don't carry). None until a ground-truth submission runs.
        "gt_state",
        # Per-decode-step cache for the GT decode score path: the row->block map
        # and candidate total derived from the (per-step-invariant) paged
        # metadata. Recomputed once per step in ``init_forward_metadata`` and
        # reused across all layers, avoiding a repeat_interleave + a device sync
        # on every layer. None until a ground-truth decode step runs.
        "gt_decode_setup",
    )

    # ---- type hints (declarations only) ----
    metadata: Optional[MetaData]
    max_bs: int
    vortex_attention_backend: str
    max_num_workloads: int
    workload_chunk_size: int
    group_size: int
    num_kv_heads: int
    num_qo_heads: int
    head_dim: int
    num_sms: int
    page_size: int
    max_num_pages: int
    max_num_pages_per_request: int
    block_size: int
    max_num_blocks: int
    max_num_blocks_per_request: int
    num_blocks_per_page: int
    num_pages_per_workload: int
    topk_val: int
    topk_ratio: float
    block_reserved_bos: int
    block_reserved_eos: int
    max_topk_val: Union[int, None]
    _aux_total_bytes: int
    _aux_total_flops: int
    sparse_prefill: bool
    tensor_list: list
    op_list: list
    output_tensor_to_op_list: list
    op_to_input_tensor_list: list
    op_to_output_tensor_list: list
    side_effect_op_ids: list
    sparse_attention_name: str
    impl_backend: str
    tensor_id_to_tensor_name_map: dict
    compilation_header_lines: list
    auxilary_func_def_lines: list
    compilation_cache_dir: str
    use_tensor_core: bool

    def __init__(self) -> None:
        for name in self.__slots__:
            if name == "_created":
                object.__setattr__(self, name, False)
            elif name == "name":
                object.__setattr__(self, name, "Indexer")
            elif name == "_aux_total_bytes":
                object.__setattr__(self, name, 0)
            elif name == "_aux_total_flops":
                object.__setattr__(self, name, 0)
            elif name == "mode":
                object.__setattr__(self, name, Mode.profile)
            elif name == "metadata":
                object.__setattr__(self, name, None)
            elif name == "gt_state":
                object.__setattr__(self, name, None)
            elif name == "gt_decode_setup":
                object.__setattr__(self, name, None)
            elif name == "sparse_prefill":
                # Default decode; create(prefill=True) flips it. Kept a real
                # bool (not UNSET) so the terminal-lowering guard
                # ``getattr(ctx, "sparse_prefill", False)`` is never a truthy
                # sentinel before create() runs.
                object.__setattr__(self, name, False)
            else:
                object.__setattr__(self, name, UNSET)

    # ------------------------------------------------------------------
    # Per-batch convenience accessors — forward to ``self.metadata``.
    # ------------------------------------------------------------------
    @property
    def batch_size(self) -> int:
        """Current batch size — proxied from ``self.metadata``.

        Kept as a property (not a slot) so the only writable copy lives on
        ``self.metadata``; ``ctx.batch_size`` is now read-only.
        """
        return 0 if self.metadata is None else self.metadata.batch_size

    def set_batch_size(self, n: int) -> None:
        """Compatibility shim — forwards to ``self.metadata.set_batch_size``."""
        if self.metadata is None:
            raise RuntimeError(
                "Context.set_batch_size called before MetaData was preallocated; "
                "did you forget `ctx.metadata = MetaData.preallocate(ctx, device=...)`?"
            )
        self.metadata.set_batch_size(n)

    # ------------------------------------------------------------------
    def create(self, parent: Any, model_runner: Any, *, overwrite: bool = False,
               prefill: bool = False) -> "Context":
        """Populate the static fields. Per-batch ``MetaData`` is allocated
        separately by the caller via ``MetaData.preallocate(ctx, device=...)``
        — see this class's docstring.

        ``prefill=True`` builds the *sparse-prefill* indexer context: rows are
        per (query-token, kv-head) so the row budget is ``max_prefill_tokens``
        (not the concurrent-request count), and the intermediate-RAGGED / CSR
        block budget is re-derived to the causal score-axis bound. See
        ``sparse_prefill`` in ``__slots__`` and memory ``sparse-prefill-gqa-design``.
        """
        if self._created and not overwrite:
            raise RuntimeError("Context.create() already called; pass overwrite=True to reinitialize.")

        self.sparse_prefill = bool(prefill)
        sa = model_runner.server_args
        max_pages_per_req = (
            (model_runner.model_config.context_len + sa.page_size - 1) // sa.page_size
            if sa.vortex_max_seq_lens < 0
            else (sa.vortex_max_seq_lens + sa.page_size - 1) // sa.page_size
        )
        # Row budget. Decode: one row per concurrent request. Prefill: one
        # pseudo-request row per query token (see the pseudo-request planner),
        # bounded by the max tokens in a prefill batch.
        max_bs = (
            int(sa.max_prefill_tokens) if prefill
            else int(model_runner.req_to_token_pool.size)
        )
        self.max_bs = max_bs

        self.workload_chunk_size = sa.vortex_workload_chunk_size

        self.group_size = parent.group_size
        self.num_kv_heads = parent.num_kv_heads
        self.num_qo_heads = parent.num_qo_heads
        self.head_dim = parent.head_dim

        self.num_sms = torch.cuda.get_device_properties(0).multi_processor_count
        self.page_size = sa.page_size
        self.block_size = sa.vortex_block_size
        self.num_blocks_per_page = self.page_size // self.block_size
        assert self.page_size % self.block_size == 0, "Page size must be a multiple of block size."
        assert self.workload_chunk_size % self.num_blocks_per_page == 0, "Workload chunk size must be a multiple of blocks per page."
        # Capacity model (adjust as needed)
        self.max_num_pages = max_pages_per_req * max_bs * self.num_kv_heads
        self.max_num_pages_per_request = max_pages_per_req
        self.max_num_blocks = self.max_num_pages * self.num_blocks_per_page
        self.max_num_blocks_per_request = self.max_num_pages_per_request * self.num_blocks_per_page
        self.num_pages_per_workload = self.workload_chunk_size // self.num_blocks_per_page

        if prefill:
            # Prefill score axis = candidate blocks summed over (query-token,
            # kv-head): per kv-head, sum_t ceil((pos_t+1)/C). Worst case is a
            # single sequence of length L=max_prefill_tokens (the sum is convex
            # in each sequence length ⇒ concentrating tokens maximizes it):
            #   sum_{p<L} ceil((p+1)/C) <= L(L-1)/(2C) + L.
            # This causal bound is far smaller than the naive
            # max_pages_per_req * max_bs * nkv * blocks_per_page that the decode
            # formula above produced, and it's what keeps the auto-allocated
            # RAGGED score buffers (leading dim = max_num_blocks) tractable.
            # Still O(nkv * L^2 / C) → query-axis chunking is the deferred
            # mitigation (memory sparse-prefill-gqa-design, Top Risk #1).
            L = int(max_bs)  # == max_prefill_tokens
            C = self.block_size
            causal_blocks = (L * (L - 1)) // (2 * C) + L + 1
            self.max_num_blocks = causal_blocks * self.num_kv_heads
            naive_cap = max_bs * self.num_kv_heads * self.max_num_blocks_per_request
            assert self.max_num_blocks <= naive_cap, (
                f"prefill max_num_blocks {self.max_num_blocks} exceeds naive cap "
                f"{naive_cap} (L={L}, C={C}, nkv={self.num_kv_heads})"
            )

        # Round trtllm's dense_block_tables row stride up to a multiple
        # of 4 int32s (= 128 bits). Each row of ``dense_block_tables`` is
        # ``max_num_blocks_per_request`` int32s wide, and the trtllm
        # indexer kernel reads
        # ``indices[pid * max_blocks_per_seq + p * nbp]`` starting at
        # ``pid * max_blocks_per_seq`` — making that product 16-byte
        # aligned for every ``pid`` lets the int32 loads coalesce into
        # 128-bit LDG.E.128 cache-line-aligned transactions on the per-
        # row index sweep, and is harmless on flashinfer where the CSR
        # indices array doesn't carry a per-row stride at all.
        # ``p < _page`` already gates the kernel's index reads against
        # the actual sequence length, so the extra padding entries are
        # never read (the planner just doesn't write to them).
        # NB: ``self.vortex_attention_backend`` is assigned further down
        # in this method, so read off ``sa`` directly.
        _attn_backend = (
            getattr(sa, "vortex_attention_backend", "flashinfer") or "flashinfer"
        )
        if _attn_backend == "trtllm" and self.max_num_blocks_per_request % 4 != 0:
            self.max_num_blocks_per_request = (
                (self.max_num_blocks_per_request + 3) // 4 * 4
            )
        self.topk_val = sa.vortex_topk_val
        self.max_topk_val = sa.vortex_max_topk_val
        self.topk_ratio = sa.vortex_topk_ratio
        self.vortex_dtype = resolve_dtype(
            getattr(sa, "vortex_dtype", "bfloat16"), default=torch.bfloat16
        )

        self.block_reserved_bos = sa.vortex_block_reserved_bos
        self.block_reserved_eos = sa.vortex_block_reserved_eos

        self.max_num_workloads = (
            (self.max_num_blocks // max(1, self.workload_chunk_size)) + max_bs * self.num_kv_heads
        )

        self.tensor_list = []
        self.op_list = []
        self.output_tensor_to_op_list = []
        self.op_to_input_tensor_list = []
        self.op_to_output_tensor_list = []
        self.side_effect_op_ids = []
        self.tensor_id_to_tensor_name_map = {}
        # Query argument name(s) for the generated ``forward(...)`` entry point.
        # MHA flows pass a single ``q``; MLA flows pass the absorbed pair
        # ``q_nope_out``/``q_pe`` and set this to ["q_nope_out", "q_pe"].
        self.query_arg_names = ["q"]
        self.compilation_header_lines = []
        self.auxilary_func_def_lines = []
        self.compilation_cache_dir = sa.vortex_compilation_cache_dir
        self.sparse_attention_name = parent.sparse_attention.__class__.__name__.lower() + f"_{uuid.uuid4().hex[:8]}"  # unique name for this attention instance
        self.impl_backend = getattr(sa, "vortex_impl_backend", "triton") or "triton"
        self.vortex_attention_backend = getattr(
            sa, "vortex_attention_backend", "flashinfer"
        ) or "flashinfer"
        self.use_tensor_core = bool(getattr(sa, "vortex_use_tensor_core", False))
        # Tensor-core (bf16-compute + tl.dot) codegen is implemented only
        # in the triton W-kernel backend. The cuda backend has its own
        # fp32-accumulation path and ignores this flag — reject the
        # combination explicitly so a misconfigured run fails loudly
        # instead of silently running fp32.
        if self.use_tensor_core and self.impl_backend != "triton":
            raise ValueError(
                "vortex_use_tensor_core is only supported with "
                f"vortex_impl_backend='triton'; got '{self.impl_backend}'."
            )
        self._created = True
        return self


# Module-level singleton (part of the public package API)
ctx: Final[Context] = Context()


def get_ctx() -> Context:
    return ctx


__all__ = ["Context", "MetaData", "ctx", "get_ctx"]
