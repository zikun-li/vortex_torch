from __future__ import annotations

"""
Support different attention backends.
Now there are two backends: FlashInfer and Triton.
FlashInfer is faster and Triton is easier to customize.
Each backend supports two operators: extend (i.e. prefill with cached prefix) and decode.
"""

import os
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, List, Optional, Union, Dict, Tuple
from functools import partial
import torch
from vortex_torch import is_hopper
from vortex_torch.abs import as_vtensor, FORMAT
from vortex_torch.indexer import Context, MetaData
from vortex_torch.indexer.compiler.compile import compile as compile_indexer
from vortex_torch.indexer.utils_sglang import (
    get_chunkwise_hn2nh_transpose,
    get_chunkwise_nh2hn_transpose,
    get_decode_planner,
    get_prefill_planner,
)
from vortex_torch.engine.sgl.attention_backend.prefill_sparse import (
    VortexSparsePrefillWrapper,
)
from vortex_torch.engine.sgl.attention_backend.prefill_select import (
    select_prefill_fast,
)
if os.environ["SGLANG_ENABLE_TORCH_COMPILE"] == "1":
    import logging

    torch._logging.set_logs(dynamo=logging.ERROR)
    torch._dynamo.config.suppress_errors = True

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.utils import is_flashinfer_available
from sglang.srt.layers.attention.flashinfer_backend import should_use_tensor_core
if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

if is_flashinfer_available():
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
        BatchPrefillWithRaggedKVCacheWrapper,
    )
    from flashinfer.cascade import merge_state
    from flashinfer.decode import _get_range_buf, get_seq_lens

@dataclass
class DecodeMetadata:
    decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper]

@dataclass
class PrefillMetadata:
    extend_no_prefix: bool


# Reuse this workspace buffer across all flashinfer wrappers
global_workspace_buffer = None


class VortexFlashInferBackend(AttentionBackend):
    """Flashinfer attention kernels."""

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        kv_last_page_len_buf: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        # Parse constants
        self.max_context_len = model_runner.model_config.context_len
        self.skip_prefill = skip_prefill
        self.is_multimodal = model_runner.model_config.is_multimodal
        assert not (
            model_runner.sliding_window_size is not None
            and model_runner.model_config.is_encoder_decoder
        ), "Sliding window and cross attention are not supported together"

        assert model_runner.sliding_window_size is None
        assert not model_runner.model_config.is_encoder_decoder 
        assert not self.skip_prefill
        assert not self.is_multimodal
        assert kv_indptr_buf is None
        assert kv_last_page_len_buf is None
        self.num_wrappers = 2
        self.dispatch_reason = None

        # Qwen2/Qwen3 models require higher flashinfer workspace size
        # if (
        #     "Qwen2ForCausalLM" in model_runner.model_config.hf_config.architectures
        #     or "Qwen3ForCausalLM" in model_runner.model_config.hf_config.architectures
        #     or "MiMoForCausalLM" in model_runner.model_config.hf_config.architectures
        # ):
        #     global_config.flashinfer_workspace_size = 512 * 1024 * 1024

        # Allocate buffers
        global global_workspace_buffer
        if global_workspace_buffer is None:
            global_workspace_buffer = torch.empty(
                512 * 1024 * 1024,
                dtype=torch.uint8,
                device=model_runner.device,
            )
        self.workspace_buffer = global_workspace_buffer
        max_bs = model_runner.req_to_token_pool.size
        
        self.num_qo_heads = model_runner.model_config.num_attention_heads // get_attention_tp_size()
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(get_attention_tp_size())
        self.group_size = self.num_qo_heads // self.num_kv_heads
        self.head_dim = model_runner.model_config.head_dim
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.decode_use_tensor_cores = should_use_tensor_core(self.data_type, self.num_qo_heads, self.num_kv_heads)
        assert self.q_data_type in [torch.bfloat16, torch.float8_e5m2, torch.float8_e4m3fn]
        assert self.data_type in [torch.bfloat16, torch.float8_e5m2, torch.float8_e4m3fn]
        self.is_fp8 = (self.data_type in [torch.float8_e5m2, torch.float8_e4m3fn])
        
        # Assign key configuration and parameters
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.page_size = model_runner.server_args.page_size
        self.block_size = model_runner.server_args.vortex_block_size
        self.layers_skip = model_runner.server_args.vortex_layers_skip
        # Opt-in GQA sparse-prefill path. When on, ``_compile`` builds a second
        # (prefill-shaped) indexer compile alongside the decode one; when off,
        # these stay None and ``forward_extend`` runs the existing dense path.
        self.sparse_prefill = bool(model_runner.server_args.vortex_sparse_prefill)
        if self.sparse_prefill:
            # Authoritative config gate (every entry path — get_engine, JSON, CLI —
            # constructs this backend, so guarding here is unbypassable).
            # check_engine_config validates the same, but only the benchmark
            # runners call it; enforce at server init so unsupported chunked /
            # radix configs can't silently mis-route to the dense path at runtime.
            _sa = model_runner.server_args
            if getattr(_sa, "chunked_prefill_size", None) != -1:
                raise ValueError(
                    "vortex_sparse_prefill requires chunked_prefill_size=-1 (got "
                    f"{getattr(_sa, 'chunked_prefill_size', None)!r}); sparse prefill "
                    "is fresh-prompt only."
                )
            if not getattr(_sa, "disable_radix_cache", False):
                raise ValueError(
                    "vortex_sparse_prefill requires disable_radix_cache=True; a "
                    "cached prompt prefix would enter a paged-prefix path the "
                    "sparse-prefill path does not handle (fresh-prompt only)."
                )
            # Query tiles are block-aligned and can't be smaller than one block, so
            # one block's queries × num_kv_heads must fit the decode planner
            # (batch<=1024, eff_bs=cap*nkv<=8192). If block_size exceeds this per-tile
            # cap there's no valid tile size — reject loudly rather than growing cap
            # past the planner limit at runtime.
            _cap = min(1024, 8192 // self.num_kv_heads)
            if self.block_size > _cap:
                raise ValueError(
                    f"vortex_sparse_prefill unsupported for block_size="
                    f"{self.block_size} with num_kv_heads={self.num_kv_heads}: a "
                    f"query tile can't be smaller than one block, so block_size must "
                    f"be <= min(1024, 8192//num_kv_heads)={_cap} "
                    f"(block_size*num_kv_heads <= 8192). Reduce block_size or nkv."
                )
        self.ctx_prefill: Optional[Context] = None
        self.compiled_indexer_prefill = None
        self.num_blocks_per_page = self.page_size // self.block_size
        assert self.page_size % self.block_size == 0, "Page size must be a multiple of block size."
        # ===========================
        # Prefill KV-indptr buffers
        # ===========================

        self.kv_indptr_prefill = torch.zeros(
            (max_bs * self.num_kv_heads + 1,),
            dtype=torch.int32,
            device=model_runner.device
        )

        # ===========================
        # Decode-path buffers live on ``self.ctx.metadata`` (pre-allocated
        # in ``_compile``). The flashinfer wrappers consume them directly
        # via ``self.ctx.metadata.dense_kv_indptr`` etc.
        # ===========================

        # ===========================
        # KV indices (prefill) — still owned by this object (the flashinfer
        # prefill wrapper has its own indptr/indices arrays).
        # ===========================

        self.kv_indices_prefill = torch.zeros(
            (
                (max_bs * self.num_kv_heads * model_runner.model_config.context_len + self.page_size - 1)
                // self.page_size,
            ),
            dtype=torch.int32,
            device=model_runner.device
        )

        # ===========================
        # KV last-page-len (prefill — decode lives on MetaData)
        # ===========================

        self.kv_last_page_len_prefill = torch.ones(
            (max_bs * self.num_kv_heads,),
            dtype=torch.int32,
            device=model_runner.device
        )

        # ===========================
        # Query/Output indptr buffers
        # ===========================

        self.qo_indptr = [
            torch.zeros(
                (max_bs + 1,),
                dtype=torch.int32,
                device=model_runner.device
            ),
            torch.zeros(
                (max_bs * self.num_kv_heads + 1,),
                dtype=torch.int32,
                device=model_runner.device
            ),
        ]

        # ===========================
        # Batch table (token-level mapping)
        # ===========================

        self.batch_table = torch.zeros(
            (model_runner.server_args.max_prefill_tokens,),
            dtype=torch.uint16,
            device=model_runner.device
        )

        
        self.prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace_buffer, "NHD", backend= "auto" if not is_hopper() else "fa3"
        )

        self.prefill_wrapper_paged = BatchPrefillWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        backend="fa2" if ((not is_hopper()) or self.is_fp8) else "fa3",
                    )
        
        self.decode_wrappers = [
            BatchDecodeWithPagedKVCacheWrapper(
                    self.workspace_buffer,
                    "NHD",
                    use_tensor_cores=self.decode_use_tensor_cores,
                ),
            BatchDecodeWithPagedKVCacheWrapper(
                    self.workspace_buffer,
                    "NHD",
                    use_tensor_cores=self.decode_use_tensor_cores,
                ),
        ]
        
        self.plan_decode = get_decode_planner(model_runner.server_args.vortex_schedule_policy)
        self.plan_prefill = get_prefill_planner()
        self.chunkwise_nh2hn_transpose = get_chunkwise_nh2hn_transpose()
        self.chunkwise_hn2nh_transpose = get_chunkwise_hn2nh_transpose()

        self.sparse_attention = model_runner.sparse_attention
        self.ctx = Context()
        self._compile(model_runner)
        # Other metadata
        self.forward_metadata: Union[PrefillMetadata, DecodeMetadata] = None
        self.decode_cuda_graph_metadata: Dict[int, List[BatchDecodeWithPagedKVCacheWrapper]] = {}
        self.plan_graph: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.cuda.CUDAGraph]]
    

    def _compile(self, model_runner: "ModelRunner") -> None:
        """Trace + compile the sparse-attention indexer.

        Always compiles the **decode** indexer (``self.compiled_indexer``). When
        ``sparse_prefill`` is on, also compiles a **prefill** indexer
        (``self.compiled_indexer_prefill``) from the *same* ``forward_indexer``
        against a second Context (``self.ctx_prefill``) whose leading BATCHED
        axis is per (query-token, kv-head) and whose terminal ``topK`` emits
        scores (the Phase-3 kernel selects). See memory ``sparse-prefill-gqa-design``.
        """
        self.compiled_indexer = self._trace_and_compile(
            self.ctx, model_runner, prefill=False
        )
        self.gt_prefill = False
        if self.sparse_prefill:
            self.ctx_prefill = Context()
            self.compiled_indexer_prefill = self._trace_and_compile(
                self.ctx_prefill, model_runner, prefill=True
            )
            # Ground-truth submission? Its indexer is the two custom kernel ops
            # (GTGroupScore -> GTTopK) which produce the selection directly (via
            # gt_score_kernels), so the prefill tile loop skips plan_decode +
            # select_prefill_fast and reads the CSR from ctx_prefill.gt_state.
            from vortex_torch.indexer import GTTopK
            self.gt_prefill = any(
                isinstance(op, GTTopK) for op in self.ctx_prefill.op_list
            )
            # Preallocated fp32 score buffer (emit-scores terminal copies into it,
            # sized to the prefill causal score-axis budget) + the exact-causal
            # block-sparse attention wrapper (Phase 1'). bf16 only for prefill.
            self.prefill_scores = torch.zeros(
                (self.ctx_prefill.max_num_blocks, 1, 1),
                dtype=torch.float32, device=model_runner.device,
            )
            self.prefill_sparse_wrapper = VortexSparsePrefillWrapper(
                self.num_qo_heads, self.num_kv_heads, self.head_dim,
                device=model_runner.device,
                workspace_buffer=self.workspace_buffer,
                q_data_type=self.q_data_type, kv_data_type=self.q_data_type,
            )

    def _trace_and_compile(self, ctx: "Context", model_runner: "ModelRunner",
                           *, prefill: bool):
        """Trace ``forward_indexer`` on zero-leading-dim dummies into ``ctx`` and
        compile it. Shared by the decode (``prefill=False``) and sparse-prefill
        (``prefill=True``) compiles — only the Context budgets + terminal
        lowering differ (both driven off ``ctx.sparse_prefill``)."""
        device = model_runner.device
        dtype = self.q_data_type
        indexer = self.sparse_attention.forward_indexer

        ctx.create(self, model_runner, prefill=prefill)
        # Allocate every per-forward-batch buffer (winfo_*, dense/sparse
        # kv_indptr+indices, kv_last_page_len) on a MetaData owned by the
        # context. The planner writes into this MetaData; the indexer kernels
        # read from it. Decode wrappers additionally take pointers into the
        # decode ctx's MetaData.
        ctx.metadata = MetaData.preallocate(ctx, device=device)
        ctx.assert_created()
        ctx.profile()

        def register(vt, name: str) -> None:
            ctx.tensor_list.append(vt)
            ctx.output_tensor_to_op_list.append(None)
            ctx.tensor_id_to_tensor_name_map[vt.tensor_id] = name

        def make_dummy(shape, fmt, tensor_id, *, tdtype=dtype, zeros=False):
            factory = torch.zeros if zeros else torch.empty
            return as_vtensor(factory(shape, device=device, dtype=tdtype), fmt, tensor_id=tensor_id)

        with torch.no_grad():
            q_dummy = make_dummy((0, self.group_size, self.head_dim), FORMAT.BATCHED, tensor_id=0)
            register(q_dummy, "q")

            o_dummy = make_dummy((0, 1, 1), FORMAT.RAGGED, tensor_id=1)
            register(o_dummy, "o")

            cache_dummy = {}
            for i, (name, (shape, cache_dtype)) in enumerate(
                self.sparse_attention.get_cache_meta_info().items()
            ):
                vt = make_dummy(
                    (0, shape[0], shape[1]),
                    FORMAT.PAGED,
                    tensor_id=2 + i,
                    tdtype=cache_dtype,
                    zeros=True,
                )
                cache_dummy[name] = vt
                register(vt, f"cache['{name}']")

            indexer(q_dummy, o_dummy, cache_dummy, ctx=ctx)

        compiled = compile_indexer(ctx)()
        ctx.summary()
        ctx.execute()
        return compiled

    
    def init_forward_metadata(self, forward_batch: ForwardBatch):
        
        assert not forward_batch.forward_mode.is_draft_extend()
        assert not forward_batch.forward_mode.is_target_verify()
        
        if forward_batch.forward_mode.is_decode_or_idle():
            
            bs = len(forward_batch.req_pool_indices)
            self.plan_decode(
                cached_seq_lens=forward_batch.seq_lens.to(torch.int32),
                req_to_token=self.req_to_token,
                req_indices=forward_batch.req_pool_indices,
                ctx=self.ctx
            )
            
            self.decode_wrappers[0].plan(
                indptr=self.ctx.metadata.dense_kv_indptr[:bs*self.num_kv_heads+1],
                indices=self.ctx.metadata.dense_kv_indices,
                last_page_len=self.ctx.metadata.kv_last_page_len[:bs*self.num_kv_heads],
                num_qo_heads=self.group_size,
                num_kv_heads=1,
                head_dim=self.head_dim,
                page_size=self.block_size,
                q_data_type=self.q_data_type,
                kv_data_type=self.data_type,
            )
            
            self.decode_wrappers[1].plan(
                indptr=self.ctx.metadata.sparse_kv_indptr[:bs*self.num_kv_heads+1],
                indices=self.ctx.metadata.sparse_kv_indices,
                last_page_len=self.ctx.metadata.kv_last_page_len[:bs*self.num_kv_heads],
                num_qo_heads=self.group_size,
                num_kv_heads=1,
                head_dim=self.head_dim,
                page_size=self.block_size,
                q_data_type=self.q_data_type,
                kv_data_type=self.data_type,
            )
            self.forward_metadata = DecodeMetadata([self.decode_wrappers[0], self.decode_wrappers[1]])

        elif forward_batch.forward_mode.is_extend():
            
            prefix_lens = forward_batch.extend_prefix_lens
            extend_no_prefix = not any(forward_batch.extend_prefix_lens_cpu)
            bs = len(forward_batch.req_pool_indices)
            
            self.plan_prefill(
                cached_seq_lens=prefix_lens,
                dense_kv_indptr=self.kv_indptr_prefill[:bs*self.num_kv_heads+1],
                dense_kv_indices=self.kv_indices_prefill,
                input_seq_lens=(forward_batch.seq_lens.to(torch.int32) - prefix_lens),
                qo_indptr_ragged=self.qo_indptr[0][:bs+1],
                qo_indptr_paged=self.qo_indptr[1][:bs*self.num_kv_heads+1],
                kv_last_page_len=self.kv_last_page_len_prefill[:bs*self.num_kv_heads],
                req_to_token=self.req_to_token,
                req_indices=forward_batch.req_pool_indices,
                batch_table=self.batch_table,
                page_size=self.page_size,
                num_kv_heads=self.num_kv_heads
            )
            
   
            self.prefill_wrapper_ragged.plan(
                self.qo_indptr[0][:bs+1],
                self.qo_indptr[0][:bs+1],
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                q_data_type=self.q_data_type,
            )
            
            self.prefill_wrapper_paged.plan(
                self.qo_indptr[1][:bs*self.num_kv_heads+1],
                self.kv_indptr_prefill[:bs*self.num_kv_heads+1],
                self.kv_indices_prefill,
                self.kv_last_page_len_prefill[:bs*self.num_kv_heads],
                self.group_size,
                1,
                self.head_dim,
                self.page_size,
                q_data_type=self.q_data_type,
                kv_data_type=self.data_type,
                custom_mask=None,
                non_blocking=True,
            )
            

            self.forward_metadata = PrefillMetadata(extend_no_prefix)

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
    ):
        pass
    
    
    def capture_plan_graph(
        self, 
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        bs: int):
        
        pass

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info,
    ):  
        assert bs == num_tokens
        
        if forward_mode.is_decode_or_idle():
            decode_wrappers = [
                BatchDecodeWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        use_cuda_graph=True,
                        use_tensor_cores=self.decode_use_tensor_cores,
                        paged_kv_indptr_buffer=self.ctx.metadata.dense_kv_indptr[:bs*self.num_kv_heads + 1],
                        paged_kv_indices_buffer=self.ctx.metadata.dense_kv_indices,
                        paged_kv_last_page_len_buffer=self.ctx.metadata.kv_last_page_len[
                            :bs*self.num_kv_heads
                        ],
                    ),
                
                BatchDecodeWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        use_cuda_graph=True,
                        use_tensor_cores=self.decode_use_tensor_cores,
                        paged_kv_indptr_buffer=self.ctx.metadata.sparse_kv_indptr[:bs*self.num_kv_heads + 1],
                        paged_kv_indices_buffer=self.ctx.metadata.sparse_kv_indices,
                        paged_kv_last_page_len_buffer=self.ctx.metadata.kv_last_page_len[
                            :bs*self.num_kv_heads
                        ],
                    ),
                
            ]

            self.plan_decode(
                cached_seq_lens=seq_lens.to(torch.int32),
                req_to_token=self.req_to_token,
                req_indices=req_pool_indices,
                ctx=self.ctx
            )
            
            decode_wrappers[0].plan(
                indptr=self.ctx.metadata.dense_kv_indptr[:bs*self.num_kv_heads+1],
                indices=self.ctx.metadata.dense_kv_indices,
                last_page_len=self.ctx.metadata.kv_last_page_len[:bs*self.num_kv_heads],
                num_qo_heads=self.group_size,
                num_kv_heads=1,
                head_dim=self.head_dim,
                page_size=self.block_size,
                q_data_type=self.q_data_type,
                kv_data_type=self.data_type,
            )
            
            decode_wrappers[1].plan(
                indptr=self.ctx.metadata.sparse_kv_indptr[:bs*self.num_kv_heads+1],
                indices=self.ctx.metadata.sparse_kv_indices,
                last_page_len=self.ctx.metadata.kv_last_page_len[:bs*self.num_kv_heads],
                num_qo_heads=self.group_size,
                num_kv_heads=1,
                head_dim=self.head_dim,
                page_size=self.block_size,
                q_data_type=self.q_data_type,
                kv_data_type=self.data_type,
            )
            
            self.decode_cuda_graph_metadata[bs] = decode_wrappers
            self.forward_metadata = DecodeMetadata(decode_wrappers)             
        else:
            raise NotImplementedError
            

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info,
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        assert forward_mode.is_decode_or_idle()
        
        self.plan_decode(
                cached_seq_lens=seq_lens.to(torch.int32),
                req_to_token=self.req_to_token,
                req_indices=req_pool_indices,
                ctx=self.ctx
            )
        
        self.decode_cuda_graph_metadata[bs][0].plan(
            indptr=self.ctx.metadata.dense_kv_indptr[:bs*self.num_kv_heads+1],
            indices=self.ctx.metadata.dense_kv_indices,
            last_page_len=self.ctx.metadata.kv_last_page_len[:bs*self.num_kv_heads],
            num_qo_heads=self.group_size,
            num_kv_heads=1,
            head_dim=self.head_dim,
            page_size=self.block_size,
            q_data_type=self.q_data_type,
            kv_data_type=self.data_type,
        )
        
        self.decode_cuda_graph_metadata[bs][1].plan(
            indptr=self.ctx.metadata.sparse_kv_indptr[:bs*self.num_kv_heads+1],
            indices=self.ctx.metadata.sparse_kv_indices,
            last_page_len=self.ctx.metadata.kv_last_page_len[:bs*self.num_kv_heads],
            num_qo_heads=self.group_size,
            num_kv_heads=1,
            head_dim=self.head_dim,
            page_size=self.block_size,
            q_data_type=self.q_data_type,
            kv_data_type=self.data_type,
        )

    def get_cuda_graph_seq_len_fill_value(self):
        
        return 1

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        
        assert not layer.is_cross_attention
        cache_loc = forward_batch.out_cache_loc

        logits_soft_cap = layer.logit_cap

        q = q.contiguous()

        # GQA sparse-prefill path: fresh prompt only, non-skip layers, and only
        # when save_kv_cache is set. The sparse path MUST write K/V + summaries to
        # the pool before the indexer runs (it reads them back), so it cannot honor
        # a no-save pass; fall back to dense (which respects save_kv_cache) instead
        # of silently mutating cache state.
        if (
            self.sparse_prefill
            and save_kv_cache
            and self.forward_metadata.extend_no_prefix
            and layer.layer_id not in self.layers_skip
        ):
            return self._forward_extend_sparse(
                q, k, v, layer, forward_batch, cache_loc, logits_soft_cap
            )

        if self.forward_metadata.extend_no_prefix:
            o = self.prefill_wrapper_ragged.forward(
                q.view(-1, layer.tp_q_head_num, layer.head_dim),
                k.view(-1, layer.tp_k_head_num, layer.head_dim),
                v.view(-1, layer.tp_v_head_num, layer.head_dim),
                causal=True,
                sm_scale=layer.scaling,
                logits_soft_cap=logits_soft_cap,
            )

        else:
            o1, s1 = self.prefill_wrapper_ragged.forward_return_lse(
                q.view(-1, layer.tp_q_head_num, layer.head_dim),
                k.view(-1, layer.tp_k_head_num, layer.head_dim),
                v.view(-1, layer.tp_v_head_num, layer.head_dim),
                causal=True,
                sm_scale=layer.scaling,
                logits_soft_cap=logits_soft_cap,
                )
            
            q_t = self.chunkwise_nh2hn_transpose(
                q.view(-1, self.num_qo_heads, self.head_dim),
                self.qo_indptr[0],
                self.batch_table,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim
            )
            
            
            k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
            k_cache = k_cache.view(-1, self.page_size, 1, self.head_dim)
            v_cache = v_cache.view(-1, self.page_size, 1, self.head_dim)
            o2, s2 = self.prefill_wrapper_paged.forward_return_lse(
                q_t,
                (k_cache, v_cache),
                causal=False,
                sm_scale=layer.scaling,
                logits_soft_cap=logits_soft_cap,
                )
            o2_t, s2_t = self.chunkwise_hn2nh_transpose(
                o2,  s2,
                self.qo_indptr[0],
                self.batch_table,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim
            )
            
            o, _ = merge_state(o1, s1, o2_t, s2_t)

        if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def _forward_extend_sparse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        cache_loc: torch.Tensor,
        logits_soft_cap,
    ):
        """GQA sparse-prefill attention (fresh prompt, non-skip layers).

        Pipeline (all pieces unit-verified in tests/prefill_sparse/):
          1. Write K/V + summaries into the pool FIRST (``set_kv_buffer`` also
             runs ``forward_cache``) so the indexer reads the current prompt.
          2. Per query token, a *pseudo-decode-request* (cached_seq_len = causal
             pos+1, sharing the request's ``req_to_token`` row) → ``plan_decode``
             on ``ctx_prefill`` populates winfo_* + causal CSR.
          3. ``compiled_indexer_prefill`` scores each (query-token, kv-head,
             candidate-block) and emits the RAGGED scores into ``prefill_scores``.
          4. ``build_head_major_selection`` → per-request head-major CSR +
             ``(nnz,1,C)`` causal mask.
          5. Per request, per query-tile: the exact-causal block-sparse wrapper.

        Query-axis chunking (Option A): each request's query axis is tiled into
        chunks of ``<= cap = min(1024, 8192//nkv)`` tokens (the decode-planner
        limit). Per tile the planner/indexer/selector run as one unit (they share
        ``ctx_prefill.metadata`` + ``prefill_scores``, overwritten per tile); the
        wrapper then attends the tile's queries to their causal KV prefix
        (``s_q <= s_kv``). Long prompts / multi-request batches are handled by the
        loop; nothing caps the total prompt length.
        """
        device = q.device
        Hq, Hkv, Dh = layer.tp_q_head_num, layer.tp_k_head_num, layer.head_dim
        q3 = q.view(-1, Hq, Dh)          # [tokens, Hq, D]
        k3 = k.view(-1, Hkv, Dh)         # [tokens, Hkv, D]  (fresh prompt = full seq)
        v3 = v.view(-1, Hkv, Dh)
        total_tokens = q3.shape[0]

        # 1) Hoist K/V + forward_cache (centroids etc.) before the indexer.
        forward_batch.token_to_kv_pool.set_kv_buffer(
            layer, cache_loc, k, v, layer.k_scale, layer.v_scale
        )

        bs = len(forward_batch.req_pool_indices)
        qo_cpu = self.qo_indptr[0][:bs + 1].cpu().tolist()   # ragged token ranges
        req_pool = forward_batch.req_pool_indices            # [bs]
        # Query-tile size (decode/compiled path): bounded by the decode-planner
        # caps (batch<=1024, eff_bs=cap*nkv<=8192), rounded DOWN to a block_size
        # multiple so every tile start ``a=k*cap`` is block-aligned. The diagonal
        # pass tiles the tile's own tokens on the LOCAL block grid [0,C,2C,...];
        # that only matches the global block grid (used by the past pass +
        # selection) when tiles start on a block boundary, else same-block
        # predecessors are dropped/double-counted. Rounding only shrinks cap.
        cap = min(1024, 8192 // self.num_kv_heads)
        cap = max(self.block_size, (cap // self.block_size) * self.block_size)
        cache = forward_batch.token_to_kv_pool.get_cache(layer.layer_id)

        o = torch.empty((total_tokens, Hq, Dh), dtype=q3.dtype, device=device)
        for r in range(bs):
            s0, s1 = qo_cpu[r], qo_cpu[r + 1]
            S_r = s1 - s0
            k_req = k3[s0:s1]            # full request K/V (fresh prompt)
            v_req = v3[s0:s1]
            req_idx_r = int(req_pool[r])
            # Tile size. The GT path does NOT use plan_decode, so it is NOT bound
            # by the decode-planner caps (1024 / 8192) — only by the score-matrix
            # memory (~tlen * nkv * s_blocks * 4 bytes, held to ~4 GB) and block
            # alignment. Bigger tiles => far fewer per-KV-head wrapper plan()
            # launches (the prefill bottleneck). The compiled path keeps the
            # decode-planner cap.
            if self.gt_prefill:
                s_blocks = max(1, (S_r + self.block_size - 1) // self.block_size)
                gt_cap = int(4.0 * 1e9 / (self.num_kv_heads * s_blocks * 4))
                tile_cap = max(self.block_size,
                               (gt_cap // self.block_size) * self.block_size)
            else:
                tile_cap = cap
            # chunk this request's query axis into <= tile_cap-token tiles
            for a in range(0, S_r, tile_cap):
                b = min(a + tile_cap, S_r)
                tlen = b - a
                C = self.block_size
                q_tile = q3[s0 + a : s0 + b]                    # [tlen, Hq, D]

                if self.gt_prefill:
                    # Ground-truth path: the two custom ops (GTGroupScore ->
                    # GTTopK) run inside compiled_indexer_prefill.forward and
                    # dispatch to gt_score_kernels (score + FlashInfer top-k +
                    # assemble), producing the head-major CSR selection directly.
                    # Raw per-tile K + geometry go via ctx_prefill.gt_state (our
                    # score kernel needs a contiguous [H_kv,S,C,D] prefix, which
                    # the paged cache / planner metadata don't carry). No
                    # plan_decode, no select_prefill_fast.
                    self.ctx_prefill.gt_state = {
                        "raw_k": k_req[:b].contiguous(),   # [b, Hkv, D] causal prefix
                        "q_offset": a,
                        "scale": layer.scaling,
                        "tlen": tlen,
                    }
                    q_idx = q_tile.reshape(-1, self.group_size, Dh)
                    self.compiled_indexer_prefill.forward(
                        q_idx, self.prefill_scores, cache, self.ctx_prefill
                    )
                    kv_indptr = self.ctx_prefill.gt_state["kv_indptr"]
                    block_ids = self.ctx_prefill.gt_state["block_ids"]
                    # run() ignores block_mask (diagonal-split); pass a placeholder.
                    block_mask = torch.empty((0, 1, C), dtype=torch.bool, device=device)
                    o[s0 + a : s0 + b] = self.prefill_sparse_wrapper.run(
                        q_tile.contiguous(),
                        k_req[:b].contiguous(),
                        v_req[:b].contiguous(),
                        kv_indptr, block_ids, block_mask, self.block_size,
                        sm_scale=layer.scaling, logits_soft_cap=logits_soft_cap,
                    )
                    continue

                # Hard guard BEFORE plan_decode: plan_decode launches a CUDA kernel
                # that WRITES dense/sparse_kv_indices + indptr + kv_last_page_len, so
                # an oversized tile would OOB there before any post-hoc check. Verify
                # the tile fits the preallocated buffers using an EXACT analytic count
                # (not read from the buffers plan_decode fills): rows = tlen (<= max_bs
                # rows of indptr/last-page) and candidate blocks = nkv * Σ_{t=a+1}^{b}
                # ceil(t/C) (<= max_num_blocks, the score/CSR axis). Holds by
                # construction when ΣS_q <= max_prefill_tokens; raise loudly otherwise.
                C = self.block_size

                def _cum_ceil(n, C=C):  # Σ_{t=1}^{n} ceil(t/C), closed form
                    q, rmd = divmod(n, C)
                    return C * q * (q + 1) // 2 + rmd * (q + 1)

                n_cand = self.num_kv_heads * (_cum_ceil(b) - _cum_ceil(a))
                if tlen > self.ctx_prefill.max_bs:
                    raise ValueError(
                        f"sparse prefill tile rows {tlen} exceed ctx_prefill.max_bs "
                        f"{self.ctx_prefill.max_bs} (a={a}, b={b}); prompt exceeds "
                        f"max_prefill_tokens."
                    )
                if n_cand > self.ctx_prefill.max_num_blocks:
                    raise ValueError(
                        f"sparse prefill tile candidate blocks {n_cand} exceed "
                        f"max_num_blocks {self.ctx_prefill.max_num_blocks} (a={a}, "
                        f"b={b}, S_r={S_r}); prompt exceeds max_prefill_tokens."
                    )
                # (1) pseudo-request plan for the tile (local causal pos+1 = a+1..b)
                cached = torch.arange(a + 1, b + 1, device=device, dtype=torch.int32)
                reqidx = torch.full((tlen,), req_idx_r, device=device, dtype=torch.int64)
                self.plan_decode(
                    cached_seq_lens=cached, req_to_token=self.req_to_token,
                    req_indices=reqidx, ctx=self.ctx_prefill,
                )
                # (2) indexer → tile scores (head-minor rows = tile_token*nkv + head)
                q_tile = q3[s0 + a : s0 + b]                    # [tlen, Hq, D]
                q_idx = q_tile.reshape(-1, self.group_size, Dh) # [tlen*nkv, group, D]
                self.compiled_indexer_prefill.forward(
                    q_idx, self.prefill_scores, cache, self.ctx_prefill
                )
                # (3) vectorized selection → head-major CSR + (nnz,1,C) mask
                kv_indptr, block_ids, block_mask = select_prefill_fast(
                    self.prefill_scores,
                    self.ctx_prefill.metadata.dense_kv_indptr,
                    tlen, self.num_kv_heads, self.block_size, a,
                    topk_val=self.ctx_prefill.topk_val,
                    topk_ratio=self.ctx_prefill.topk_ratio,
                    reserved_bos=self.ctx_prefill.block_reserved_bos,
                    reserved_eos=self.ctx_prefill.block_reserved_eos,
                )
                # (4) attention: tile queries vs causal KV prefix (k/v up to token b)
                o[s0 + a : s0 + b] = self.prefill_sparse_wrapper.run(
                    q_tile.contiguous(),
                    k_req[:b].contiguous(),
                    v_req[:b].contiguous(),
                    kv_indptr, block_ids, block_mask, self.block_size,
                    sm_scale=layer.scaling, logits_soft_cap=logits_soft_cap,
                )
        return o.view(-1, Hq * Dh)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        """
        Decode-time forward pass with optional sparse attention.
        Expects KV to be sourced from token_to_kv_pool; can also save new KV.
        """

        # Sanity checks and setup
        assert not layer.is_cross_attention
        cache_loc = forward_batch.out_cache_loc

        # Optionally write incoming K/V to decode cache
        if k is not None:
            assert v is not None
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )

        # Read Cache from memory pool
        cache = forward_batch.token_to_kv_pool.get_cache(layer.layer_id)
        
        cache_k = cache["k"].view(-1, self.block_size, 1, self.head_dim)
        cache_v = cache["v"].view(-1, self.block_size, 1, self.head_dim)

        # Use the *_float scalars (not layer.k_scale / layer.v_scale, which are
        # GPU tensors): the flashinfer decode wrapper expects python-float
        # scales, and reading the tensor would force a device->host sync that
        # breaks cuda-graph capture. For a bf16 KV cache these are the 1.0
        # default (no-op); for an fp8 cache they dequantize the stored K/V,
        # matching the div() applied on the write side in set_kv_buffer.
        k_scale = layer.k_scale_float if layer.k_scale_float is not None else 1.0
        v_scale = layer.v_scale_float if layer.v_scale_float is not None else 1.0

        # Decide whether to use sparsity on this layer
        use_sparsity = (layer.layer_id not in self.layers_skip)

        if use_sparsity:
            # Prepare Q in grouped shape expected by sparse path
            q = q.contiguous().view(-1, self.group_size, layer.head_dim)

            # Build sparse indices into paged KV buffers
            self.compiled_indexer.forward(
                q=q,
                o=self.forward_metadata.decode_wrappers[1]._paged_kv_indices_buf,
                cache=cache,
                ctx=self.ctx
            )

            # Sparse attention compute
            o = self.forward_metadata.decode_wrappers[1].forward(
                q,
                (cache_k, cache_v),
                sm_scale=layer.scaling,
                logits_soft_cap=layer.logit_cap,
                k_scale=k_scale,
                v_scale=v_scale,
            )

        else:
            # Dense attention path
            o = self.forward_metadata.decode_wrappers[0].forward(
                q.contiguous().view(-1, self.group_size, layer.head_dim),
                (cache_k, cache_v),
                sm_scale=layer.scaling,
                logits_soft_cap=layer.logit_cap,
                k_scale=k_scale,
                v_scale=v_scale,
            )

        # Restore to merged head dimension
        return o.view(-1, layer.tp_q_head_num * layer.head_dim)