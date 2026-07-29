from __future__ import annotations
from typing import Any, Final
import torch
import uuid
from ..utils import UNSET, Mode, resolve_dtype
from ..abs import ContextBase


class Context(ContextBase):
    """
    Mutable, single-instance cache context; populate later via .create(...).

    Beyond the minimal runtime knobs (page/block layout, head shape) this
    context also carries the **graph** and **codegen** state used by the
    cache compiler — mirroring ``vortex_torch.indexer.context.Context``.

    The graph state (``tensor_list``, ``op_list``, ``op_to_input_tensor_list``,
    ``op_to_output_tensor_list``, ``output_tensor_to_op_list``) is populated
    during the ``profile`` phase as cache ops register themselves; the
    codegen state (``compilation_header_lines``, ``auxilary_func_def_lines``,
    ``tensor_id_to_tensor_name_map``, ``compilation_cache_dir``,
    ``sparse_attention_name``, ``impl_backend``) is consumed by
    ``vortex_torch.cache.compiler``.
    """

    __slots__ = ContextBase.__slots__ + (
        # page & block infomation
        "max_new_tokens_per_batch", "page_size", "total_num_pages", "block_size",
        "num_blocks_per_page", "total_num_blocks",

        # model infomation
        "head_dim", "head_num",

        # hardware
        "num_sms",

        # auxilary memory / flops in graph
        "_aux_total_bytes", "_aux_total_flops",

        # graph state
        "tensor_list", "op_list",
        "output_tensor_to_op_list",
        "op_to_input_tensor_list", "op_to_output_tensor_list",

        # codegen state
        "sparse_attention_name", "impl_backend",
        "tensor_id_to_tensor_name_map",
        "compilation_header_lines", "auxilary_func_def_lines",
        "compilation_cache_dir",
    )

    # --- page & block ---
    max_new_tokens_per_batch: int   #: Max number of new tokens per batch.
    page_size: int                  #: Page size used for memory paging.
    total_num_pages: int            #: Total available pages in the cache.
    block_size: int                 #: Block size (page_size % block_size == 0).
    num_blocks_per_page: int        #: ``page_size // block_size``.
    total_num_blocks: int           #: ``num_blocks_per_page * total_num_pages``.

    # --- model ---
    head_dim: int                   #: Dimension per attention head.
    head_num: int                   #: Number of (KV) heads.

    # --- hardware ---
    num_sms: int                    #: Number of streaming multiprocessors.

    # --- auxiliary ---
    _aux_total_bytes: int           #: Accumulated auxiliary memory in bytes.
    _aux_total_flops: int           #: Accumulated auxiliary FLOPs.

    # --- graph state ---
    tensor_list: list               #: All vTensors that flow through the graph.
    op_list: list                   #: All registered vOps in profile order.
    output_tensor_to_op_list: list  #: For each tensor_id, the producing op_id (or ``None``).
    op_to_input_tensor_list: list   #: For each op_id, the list of input tensor_ids.
    op_to_output_tensor_list: list  #: For each op_id, the (single) output tensor_id wrapped in a list.

    # --- codegen state ---
    sparse_attention_name: str         #: Unique name for the cache-pipeline instance.
    impl_backend: str                  #: Implementation backend (default ``"triton"``).
    tensor_id_to_tensor_name_map: dict #: tensor_id -> Python name in the generated code.
    compilation_header_lines: list     #: Lines inserted at the top of the generated module.
    auxilary_func_def_lines: list      #: Auxiliary function/kernel definitions to embed.
    compilation_cache_dir: str         #: Where to write the generated module.


    def __init__(self) -> None:
        # Start as an empty shell (no big allocations).
        for name in self.__slots__:
            if name == "_created":
                object.__setattr__(self, name, False)
            elif name == "name":
                object.__setattr__(self, name, "Cache")
            elif name == "_aux_total_bytes":
                object.__setattr__(self, name, 0)
            elif name == "_aux_total_flops":
                object.__setattr__(self, name, 0)
            elif name == "mode":
                object.__setattr__(self, name, Mode.profile)
            else:
                object.__setattr__(self, name, UNSET)


    def create(self, parent: Any, model_runner: Any, *, overwrite: bool = False) -> "Context":
        """
        Populate this instance once (no locking). Set overwrite=True to allow re-init.
        NOTE: Without locking, concurrent callers may race; call from a single thread.
        """
        if self._created and not overwrite:
            raise RuntimeError("Context.create() already called; pass overwrite=True to reinitialize.")

        sa = model_runner.server_args
        self.vortex_dtype = resolve_dtype(
            getattr(sa, "vortex_dtype", "bfloat16"), default=torch.bfloat16
        )
        self.page_size = parent.page_size
        self.block_size = parent.block_size
        assert self.page_size % self.block_size == 0, "Page size must be a multiple of block size for block-sparse attention"
        self.num_blocks_per_page = self.page_size // self.block_size
        self.total_num_blocks = self.num_blocks_per_page * parent.num_pages
        self.total_num_pages = parent.num_pages
        self.max_new_tokens_per_batch = max(sa.max_prefill_tokens, model_runner.req_to_token_pool.size)
        self.head_num = parent.head_num
        self.head_dim = parent.head_dim
        self.num_sms = torch.cuda.get_device_properties(0).multi_processor_count

        # --- graph state (fresh per instance) ---
        self.tensor_list = []
        self.op_list = []
        self.output_tensor_to_op_list = []
        self.op_to_input_tensor_list = []
        self.op_to_output_tensor_list = []

        # --- codegen state (fresh per instance) ---
        self.tensor_id_to_tensor_name_map = {}
        self.compilation_header_lines = []
        self.auxilary_func_def_lines = []
        self.compilation_cache_dir = getattr(sa, "vortex_compilation_cache_dir", None)
        self.sparse_attention_name = (
            parent.__class__.__name__.lower() + f"_cache_{uuid.uuid4().hex[:8]}"
        )
        # Cache compiler only supports triton for now — the indexer is the
        # piece that picks up vortex_impl_backend. Hard-code triton here so
        # passing vortex_impl_backend="cuda" still works end-to-end (only
        # the indexer kernel switches backends).
        self.impl_backend = "triton"

        self._created = True
        return self


# Module-level singleton mirroring ``indexer.context.ctx``.
ctx: Final[Context] = Context()


def get_ctx() -> Context:
    return ctx


__all__ = ["Context", "ctx", "get_ctx"]
    

