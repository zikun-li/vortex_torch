"""gt_topk — launcher emitter (Schedule.S, JIT-bypass terminal op).

Emits a plain-Python call to the ground-truth top-k selection glue in
``vortex_torch.engine.sgl.attention_backend.gt_prefill`` (FlashInfer
``top_k_ragged_transform`` + ``assemble_block_ids`` from ``gt_score_kernels``).

Per-phase lowering (chosen at codegen time off ``ctx.sparse_prefill``):
  * prefill → ``gt_topk_prefill`` (our FlashInfer top-k + assemble; writes the
    head-major CSR selection to ``ctx.gt_state`` for the sparse-prefill wrapper).
  * decode  → ``gt_topk_decode`` (placeholder until the GT decode kernel lands;
    raises at runtime).
"""
from ..graph import Graph
from ...context import Context
from ....utils import INDENT
from ....abs import FORMAT
from ...gt_topk import GTTopK
from .topk import generate_topk_impl


def generate_gttopk_impl(graph: Graph, op_id: int, ctx: Context) -> str:
    op = graph.op_list[op_id]
    assert issubclass(op.__class__, GTTopK), f"Expected a GTTopK op, got {op}"
    in_tensor_id = graph.op_to_input_tensor_list[op_id][0]    # block scores
    out_tensor_id = graph.op_to_output_tensor_list[op_id][0]  # o (selection sink)
    t_i = graph.tensor_list[in_tensor_id]
    assert t_i._format == FORMAT.RAGGED, (
        f"Expected ragged score input for GTTopK, got {t_i._format}"
    )

    if not getattr(ctx, "sparse_prefill", False):
        # Decode: reuse the built-in exact top-k selector (writes sparse_kv_indices
        # from the GTGroupScore decode reference scores). Identical to topK decode.
        return generate_topk_impl(graph, op_id, ctx)

    # Prefill: our FlashInfer top-k + assemble (writes the CSR to ctx.gt_state).
    ctx.compilation_header_lines.extend([
        "from vortex_torch.engine.sgl.attention_backend.gt_prefill import "
        "gt_topk_prefill as _gt_topk_prefill",
    ])
    return (
        f"{INDENT}_gt_topk_prefill(\n"
        f"{INDENT * 2}tensor_{in_tensor_id},\n"
        f"{INDENT * 2}tensor_{out_tensor_id},\n"
        f"{INDENT * 2}ctx,\n"
        f"{INDENT})"
    )
