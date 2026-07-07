"""gt_score — launcher emitter (Schedule.S, JIT-bypass).

Emits a plain-Python call to the ground-truth block-score kernel in
``vortex_torch.engine.sgl.attention_backend.gt_runtime`` (which wraps the
external ``gt_score_kernels`` package). No fused Triton is emitted for this op —
this IS the "operator that bypasses vortex compilation" mechanism.

Per-phase lowering (chosen at codegen time off ``ctx.sparse_prefill``):
  * prefill → ``gt_group_score_prefill`` (our fused score kernel).
  * decode  → ``gt_group_score_decode`` (placeholder until the GT decode kernel
    lands; raises at runtime).
"""
from ..graph import Graph
from ...context import Context
from ....utils import INDENT
from ....abs import FORMAT
from ...gt_score import GTGroupScore


def generate_gtgroupscore_impl(graph: Graph, op_id: int, ctx: Context) -> str:
    op = graph.op_list[op_id]
    assert issubclass(op.__class__, GTGroupScore), (
        f"Expected a GTGroupScore op, got {op}"
    )
    q_tensor_id = graph.op_to_input_tensor_list[op_id][0]     # q (query)
    k_tensor_id = graph.op_to_input_tensor_list[op_id][1]     # cache["k"] (paged)
    out_tensor_id = graph.op_to_output_tensor_list[op_id][0]  # block scores
    t_o = graph.tensor_list[out_tensor_id]
    assert t_o._format == FORMAT.RAGGED, (
        f"Expected ragged score output for GTGroupScore, got {t_o._format}"
    )

    ctx.compilation_header_lines.extend([
        "from vortex_torch.engine.sgl.attention_backend.gt_runtime import "
        "gt_group_score_prefill as _gt_group_score_prefill, "
        "gt_group_score_decode as _gt_group_score_decode",
    ])
    fn = "_gt_group_score_prefill" if getattr(ctx, "sparse_prefill", False) \
        else "_gt_group_score_decode"

    # prefill reads the contiguous per-tile K from ctx.gt_state (ignores k here);
    # decode reads the paged cache["k"] directly. Pass both so one signature works.
    return (
        f"{INDENT}{fn}(\n"
        f"{INDENT * 2}tensor_{q_tensor_id},\n"
        f"{INDENT * 2}tensor_{k_tensor_id},\n"
        f"{INDENT * 2}tensor_{out_tensor_id},\n"
        f"{INDENT * 2}ctx,\n"
        f"{INDENT})"
    )
