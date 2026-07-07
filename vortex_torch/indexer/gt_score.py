r"""Ground-truth group-level block scoring (Schedule.S, custom-kernel op).

``GTGroupScore`` computes the **exact** softmax attention block score that the
``naive_ground_truth_topk`` submission builds from an 11-op compiled chain
(GeMM + stable softmax + sum-over-group + max-over-block), but as a single
``Schedule.S`` op whose ``custom_impl`` codegen dispatches to the fused kernels
in the sibling ``gt_score_kernels`` package (JIT-bypass — no fused Triton is
emitted for this op).

Aggregation: **sum over the GQA query-head group, then max over the block
tokens** → one scalar score per (query, kv-head, block).

Naming: the ``Group`` in the name is the GQA-group reduction. A future
non-head-reduced variant (per-query-head selection) will live alongside as
``GTHeadScore``; keeping the reduction explicit here leaves that a clean
sibling.

The score op is generic over phase; the actual kernel is chosen in
``compiler/custom_impl/gt_score.py`` (prefill → ``group_scores_prefill``;
decode → reference exact-softmax until the GT decode kernel lands).
"""
import torch
from typing import FrozenSet, Optional
from .context import Context
from ..abs import vTensor, FORMAT, vOp
from ..utils import Schedule


class GTGroupScore(vOp):
    r"""
    Ground-truth group-level block score.

    :Math:
        For query token :math:`i` in kv-group :math:`g` and block :math:`b`,
        with true softmax weights :math:`p_{i,h,t}` over all causal keys
        (:math:`h` ranges over the group's query heads, :math:`t` over block
        :math:`b`'s tokens),

        .. math::

            \text{score}_{i,g,b} =
            \max_{t\in b}\ \sum_{h\in g}\ p_{i,h,t}.
    :__init__: ``GTGroupScore()`` — no arguments.
    :__call__: ``score = op(q, k, ctx=ctx)`` — ``q`` ``[B|S, group, D]``,
        ``k`` = ``cache["k"]`` ``[S, block_size, D]`` (auto-injected); returns
        ``score`` ``[S, 1, 1]`` RAGGED (one scalar per candidate block).
    :Note: ``Schedule.S`` — lowered by ``custom_impl/gt_score.py`` to the
        external ``gt_score_kernels`` kernel, not a fused Triton kernel.
    """

    _supported_q_formats: FrozenSet[FORMAT] = frozenset({FORMAT.BATCHED, FORMAT.RAGGED})

    def __init__(self):
        super().__init__()
        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[vTensor] = None
        self.schedule = Schedule.S

    # ---------------- profile ----------------
    def profile(self, q: vTensor, k: vTensor, ctx: Context) -> vTensor:
        r"""Trace-time: validate ``q`` ``[B|S, group, D]`` / ``k``
        ``[S, block_size, D]``, register the op, and return a RAGGED
        ``[S, 1, 1]`` score ``vTensor`` (one scalar per candidate block)."""
        prefix = self._prefix()

        assert isinstance(q, vTensor), f"{prefix}profile expects q to be vTensor, got {type(q)}"
        assert isinstance(k, vTensor), f"{prefix}profile expects k to be vTensor, got {type(k)}"
        assert q.dim() == 3 and k.dim() == 3, (
            f"{prefix}expected 3D inputs; got q.ndim={q.dim()}, k.ndim={k.dim()}"
        )
        assert q.shape[2] == k.shape[2], (
            f"{prefix}head_dim mismatch: q.shape[2]={q.shape[2]} vs k.shape[2]={k.shape[2]}"
        )
        assert q._format in self._supported_q_formats, (
            f"{prefix}no implementation for q._format={q._format}. "
            f"Supported: {sorted(self._supported_q_formats, key=lambda f: f.value)}"
        )

        # The score is a per-candidate-block scalar, always RAGGED (one row per
        # (query, kv-head), variable causal length) — matches the topK terminal
        # contract [S, 1, 1].
        self.output_format = FORMAT.RAGGED
        self.output_buffer = vTensor(
            shape=(0, 1, 1),
            dtype=ctx.vortex_dtype,
            device=q.device,
            _format=self.output_format,
            tensor_id=len(ctx.tensor_list),
        )

        ctx.tensor_list.append(self.output_buffer)
        ctx.output_tensor_to_op_list.append(len(ctx.op_list))
        ctx.op_list.append(self)
        ctx.op_to_input_tensor_list.append([q.tensor_id, k.tensor_id])
        ctx.op_to_output_tensor_list.append([self.output_buffer.tensor_id])

        return self.output_buffer
