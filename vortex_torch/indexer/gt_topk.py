r"""Ground-truth top-k block selection (Schedule.S, custom-kernel terminal op).

``GTTopK`` is the terminal op for a ground-truth ``forward_indexer``: it turns
per-block scores into the sparse set of blocks each (query, kv-head) attends to,
using **our** FlashInfer-based top-k (``top_k_ragged_transform``) +
``assemble_block_ids`` for prefill. Like the built-in ``topK`` it claims ``o``
and force-includes the reserved BOS/EOS blocks + the causal diagonal.

Kept distinct from the built-in ``topK`` so the existing emit-scores prefill
path (used by other sparse-prefill submissions) is untouched. The concrete
kernel is chosen in ``compiler/custom_impl/gt_topk.py``:
  * prefill → FlashInfer ``top_k_ragged_transform`` + ``assemble_block_ids``
    (writes head-major CSR ``block_ids`` into ``o`` + ``kv_indptr`` into
    ``ctx.metadata``);
  * decode → the existing exact selector (``generate_topk_impl`` body), until a
    FlashInfer decode top-k is wired.
"""
from typing import FrozenSet
from ..abs import vOp, vTensor, FORMAT
from .context import Context
from ..utils import Schedule


class GTTopK(vOp):
    r"""
    Ground-truth top-k block selector (terminal op).

    :Math:
        Same selected set as :class:`vortex_torch.indexer.topK` — reserved
        BOS prefix :math:`\mathcal{B}`, reserved EOS/diagonal suffix
        :math:`\mathcal{E}`, and the top-:math:`k` of the remaining
        strictly-past blocks by score:

        .. math::

            \mathcal{S} = \mathcal{B}\cup\mathcal{E}\cup
            \operatorname*{top\text{-}k}_{\,p\notin\mathcal{B}\cup\mathcal{E}} X_p.
    :__init__: ``GTTopK()`` — no arguments; budget ``topk_val`` / ratio and the
        reserved BOS/EOS counts are read from :class:`Context` at runtime.
    :__call__: ``op(score, o, ctx=ctx)`` — ``score`` ``[S, 1, 1]`` RAGGED;
        selected block ids written **in place** into ``o``. Returns nothing.
    :Note: ``Schedule.S``; lowered by ``custom_impl/gt_topk.py`` to our
        FlashInfer top-k (prefill) / the exact selector (decode).
    """

    _supported_formats: FrozenSet[FORMAT] = frozenset({FORMAT.RAGGED})

    def __init__(self):
        super().__init__()
        self.schedule = Schedule.S

    # ---------------- profile ----------------
    def profile(self, x: vTensor, o: vTensor, ctx: Context) -> None:
        r"""Trace-time: validate ``x`` ``[S, 1, 1]`` and ``o`` (both RAGGED
        ``vTensor`` on the same device) and register the op. Allocates nothing;
        ``o`` is filled in place at execute time (same contract as ``topK``)."""
        prefix = self._prefix()

        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert isinstance(o, vTensor), f"{prefix}profile expects o to be vTensor, got {type(o)}"
        assert x.dim() == 3, (
            f"{prefix}expected x to be 3D, got ndim={x.dim()} shape={tuple(x.shape)}"
        )
        assert o.dim() == 3, (
            f"{prefix}expected o to be 3D, got ndim={o.dim()} shape={tuple(o.shape)}"
        )
        assert x.shape[1] == 1 and x.shape[2] == 1, (
            f"{prefix}expected x.shape[1] == x.shape[2] == 1, got {tuple(x.shape)}"
        )
        assert x._format in self._supported_formats, (
            f"{prefix}no implementation for x._format={x._format}. "
            f"Supported: {sorted(self._supported_formats, key=lambda f: f.value)}"
        )
        assert x.device == o.device, (
            f"{prefix}x and o must be on the same device "
            f"(x.device={x.device}, o.device={o.device})"
        )

        # Claim ``o`` as this op's output (mirrors topK — allocates nothing).
        ctx.output_tensor_to_op_list[o.tensor_id] = len(ctx.op_list)
        ctx.op_list.append(self)
        ctx.op_to_input_tensor_list.append([x.tensor_id])
        ctx.op_to_output_tensor_list.append([o.tensor_id])
