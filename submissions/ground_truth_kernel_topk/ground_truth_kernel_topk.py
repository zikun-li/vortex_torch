import torch
from typing import Dict

from vortex_torch.flow import vFlow, register
from vortex_torch.indexer import GTGroupScore, GTTopK
from vortex_torch.cache import Fill as CFill
from vortex_torch.abs import ContextBase


@register("ground_truth_kernel_topk_sub")
class GroundTruthKernelTopK(vFlow):
    r"""
    Ground-truth top-k routing backed by the ``gt_score_kernels`` kernels.

    Semantically identical to ``naive_ground_truth_topk`` — it selects the exact
    ground-truth top-k blocks by the true softmax attention score (sum over the
    GQA query-head group, max over the block tokens) — but the entire indexer is
    just **two custom ops**, ``GTGroupScore`` and ``GTTopK``, each of which
    **bypasses vortex JIT compilation** and dispatches to our fused kernels:

      * prefill: ``group_scores_prefill`` (score) + FlashInfer
        ``top_k_ragged_transform`` + ``assemble_block_ids`` (top-k), feeding the
        exact sparse-prefill wrapper;
      * decode: reserved for the GT decode kernels (future work).

    Requires the sparse-prefill regime (fresh prompt): ``vortex_sparse_prefill``,
    ``chunked_prefill_size=-1``, ``disable_radix_cache=true``.
    """

    def __init__(self):
        super().__init__()
        self.gtscore = GTGroupScore()   # exact group-level block score (our kernel)
        self.gttopk = GTTopK()          # FlashInfer top-k + assemble (our kernels)
        # No cache summary is built (scores come straight from raw K); the cache
        # program still needs one produced output, so write a constant dummy.
        self.fill = CFill(0.0)

    def forward_indexer(
        self,
        q: torch.Tensor,
        o: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        ctx: ContextBase,
    ):
        score = self.gtscore(q, cache["k"], ctx=ctx)   # [S, 1, 1] ground-truth block score
        self.gttopk(score, o, ctx=ctx)                 # exact top-k block selection

    def forward_cache(
        self,
        cache: Dict[str, torch.Tensor],
        loc: torch.Tensor,
        ctx: ContextBase,
    ):
        self.fill(cache["dummy"], loc=loc, ctx=ctx)

    def create_cache(self, block_size: int, head_dim: int):
        # No custom fields ("k"/"v" are auto-provided); a (1,1) dummy keeps the
        # cache program well-formed.
        return {"dummy": (1, 1)}
