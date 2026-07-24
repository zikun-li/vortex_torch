import math
import torch
from typing import Dict

from vortex_torch.flow import vFlow, register
from vortex_torch.indexer import (
    topK, GeMM, Max, Sum, Add, Multiply, Exp, Log,
)
from vortex_torch.cache import Fill as CFill
from vortex_torch.abs import ContextBase


@register("naive_ground_truth_topk_sub")
class NaiveGroundTruthTopK(vFlow):
    r"""
    Naive top-k routing (approximate reference — see the caveat below).

    This flow computes attention logits from the raw queries and per-token
    keys (no centroid / envelope approximation), turns them into a per-head
    softmax over the keys, then selects the top-k blocks. It is **not** an
    exact ground-truth reference: the softmax denominator excludes the reserved
    BOS/EOS blocks, which perturbs the middle-block ranking under GQA (see the
    caveat after the pipeline table). For exact ground truth, use the
    ``ground_truth_kernel_topk`` submission / ``gt_score_kernels`` path.

    Because the router selects at *group-level block* granularity, the
    per-(token, query-head) attention weights are aggregated in two stages:

    1. **over the GQA group** — ``Sum`` over the query-head axis (sum-pool),
    2. **over the block** — ``Max`` over the within-block token axis
       (max-pool),

    yielding a single scalar score per block that is fed to :class:`topK`.

    Shapes (indexer view)
    ---------------------
    - Queries ``q``: ``[B, H_q, D]``.
    - ``cache["k"]``: the auto-provided per-token key cache, inner shape
      ``(block_size, head_dim)`` → viewed as ``[S, block_size, D]`` in
      :meth:`forward_indexer`. Here :math:`S` is the leading (packed) page
      axis — "the number of blocks for this sequence".

    No custom cache fields are declared: the ground-truth scores are
    computed directly from the raw keys, so :meth:`forward_cache` is a
    no-op (``cache["k"]`` is populated by the engine regardless).

    Scoring pipeline (:meth:`forward_indexer`)
    ------------------------------------------
    Let ``scale = 1/sqrt(head_dim)``.

    ================  ==========================================  ======================
    step              op                                          shape
    ================  ==========================================  ======================
    1  logits         ``GeMM(q, cache["k"])``                     ``[S, block_size, H_q]``
    2  per-block max  ``Max(dim=1)``                              ``[S, 1, H_q]``
    3  global max M   ``Max(dim=0)``  (S-barrier)                 ``[., 1, H_q]``
    4  shift          ``Add(1, -1)(logits, M)``  = logits - M     ``[S, block_size, H_q]``
    5  exp            ``Exp(beta=scale)``                         ``[S, block_size, H_q]``
    6  partial denom  ``Sum(dim=1)``                              ``[S, 1, H_q]``
    7  global denom Z ``Sum(dim=0)``  (S-barrier)                 ``[., 1, H_q]``
    8  reciprocal     ``Exp(beta=-1)(Log(Z))``  = 1/Z             ``[., 1, H_q]``
    9  softmax w      ``Multiply(e, 1/Z)``                        ``[S, block_size, H_q]``
    10 sum over group ``Sum(dim=2)``  (sum-pool over H_q)         ``[S, block_size, 1]``
    11 max over block ``Max(dim=1)``  (max-pool over tokens)      ``[S, 1, 1]``
    12 select         ``topK(score, o)``                          sparse indices
    ================  ==========================================  ======================

    **Exactness caveat.** The two ``dim=0`` reductions (steps 3, 7) are
    cross-block reductions that exclude the reserved BOS/EOS blocks — the
    framework's ``reduce_dim0`` kernel trims the first ``block_reserved_bos``
    and last ``block_reserved_eos`` pages. So the softmax denominator ``Z``
    (step 7) is normalized over the **middle** blocks only, not over all keys.
    This does not simply cancel: a per-head constant rescale would leave a
    single head's block ranking unchanged, but the per-head weights are summed
    across the GQA group (step 10) each carrying a *different* ``Z``, so the
    middle-block ranking can diverge from a true full-key softmax. Hence this
    flow is an **approximate** reference. The exact ground truth normalizes the
    softmax over all keys (BOS/EOS included) and lives in the
    ``ground_truth_kernel_topk`` submission / ``gt_score_kernels`` path; the
    standard indexer ops cannot express that (``dim=0`` reductions always trim
    the reserved pages).

    The ``1/sqrt(head_dim)`` scale is folded into the numerator ``Exp``
    (step 5): subtracting the global max ``M`` first makes the exponential
    numerically stable for any logit magnitude.
    """

    def __init__(self):
        super().__init__()

        # --- Indexer-side ops (one instance per call site) ---
        self.gemm = GeMM()                    # 1  q . k  -> per-token, per-head logits
        self.max_tok = Max(dim=1)             # 2  max over tokens within a block
        self.max_glob = Max(dim=0)            # 3  global max per head (stability)
        self.sub = Add(alpha=1.0, beta=-1.0)  # 4  logits - M
        # step 5 (numerator Exp with 1/sqrt(head_dim) scale) built in create_cache
        self.sum_tok = Sum(dim=1)             # 6  partial denom over tokens
        self.sum_glob = Sum(dim=0)            # 7  global denom per head (Z)
        self.log_z = Log()                    # 8a log(Z)
        self.exp_recip = Exp(beta=-1.0)       # 8b exp(-log Z) = 1/Z
        self.mul_norm = Multiply()            # 9  e * (1/Z) -> softmax weights
        self.sum_group = Sum(dim=2)           # 10 sum-pool over the GQA group (H_q)
        self.max_block = Max(dim=1)           # 11 max-pool over the block tokens
        self.output_func = topK()             # 12 exact ground-truth top-k

        # --- Cache-side op ---
        # We read cache["k"] directly and build no summary, but the cache
        # program needs one produced-but-unconsumed tensor, so write a
        # constant into a (1, 1) dummy field (cheapest valid cache pass).
        self.fill = CFill(0.0)

    def forward_indexer(
        self,
        q: torch.Tensor,
        o: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        ctx: ContextBase,
    ):
        r"""Compute (approximate) block scores and select the top-k blocks.

        Approximate because the softmax denominator excludes reserved BOS/EOS
        blocks — see the class docstring's exactness caveat."""
        # 1. per-token, per-head logits: [S, block_size, H_q]
        logits = self.gemm(q, cache["k"], ctx=ctx)

        # 2-3. global max per head for numerical stability: [., 1, H_q]
        m_tok = self.max_tok(logits, ctx=ctx)
        m_glob = self.max_glob(m_tok, ctx=ctx)

        # 4-5. stable exp(scale * (logits - M)): [S, block_size, H_q]
        shifted = self.sub(logits, m_glob, ctx=ctx)
        e = self.exp_num(shifted, ctx=ctx)

        # 6-8. softmax denominator per head and its reciprocal: [., 1, H_q].
        #      NOTE: sum_glob (dim=0) excludes reserved BOS/EOS blocks, so Z is
        #      normalized over the middle blocks only (approximate; see caveat).
        z_tok = self.sum_tok(e, ctx=ctx)
        z_glob = self.sum_glob(z_tok, ctx=ctx)
        z_log = self.log_z(z_glob, ctx=ctx)
        z_inv = self.exp_recip(z_log, ctx=ctx)

        # 9. per-head softmax weights (middle-normalized): [S, block_size, H_q]
        w = self.mul_norm(e, z_inv, ctx=ctx)

        # 10. sum-pool over the GQA group (H_q): [S, block_size, 1]
        grp = self.sum_group(w, ctx=ctx)

        # 11. max-pool over the block tokens: [S, 1, 1]
        score = self.max_block(grp, ctx=ctx)

        # 12. select the ground-truth top-k blocks
        self.output_func(score, o, ctx=ctx)

    def forward_cache(
        self,
        cache: Dict[str, torch.Tensor],
        loc: torch.Tensor,
        ctx: ContextBase,
    ):
        r"""
        No summary is built (ground-truth scores are read straight from
        ``cache["k"]`` at index time). Write a constant into the ``dummy``
        field only so the cache graph has one produced output.
        """
        self.fill(cache["dummy"], loc=loc, ctx=ctx)

    def create_cache(self, block_size: int, head_dim: int):
        r"""
        Declare no custom cache fields (``cache["k"]`` is auto-provided) and
        build the scale-dependent numerator ``Exp`` now that ``head_dim`` is
        known. ``scale = 1/sqrt(head_dim)`` is folded into ``Exp``'s ``beta``.
        """
        self.exp_num = Exp(beta=1.0 / math.sqrt(head_dim))  # 5  exp(scale * x)
        return {
            "dummy": (1, 1),  # unused; keeps the cache program well-formed
        }
