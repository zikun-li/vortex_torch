from __future__ import annotations

"""GQA group-wise, token-level sparse **prefill** attention (exact causal).

Standalone, engine-independent core of vortex's sparse-prefill path. Wraps
FlashInfer 0.6.3's `BlockSparseAttentionWrapper` (the *basic* wrapper) and runs
one *fresh-prompt* sequence at a time.

Design (see memory `sparse-prefill-gqa-design`):
  * **Kernel = looped basic `BlockSparseAttentionWrapper`.** Its `plan()` takes
    **CSR directly** (`indptr`, `indices`=block-ids, `C`=page_size) — public API,
    no dense mask, no internal-plan vendoring, no block→token expansion. Group-wise
    ⇒ loop over the `num_kv_heads` KV groups (per call: num_qo_heads=group_size,
    num_kv_heads=1).
  * **R=1, per-query-token, group-wise selection.** Each query token (row-block of
    size 1) and each KV group picks its own set of KV blocks; the `group_size` query
    heads in a group share.
  * **Exact causal via a custom (nnz, 1, C) mask, `causal=False`.** With R=1 the
    wrapper's built-in causal only masks at block granularity (diagonal block would
    leak C-1 future tokens for C>1). Instead the selection carries, per non-zero
    block: **all-ones for strictly-past blocks, causal-triangular for the query's own
    diagonal block** (keep kv position p iff `block*C+p <= query_pos`). `plan()`
    packs it. The diagonal block is always included (mandatory local, Q8).
  * **Ragged tail.** The basic wrapper needs `N % C == 0`, so K/V are padded to a
    multiple of C. The last (partial) block is only ever a query's diagonal block
    (never strictly-past in fresh prompt), so its triangular mask zeros the padding.

The wrapper uses **NHD** tensors (`[S, H, D]`) directly (no transpose needed).

The per-head CSR + custom mask are *inputs* to :meth:`run` — produced at integration
time by the prefill selection kernel (from indexer scores). :func:`build_selection`
is a torch reference builder (from a dense boolean selection) used by tests and as
the spec the Triton selection kernel must match.

Verified exact (<=2e-3) vs a FlexAttention token-causal oracle for C in {1..128}
(`tests/prefill_sparse/test_parity.py`).
"""

from typing import Optional

import torch

import flashinfer
from flashinfer.cascade import merge_state


_WORKSPACE_BYTES = 256 * 1024 * 1024


def build_selection(
    sel_past: torch.Tensor,   # [H_kv, S_q, NB] bool — STRICTLY-PAST selected blocks
    C: int,
    *,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference builder: dense strictly-past selection → head-major CSR + custom mask.

    Returns ``(kv_indptr, block_ids, block_mask)``:
      * ``kv_indptr``  : int32 ``[H_kv*S_q + 1]``, head-major (head g owns rows
        ``[g*S_q, (g+1)*S_q)``); CSR row pointers over query tokens.
      * ``block_ids``  : int32 ``[nnz]`` selected block column ids (strictly-past
        blocks **plus** each query's diagonal block ``i//C``), per row.
      * ``block_mask`` : bool ``[nnz, 1, C]`` — all-ones for past blocks,
        causal-triangular for the diagonal block.

    This is the contract the Phase-3 Triton selection kernel must reproduce. It uses
    plain torch (engine-side; the "no native torch" rule is for the indexer, not here).
    """
    device = device or sel_past.device
    H_kv, S_q, NB = sel_past.shape
    diag = torch.arange(S_q, device=device) // C          # [S_q] each query's own block
    cols = torch.arange(NB, device=device)

    indptr = [0]
    block_ids: list[int] = []
    masks: list[torch.Tensor] = []
    ones_row = torch.ones(C, dtype=torch.bool)
    p = torch.arange(C)
    sel_cpu = sel_past.cpu()
    diag_cpu = diag.cpu()
    for g in range(H_kv):
        for i in range(S_q):
            d = int(diag_cpu[i])
            past = torch.nonzero(sel_cpu[g, i]).flatten().tolist()
            past = [c for c in past if c < d]             # enforce strictly-past
            for c in past:
                block_ids.append(c)
                masks.append(ones_row)
            # diagonal block always included, causal-triangular
            block_ids.append(d)
            masks.append(p <= (i - d * C))
            indptr.append(len(block_ids))

    kv_indptr = torch.tensor(indptr, dtype=torch.int32, device=device)
    block_ids_t = torch.tensor(block_ids, dtype=torch.int32, device=device)
    block_mask = torch.stack(masks).view(-1, 1, C).to(device)
    return kv_indptr, block_ids_t, block_mask


class VortexSparsePrefillWrapper:
    """Exact GQA sparse-prefill attention for one fresh-prompt sequence.

    Reusable across layers/requests; owns one basic `BlockSparseAttentionWrapper`
    (re-planned per KV group) and a shared workspace.
    """

    def __init__(
        self,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        device: torch.device,
        workspace_buffer: Optional[torch.Tensor] = None,
        backend: str = "auto",
        q_data_type: torch.dtype = torch.bfloat16,
        kv_data_type: torch.dtype = torch.bfloat16,
    ):
        assert num_qo_heads % num_kv_heads == 0
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.group_size = num_qo_heads // num_kv_heads
        self.head_dim = head_dim
        self.q_data_type = q_data_type
        self.kv_data_type = kv_data_type

        if workspace_buffer is None:
            workspace_buffer = torch.empty(
                _WORKSPACE_BYTES, dtype=torch.uint8, device=device
            )
        # Diagonal-split with NO custom mask (the custom-mask plan() has a ~fixed
        # ~90ms cost regardless of mask size — it was 99.8% of prefill). Two
        # SEPARATE workspaces so the two passes' async plan/run never contend
        # before merge_state:
        #   * ``_wrapper``  — strictly-past blocks: maskless block-sparse CSR.
        #   * ``_ragged``   — diagonal block: block-local CAUSAL ragged prefill
        #     (flashinfer's built-in causal, no mask) — each query attends
        #     causally within its own C-block. All heads in one call.
        self._wrapper = flashinfer.BlockSparseAttentionWrapper(
            workspace_buffer, backend=backend
        )
        self._ragged = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device=device), "NHD",
        )

    def run(
        self,
        q: torch.Tensor,          # [S_q, H_q, D]  (NHD)
        k: torch.Tensor,          # [S_kv, H_kv, D] (NHD)
        v: torch.Tensor,          # [S_kv, H_kv, D] (NHD)
        kv_indptr: torch.Tensor,  # [H_kv*S_q + 1] int32, head-major CSR
        block_ids: torch.Tensor,  # [nnz] int32
        block_mask: torch.Tensor, # [nnz, 1, C] bool
        block_size: int,          # C
        *,
        sm_scale: Optional[float] = None,
        logits_soft_cap: Optional[float] = None,
    ) -> torch.Tensor:            # [S_q, H_q, D] (NHD)
        s_q, h_q, d = q.shape
        s_kv, h_kv, _ = k.shape
        C = block_size
        G = self.group_size
        assert h_q == self.num_qo_heads and h_kv == self.num_kv_heads and d == self.head_dim
        # Query-axis chunking: a query tile (s_q queries) attends to its causal KV
        # prefix (s_kv tokens, s_kv >= s_q). s_q == s_kv is the whole-request case.
        assert s_q <= s_kv, "query tile length must not exceed the KV prefix length"
        assert kv_indptr.numel() == self.num_kv_heads * s_q + 1

        device = q.device
        # The diagonal pass tiles this tile's own tokens on the LOCAL block grid,
        # which only matches the global grid when the tile starts on a block
        # boundary. The tile start is a = s_kv - s_q; require it block-aligned
        # (the caller sizes query tiles to a block_size multiple).
        assert (s_kv - s_q) % C == 0, (
            f"query tile must start on a block boundary: (s_kv-s_q)={s_kv - s_q} "
            f"not a multiple of C={C}"
        )
        # Tile's own tokens = its causal-prefix suffix k[s_kv-s_q : s_kv] (tiles are
        # block-aligned, so the diagonal blocks live entirely in this suffix). Used
        # by the block-local-causal diagonal pass — capture BEFORE padding.
        k_diag = k[s_kv - s_q : s_kv].contiguous()            # [s_q, H_kv, D]
        v_diag = v[s_kv - s_q : s_kv].contiguous()

        # Pad K/V to a multiple of C for the block-sparse past pass.
        NB = (s_kv + C - 1) // C
        n_pad = NB * C
        if n_pad != s_kv:
            k = torch.cat([k, k.new_zeros(n_pad - s_kv, h_kv, d)], dim=0)
            v = torch.cat([v, v.new_zeros(n_pad - s_kv, h_kv, d)], dim=0)

        # === (b) DIAGONAL pass — block-local causal, ALL heads, no mask ==========
        # Each query attends causally within its own C-block. Partition the tile's
        # own tokens into C-blocks (last may be partial) and run flashinfer's
        # built-in causal ragged prefill (one call, all heads). Its LSE is on the
        # same scale as the block-sparse past pass, so merge_state combines them.
        n_blk = (s_q + C - 1) // C
        blk_ptr = (torch.arange(n_blk + 1, dtype=torch.int32, device=device) * C)
        blk_ptr[-1] = s_q                                     # last block may be partial
        self._ragged.plan(
            blk_ptr, blk_ptr, h_q, h_kv, self.head_dim, causal=True,
            sm_scale=sm_scale, logits_soft_cap=logits_soft_cap,
            q_data_type=self.q_data_type, kv_data_type=self.kv_data_type,
        )
        o_diag, lse_diag = self._ragged.run(q, k_diag, v_diag, return_lse=True)  # [s_q,H_q,D],[s_q,H_q]

        # === (a) STRICTLY-PAST pass — maskless block-sparse CSR, per kv-head ======
        # zeros + lse=-inf so rows/heads with no strictly-past block merge to the
        # diagonal cleanly (exp(-inf)=0 weight; zeros avoid 0*nan). o_past matches
        # o_diag dtype for merge_state.
        o_past = torch.zeros((s_q, h_q, d), dtype=o_diag.dtype, device=device)
        lse_past = torch.full((s_q, h_q), float("-inf"), dtype=torch.float32, device=device)
        any_past = False
        for g in range(self.num_kv_heads):
            r0, r1 = g * s_q, (g + 1) * s_q
            indptr_g = kv_indptr[r0 : r1 + 1]
            start, end = int(indptr_g[0]), int(indptr_g[-1])
            indptr_g = (indptr_g - indptr_g[0])
            ids_g = block_ids[start:end]
            # Drop the diagonal (last block of each CSR row — build_selection emits
            # it last) → strictly-past blocks only, attended fully (no mask).
            diag_at = (indptr_g[1:] - 1).long()
            keep = torch.ones(ids_g.numel(), dtype=torch.bool, device=device)
            keep[diag_at] = False
            past_ids = ids_g[keep].contiguous()
            past_ptr = torch.cat([
                indptr_g.new_zeros(1),
                (indptr_g[1:] - indptr_g[:-1] - 1).cumsum(0).to(indptr_g.dtype),
            ]).contiguous()

            sl = slice(g * G, (g + 1) * G)
            if past_ids.numel() == 0:
                lse_past[:, sl] = float("-inf")              # merge → diagonal only
                continue
            any_past = True
            q_g = q[:, sl, :].contiguous()
            k_g = k[:, g : g + 1, :].contiguous()
            v_g = v[:, g : g + 1, :].contiguous()
            self._wrapper.plan(
                past_ptr, past_ids, s_q, n_pad, 1, C, G, 1, self.head_dim,
                mask=None, causal=False,
                sm_scale=sm_scale, logits_soft_cap=logits_soft_cap,
                q_data_type=self.q_data_type, kv_data_type=self.kv_data_type,
            )
            o_pg, lse_pg = self._wrapper.run(q_g, k_g, v_g, return_lse=True)
            o_past[:, sl, :] = o_pg
            lse_past[:, sl] = lse_pg

        if not any_past:
            return o_diag                                    # nothing strictly-past
        o, _ = merge_state(o_past, lse_past, o_diag, lse_diag)
        return o.to(q.dtype)
