from __future__ import annotations

"""Phase-3 sparse-prefill selection: scores → CSR block-ids + ``(nnz,1,C)`` mask.

Turns the per-(kv-head, query-token, candidate-block) scores emitted by the
prefill indexer (``compiled_indexer_prefill``, whose terminal ``topK`` lowers to
"emit scores") into the head-major CSR + custom causal mask that
:class:`VortexSparsePrefillWrapper` consumes.

Selection semantics mirror the **decode** top-k convention
(``custom_ops/topk_output/_reference.py`` + ``planner_sglang.py``) mapped onto
the per-query-token causal layout:

  * Each row = (kv-head ``g``, query-token ``i``). Its candidate blocks are the
    causal prefix ``0 .. d`` where ``d = i // C`` (``d+1`` candidates); candidate
    index == local block id (fresh contiguous prompt).
  * The **diagonal** block ``d`` (the query's own block) is ALWAYS kept
    (mandatory local — the decode ``eos`` slot), with a causal-triangular mask
    ``p <= i - d*C``.
  * The first ``reserved_bos`` blocks (``0 .. bos-1``, the BOS sink) are kept iff
    ``reserved_bos > 0`` (all-ones mask), matching decode's front reservation.
  * The budget is ``sparse_count = max(topk_val + bos + 1, floor((d+1)*ratio))``
    clamped to ``d+1`` (the ``+1`` is the diagonal/eos). ``middle_k =
    sparse_count - bos - 1`` blocks are top-k'd by score over the strictly-past
    middle ``[bos, d-1]``.

Output per row: ``sorted(bos ∪ middle-top-k)`` strictly-past block ids (ascending)
followed by the diagonal ``d``; masks are all-ones for past blocks and
causal-triangular for the diagonal — byte-identical to
:func:`prefill_sparse.build_selection` for the same selection.

:func:`select_prefill_torch` is the reference/spec. :func:`select_prefill_fast`
is the vectorized GPU implementation used on the hot path (same output, no python
per-row loop) — it operates on ONE query tile (``[q_offset, q_offset+tile_len)``)
of a request, reading the tile's head-minor scores and emitting head-major CSR +
mask for :class:`VortexSparsePrefillWrapper`.
"""

import math
from typing import Optional

import torch


def _row_budget(count: int, *, topk_val: int, topk_ratio: float,
                reserved_bos: int) -> tuple[int, int]:
    """Return ``(bos, middle_k)`` for a row with ``count = d+1`` candidate blocks.

    Mirrors the decode budget (``max(static, dynamic)`` clamped to the available
    block count) with the diagonal counted as the single ``eos`` slot.
    """
    d = count - 1
    bos = reserved_bos if (reserved_bos > 0 and d >= 1) else 0
    bos = min(bos, max(0, d))                      # can't reserve past what exists
    static_budget = topk_val + bos + 1             # +1 = diagonal (eos)
    dynamic_budget = int(math.floor(count * topk_ratio))
    sparse_count = max(static_budget, dynamic_budget)
    sparse_count = min(sparse_count, count)
    middle_k = sparse_count - bos - 1
    middle_k = max(0, min(middle_k, d - bos))      # available middle = [bos, d-1]
    return bos, middle_k


def select_prefill_torch(
    scores: torch.Tensor,       # [total, 1, 1] RAGGED, head-major per (g, i)
    indptr: torch.Tensor,       # [H_kv*S_q + 1] int, per-row candidate ranges
    q_positions: torch.Tensor,  # [H_kv*S_q] int, within-request position i of each row
    block_size: int,            # C
    *,
    topk_val: int,
    topk_ratio: float = 0.0,
    reserved_bos: int = 1,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference selection. Returns ``(kv_indptr, block_ids, block_mask)`` exactly
    as :func:`prefill_sparse.build_selection` (head-major CSR + ``(nnz,1,C)`` mask).
    """
    device = device or scores.device
    C = block_size
    n_rows = indptr.numel() - 1
    indptr_cpu = indptr.detach().cpu().tolist()
    qpos_cpu = q_positions.detach().cpu().tolist()
    scores_flat = scores.reshape(-1).detach().to(torch.float32).cpu()

    p = torch.arange(C)
    ones_row = torch.ones(C, dtype=torch.bool)

    out_indptr = [0]
    block_ids: list[int] = []
    masks: list[torch.Tensor] = []

    for r in range(n_rows):
        base, end = indptr_cpu[r], indptr_cpu[r + 1]
        count = end - base                         # d + 1 candidate blocks
        i = qpos_cpu[r]
        d = i // C
        assert count == d + 1, (
            f"row {r}: candidate count {count} != d+1 ({d + 1}) "
            f"(i={i}, C={C}) — indptr/q_positions disagree"
        )
        bos, middle_k = _row_budget(
            count, topk_val=topk_val, topk_ratio=topk_ratio, reserved_bos=reserved_bos,
        )

        selected: list[int] = list(range(bos))     # BOS sink: blocks 0..bos-1
        if middle_k > 0:
            # strictly-past middle candidates = local block ids [bos, d-1]
            lo, hi = bos, d                         # exclusive hi = d (diagonal)
            mids = scores_flat[base + lo : base + hi]
            k = min(middle_k, hi - lo)
            if k > 0:
                top = torch.topk(mids, k).indices.tolist()
                selected.extend(lo + t for t in top)
        selected = sorted(set(selected))

        for c in selected:                          # strictly-past → all-ones
            block_ids.append(c)
            masks.append(ones_row)
        # diagonal block d → causal-triangular (keep kv positions 0..i-d*C)
        block_ids.append(d)
        masks.append(p <= (i - d * C))
        out_indptr.append(len(block_ids))

    kv_indptr = torch.tensor(out_indptr, dtype=torch.int32, device=device)
    block_ids_t = torch.tensor(block_ids, dtype=torch.int32, device=device)
    block_mask = torch.stack(masks).view(-1, 1, C).to(device)
    return kv_indptr, block_ids_t, block_mask


_NEG = -1e30


def select_prefill_fast(
    scores: torch.Tensor,        # [total, 1, 1] tile scores, HEAD-MINOR rows (j*H_kv+g)
    dense_kv_indptr: torch.Tensor,  # tile CSR, row = tile_local_token*H_kv + kv_head
    tile_len: int,               # number of query tokens in this tile
    H_kv: int,
    block_size: int,             # C
    q_offset: int,               # global-in-request position of the tile's first query
    *,
    topk_val: int,
    topk_ratio: float = 0.0,
    reserved_bos: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized GPU selection for one query tile. Byte-identical output to
    :func:`select_prefill_torch` fed the head-major gather with
    ``q_positions = q_offset + arange(tile_len)`` — no python per-row loop.

    Returns head-major ``(kv_indptr [H_kv*tile_len+1], block_ids [nnz],
    block_mask [nnz,1,C])``.
    """
    device = scores.device
    C = block_size
    n_rows = H_kv * tile_len
    scores_flat = scores.reshape(-1)

    ro = torch.arange(n_rows, device=device)
    g = ro // tile_len                      # head (head-major output order)
    j = ro % tile_len                       # tile-local query index
    q_pos = (q_offset + j).to(torch.int64)  # position within request
    d = q_pos // C                          # diagonal block index
    num_causal = d + 1
    src_row = (j * H_kv + g).to(torch.int64)   # head-minor source row
    base = dense_kv_indptr[src_row].to(torch.int64)

    max_blocks = int(d.max().item()) + 1
    cols = torch.arange(max_blocks, device=device)                 # [max_blocks]
    valid = cols[None, :] < num_causal[:, None]                    # b < d+1
    flat = base[:, None] + cols[None, :]
    flat_c = torch.where(valid, flat, torch.zeros_like(flat))
    S = scores_flat[flat_c].to(torch.float32)
    S = torch.where(valid, S, torch.full_like(S, _NEG))

    # --- per-row budget (mirrors _row_budget, vectorized) ---
    bos = torch.where(d >= 1,
                      torch.full_like(d, reserved_bos if reserved_bos > 0 else 0),
                      torch.zeros_like(d))
    bos = torch.minimum(bos, torch.clamp(d, min=0))
    static = topk_val + bos + 1
    dynamic = torch.floor(num_causal.to(torch.float64) * topk_ratio).to(torch.int64)
    sparse_count = torch.minimum(torch.maximum(static, dynamic), num_causal)
    middle_k = torch.clamp(sparse_count - bos - 1, min=0)
    middle_k = torch.minimum(middle_k, torch.clamp(d - bos, min=0))

    # --- top-middle_k over the strictly-past middle [bos, d-1] ---
    is_middle = (cols[None, :] >= bos[:, None]) & (cols[None, :] < d[:, None]) & valid
    Smid = torch.where(is_middle, S, torch.full_like(S, _NEG))
    sorted_vals, sorted_idx = torch.sort(Smid, dim=1, descending=True)
    rank = torch.arange(max_blocks, device=device)[None, :]
    keep = (rank < middle_k[:, None]) & (sorted_vals > _NEG)

    sel = torch.zeros(n_rows, max_blocks, dtype=torch.bool, device=device)
    sel.scatter_(1, sorted_idx, keep)                    # middle top-k
    sel |= cols[None, :] < bos[:, None]                  # BOS sink
    sel[ro, d] = True                                    # diagonal (always)

    # --- CSR assembly (nonzero → col-ascending per row, diagonal last) ---
    counts = sel.sum(dim=1)
    kv_indptr = torch.cat([
        torch.zeros(1, dtype=torch.int32, device=device),
        counts.cumsum(0).to(torch.int32),
    ])
    nz = sel.nonzero(as_tuple=False)                     # [nnz, 2] (row, block)
    r_nz, block_ids = nz[:, 0], nz[:, 1].to(torch.int32)
    # Unified mask: all-ones for past (q_pos - b*C >= C-1), triangular for diagonal.
    limit = q_pos[r_nz] - nz[:, 1] * C                   # [nnz]
    block_mask = (torch.arange(C, device=device)[None, :] <= limit[:, None]).view(-1, 1, C)
    return kv_indptr, block_ids, block_mask


def build_head_major_selection(
    scores: torch.Tensor,        # [total, 1, 1] RAGGED, HEAD-MINOR global rows
    dense_kv_indptr: torch.Tensor,  # planner CSR, row = global_token*H_kv + kv_head
    qo_indptr: torch.Tensor,     # [n_req + 1] per-request global token ranges
    H_kv: int,
    block_size: int,             # C
    *,
    topk_val: int,
    topk_ratio: float = 0.0,
    reserved_bos: int = 1,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """P5-facing selection. The planner/indexer emit scores in **head-minor**
    global row order (``row = global_token*H_kv + kv_head``); the
    :class:`VortexSparsePrefillWrapper` wants **head-major per-request** CSR
    (head ``g`` owns rows ``[g*S_r, (g+1)*S_r)``, ``q_pos = i``).

    For each request this gathers the head-minor global score slices into
    head-major order and runs :func:`select_prefill_torch`, returning one
    ``(kv_indptr, block_ids, block_mask)`` triple per request — ready to feed
    the wrapper directly.
    """
    device = scores.device
    scores_flat = scores.reshape(-1)
    indptr_cpu = dense_kv_indptr.detach().cpu().tolist()
    qo_cpu = qo_indptr.detach().cpu().tolist()
    n_req = len(qo_cpu) - 1

    out: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for r in range(n_req):
        qo_start, qo_end = qo_cpu[r], qo_cpu[r + 1]
        S_r = qo_end - qo_start
        sub_slices: list[torch.Tensor] = []
        sub_indptr = [0]
        sub_qpos: list[int] = []
        for g in range(H_kv):                       # head-major output order
            for i in range(S_r):
                row = (qo_start + i) * H_kv + g      # head-minor global row
                b0, b1 = indptr_cpu[row], indptr_cpu[row + 1]
                sub_slices.append(scores_flat[b0:b1])
                sub_indptr.append(sub_indptr[-1] + (b1 - b0))
                sub_qpos.append(i)
        sub_scores = (
            torch.cat(sub_slices) if sub_slices
            else scores_flat.new_zeros(0)
        ).view(-1, 1, 1)
        kv_indptr, block_ids, block_mask = select_prefill_torch(
            sub_scores,
            torch.tensor(sub_indptr, dtype=torch.int32, device=device),
            torch.tensor(sub_qpos, dtype=torch.int32, device=device),
            block_size,
            topk_val=topk_val, topk_ratio=topk_ratio, reserved_bos=reserved_bos,
            device=device,
        )
        out.append((kv_indptr, block_ids, block_mask))
    return out
