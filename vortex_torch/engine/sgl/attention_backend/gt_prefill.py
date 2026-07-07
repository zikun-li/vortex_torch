from __future__ import annotations

"""Runtime glue for the ground-truth indexer ops (``GTGroupScore`` / ``GTTopK``).

These are the plain-Python callables that the ``Schedule.S`` custom_impl codegen
(``indexer/compiler/custom_impl/gt_score.py`` / ``gt_topk.py``) emits calls to —
they **bypass** the JIT-compiled Triton path and dispatch to the fused kernels in
the sibling ``gt_score_kernels`` package.

Data flow (prefill): the compiled indexer's ``forward()`` runs the two ops in
order; ``gt_group_score_prefill`` computes the exact group-level block scores and
stashes them on ``ctx.gt_state``; ``gt_topk_prefill`` reads them, runs FlashInfer
top-k + ``assemble_block_ids``, and stashes the head-major CSR selection back on
``ctx.gt_state`` for ``_forward_extend_sparse`` to feed the sparse-prefill wrapper.

The raw per-tile K and geometry come via ``ctx.gt_state`` (set by
``_forward_extend_sparse``) because our score kernel needs a contiguous
``[H_kv, S, C, D]`` key prefix, which neither the paged ``cache`` view nor the
planner metadata provides directly.

Decode ground-truth kernels do not exist yet; the decode lowerings raise a clear
error until they land (this session is prefill-only; see the plan).
"""

import math

import torch


def _keys_to_hscd(raw_k: torch.Tensor, C: int) -> torch.Tensor:
    """``[S_kv, H_kv, D]`` contiguous prefix → ``[H_kv, S, C, D]`` (S = ceil(S_kv/C)).

    Pads the token axis up to a block multiple; the pad tokens only ever land in
    a query's own (diagonal) block, which the wrapper masks causally, so their
    contribution to the block score is harmless (they are strictly-future)."""
    s_kv, h_kv, d = raw_k.shape
    S = (s_kv + C - 1) // C
    if S * C != s_kv:
        pad = raw_k.new_zeros(S * C - s_kv, h_kv, d)
        raw_k = torch.cat([raw_k, pad], dim=0)
    return raw_k.view(S, C, h_kv, d).permute(2, 0, 1, 3).contiguous()  # [H_kv,S,C,D]


def gt_group_score_prefill(q, k, score_out, ctx) -> None:
    """GTGroupScore prefill launcher: exact group-level block scores via
    ``gt_score_kernels.group_scores_prefill``. Stashes the dense
    ``[tlen, H_kv, S]`` fp32 scores on ``ctx.gt_state['scores']`` for GTTopK.

    ``q`` is the forward() query arg ``[tlen*H_kv, group, D]`` (token-major,
    head-minor rows); ``k`` (paged cache["k"]) is ignored — the contiguous
    per-tile K prefix is read from ``ctx.gt_state``. ``score_out`` (the framework
    intermediate buffer) is unused — the score is passed to GTTopK via
    ``ctx.gt_state`` to avoid a layout round-trip through the RAGGED
    ``[max_num_blocks,1,1]`` buffer."""
    from gt_score_kernels.group_score_topk.prefill import group_scores_prefill

    st = ctx.gt_state
    C = ctx.block_size
    H_kv, G, D = ctx.num_kv_heads, ctx.group_size, ctx.head_dim
    tlen = st["tlen"]

    q4 = q.reshape(tlen, H_kv, G, D)                 # token-major, head-minor
    k4 = _keys_to_hscd(st["raw_k"], C)               # [H_kv, S, C, D]
    scores = group_scores_prefill(
        q4, k4, scale=st["scale"], block_size=C, q_pos0=st["q_offset"],
    )                                                # [tlen, H_kv, S] fp32
    st["scores"] = scores


def gt_topk_prefill(score_in, o, ctx) -> None:
    """GTTopK prefill launcher: FlashInfer ragged top-k + ``assemble_block_ids``
    over the strictly-past middle, producing the head-major CSR selection
    (``block_ids`` diagonal-last, ``kv_indptr``) the sparse-prefill wrapper
    consumes. Stashes them on ``ctx.gt_state``.

    Mirrors ``gt_score_kernels.chunked``'s per-chunk selection, reordered to the
    wrapper's HEAD-MAJOR row layout (head ``g`` owns rows ``[g*tlen,(g+1)*tlen)``)
    and honoring the reserved BOS/EOS convention (fixed-k budget; ``topk_ratio``
    support is a follow-up)."""
    import flashinfer.topk as fitk
    from gt_score_kernels.group_score_topk.assemble import assemble_block_ids

    st = ctx.gt_state
    scores = st["scores"]                            # [tlen, H_kv, S] fp32
    tlen, H_kv, S = scores.shape
    dev = scores.device
    C = ctx.block_size
    bos, eos = ctx.block_reserved_bos, ctx.block_reserved_eos
    k_blocks = max(1, int(ctx.topk_val))
    a = st["q_offset"]

    # Head-major rows: row r -> head g=r//tlen, tile-local token j=r%tlen.
    sf = scores.permute(1, 0, 2).reshape(H_kv * tlen, S).to(torch.float32)  # [R,S]
    R = sf.shape[0]
    j = torch.arange(R, device=dev) % tlen
    q_pos = a + j
    d = q_pos // C
    n_blocks = (d + 1).to(torch.int32)               # causal blocks per row
    n_mid = (n_blocks - eos - bos).clamp(min=0)
    k_take = torch.minimum(torch.full_like(n_mid, k_blocks), n_mid)

    # FlashInfer ragged top-k over the strictly-past middle [bos, d-eos].
    inp = sf[:, bos:]
    if inp.shape[1] < k_blocks:
        pad = torch.full((R, k_blocks - inp.shape[1]), float("-inf"), device=dev)
        inp = torch.cat([inp, pad], dim=1)
    lengths = n_mid.clamp(min=1).to(torch.int32)
    offsets = torch.full((R,), bos, dtype=torch.int32, device=dev)
    out_mid = fitk.top_k_ragged_transform(inp.contiguous(), offsets, lengths, k_blocks)

    block_ids, kv_indptr, _ = assemble_block_ids(
        out_mid, n_blocks, k_take, bos=bos, eos=eos, k_blocks=k_blocks,
    )
    st["kv_indptr"] = kv_indptr
    st["block_ids"] = block_ids


# --------------------------------------------------------------------------- #
# Decode: correctness-first torch REFERENCE (a fast GT decode kernel is future
# work). Computes the exact group-level block score for every decode row over the
# paged cache, in the head-minor CSR layout the built-in top-k selector reads
# (GTTopK decode reuses that selector). Vectorized (padded-ragged) to avoid a
# per-row python loop; still a reference — expect it to be the decode bottleneck.
# --------------------------------------------------------------------------- #
def gt_group_score_decode(q, k, score_out, ctx) -> None:
    """GTGroupScore decode reference. ``q`` = ``[B*H_kv, group, D]`` (row =
    request*H_kv + kv_head, head-minor); ``k`` = paged ``cache["k"]``. Writes one
    exact block score per candidate into ``score_out`` at the row's
    ``dense_kv_indptr`` slot (sum over the GQA group of the true softmax weights,
    max over the block tokens), matching the ``naive_ground_truth_topk`` math."""
    md = ctx.metadata
    C, D, G = ctx.block_size, ctx.head_dim, ctx.group_size
    R = int(md.batch_size) * ctx.num_kv_heads
    if R == 0:
        return
    dev = q.device
    scale = 1.0 / math.sqrt(D)

    indptr = md.dense_kv_indptr[: R + 1].to(torch.int64)          # [R+1]
    indices = md.dense_kv_indices                                 # [total] paged block ids
    last_len = md.kv_last_page_len[:R].to(torch.int64)            # [R]
    kv = k.reshape(-1, C, D)                                      # [num_blocks, C, D]
    q3 = q.reshape(R, G, D).to(torch.float32)                    # [R, G, D]

    counts = (indptr[1:] - indptr[:-1])                           # [R] candidate blocks/row
    max_nc = int(counts.max().item())
    col = torch.arange(max_nc, device=dev)                        # [max_nc]
    blk_valid = col[None, :] < counts[:, None]                    # [R, max_nc]
    flat = indptr[:-1][:, None] + col[None, :]                    # [R, max_nc] index into `indices`
    flat_c = torch.where(blk_valid, flat, torch.zeros_like(flat))
    cand = indices[flat_c].to(torch.int64)                        # [R, max_nc] paged block ids

    Kc = kv[cand].to(torch.float32)                              # [R, max_nc, C, D]
    logits = torch.einsum("rgd,rncd->rgnc", q3, Kc) * scale      # [R, G, max_nc, C]

    # token-validity: block valid AND (not-last-block OR token < last_page_len).
    is_last = (col[None, :] == (counts[:, None] - 1))            # [R, max_nc]
    tok = torch.arange(C, device=dev)                            # [C]
    tok_valid = blk_valid[:, :, None] & (
        (~is_last[:, :, None]) | (tok[None, None, :] < last_len[:, None, None])
    )                                                            # [R, max_nc, C]

    neg = torch.finfo(torch.float32).min
    logits = logits.masked_fill(~tok_valid[:, None], neg)
    p = torch.softmax(logits.reshape(R, G, max_nc * C), dim=-1).reshape(R, G, max_nc, C)
    w = p.sum(dim=1)                                             # [R, max_nc, C] sum over group
    w = w.masked_fill(~tok_valid, 0.0)
    blk_score = w.max(dim=2).values                             # [R, max_nc] max over block tokens

    sel = blk_valid.reshape(-1)
    score_out.view(-1)[flat.reshape(-1)[sel]] = blk_score.reshape(-1)[sel].to(score_out.dtype)
