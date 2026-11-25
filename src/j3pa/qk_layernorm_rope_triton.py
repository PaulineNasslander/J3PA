"""QK-LayerNorm and RoPE of PointSelfAttn fused into one Triton kernel, with a fused backward kernel.

Selected by PointSelfAttn's `qk_layernorm_impl="triton_fused_rope"` when QK-LayerNorm and RoPE are enabled.

What it computes
    In PointSelfAttn, q and k each go through a LayerNorm over the head dim (QK-LayerNorm) and are then rotated by RoPE:

        u = LayerNorm(x)                                    per row: one head of one token, head_dim values
        y[2i]   = cos(a_i) * u[2i] - sin(a_i) * u[2i+1]     every channel pair (2i, 2i+1) is rotated by its angle a_i
        y[2i+1] = sin(a_i) * u[2i] + cos(a_i) * u[2i+1]
        a_i     = frequency_i * position[axis_i]            the frequency and position axis (x, y or z) of the pair, as
                                                            in RoPE, and the position of the row's token

Why it is faster
    The separate implementation writes u, computes cos and sin of every channel of every token as two full tensors,
    rounds them to bfloat16 and rotates q and k with several elementwise kernels, and does all of that again in the
    backward pass. Here q and k are read once, the angles are computed in registers, once per channel pair and shared by
    q and k, and y is written once. The backward pass recomputes the angles instead of keeping cos and sin.

Numerics
    Everything is computed in float32, with the same float32 angles and the same cos and sin values as RoPE, and rounded
    once to the input dtype. The separate implementation also rounds u, cos, sin and every product and sum of
    RoPE.apply to bfloat16, so the results are not bit-identical to it, but closer to the exact result.

Tiles
    q and k have shape (groups, heads, group_size, head_dim), see PointSelfAttn, and are strided views of the QKV
    projection whose last dim is contiguous. A tile is BLOCK_TOKENS consecutive tokens of one head of one attention
    group: every row of a tile has the same head, so the RoPE frequencies are one vector per tile, not one per row.
    Rows are numbered as in a contiguous tensor of that shape, row = (group * heads + head) * group_size + token, which
    is also the layout of the outputs and of the saved statistics.
"""

import functools

import torch
import triton
import triton.language as tl
from torch import nn

from j3pa.rope import RoPE

# Elements of a tile per tensor (TILE // head_dim tokens) and warps per program. On an A100 these were the fastest for
# head_dim 64; other values within a factor of two are within a few percent.
_FORWARD_TILE = 2048
_BACKWARD_TILE = 2048
_FORWARD_NUM_WARPS = 4
_BACKWARD_NUM_WARPS = 8
# Backward programs per streaming multiprocessor, see j3pa/qk_layernorm_triton.py.
_BACKWARD_PROGRAMS_PER_SM = 2


@triton.jit
def _token_offsets(group, head, tokens, stride_group, stride_head, stride_token):
    """Offset (in elements) of the first value of each token's row of one head, in a tensor with these strides. In int64:
    in large batches the offsets into the QKV projection exceed 2**31."""
    return group.to(tl.int64) * stride_group + head.to(tl.int64) * stride_head + tokens.to(tl.int64) * stride_token


@triton.jit
def _rope_cos_sin(
    group, head, tokens, token_mask, group_size,
    pos_ptr,  # (groups * group_size, 3) position of every token.
    angles_ptr, angle_idx_ptr, pos_idx_ptr,  # The RoPE module's buffers.
    D: tl.constexpr,
    BLOCK_PAIRS: tl.constexpr,  # BLOCK_D // 2
):
    """cos and sin of the RoPE angle of every (token, channel pair) of a tile: two (BLOCK_TOKENS, BLOCK_PAIRS) tensors."""
    # RoPE gives every channel of the full feature (heads * head_dim channels) an angle index into its frequencies and a
    # position axis (0, 1, 2 for x, y, z). Both channels of a pair share them, so look them up for the first channel of
    # every pair of the tile's head.
    pairs = tl.arange(0, BLOCK_PAIRS)
    pair_mask = pairs < D // 2
    channel = head * D + 2 * pairs
    frequency = tl.load(angles_ptr + tl.load(angle_idx_ptr + channel, mask=pair_mask, other=0), mask=pair_mask, other=0.0)
    axis = tl.load(pos_idx_ptr + channel, mask=pair_mask, other=0)

    # The (x, y, z) position of every token, and per pair the coordinate of its axis. Positions are given in the padded
    # token order of the attention groups: token t of group g is position g * group_size + t.
    point = (group.to(tl.int64) * group_size + tokens) * 3
    x = tl.load(pos_ptr + point, mask=token_mask, other=0.0)
    y = tl.load(pos_ptr + point + 1, mask=token_mask, other=0.0)
    z = tl.load(pos_ptr + point + 2, mask=token_mask, other=0.0)
    position = tl.where(axis[None, :] == 0, x[:, None], tl.where(axis[None, :] == 1, y[:, None], z[:, None]))

    # The same float32 product as RoPE._compute_cos_sin. tl.cos and tl.sin give the same values as torch.cos and
    # torch.sin (the accurate CUDA functions, not the fast approximations).
    angle = frequency[None, :] * position
    return tl.cos(angle), tl.sin(angle)


@triton.jit
def _forward_one(
    x_ptr, y_ptr, weight_ptr, bias_ptr, mean_ptr, rstd_ptr,
    x_offsets, rows, token_mask, cos, sin, eps,
    D: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_TOKENS: tl.constexpr,
):
    """LayerNorm and RoPE of one tile of q or of k."""
    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < D  # A head_dim that is not a power of two leaves columns of the block unused.
    mask = token_mask[:, None] & col_mask[None, :]

    # The LayerNorm as in j3pa/qk_layernorm_triton.py, but its output u stays in float32 registers.
    x = tl.load(x_ptr + x_offsets[:, None] + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1) / D
    x_centered = tl.where(mask, x - mean[:, None], 0.0)
    rstd = tl.rsqrt(tl.sum(x_centered * x_centered, axis=1) / D + eps)
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    u = x_centered * rstd[:, None] * weight[None, :] + bias[None, :]

    # RoPE: split every row into the first and the second channel of its pairs, (BLOCK_TOKENS, BLOCK_D) ->
    # 2 x (BLOCK_TOKENS, BLOCK_D // 2), rotate every pair by its angle and put the channels back in their places.
    u_first, u_second = tl.split(tl.reshape(u, (BLOCK_TOKENS, BLOCK_D // 2, 2)))
    y = tl.interleave(cos * u_first - sin * u_second, sin * u_first + cos * u_second)

    # Round once to the output dtype. The output is contiguous: row r starts at r * D.
    tl.store(y_ptr + rows.to(tl.int64)[:, None] * D + cols[None, :], y.to(y_ptr.dtype.element_ty), mask=mask)
    # The backward pass recomputes the normalised input from x and these two statistics.
    tl.store(mean_ptr + rows, mean, mask=token_mask)
    tl.store(rstd_ptr + rows, rstd, mask=token_mask)


@triton.jit
def _forward_kernel(
    q_ptr, k_ptr, q_out_ptr, k_out_ptr,
    q_weight_ptr, q_bias_ptr, k_weight_ptr, k_bias_ptr,
    q_mean_ptr, q_rstd_ptr, k_mean_ptr, k_rstd_ptr,  # Written: LayerNorm statistics of every row.
    pos_ptr, angles_ptr, angle_idx_ptr, pos_idx_ptr,
    stride_group, stride_head, stride_token,  # Strides of q and of k, which are the same (views of the QKV projection).
    num_heads, group_size, q_eps, k_eps,
    D: tl.constexpr,  # head_dim, the size normalised over.
    BLOCK_D: tl.constexpr,  # D rounded up to a power of two, as Triton's block sizes must be.
    BLOCK_TOKENS: tl.constexpr,  # Tokens per tile.
):
    # Program (group * num_heads + head, block) handles the tokens [block * BLOCK_TOKENS, (block + 1) * BLOCK_TOKENS)
    # of that head of that attention group, in q and in k.
    group_head = tl.program_id(0)
    group = group_head // num_heads
    head = group_head % num_heads
    tokens = tl.program_id(1) * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    token_mask = tokens < group_size  # The last tile of a group may run past its last token.
    rows = group_head * group_size + tokens

    # The angles depend only on the head and the token, so q and k share them.
    cos, sin = _rope_cos_sin(group, head, tokens, token_mask, group_size, pos_ptr, angles_ptr, angle_idx_ptr, pos_idx_ptr,
                             D, BLOCK_D // 2)

    x_offsets = _token_offsets(group, head, tokens, stride_group, stride_head, stride_token)
    _forward_one(q_ptr, q_out_ptr, q_weight_ptr, q_bias_ptr, q_mean_ptr, q_rstd_ptr, x_offsets, rows, token_mask, cos, sin,
                 q_eps, D, BLOCK_D, BLOCK_TOKENS)
    _forward_one(k_ptr, k_out_ptr, k_weight_ptr, k_bias_ptr, k_mean_ptr, k_rstd_ptr, x_offsets, rows, token_mask, cos, sin,
                 k_eps, D, BLOCK_D, BLOCK_TOKENS)


@triton.jit
def _backward_one(
    x_ptr, dy_ptr, dx_ptr, weight_ptr, mean_ptr, rstd_ptr,
    x_offsets, dy_offsets, rows, token_mask, cos, sin,
    D: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_TOKENS: tl.constexpr,
):
    """Gradient of the input of one tile of q or of k. Returns the tile's contributions to the gradients of the
    LayerNorm weight and bias (not yet summed over the rows)."""
    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < D
    mask = token_mask[:, None] & col_mask[None, :]
    x = tl.load(x_ptr + x_offsets[:, None] + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + dy_offsets[:, None] + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + rows, mask=token_mask, other=0.0)
    rstd = tl.load(rstd_ptr + rows, mask=token_mask, other=0.0)
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

    # RoPE backward. A rotation is undone by the opposite rotation, so the gradient of u is the gradient of y rotated
    # back by -a: du[2i] = cos * dy[2i] + sin * dy[2i+1] and du[2i+1] = cos * dy[2i+1] - sin * dy[2i].
    dy_first, dy_second = tl.split(tl.reshape(dy, (BLOCK_TOKENS, BLOCK_D // 2, 2)))
    du = tl.interleave(cos * dy_first + sin * dy_second, cos * dy_second - sin * dy_first)

    # The LayerNorm backward of j3pa/qk_layernorm_triton.py with du as the output gradient: recompute the normalised
    # input x_hat, then dx = (g - mean(g) - x_hat * mean(g * x_hat)) / std with g = du * weight and means over the row.
    x_hat = tl.where(mask, (x - mean[:, None]) * rstd[:, None], 0.0)
    g = du * weight[None, :]
    mean_g = tl.sum(g, axis=1) / D
    mean_g_x_hat = tl.sum(g * x_hat, axis=1) / D
    dx = (g - mean_g[:, None] - x_hat * mean_g_x_hat[:, None]) * rstd[:, None]
    tl.store(dx_ptr + rows.to(tl.int64)[:, None] * D + cols[None, :], dx.to(dx_ptr.dtype.element_ty), mask=mask)

    # The weight gradient is the sum over the rows of du * x_hat, the bias gradient that of du.
    return du * x_hat, du


@triton.jit
def _backward_kernel(
    q_ptr, k_ptr,  # The inputs of the forward pass.
    q_out_grad_ptr, k_out_grad_ptr,  # Gradients of the outputs, coming from the attention.
    q_grad_ptr, k_grad_ptr,  # Written: gradients of q and k.
    q_weight_ptr, k_weight_ptr, q_mean_ptr, q_rstd_ptr, k_mean_ptr, k_rstd_ptr,
    pos_ptr, angles_ptr, angle_idx_ptr, pos_idx_ptr,
    q_dweight_ptr, q_dbias_ptr, k_dweight_ptr, k_dbias_ptr,  # Written: (num_programs, D) partial sums per program.
    x_stride_group, x_stride_head, x_stride_token,  # Strides of q and of k.
    dy_stride_group, dy_stride_head, dy_stride_token,  # Strides of both output gradients (also the same).
    num_heads, group_size, num_tiles, token_blocks,
    D: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_TOKENS: tl.constexpr,
):
    # As in j3pa/qk_layernorm_triton.py: program p handles the tiles p, p + num_programs, p + 2 * num_programs, ... of
    # q and of k, and adds up the weight and bias gradients of its rows in registers. Tile t is token block
    # t % token_blocks of head t // token_blocks (counting over all groups).
    q_dweight = tl.zeros((BLOCK_TOKENS, BLOCK_D), dtype=tl.float32)
    q_dbias = tl.zeros((BLOCK_TOKENS, BLOCK_D), dtype=tl.float32)
    k_dweight = tl.zeros((BLOCK_TOKENS, BLOCK_D), dtype=tl.float32)
    k_dbias = tl.zeros((BLOCK_TOKENS, BLOCK_D), dtype=tl.float32)

    for tile in range(tl.program_id(0), num_tiles, tl.num_programs(0)):
        group_head = tile // token_blocks
        group = group_head // num_heads
        head = group_head % num_heads
        tokens = (tile % token_blocks) * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
        token_mask = tokens < group_size
        rows = group_head * group_size + tokens

        # Recompute the angles, shared by q and k, instead of keeping cos and sin from the forward pass.
        cos, sin = _rope_cos_sin(group, head, tokens, token_mask, group_size, pos_ptr, angles_ptr, angle_idx_ptr,
                                 pos_idx_ptr, D, BLOCK_D // 2)

        x_offsets = _token_offsets(group, head, tokens, x_stride_group, x_stride_head, x_stride_token)
        dy_offsets = _token_offsets(group, head, tokens, dy_stride_group, dy_stride_head, dy_stride_token)
        dweight, dbias = _backward_one(q_ptr, q_out_grad_ptr, q_grad_ptr, q_weight_ptr, q_mean_ptr, q_rstd_ptr, x_offsets,
                                       dy_offsets, rows, token_mask, cos, sin, D, BLOCK_D, BLOCK_TOKENS)
        q_dweight += dweight
        q_dbias += dbias
        dweight, dbias = _backward_one(k_ptr, k_out_grad_ptr, k_grad_ptr, k_weight_ptr, k_mean_ptr, k_rstd_ptr, x_offsets,
                                       dy_offsets, rows, token_mask, cos, sin, D, BLOCK_D, BLOCK_TOKENS)
        k_dweight += dweight
        k_dbias += dbias

    # Sum over the rows of the tile, and write this program's partial sums.
    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < D
    partial = tl.program_id(0) * D + cols
    tl.store(q_dweight_ptr + partial, tl.sum(q_dweight, axis=0), mask=col_mask)
    tl.store(q_dbias_ptr + partial, tl.sum(q_dbias, axis=0), mask=col_mask)
    tl.store(k_dweight_ptr + partial, tl.sum(k_dweight, axis=0), mask=col_mask)
    tl.store(k_dbias_ptr + partial, tl.sum(k_dbias, axis=0), mask=col_mask)


@functools.cache
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _block_sizes(head_dim: int, tile: int) -> tuple[int, int]:
    """Block size of the head dim (a power of two) and the number of tokens of a tile with about `tile` elements."""
    block_d = triton.next_power_of_2(head_dim)
    return block_d, max(1, tile // block_d)


class _QKLayerNormRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, q_weight, q_bias, k_weight, k_bias, pos, angles, angle_idx, pos_idx, q_eps: float, k_eps: float):
        num_groups, num_heads, group_size, head_dim = q.shape
        block_d, block_tokens = _block_sizes(head_dim, _FORWARD_TILE)
        # Contiguous outputs, as the separate implementation gives them to the attention.
        q_out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        k_out = torch.empty_like(q_out)
        # LayerNorm statistics of every row, for the backward pass: q mean, q 1 / std, k mean, k 1 / std.
        stats = torch.empty((4, *q.shape[:-1]), dtype=torch.float32, device=q.device)
        _forward_kernel[(num_groups * num_heads, triton.cdiv(group_size, block_tokens))](
            q, k, q_out, k_out,
            q_weight, q_bias, k_weight, k_bias,
            *stats,
            pos, angles, angle_idx, pos_idx,
            *q.stride()[:3],
            num_heads, group_size, q_eps, k_eps,
            D=head_dim, BLOCK_D=block_d, BLOCK_TOKENS=block_tokens, num_warps=_FORWARD_NUM_WARPS,
        )
        ctx.save_for_backward(q, k, q_weight, k_weight, stats, pos, angles, angle_idx, pos_idx)
        return q_out, k_out

    @staticmethod
    def backward(ctx, q_out_grad, k_out_grad):
        q, k, q_weight, k_weight, stats, pos, angles, angle_idx, pos_idx = ctx.saved_tensors
        # The kernel reads both output gradients with one set of strides and every row as head_dim contiguous values.
        # The attention backward hands them over like that; anything else is copied first.
        if q_out_grad.stride() != k_out_grad.stride() or q_out_grad.stride(-1) != 1:
            q_out_grad, k_out_grad = q_out_grad.contiguous(), k_out_grad.contiguous()
        num_groups, num_heads, group_size, head_dim = q.shape
        block_d, block_tokens = _block_sizes(head_dim, _BACKWARD_TILE)
        token_blocks = triton.cdiv(group_size, block_tokens)
        num_tiles = num_groups * num_heads * token_blocks
        num_programs = min(num_tiles, _BACKWARD_PROGRAMS_PER_SM * _num_sms(q.device.index))
        q_grad = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        k_grad = torch.empty_like(q_grad)
        # Every program's partial sums of the q weight, q bias, k weight and k bias gradients.
        partials = torch.empty((4, num_programs, head_dim), dtype=torch.float32, device=q.device)
        _backward_kernel[(num_programs,)](
            q, k, q_out_grad, k_out_grad, q_grad, k_grad,
            q_weight, k_weight, *stats,
            pos, angles, angle_idx, pos_idx,
            *partials,
            *q.stride()[:3],
            *q_out_grad.stride()[:3],
            num_heads, group_size, num_tiles, token_blocks,
            D=head_dim, BLOCK_D=block_d, BLOCK_TOKENS=block_tokens, num_warps=_BACKWARD_NUM_WARPS,
        )
        # Add up the partial sums of the programs (in a fixed order, so deterministically).
        q_dweight, q_dbias, k_dweight, k_dbias = partials.sum(1).to(q_weight.dtype)
        # One gradient per input of forward. The positions have none, as in RoPE.cos_sin (no_grad without the learnable
        # scale), and neither do the RoPE buffers and the eps.
        return q_grad, k_grad, q_dweight, q_dbias, k_dweight, k_dbias, None, None, None, None, None, None


def qk_layernorm_rope(
    q: torch.Tensor, k: torch.Tensor, q_ln: nn.LayerNorm, k_ln: nn.LayerNorm, rope: RoPE, pos: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """QK-LayerNorm followed by RoPE, as in PointSelfAttn:

        q, k = q_ln(q).to(q.dtype), k_ln(k).to(k.dtype)
        cos, sin = rope.cos_sin(pos, q.dtype), viewed and permuted to the shape of q
        q, k = RoPE.apply(q, cos, sin), RoPE.apply(k, cos, sin)

    Args:
        q, k: (groups, heads, group_size, head_dim) views of the QKV projection, as in PointSelfAttn.
        q_ln, k_ln: The QK-LayerNorm modules.
        rope: The layer's RoPE module.
        pos: (groups * group_size, 3) position of every token in the padded order of the attention groups, i.e.
             `pt_pos[fused_pad_idx]` in PointSelfAttn.

    Returns:
        q and k, contiguous, in their dtype.
    """
    assert q.dim() == 4 and q.shape == k.shape and q.dtype == k.dtype, "Expects q and k as in PointSelfAttn."
    num_groups, num_heads, group_size, head_dim = q.shape
    assert head_dim % 2 == 0, "RoPE rotates channel pairs."
    assert rope._alpha is None, "The learnable RoPE scale is not supported."
    assert rope._pos_dim == 3 and rope._dim == num_heads * head_dim, "Expects the layer's RoPE module."
    assert pos.shape == (num_groups * group_size, 3), "Expects one (x, y, z) position per token."
    # The kernels read q and k with one set of strides and every row as head_dim contiguous values.
    if q.stride() != k.stride() or q.stride(-1) != 1:
        q, k = q.contiguous(), k.contiguous()
    # RoPE multiplies its float32 frequencies with the positions in float32.
    pos = pos.float().contiguous()
    return _QKLayerNormRoPE.apply(
        q, k, q_ln.weight, q_ln.bias, k_ln.weight, k_ln.bias, pos, rope._angles, rope._angle_idx, rope._pos_idx,
        q_ln.eps, k_ln.eps,
    )
