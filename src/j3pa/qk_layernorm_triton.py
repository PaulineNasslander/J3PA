"""QK Layernorm in Triton for our specific model.

Dirty, will fix later, but if it works we can use.

It exploits the non-contiguous memory layout our our tensors in the point transformer directly
and outputs into bfloat16. By avoiding copies things are much faster than in pytorch.
"""

import functools

import torch
import triton
import triton.language as tl

# Probably doesn't matter much within a reasonable range. Claude suggested these for the memory bandwidth of A100s.
_FORWARD_TILE = 4096
_BACKWARD_TILE = 2048
_NUM_WARPS = 4

# Backward programs per streaming multiprocessor. Enough to keep the GPU busy, few enough to make the partial sums of
# the weight and bias gradients cheap to add up. We don't want to end up with monster tensors if we do the summations
# too granular in the backward pass.
_BACKWARD_PROGRAMS_PER_SM = 4


@triton.jit
def _row_offsets(
    rows, num_heads, group_size, stride_group, stride_head, stride_token):
    """Offset (in elements) of the first value of each row of a (groups, heads, group_size, head_dim) tensor.

    Each "row" in our definition is a C//H vector from one of the attention heads. We have groups*heads*group_size different
    rows. Rows are numbered = (group * num_heads + head) * group_size + token. The tensors can have any strides, but its last axis
    must be contiguous.
    """
    # Token in the group
    token = rows % group_size
    head = (rows // group_size) % num_heads
    group = rows // (group_size * num_heads)
    # Do Int64 to be safe on large tensors with tons of elements. Getting a billion elements is not impossible.
    # Compute the actual index in the underlying (NG, G, H, C//H) tensor.
    return group.to(tl.int64) * stride_group + head.to(tl.int64) * stride_head + token.to(tl.int64) * stride_token


@triton.jit
def _forward_kernel(
    x_ptr, # Input
    y_ptr, # Empty tensor where we write outputs
    weight_ptr, # Learnable gamma
    bias_ptr, # learnable beta
    mean_ptr, # Empty tensor where we will write the mean. We need to keep this for backward.
    rstd_ptr, # Empty tensor where we write inverse standard deviations. We need to keep this for backward.
    x_stride_group, # Memory stride from one index in the NG axis of the (NG, H, G, C//H) tensor.
    x_stride_head, # Memory stride from one index in the H axis of the (NG, H, G, C//H) tensor.
    x_stride_token, # Memory stride from one index in the G axis of the (NG, H, G, C//H) tensor.
    num_heads, # Obvious.
    group_size, 
    num_rows,
    eps, # Epsilon for safe division by the standard deviation.
    D: tl.constexpr,  # head_dim, the size normalised over.
    BLOCK_D: tl.constexpr,  # D rounded up to a power of two, as Triton's block sizes must be.
    BLOCK_ROWS: tl.constexpr,  # Rows per program.
):
    # (We define tokens as one C//H token here, not a full original token. I should clean up my messy notation.)
    # We process the layernorm in chunks of BLOCK_ROWS tokens and each compute block will process 
    # the full channel count. It has to be like this because we need to normalize over all channels in
    # each individual token so we can't split the channels across different kernel calls.
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, BLOCK_D)
    row_mask = rows < num_rows  # The last tile may run past the last row,
    col_mask = cols < D  # and a head_dim that is not a power of two leaves columns unused.
    mask = row_mask[:, None] & col_mask[None, :]

    # We convert our row index into an index for each of the (NG, H, G) axes, and then use the actual
    # strides of each axis in the pytorch tensor to make sure we read the correct memory locations.
    # x is a (BLOCK_ROWS, BLOCK_D) tensor with 0s in out-of-memory locations.
    x_offsets = _row_offsets(rows, num_heads, group_size, x_stride_group, x_stride_head, x_stride_token)

    # Accumulate the mean and inverse standard deviation in fp32 for bfloat16 inputs.
    x = tl.load(x_ptr + x_offsets[:, None] + cols[None, :], mask=mask, other=0.0).to(tl.float32)

    # Mean and biased variance of every row, as in nn.LayerNorm. The masked-out entries are 0, so they do not change
    # the sum for the mean, and are zeroed again after centering so that they do not count in the variance either.
    mean = tl.sum(x, axis=1) / D
    x_centered = tl.where(mask, x - mean[:, None], 0.0)
    rstd = tl.rsqrt(tl.sum(x_centered * x_centered, axis=1) / D + eps)

    # The weight and bias are just C//H tensors so we can just index them with cols.
    # We upcast the weights to fp32 so we can compute the actual layernorm output in fp32. Pytorch outputs in fp32
    # for layer_norm, but we will instantly cast it back to bfloat16 in the next layer. We compute in fp32 but write
    # the output to a bfloat16 tensor which is faster and equivalent for our usecase.
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    y = x_centered * rstd[:, None] * weight[None, :] + bias[None, :]

    # Indexing the output tensor is easy because we explicitly design it so it is contiguous in our definition of rows.
    # This was not the case for the reading of the input tensor.
    y_offsets = rows.to(tl.int64)[:, None] * D + cols[None, :]

    # Cast to the element of our output tensor, which is bfloat16 as specified above.
    tl.store(y_ptr + y_offsets, y.to(y_ptr.dtype.element_ty), mask=mask)

    # The backward pass recomputes the normalised input from x and these two statistics.
    tl.store(mean_ptr + rows, mean, mask=row_mask)
    tl.store(rstd_ptr + rows, rstd, mask=row_mask)


@triton.jit
def _backward_kernel(
    x_ptr, # The input to layernorm from the forward pass.
    dy_ptr, # Gradient coming from the the next layer back up. (The output.)
    dx_ptr, # Empty tensor where we will write the gradient from this layer so the next one can use it.
    weight_ptr, # The current weight.
    mean_ptr, # Mean used in forward pass. Needed to backprop through.
    rstd_ptr, # Inverse std used in forward pass. Needed to backprop through.
    dweight_partial_ptr, # Summed gradients on the weights for chunks of tokens
    dbias_partial_ptr, # Summed gradients on the bias for chunks of tokens
    x_stride_group, # Actual memory strides of our input tensors.
    x_stride_head, 
    x_stride_token,
    dy_stride_group, # Actual memory strides of the backpropped tensors.
    dy_stride_head,
    dy_stride_token,
    num_heads,
    group_size,
    num_rows, 
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    # Program p handles the tiles p, p + num_programs, p + 2 * num_programs, ... For the weight and bias gradients,
    # which are sums over all rows, it adds up the contributions of its rows in registers and writes the result to
    # its own row of the partial-sum buffers.
    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < D
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

    # We 
    dweight_sum = tl.zeros((BLOCK_ROWS, BLOCK_D), dtype=tl.float32)
    dbias_sum = tl.zeros((BLOCK_ROWS, BLOCK_D), dtype=tl.float32)

    # The reasoning behind the loop here is that we ultimately need one global gradient on the weight and bias which is the
    # sum of the contributions from all the different tokens. We want each triton program to process several vectorized blocks
    # and sum the partial results of the gradients so that we don't need massive output tensors (one per block).
    for first_row in range(tl.program_id(0) * BLOCK_ROWS, num_rows, tl.num_programs(0) * BLOCK_ROWS):
        rows = first_row + tl.arange(0, BLOCK_ROWS)
        row_mask = rows < num_rows
        mask = row_mask[:, None] & col_mask[None, :]

        # Load the input and the output gradient (both strided) and the statistics of the rows.
        x_offsets = _row_offsets(rows, num_heads, group_size, x_stride_group, x_stride_head, x_stride_token)
        dy_offsets = _row_offsets(rows, num_heads, group_size, dy_stride_group, dy_stride_head, dy_stride_token)
        x = tl.load(x_ptr + x_offsets[:, None] + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + dy_offsets[:, None] + cols[None, :], mask=mask, other=0.0).to(tl.float32)

        # We have one mean per token and not channel so we need to index with rows.
        mean = tl.load(mean_ptr + rows, mask=row_mask, other=0.0)
        rstd = tl.load(rstd_ptr + rows, mask=row_mask, other=0.0)

        # Recompute the normalised input x_hat = (x - mean) / std, which the forward pass did not keep.
        # This feels like a bit of compute waste since we calculated it in the forward, but I see no other way.
        # Keeping this in memoyr for every token from the forward would be suicide.
        x_hat = tl.where(mask, (x - mean[:, None]) * rstd[:, None], 0.0)

        # y = x_hat * weight + bias
        # dx_hat = dy * weight
        # dx = (g - mean(g) - x_hat * mean(g * x_hat)) / std
        # with the means over the row. The two subtracted terms are the contributions through the mean and the std.
        g = dy * weight[None, :]
        mean_g = tl.sum(g, axis=1) / D
        mean_g_x_hat = tl.sum(g * x_hat, axis=1) / D
        dx = (g - mean_g[:, None] - x_hat * mean_g_x_hat[:, None]) * rstd[:, None]

        # The output write is contiguous.
        dx_offsets = rows.to(tl.int64)[:, None] * D + cols[None, :]
        tl.store(dx_ptr + dx_offsets, dx.to(dx_ptr.dtype.element_ty), mask=mask)

        # The weight gradient is the sum over the rows of dy * x_hat, the bias gradient that of dy.
        dweight_sum += dy * x_hat
        dbias_sum += dy

    # Sum over the rows of the tile, and write this program's partial sums.
    partial_offsets = tl.program_id(0) * D + cols
    tl.store(dweight_partial_ptr + partial_offsets, tl.sum(dweight_sum, axis=0), mask=col_mask)
    tl.store(dbias_partial_ptr + partial_offsets, tl.sum(dbias_sum, axis=0), mask=col_mask)


@functools.cache
def _num_sms(device_index: int) -> int:
    """Count streaming multiprocessors of the GPU"""
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _block_sizes(head_dim: int, tile: int) -> tuple[int, int]:
    """Block size of the head dim (must be a power of two in Triton) and the number of rows of a compute tile."""
    block_d = triton.next_power_of_2(head_dim)
    return block_d, max(1, tile // block_d)


class _LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
        _, num_heads, group_size, head_dim = x.shape
        block_d, block_rows = _block_sizes(head_dim, _FORWARD_TILE)

        # We make the output contiguous.
        # Store directly in bfloat16; the next operation would otherwise immediately cast a float32 result.
        y = torch.empty(x.shape, dtype=x.dtype, device=x.device)

        # These bad boys should be in fp32 since we sum a bunch of channels to compute them.
        mean = torch.empty(x.shape[:-1], dtype=torch.float32, device=x.device)
        rstd = torch.empty_like(mean)
        num_rows = mean.numel()
        _forward_kernel[(triton.cdiv(num_rows, block_rows),)](
            x,
            y,
            weight,
            bias,
            mean,
            rstd,
            *x.stride()[:3], #(NG, H, G, C//H) tensors. We only need stides for (NG, H, G)
            num_heads,
            group_size,
            num_rows,
            eps,
            D=head_dim, BLOCK_D=block_d, BLOCK_ROWS=block_rows, num_warps=_NUM_WARPS,
        )

        # We don't need the bias term stored for the backward. 
        ctx.save_for_backward(x, weight, mean, rstd)
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x, weight, mean, rstd = ctx.saved_tensors

        # The backward kernel assumes a contiguous last dimension, so copy gradients when that is not true.
        if not dy.stride(-1) == 1:
            dy = dy.contiguous()
        _, num_heads, group_size, head_dim = x.shape
        block_d, block_rows = _block_sizes(head_dim, _BACKWARD_TILE)

        # Each row is one C//H sized token.
        num_rows = mean.numel()
        num_programs = min(triton.cdiv(num_rows, block_rows), _BACKWARD_PROGRAMS_PER_SM * _num_sms(x.device.index))

        # The tensor in which we will write the output derivative.
        dx = torch.empty(x.shape, dtype=x.dtype, device=x.device)

        # Tensors containing the derivatives w.r.t. the weights so we can update them.
        # We get gradients from each chunk of tokens (and the kernel already sums them). We will
        # sum over all the chunks in the end to get the final derivative.
        dweight_partial = torch.empty((num_programs, head_dim), dtype=torch.float32, device=x.device)
        dbias_partial = torch.empty_like(dweight_partial)

        _backward_kernel[(num_programs,)](
            x,
            dy,
            dx,
            weight,
            mean,
            rstd,
            dweight_partial,
            dbias_partial,
            *x.stride()[:3],
            *dy.stride()[:3],
            num_heads,
            group_size,
            num_rows,
            D=head_dim,
            BLOCK_D=block_d,
            BLOCK_ROWS=block_rows,
            num_warps=_NUM_WARPS,
        )

        # This reduction avoids cross-program atomics and is deterministic.
        dweight = dweight_partial.sum(0).to(weight.dtype)
        dbias = dbias_partial.sum(0).to(weight.dtype)

        # We must return one gradient for each forward input. Epsilon has no gradient so we return None.
        return dx, dweight, dbias, None


def layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    """Layernorm, but only use for QK-layernorm in point transformer. 

    We know what channel we'll use it on so we dont need to bother with "normalized_shape" like in the 
    torch.nn.functional function.
    """
    assert x.dim() == 4, "Expected x to have shape (NG, H, G, C // H)."
    assert weight.shape == bias.shape == x.shape[-1:], "weight and bias must match the input's final dimension."
    # The kernels read every row as head_dim contiguous values.
    x = x if x.stride(-1) == 1 else x.contiguous()
    return _LayerNorm.apply(x, weight, bias, eps)
