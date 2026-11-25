"""Implementation of Z-order curves.

N-dimensional unlike the pure 3D implementation in PTV3.
"""

import functools

import torch


def _interleave_two_zero_bits_low(x: torch.Tensor):
    """Interleave each bit in an integer with two zero bits.

    111 -> 001001001

    This function is only correct for integers below 2**16, as soon as any bit beyond the 17th is
    nonzero we need to do an additional bit shift operation.

    Modifies the tensor in place because gotta go fast.
    """
    # Note that all operations are fused in place to be fast.
    x |= x << 16
    x &= 0x1F0000FF0000FF
    x |= x << 8
    x &= 0x100F00F00F00F00F
    x |= x << 4
    x &= 0x10C30C30C30C30C3
    x |= x << 2
    x &= 0x1249249249249249
    return x


def _interleave_two_zero_bits_high(x: torch.Tensor):
    """Interleave each bit in an interger with two zero bits.

    Works for integers up to 2**24.
    """
    x |= x << 32
    x &= 0x1F00000000FFFF
    return _interleave_two_zero_bits_low(x)


def _interleave_one_zero_bit(x: torch.Tensor):
    """Interleave each bit in an integer with one zero bit.

    111 -> 010101

    This function is correct for integers up to 2**32.

    Modifies the tensor in place because gotta go fast.
    """
    # Note that all operations are fused in place to be fast.
    x |= x << 16
    x &= 0x0000FFFF0000FFFF
    x |= x << 8
    x &= 0x00FF00FF00FF00FF
    x |= x << 4
    x &= 0x0F0F0F0F0F0F0F0F
    x |= x << 2
    x &= 0x3333333333333333
    x |= x << 1
    x &= 0x5555555555555555
    return x


# @torch.compile
def encode_zorder_fast_3d(coords: torch.Tensor, num_bits: int):
    """Encode 3D coordinates into a Z-order curve.

    The algorithm starts by spacing two zero bits between all bits for each coordinate. Then we
    shift one more bit to the left for y and two for z. Then bit-wise or just gives the z-order code

    Args:
        coords: Tensor of 3D coordinates (..., 3).
        num_bits: Number of bits needed to represent each dimension. To be fast the code does not
                  check if this is a lie of not so please be careful with this argument.

    Returns:
        Z-order encoding of all coordinates. Shape (...)
    """
    assert coords.shape[-1] == 3, "This method is only appropriate for 3D positions."
    assert num_bits < 24, "In 3 dimensions we can only process up to 24 bits per dimension."

    # If we have only 10 or fewer bits per dimension we can still represent each dimension as a 32
    # bit integer after interleaving. Doing the bit shifts in int32 is faster and we gotta go fast.
    if num_bits < 11:
        interleaved = _interleave_two_zero_bits_low(coords.int() if coords.dtype != torch.int else coords.clone())
    elif num_bits < 17:
        interleaved = _interleave_two_zero_bits_low(coords.long() if coords.dtype != torch.long else coords.clone())
    else:
        interleaved = _interleave_two_zero_bits_high(coords.long() if coords.dtype != torch.long else coords.clone())

    # Move the y and z axes by 1 and 2 bits so we can interleave the bits of all dimensions.
    interleaved[..., 1] <<= 1
    interleaved[..., 2] <<= 2
    return interleaved[..., 0] | interleaved[..., 1] | interleaved[..., 2]


def encode_zorder_fast_2d(coords: torch.Tensor, num_bits: int):
    """Encode 3D coordinates into a Z-order curve.

    The algorithm starts by spacing a single zero bit between all bits for the coordinates. Then we
    shift one more bit for y and a bitwise or between x and y gives us the z-order curve.

    Args:
        coords: Tensor of 2D coordinates (..., 2).
        num_bits: Number of bits needed to represent each dimension. The caller must ensure this
                  is correct; an undersized value produces incorrect results.

    Returns:
        Z-order encoding of the 2D coordinates. Shape (...)
    """
    assert coords.shape[-1] == 2, "This method is only appropriate for 2D positions."
    assert num_bits < 32, "In 2 dimensions we can only process up to 31 bits per dimension."
    if num_bits < 17:
        interleaved = _interleave_one_zero_bit(coords.int() if coords.dtype != torch.int else coords.clone())
    else:
        interleaved = _interleave_one_zero_bit(coords.long() if coords.dtype != torch.long else coords.clone())
    interleaved[..., 1] <<= 1
    return interleaved[..., 0] | interleaved[..., 1]


def encode_zorder(coords: torch.Tensor, num_bits: int):
    assert not coords.is_floating_point()
    num_dims = coords.shape[-1]
    if num_dims == 3:
        return encode_zorder_fast_3d(coords, num_bits)
    return _encode_zorder_slow(coords, num_bits)


def _encode_zorder_slow(coords: torch.Tensor, num_bits: int):
    """Compute Z-order curve in N dimensions.

    We can use this slow-ass function to build a lookup table to make it fast or you can do the
    work to figure out the correct hexadecimals to shift n-1 bits for your n dimensional Z order
    encoding.

    Args:
        coords: Shape (..., N) for N dimensional coordinates.

    Returns:
        Z-order encoding of coordinates Shape (...)
    """
    assert coords.dtype == torch.int64
    num_dims = coords.shape[-1]
    # Torch does not support uint64 so we lose 1 bit.
    assert num_bits * num_dims < 63
    out = torch.zeros_like(coords)
    for b in range(num_bits):
        bit_mask = 1 << b

        # Filter the i-th bit in each coordinate and move it to the right.
        # (..., N) with 1s and 0s.
        # Note that bracket is essential here for order of operations.
        b_bits = (coords & bit_mask) >> b

        # Turn on the interleaved bits
        offsets = num_dims * b + torch.arange(num_dims, device=out.device)
        out |= b_bits << offsets
    return functools.reduce(torch.bitwise_or, out.unbind(dim=-1))


if __name__ == "__main__":
    coords = torch.randint(0, 100, (10, 10, 10, 3))
    out_1 = _encode_zorder_slow(coords, num_bits=8)
    out_2 = encode_zorder_fast_3d(coords, num_bits=8)
    print(out_1 - out_2)
