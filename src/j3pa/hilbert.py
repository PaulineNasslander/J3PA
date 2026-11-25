import torch


def gray_to_binary_int(gray: torch.Tensor) -> torch.Tensor:
    """Convert gray code to binary using bitwise operations."""
    # For gray to binary conversion: bin[i] = gray[i] XOR bin[i-1]
    # Start from the most significant bit
    binary = gray.clone()
    shift = 1
    while shift < 64:  # Maximum bits in int64
        binary = binary ^ (binary >> shift)
        shift <<= 1
    return binary


# @torch.compile
def encode_hilbert_bitwise(coords: torch.Tensor, num_bits: int) -> torch.Tensor:
    """Convert discretized coordinates into a point along an appropriate order Hilbert Curve.

    (..., N) -> (...)

    Implementation of the algorithm from Skilling, J. (2004, April). Programming the Hilbert curve.
    AIP Conference Proceedings (Vol. 707, No. 1, pp. 381-387). American Institute of Physics.

    Warning: Pyorch does not support uint64. This means we can only use 63 bits for encoding
    Hilbert integers.

    Unlike the official implementation of PTV3, which is a dirty port of
    https://github.com/PrincetonLIPS/numpy-hilbert-curve, we don't make uint8s act as if they are
    bits but use actual bitwise operations. For representative point clouds, GPU encoding on an
    A100 with this implementation is roughly 3x faster than the PTV3 implementation.

    Args:
        coords: Tensor containing discretized coordinates (grid cells) along each of the
                dimensions. Seeing an N-D grid as a discretized hypercube in N dimensions this
                determines the point locations in the hypercube.
                Shape: (..., N) for N dimensional coordinates.
        num_bits: Number of bits needed to represent each coordinate. In other words, the largest
                  coordinate along each axis (or the number of grid cells in the hypercube) is
                  2**(num_bits).
    Returns:
        int64 tensor of shape (...) in which the N dimensional coordinates have been compressed
    """
    assert len(coords.shape) >= 2, f"We expect a tensor of shape (..., N) but got {coords.shape=}"
    # assert (
    #    coords.dtype == torch.int64
    # ), f"Expected coordinates in torch.int64 but got {coords.dtype=}"
    num_dims = coords.shape[-1]
    assert num_dims * num_bits <= 63, (
        f"We can only use up to 63 bits to represent Hilbert integers because the best we can do is int64 in Pytorch. Got {num_dims*num_bits=}"
    )
    assert coords.max() < 2**num_bits, "The requested bit depth is not sufficient to process the given coordinates."

    # TODO CAN WE REMOVE THIS CLONE AND DETERMINE WHEN TO DO INT AND LONG?
    coords = coords.to(dtype=torch.int64, copy=True)

    batch_shape = coords.shape[:-1]

    # Flatten batch dimensions for easier processing
    # Flatten (..., N) -> (M, N) with M the flattening of all (...) dimensions.
    coords_flat = coords.view(-1, num_dims)  # (batch_size, num_dims)
    M = coords_flat.shape[0]

    # We'll build the result by processing bits from the most significant (leftmost) to the least
    # significant (rightmost) bit.
    # This is an inversion of the decoding loop given in the paper referenced above. In the paper
    # they start at the highest bits for decoding. Here we start at the lowest ones for encoding.
    # Iterate forwards through the bits and dimensions. This part is very unintuitive so read the 
    # paper referenced above if you want to understand it fully.
    # This loop can not be inverted.
    # This loop is borderline impossible to vectorize without expanding the tensors to have two
    # additional axes because of strong interloop dependencies. Note that there can never be more
    # than 62 iterations in the combined double loop so the overhead is not enormous.
    # Note that the loop is not symmetric.
    for b in range(num_bits - 1):
        # We start at the most significant bits.
        bit_mask = 1 << num_bits - b - 1

        # All bits of lower order than b in dimension 0 must be inverted where dimension d has a
        # nonzero bit b. By substracting one from 10000000 you get 01111111.
        lower_mask = bit_mask - 1

        for d in range(num_dims):
            # Points for which bit d is on.
            bit_d_on = (coords_flat[:, d] & bit_mask) != 0  # (M,)

            # This gives a tensor with 1s (bitwise) at every bit location right of b, where b was
            # also on in the original coordinate. Then we invert the bits below b in the original
            # coordinate tensor by xoring.
            xor_mask = torch.where(bit_d_on, lower_mask, 0)

            # Inplace operations are essential. The three in-place xors in this loop make the code
            # >10% faster in total.
            coords_flat[:, 0] ^= xor_mask

            # On the other hand, when the bth bit of dimension d is off we should swap all of the
            # lower order bits representing this dimension's coordinate with those of the 0th
            # coordinate.
            diff_bits = (coords_flat[:, 0] ^ coords_flat[:, d]) & lower_mask
            swap_mask = torch.where(~bit_d_on, diff_bits, 0)

            # Do the switcheroo.
            coords_flat[:, 0] ^= swap_mask
            coords_flat[:, d] ^= swap_mask

    # Interleave bits from all dimensions to create Hilbert index. Like in Skillings paper, if we
    # have n dimensions represented by p bits each we built a n*p bit hilbert integer. The hilbert
    # integer will consist of chunks n bits with the most significant n bits of each dimension,
    # then the next most significant bit of each dimension and so on (after the black magic above
    # that is). This will result in a "gray code" of bits that we will have to turn into an
    # integer.
    hilbert_indices = torch.zeros(M, dtype=torch.int64, device=coords.device)

    # Maximum 63 iterations.
    for b in range(num_bits):
        bit_pos = num_bits - 1 - b
        bit_mask = 1 << bit_pos

        # The order of dimensions is in principle arbitrary as long as all relative bit flips
        # between them are correct. Note that if you were to start the outer loop above from the
        # last bit and go backwards, you will implicitly change the orders of dimensions and fail
        # the test cases.
        for d in range(num_dims):
            # Extract bit from dimension d for each point and move it to the right (so this is
            # now basically 00...001 or 00...000 either on or off.
            bit_val = (coords_flat[:, d] & bit_mask) >> bit_pos

            # Put the extracted bits in the correct position in the Hilbert indices. Remember that
            # we interleave chunks of num_dims with a bit from each of the dimensions.
            hilbert_bit_pos = num_dims * num_bits - 1 - (b * num_dims + d)
            hilbert_indices |= bit_val << hilbert_bit_pos

    # The result of the above bit interleaving will result in a "gray code" of bits that we will
    # have to turn into an integer.
    hilbert_indices = gray_to_binary_int(hilbert_indices)

    # Reshape back to original batch shape. The last dimension disappeared because all dimensions
    # are collapsed into a single integer.
    return hilbert_indices.view(batch_shape)


# def encode_bitwise_optimized(coords: torch.Tensor, num_bits: int) -> torch.Tensor:
#    """
#    More optimized version using vectorized bitwise operations where possible.
#    """
#    assert len(coords.shape) >= 2, f"Expected shape (..., N) but got {coords.shape=}"
#    assert coords.dtype == torch.int64, f"Expected int64 but got {coords.dtype=}"
#
#    batch_shape = coords.shape[:-1]
#    num_dims = coords.shape[-1]
#
#    assert num_dims * num_bits <= 63, f"Can only use 63 bits, got {num_dims * num_bits}"
#
#    coords_flat = coords.view(-1, num_dims)
#    batch_size = coords_flat.shape[0]
#
#    # Process the Hilbert transformation
#    for b in range(num_bits - 1):
#        bit_mask = 1 << num_bits - 1 - b
#
#        # Create mask for all lower bits
#        lower_mask = bit_mask - 1
#
#        for d in range(num_dims):
#            # Check if current bit is set in dimension d
#            bit_d_on = (coords_flat[:, d] & bit_mask) != 0
#
#            # XOR operation: flip lower bits of dimension 0 when bit d is set
#            flip_mask = torch.where(bit_d_on, lower_mask, 0)
#            coords_flat[:, 0] ^= flip_mask
#
#            # Swap operation: swap differing lower bits between dim 0 and d when bit d is clear
#            # Find lower bits that differ between dimensions 0 and d
#            diff_lower = (coords_flat[:, 0] ^ coords_flat[:, d]) & lower_mask
#
#            # Only apply swap where bit is clear
#            swap_mask = torch.where(~bit_d_on, diff_lower, 0)
#
#            coords_flat[:, 0] ^= swap_mask
#            coords_flat[:, d] ^= swap_mask
#
#    # Interleave bits to form Hilbert indices
#    hilbert_indices = torch.zeros(batch_size, dtype=torch.int64, device=coords.device)
#
#    # Use bit manipulation to interleave
#    for b in range(num_bits):
#        source_bit_pos = num_bits - 1 - b
#        for d in range(num_dims):
#            # Extract the b-th bit from dimension d
#            bit_value = (coords_flat[:, d] >> source_bit_pos) & 1
#
#            # Place in interleaved position
#            target_bit_pos = num_dims * num_bits - 1 - (b * num_dims + d)
#            hilbert_indices |= bit_value << target_bit_pos
#
#    # Convert from Gray code to binary
#    hilbert_indices = gray_to_binary_int(hilbert_indices)
#
#    return hilbert_indices.view(batch_shape)
