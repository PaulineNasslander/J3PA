"""Implementation of the positional encodings."""

import math

import torch
from torch import nn


def _frequencies_orig(embed_dim: int, pos_dim: int, max_freq: int) -> torch.Tensor:
    """Frequencies used in the positional encoding of the original transformer paper.

    See Attention Is All You Need (https://arxiv.org/abs/1706.03762). These positional encodings
    are also used in DETR (https://arxiv.org/abs/2005.12872).

    Args:
        embed_dim: Dimensionality of the embeddings. The positional encoding will have the same
                   dimensionality.
        pos_dim: Dimensionality of the positions to encode. For instance 1 for text, 2 for images
                 and 3 for lidar.
        max_freq: Maximum frequency value. 1e4 in the original paper.

    Returns:
        Frequencies for positional encoding. Shape (embed_dim,)
    """
    log = math.log(max_freq)
    axis_dim = math.ceil(embed_dim / pos_dim)
    fqs = torch.arange(axis_dim // 2, dtype=torch.float)
    fqs = torch.exp(-2 * (fqs / axis_dim) * log)
    return fqs


def _frequencies_perceiver(embed_dim: int, max_freq: int) -> torch.Tensor:
    """Frequencies used in the positional encoding of the Perceiver paper.

    See https://arxiv.org/abs/2103.03206. In Perceiver, the input positions are expected to be
    normalized to the [-1, 1] interval. Account for this when using this.

    Args:
        embed_dim, max_freq: See _frequencies_orig.

    Returns:
        See _frequencies_orig
    """
    # In the original perceiver paper the number of frequencies is a hyperparamter
    fqs = torch.pi * torch.linspace(0, max_freq, embed_dim // 2, dtype=torch.float).flip(dims=[-1])
    return fqs


def _frequencies(embed_dim: int, pos_dim: int, max_freq: int, perceiver: bool) -> torch.Tensor:
    """Route to the correct frequency function."""
    if perceiver:
        return _frequencies_perceiver(embed_dim, max_freq)
    return _frequencies_orig(embed_dim, pos_dim, max_freq)


class PEStatic(nn.Module):
    """Completely precomputed positional encoder.

    This class should be used when the positional encoding can be fully known beforehand. This is
    for example the case for images of fixed resolution, or voxel grids without an additional
    position offset.

    When it can be used this class will result in more efficient positional encoding than
    PEDynamic.
    """

    # pylint: disable=too-many-locals
    def __init__(self, embed_dim: int, axis_sizes: list[int], max_freq: int = 1e4, perceiver: bool = False):
        """Initialize.

        Args:
            embed_dim: Dimensionality of the embeddings. The positional encodings will be of the
                       same size.
            axis_sizes: Sizes of the respective axis to positionally encode. For an image this
                        would be the number of height and width pixels.
            max_freq: Maximum frequency to use.
            perceiver: Whether to use Perceiver positional encodings. If False those from Attention
                       is All you Need are used.
        """
        super().__init__()

        # Add additional frequencies if the dimensionalities do not line up. We remove these later.
        dim_offset = embed_dim % (2 * len(axis_sizes))
        fqs = _frequencies(embed_dim + dim_offset, len(axis_sizes), max_freq, perceiver).unsqueeze(0)  # (1, embed_dim/2*pos_dim)

        # Make positional encodings along each dimension.
        axis_pes = []
        for ax in axis_sizes:
            pos = torch.arange(ax, dtype=torch.float).unsqueeze(1)  # (pos, 1)
            pe = pos * fqs
            axis_pes.append(torch.cat([torch.sin(pe), torch.cos(pe)], dim=-1))

        # Expand all the 1D positional encodings so they can be stacked along all dimensions.
        for i, _ in enumerate(axis_pes):
            repeats = axis_sizes.copy() + [1]
            repeats[i] = 1
            view_shape = [1 for _ in axis_sizes] + [axis_pes[i].shape[-1]]
            view_shape[i] = axis_sizes[i]
            axis_pes[i] = axis_pes[i].view(*view_shape).repeat(*repeats)

        # Stack and flatten instead of concatenate to interleave the dimensions. This allows a
        # clean dimensionality cutoff that does not favor any particular dimension.
        pe_full = torch.stack(axis_pes, dim=-1)  # (X1, X2, .., XN, embed_dim/N, N)
        pe_full = pe_full.flatten(start_dim=-2, end_dim=-1)

        # Now make a cutoff if needed.
        pe_full = pe_full[..., :embed_dim]

        # Add a batch axis and store the precompute PEs.
        pe_full = pe_full.unsqueeze(0)
        self.register_buffer("_pe", pe_full)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Build the positional encoding.

        It is precomputed and just needs to be repeated along the batch axis.

        Args:
            x: Feature tensor to be positional encoded. For instance image tensor of shape
               (B, H, W, C).

        Returns:
            Positional encoding tensor. Shape (B, X1, ..., XN, C)
        """
        B = x.shape[0]
        return self._pe.repeat(B, *[1] * (x.dim() - 1))


class PEDynamic(nn.Module):
    """Dynamically computed positional encodings.

    This class computes a positional encoding dynamically, given a tensor of positions. It is more
    flexible thant PEStatic, but slower when the latter can be used. For lidar one typically has
    positions that change between point clouds, making this the go-to class.
    """

    def __init__(self, embed_dim: int, pos_dim: int, max_freq: float = 1e4, perceiver: bool = False):
        """Initialize.

        Args:
            See PEStatic.
        """
        super().__init__()
        # Keep track of the embeddings dimensionality and compute an offset in case it does not
        # line up with the positional encodings. If so, too many positional encodings will be
        # computed and cut off in the end.
        dim_offset = embed_dim % (pos_dim * 2)
        self._embed_dim = embed_dim
        fqs = _frequencies(embed_dim + dim_offset, pos_dim, max_freq, perceiver).unsqueeze(0)
        self.register_buffer("_fqs", fqs)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        """Dynamically compute positional encodings.

        Args:
            pos: Tensor with positions, based on which positional encodings are computed. Shape
                 (..., N) for N dimensional positions.

        Returns:
            Positional encoding tensor. Shape (..., self._embed_dim).
        """
        pe = pos.unsqueeze(-1) * self._fqs  # (..., N, 1) * (1, D/N/2) -> (..., N, D/N/2)
        pe = pe.flatten(start_dim=-2, end_dim=-1)  # (..., N, D/N/2) -> (..., D/2)
        pe = torch.cat([torch.sin(pe), torch.cos(pe)], dim=-1)  # (..., D)

        # Cut off the extra dimensions.
        # It is best to do the redundant computation and cut off after the concatentation so we
        # only drop cosines and for different axes. This way each dimension has about the same
        # number of frequencies left.
        return pe[..., : self._embed_dim]


if __name__ == "__main__":
    # Do some plotting.
    import matplotlib.pyplot as plt

    encoder_dynamic = PEDynamic(embed_dim=36, pos_dim=3, max_freq=1e4, perceiver=False)
    _x = torch.linspace(-200, 200, 1000)
    _y = torch.linspace(-100, 100, 1000)
    _z = torch.linspace(-10, 10, 1000)

    points = torch.stack([_x, _y, _z], dim=-1).unsqueeze(0)  # (1, 1000, 3)
    pe_dyn = encoder_dynamic(points)
    for j in range(pe_dyn.shape[-1] // 6):
        plt.plot(_x, pe_dyn[0, :, 3 * j])
    plt.savefig("pe_dynamic_sin.pdf")
    plt.clf()

    encoder_static = PEStatic(embed_dim=36, axis_sizes=[100, 100, 10], max_freq=1e4, perceiver=False)
    points = torch.rand(1, 100, 100, 10, 64)

    pe_stat = encoder_static(points)
    for j in range(pe_stat.shape[-1] // 6):
        plt.plot(torch.arange(100), pe_stat[0, :, 0, 0, 3 * j])
    plt.savefig("pe_static_sin.pdf")
