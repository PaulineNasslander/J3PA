"""Rotary Positional Embeddings (RoPE).

Generic N-dimensional extension of the RoFormer paper https://arxiv.org/abs/2104.09864.
"""

import math

import torch
from torch import nn

from constants import INITIAL_ALPHA


class RoPE(nn.Module):
    """Rotary Positional Embeddings.

    For the N dimensional case
    """

    def __init__(self, dim: int, pos_dim: int, max_freq: float = 1e4, learnable_scale: bool = False, num_heads: int = None):
        """Initialize.

        Args:
            dim: Model dimensionality. Must be divisible by 2 to keep two-dimensional rotation
                 structure.
            pos_dim: Dimensionality of positions that will be embedded.
            max_freq: Maximum frequency for producing angles.
            learnable_scale: Whether to add a learnable scale to the rotation basis or not. Not
                             taken from any paper but intuitively makes sense to adjust the basis
                             scale to the problem at hand on unstructured data like point clouds.
        """
        assert dim % 2 == 0
        super().__init__()
        self._pos_dim = pos_dim
        self._dim = dim

        # We want to apply separate rotations to embed the positions along each of the axes.
        # The angles are always repeated in groups of two for the 2D rotation structure (see paper)
        # and we will also repeat them for each dimension.
        # Note that to be able to deal with arbitrary model dimensionalities we have to take the
        # ceil. We'll truncate to the desiresd dimensionality in the forward.
        if num_heads is None:
            angles_per_axis = int(math.ceil(dim / (2 * pos_dim)))
        else:
            dh = dim // num_heads
            angles_per_axis = int(math.ceil(dh/(2*pos_dim))*num_heads)

        angles = max_freq ** (-2 * torch.arange(angles_per_axis) / (dim / pos_dim))

        if num_heads is None:
            # [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, ....]
            angle_idx = torch.arange(angles_per_axis).repeat_interleave(2*pos_dim)[:dim]
            # [x, x, y, y, z, z, x, x, y, y, ....]
            pos_idx = torch.arange(pos_dim).repeat_interleave(2).repeat(angles_per_axis)[:dim]
        else:
            head_idx = torch.arange(dim) // dh
            pair_idx = (torch.arange(dim) % dh)//2
            angle_idx = (pair_idx //pos_dim) * num_heads + head_idx
            pos_idx = (pair_idx + head_idx) % pos_dim

        # Note that this repeat pattern will result in (theta_1, theta_1, theta_1, ... theta_2,
        # theta_2, theta_2, ...) with each angle repeated 2*pos_dim times so that it can be applied
        # to successive dimensions.
        #angles = angles.repeat_interleave(2 * pos_dim)
        self.register_buffer("_angles", angles)
        self.register_buffer("_angle_idx", angle_idx)
        self.register_buffer("_pos_idx", pos_idx)
        self._cache = None
        self._alpha = nn.Parameter(torch.full((1,), INITIAL_ALPHA, dtype=torch.float32)) if learnable_scale else None

    @torch.autocast("cuda", enabled=False)
    def _compute_cos_sin(self, pos: torch.Tensor, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute RoPE sines and cosines.

        Returned in same dtype as feature tensor.
        """
        assert pos.shape[-1] == self._pos_dim, (
            f"The dimensionality of the coordinates RoPE was initialized for "
            f"({self._pos_dim}) must match those of the given coordinates "
            f"({pos.shape[-1]})."
        )
        angles = self._angles[self._angle_idx].unsqueeze(0) * pos[..., self._pos_idx]


        # Apply learnable basis scale
        angles = angles * self._alpha if self._alpha is not None else angles

        cos = angles.cos().to(dtype)
        sin = angles.sin().to(dtype)
        return cos, sin

    @torch.autocast("cuda", enabled=False)
    def cos_sin(self, pos: torch.Tensor, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """Cosine and sine values for `pos`, in `feat`'s dtype, without touching the cache.

        Args:
            feat: Feature tensor used for its dimensionality and dtype.
            pos: Positions used to compute the RoPE matrices.
        """
        if self._alpha is None:
            with torch.no_grad():
                return self._compute_cos_sin(pos, dtype)
        return self._compute_cos_sin(pos, dtype)

    @torch.autocast("cuda", enabled=False)
    def cache_cos_sin(self, pos: torch.Tensor, dtype) -> None:
        """Cache cosine and sine values for later use.

        Args:
            feat: Feature tensor used for its dimensionality and dtype.
            pos: Positions used to compute the RoPE matrices.
        """
        self._cache = self.cos_sin(pos, dtype)

    @staticmethod
    def apply(feat: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Rotate `feat` with cos/sin that broadcast against its shape.

        Channel pairs never straddle a head boundary while the per-head width is even, so this may
        be applied either before or after the head split.
        """
        interleaved_feat = torch.stack([-feat[..., 1::2], feat[..., ::2]], dim=-1).flatten(-2)
        return cos * feat + sin * interleaved_feat

    def apply_precomputed(self, feat: torch.Tensor) -> torch.Tensor:
        """Apply RoPE with precomputed sines and cosines.

        Args:
            feat: (..., C)

        Returns:
            Pairwise rotated position-aware features (..., C).
        """
        assert self._cache
        return self.apply(feat, *self._cache)

    @property
    def is_cached(self) -> bool:
        """Check if precomputed cache exists."""
        return self._cache is not None

    def clear_cache(self) -> bool:
        del self._cache
        self._cache = None

    def forward(self, feat: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Apply RoPE to features given positions.

        The computation is efficient and exploits the sparisity of the RoPE matrices.

        Args:
            feat (..., dim)
            pos: (..., pos_dim)

        Returns:
            Features with positional embeddings (..., dim)
        """
        cos, sin = self._compute_cos_sin(feat, pos)

        # Efficient unpacking exploiting sparsity of the 2x2 block matrix structure in RoPE.
        # Truncate to the actual dimensionality of the model. The first position dimensions get
        # more positional embedding dimensions if truncation happens.
        # Warning: Don't do cat([...], -1) here because then there is no interleaving.
        interleaved_feat = torch.stack([-feat[..., 1::2], feat[..., ::2]], dim=-1).flatten(-2)
        return cos * feat + sin * interleaved_feat


if __name__ == "__main__":
    _feat = torch.randn(2, 1000, 64 * 3)
    _pos = torch.randn(2, 1000, 3)
    r = RoPE(64 * 3, 3)
    r(_feat, _pos)
    print(f"{r(_feat, _pos).shape=}")
    _feat = torch.randn(2, 1000, 128)
    _pos = torch.randn(2, 1000, 3)
    r = RoPE(128, 3)
    print(f"{r(_feat, _pos).shape=}")
    r.cache_cos_sin(_feat, _pos)
    assert torch.allclose(r(_feat, _pos), r.apply_precomputed(_feat))
