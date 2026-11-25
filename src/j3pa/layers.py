"""Common layers that can be reused."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def log_voxel_size_coordinate(voxel_size: torch.Tensor, default_voxel_size: float, voxel_size_range: tuple[float, float] | None) -> torch.Tensor:
    """log2 voxel size mapped linearly so that the range spans [-1, 1]. Without a range: log2(voxel_size / default)."""
    if voxel_size_range is None or voxel_size_range[0] == voxel_size_range[1]:
        return (voxel_size.float() / default_voxel_size).log2()
    low, high = math.log2(voxel_size_range[0]), math.log2(voxel_size_range[1])
    return (voxel_size.float().log2() - (low + high) / 2) / ((high - low) / 2)


class VoxelSizeEmbedding(nn.Module):
    """Fourier features of the normalised log voxel size (see log_voxel_size_coordinate) followed by a small MLP."""

    def __init__(self, out_dim: int, num_freqs: int = 8, hidden_dim: int = 64, out_init_std: float | None = None):
        """
        Args:
            out_init_std: Std of the output layer weights, zeros if 0. None keeps the default init.
        """
        super().__init__()
        # 0.25*pi .. 2*pi: at most two periods over [-1, 1], so the embedding stays smooth in the voxel size.
        self.register_buffer("_freqs", math.pi * 2 ** torch.linspace(-2, 1, num_freqs), persistent=False)
        self._mlp = nn.Sequential(nn.Linear(2 * num_freqs + 1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim))
        if out_init_std is not None:
            nn.init.normal_(self._mlp[-1].weight, std=out_init_std)
            nn.init.zeros_(self._mlp[-1].bias)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """(B,) normalised log voxel sizes -> (B, out_dim)."""
        u = u.float().unsqueeze(-1)
        phase = u * self._freqs
        return self._mlp(torch.cat([u, phase.sin(), phase.cos()], dim=-1))


def build_mlp(in_dim: int, hidden_dim: int, out_dim: int, num_layers, prenorm: bool):
    in_dims = [in_dim] + [hidden_dim] * (num_layers - 1)
    out_dims = [hidden_dim] * (num_layers - 1) + [out_dim]
    layers = []
    for i, (in_d, out_d) in enumerate(zip(in_dims, out_dims)):
        layers.append(nn.Linear(in_d, out_d))
        if i < len(in_dims) - 1:
            layers.append(nn.GELU())
    if prenorm:
        layers = [nn.LayerNorm(in_dim)] + layers
    return nn.Sequential(*layers)


class FiLM(nn.Module):
    def __init__(self, condition_dim: int, encoder_dim: int, use_encoder_in_condition: bool = False):
        super().__init__()

        # TODO We could go for a pure embedding FiLM when we don't mix in the encoder.
        # Affine multiplier and bias need a total of 2*encoder_dim
        self._use_encoder = use_encoder_in_condition
        in_dim = condition_dim + encoder_dim if self._use_encoder else condition_dim
        self._mlp = build_mlp(in_dim=in_dim, hidden_dim=encoder_dim*2, out_dim=encoder_dim*2, num_layers=2, prenorm=False)
        self._encoder_dim = encoder_dim

        # Initialize FiLM to start at alpha = identity and beta = 0
        nn.init.zeros_(self._mlp[-1].weight)
        with torch.no_grad():
            b = self._mlp[-1].bias
            b[:encoder_dim].fill_(1.0)
            b[encoder_dim:].zero_()

    def forward(
        self,
        condition_embed: torch.Tensor,
        encoder_out: torch.Tensor,
        encoder_batch_indices: torch.Tensor
    ) -> torch.Tensor:
        assert encoder_out.shape[-1] == self._encoder_dim
        assert encoder_batch_indices.shape[0] == encoder_out.shape[0]

        # For efficiency we apply batch indices only after summing the embeddings and making the FiLM if we can.
        mlp_in = torch.cat([condition_embed[encoder_batch_indices], encoder_out], dim=-1) if self._use_encoder else condition_embed
        mlp_out = self._mlp(mlp_in).view(*mlp_in.shape[:-1], 2, self._encoder_dim)
        mlp_out = mlp_out[encoder_batch_indices] if not self._use_encoder else mlp_out

        # TODO Memory layout is not very efficient.
        alpha, beta = mlp_out.unbind(-2)
        return alpha*encoder_out + beta


class ContextCombiner(nn.Module):
    """Fuses the decoder output and down-path features of each stage into one context feature."""

    def __init__(self, enc_stage_dims, checkpoint=True):
        super().__init__()
        self._checkpoint = checkpoint
        self._mlps = nn.ModuleList([build_mlp(2*ed, 2*ed, ed, num_layers=3, prenorm=False) for ed in enc_stage_dims])

    @staticmethod
    def _stage_forward(mlp, up_feat, down_feat):
        return mlp(torch.cat([up_feat, down_feat], dim=-1))

    def forward(self, enc_up_feat, enc_down_feat):
        assert len(enc_up_feat) == len(enc_down_feat) == len(self._mlps)
        if self._checkpoint:
            return [
                checkpoint(self._stage_forward, mlp, up, down, use_reentrant=False, preserve_rng_state=True)
                for mlp, up, down in zip(self._mlps, enc_up_feat, enc_down_feat, strict=True)
            ]

        return [self._stage_forward(mlp, up, down) for mlp, up, down in zip(self._mlps, enc_up_feat, enc_down_feat, strict=True)]
