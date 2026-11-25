"""Embeddings of the mask sizes."""

import torch
from torch import nn
from j3pa.layers import build_mlp

class MaskEmbeddingTable(nn.Module):
    """Factorized embedding table along x, y, and z."""
    def __init__(self, min_size: int, max_size: int, dim: int, init_std: float | None = 0.02):
        super().__init__()
        self._range = max_size - min_size + 1

        # Separate embeddings in x, y, z
        self._embedding_table = nn.Embedding(self._range*3, dim)
        if init_std is not None:
            nn.init.normal_(self._embedding_table.weight, std=init_std)
        self._min_size = min_size
        self._max_size = max_size

    def forward(self, mask_sizes):
        """
        Args:
            mask_sizes (N, 3).
        
        Returns:
            mask embeddings (N, C).
        """
        assert torch.all(mask_sizes <= self._max_size) and torch.all(mask_sizes >= self._min_size)
        assert mask_sizes.dtype == torch.long

        # We need to offset y indices by range and z by 2*range to use the full embedding table.
        offsets = torch.arange(mask_sizes.shape[-1], device=mask_sizes.device) * self._range

        # (N, 3) -> (N, 3, D) -> sum(1) -> (N, D)
        # Avoid modifying mask_sizes in place because it may mess with the tensor if it is to be used later.
        factorized_emb = self._embedding_table(mask_sizes - self._min_size + offsets)
        return factorized_emb.sum(-2)

class MaskEmbeddingMLP(nn.Module):
    """Simple MLP embedding.
    
    The advantage compared to an embedding table could be that it has an implicit prior of similar
    mask sizes modulating outputs similarly because of the smoothness of the MLP.
    """
    def __init__(self, dim: int):
        super().__init__()
        self._mlp = build_mlp(3, dim, dim, num_layers=2, prenorm=False)

    def forward(self, mask_sizes):
        """
        Args:
            mask_sizes (N, 3)

        Returns:
            mask embeddings (N, C)
        """
        # Logarithm for slow variation prior and therefore training stability.
        assert torch.all(mask_sizes >= 1.0)
        return self._mlp(mask_sizes.log2())


class MaskEmbedding(nn.Module):
    """One embedding for each stage."""
    def __init__(self, stage_dims: list[int], min_sizes: list[int], max_sizes: list[int], embedding_table=True, init_std: float | None = 0.02):
        assert len(stage_dims) == len(min_sizes)
        assert len(min_sizes) == len(max_sizes)
        super().__init__()
        if embedding_table:
            self._embeddings = nn.ModuleList([MaskEmbeddingTable(min_s, max_s, d, init_std=init_std) for min_s, max_s, d in zip(min_sizes, max_sizes, stage_dims)])
        else:
            self._embeddings = nn.ModuleList([MaskEmbeddingMLP(d) for d in stage_dims])

    def forward(self, mask_sizes: list[torch.Tensor]):
        assert len(mask_sizes) == len(self._embeddings)
        return [emb(ms) for emb, ms in zip(self._embeddings, mask_sizes)]
