"""FFN with SwiGLU."""

from collections.abc import Callable

import torch
from torch import nn


class GLUFFN(nn.Module):
    """Gated Linear Unit feedforward network."""

    def __init__(self, dim: int, activation: Callable, ff_mult: float = (8 / 3), nearest_multiplier: int = 64):
        """Initialize.

        Args:
            obvious.
            nearest_multiplier: Nearest multiplier for rounding up the hidden dimension.
        """
        super().__init__()
        assert ff_mult > 1, "Nobody bottlenecks the FFN."
        if (dim * ff_mult) % nearest_multiplier == 0:
            hidden_dim = int(dim * ff_mult)
        else:
            hidden_dim = int((dim * ff_mult // nearest_multiplier) * nearest_multiplier + nearest_multiplier)

        # Layernorm can not absorb the bias here because we expand the dimensionality.
        self._lin_1 = nn.Linear(dim, hidden_dim, bias=True)
        self._lin_2 = nn.Linear(dim, hidden_dim, bias=True)
        self._lin_3 = nn.Linear(hidden_dim, dim, bias=True)
        self._activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (..., C)

        Returns:
            x: (..., C)
        """
        return self._lin_3(self._activation(self._lin_1(x)) * self._lin_2(x))


class SwiGLUFFN(GLUFFN):
    """GLU with swish (SiLU) activation.

    Standard in publicly-discussed large LLM implementations.
    """

    def __init__(self, *args, **kwargs):
        """Initialize."""
        super().__init__(*args, activation=nn.SiLU(), **kwargs)


if __name__ == "__main__":
    _x = torch.randn(3, 1000, 128)
    _swiglu = SwiGLUFFN(dim=128, nearest_multiplier=128)
    print(f"{_swiglu(_x).shape=}")
