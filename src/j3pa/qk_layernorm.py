"""QK-LayerNorm of the encoder attention: a LayerNorm over the head dim of every query and key.

The learnable parameters always live in the attention layer's two nn.LayerNorm modules (`_q_ln` and `_k_ln`) and the
implementations below only read them. The state dict, the initialisation and the parameter order (which the optimizer
state follows) are therefore the same for every implementation, so a checkpoint trained with one loads into a model
using any other.

Every implementation computes what "torch" does: a LayerNorm over the last dim in float32 with the modules' weight,
bias and eps, rounded to the input dtype, with gradients for the inputs and the parameters. Only rounding may differ.
q and k are views of the QKV projection that are contiguous in the last dim only, and the norm also runs under no_grad
(teacher) and in checkpoint recomputation. To add an implementation, register a function with the signature of
`_qk_layernorm_torch` in QK_LAYERNORM_IMPLS.

`triton_fused_rope` selects joint QK-LayerNorm/RoPE in PointSelfAttn. Norm-only calls, including register tokens,
use the ordinary Triton LayerNorm.
"""

import functools
from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F

QKLayerNormFn = Callable[[torch.Tensor, torch.Tensor, nn.LayerNorm, nn.LayerNorm], tuple[torch.Tensor, torch.Tensor]]


def _qk_layernorm_torch(
    q: torch.Tensor, k: torch.Tensor, q_ln: nn.LayerNorm, k_ln: nn.LayerNorm
) -> tuple[torch.Tensor, torch.Tensor]:
    """Plain PyTorch: autocast runs the LayerNorms in float32 and the results are cast back to the input dtype."""
    return q_ln(q).to(q.dtype), k_ln(k).to(k.dtype)


def _layer_norm_float32(x: torch.Tensor, weight: torch.Tensor | None, bias: torch.Tensor | None, eps: float) -> torch.Tensor:
    return F.layer_norm(x.float(), x.shape[-1:], weight, bias, eps).to(x.dtype)


def _qk_layernorm_math(q, k, q_weight, q_bias, k_weight, k_bias, q_eps: float, k_eps: float):
    return _layer_norm_float32(q, q_weight, q_bias, q_eps), _layer_norm_float32(k, k_weight, k_bias, k_eps)


@functools.cache
def _compiled_qk_layernorm_math():
    # Compiled on first use, so torch.compile is never touched with the other implementations. The number of
    # attention groups changes every batch, and with dynamic shapes one graph serves all of them and all stages.
    return torch.compile(_qk_layernorm_math, dynamic=True)


def _qk_layernorm_compiled(
    q: torch.Tensor, k: torch.Tensor, q_ln: nn.LayerNorm, k_ln: nn.LayerNorm
) -> tuple[torch.Tensor, torch.Tensor]:
    """The same math fused by torch.compile, without separate casts and copies of the strided q and k views."""
    return _compiled_qk_layernorm_math()(q, k, q_ln.weight, q_ln.bias, k_ln.weight, k_ln.bias, q_ln.eps, k_ln.eps)


def _qk_layernorm_triton(
    q: torch.Tensor, k: torch.Tensor, q_ln: nn.LayerNorm, k_ln: nn.LayerNorm
) -> tuple[torch.Tensor, torch.Tensor]:
    """Custom Triton kernels (j3pa/qk_layernorm_triton.py) that read the strided views once, without float32 copies."""
    # Imported on first use, so that the other implementations never need Triton.
    from j3pa.qk_layernorm_triton import layer_norm

    return layer_norm(q, q_ln.weight, q_ln.bias, q_ln.eps), layer_norm(k, k_ln.weight, k_ln.bias, k_ln.eps)


QK_LAYERNORM_IMPLS: dict[str, QKLayerNormFn] = {
    "torch": _qk_layernorm_torch,
    "compiled": _qk_layernorm_compiled,
    "triton": _qk_layernorm_triton,
    "triton_fused_rope": _qk_layernorm_triton,
}
