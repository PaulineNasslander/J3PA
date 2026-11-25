"""ChadamW > AdamW

AdamW but without weight decay where you don't want it.
"""

from typing import Dict, Iterable, List, Tuple

import torch


def _decay_groups(
    named_params: Iterable[Tuple[str, torch.nn.Parameter]],
    lr: float,
    weight_decay: float,
    no_decay_keywords: List | None = None,
) -> List[Dict]:
    """Determine groups of parameters with- and without weight decay.

    We don't want weight decay on biases, layernorm affines, nor learnable embeddings.

    Args:
        params: parameters of the model which would normally be passed to instantiate an optimizer.
        lr, weight_decay: Obvious.
        no_decay_keywords: Extra keywords to search for in layer names. The ones flagged will not
                           get weight decay.
    """
    # Collect the parameters we will decay and those we won't decay.
    decay = []
    no_decay = []

    for n, p in named_params:
        if not p.requires_grad:
            continue

        # No decay for any 1D parameter is the silver bullet. Catches any bias, layernorm weight,
        # as well as typical nn.Parameter's one would manually initialize for manual embeddings.
        if p.ndim <= 1:
            no_decay.append(p)
            continue

        # Fall back on custom keywords in case they were provided.
        if no_decay_keywords and any(kw.lower() in n.lower() for kw in no_decay_keywords):
            no_decay.append(p)
            continue

        # Weight decay everything not explicitly filtered.
        decay.append(p)

    groups = [
        {"params": decay, "weight_decay": weight_decay, "lr": lr},
        {"params": no_decay, "weight_decay": 0.0, "lr": lr},
    ]
    return groups


class ChadamW(torch.optim.AdamW):
    """AdamW but without weight decay where you don't want it."""

    def __init__(self, named_params, lr, weight_decay, *args, no_decay_keywords=None, **kwargs):
        """Initialize.

        Args:
            same as torch.optim.AdamW
            no_decay_keywords: Extra keywords for not applying decay. If the name of any parameter
                               or layer includes one of these keywords it will not include any
                               weight decay.
        """
        # Check whether params is named parameters of not.
        decay_lr_wd_groups = _decay_groups(named_params, lr, weight_decay, no_decay_keywords=no_decay_keywords)
        super().__init__(decay_lr_wd_groups, *args, **kwargs)


if __name__ == "__main__":
    from torch import nn

    _model = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 10))
    ChadamW(_model.parameters(), lr=1e-4, weight_decay=1e-2)
