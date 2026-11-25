"""Parameterless upcasting of encoder features to the input point resolution.

`evaluation.od_head.ODHead.parameterless_upcast` stops at stage 0, i.e. the
voxel-pooled cloud the encoder actually attends over. Final segmentation numbers
have to be reported per *input* point against the official labels, so this module
carries the same nearest-voxel broadcast one level further, through the encoder's
initial pooling.

No parameters and no interpolation are involved: every point simply receives the
features of the voxel it fell into at each stage, concatenated across stages.
"""

import torch


def composed_up_indices(up_indices: list[torch.Tensor]) -> list[torch.Tensor]:
    """Indices mapping each input point to its voxel at every encoder stage.

    The encoder returns one hop per stage: ``up_indices[0]`` sends input points to
    stage-0 voxels, ``up_indices[i]`` sends stage-(i-1) voxels to stage-i voxels.
    Composing them left to right gives a direct input -> stage-i lookup.

    Args:
        up_indices: Per-stage upsample indices as returned by `PointTransformer`.

    Returns:
        List of (N_input,) long tensors, one per stage.
    """
    composed = [up_indices[0]]
    for stage_idx in range(1, len(up_indices)):
        composed.append(up_indices[stage_idx][composed[-1]])
    return composed


def upcast_to_input_resolution(
    features: list[torch.Tensor],
    up_indices: list[torch.Tensor],
    pos_at_input: torch.Tensor,
) -> torch.Tensor:
    """Broadcast multi-stage features onto the input points and concatenate them.

    Args:
        features: Per-stage features, ``features[i]`` of shape (M_i, C_i).
        up_indices: Per-stage upsample indices from the encoder.
        pos_at_input: (N, 3) xyz of the input points, appended as extra channels
            exactly as the stage-0 upcast does.

    Returns:
        (N, sum(C_i) + 3) tensor in the dtype of ``features[0]``.
    """
    assert len(features) == len(up_indices), (
        f"Expected one upsample index per stage, got {len(up_indices)} for {len(features)} stages"
    )
    num_points = pos_at_input.shape[0]
    total_dim = sum(f.shape[-1] for f in features) + 3

    out = torch.empty(num_points, total_dim, dtype=features[0].dtype, device=features[0].device)

    col = 0
    # Deepest stage first, matching the column order of the stage-0 upcast so a
    # head trained against one can be pointed at the other.
    composed = composed_up_indices(up_indices)
    for feat, idx in zip(reversed(features), reversed(composed), strict=True):
        dim = feat.shape[-1]
        assert idx.shape[0] == num_points, (
            f"Composed index has {idx.shape[0]} entries but there are {num_points} input points"
        )
        out[:, col : col + dim] = feat[idx]
        col += dim
    out[:, col:] = pos_at_input
    return out
