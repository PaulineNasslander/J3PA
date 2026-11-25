"""Helpers shared by the probes and evaluations (augmentation, FOV filtering, loss averaging, dataloader cleanup)."""

import gc

import torch

from constants import YAW_ROTATION_AUG_RANGE, XYZ_SLICE


def _flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


def _calc_mavg_loss(prev_loss, new_loss, mavg_factor=0.9):
    """Calculate moving average of loss."""
    if prev_loss is None:
        return new_loss
    return mavg_factor * prev_loss + (1 - mavg_factor) * new_loss


def _cleanup_dataloader(dataloader):
    """Properly cleanup a DataLoader and free memory."""
    try:
        if hasattr(dataloader, "_iterator") and dataloader._iterator is not None:
            dataloader._iterator = None
    except Exception:
        pass

    del dataloader
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# pylint: disable=too-many-locals
def rotate_cloud(point_cloud, annotations):
    """Rotate entire point cloud and annotations around z axis.

    Params:
    point_cloud: tensor of shape (B,N,C) with padded point clouds
    annotations: Dict with 'center' (B, M, 3), 'size' (B, M, 3), 'quats' (B, M, 4)

    Returns:
    rotated_point_cloud: tensor of shape (B,N,3) with rotated points
    rotated_annotations: Dict with rotated 'center', 'size', 'quats'
    R: rotation matrices applied to point clouds (B, 3, 3)

    NOTE: padded points and annotations are handled correctly since rotation on a 0 vector is 0
    """
    B, _, C = point_cloud.shape
    xyz_slice = slice(0, 3) if C == 3 else XYZ_SLICE

    points_3d = point_cloud[..., xyz_slice]
    device = points_3d.device

    # Random rotation angles [-0.3, 0.3] rad
    rotation_radians = torch.empty(B, device=point_cloud.device).uniform_(*YAW_ROTATION_AUG_RANGE)
    cos_a, sin_a = torch.cos(rotation_radians), torch.sin(rotation_radians)

    # Z-axis rotation matrices (3, 3)
    R = torch.zeros(B, 3, 3, device=device)
    R[:, 0, 0], R[:, 0, 1] = cos_a, -sin_a
    R[:, 1, 0], R[:, 1, 1] = sin_a, cos_a
    R[:, 2, 2] = 1.0

    rotated_points = points_3d @ R.transpose(-1, -2)
    point_cloud[..., xyz_slice] = rotated_points

    # Rotate annotations
    rotated_annotations = annotations.copy()
    rotated_annotations["center"] = annotations["center"] @ R.transpose(-1, -2)

    # Convert angles to quaternions [w,x,y,z] for Z-rotation
    half_angles = rotation_radians * 0.5
    rot_quats = torch.stack(
        [
            torch.cos(half_angles),
            torch.zeros_like(rotation_radians),
            torch.zeros_like(rotation_radians),
            torch.sin(half_angles),
        ],
        -1,
    )

    # Quaternion multiplication: q_new = q_rot * q_old
    q = annotations["quats"]
    rotated_annotations["quats"] = torch.stack(
        [
            rot_quats[:, None, 0:1] * q[..., 0:1] - rot_quats[:, None, 3:4] * q[..., 3:4],
            rot_quats[:, None, 0:1] * q[..., 1:2] + rot_quats[:, None, 3:4] * q[..., 2:3],
            rot_quats[:, None, 0:1] * q[..., 2:3] - rot_quats[:, None, 3:4] * q[..., 1:2],
            rot_quats[:, None, 0:1] * q[..., 3:4] + rot_quats[:, None, 3:4] * q[..., 0:1],
        ],
        -1,
    ).squeeze(-2)

    return point_cloud, rotated_annotations, R


def limit_pc_to_fov(point_cloud, field_of_view):
    """Limit boxes to field of view.

    point_cloud: (B, N, C) point features with optional leading time, padded with zeros
    field_of_view: tensor with fov information of shape (B, 2) horizontal, vertical in degrees

    Only correct when y points forward (ZOD's lidar frame): the azimuth is measured from +y. Datasets with x
    forward must return a 360 degree FOV, which skips this, and cut to their annotated region themselves.
    TODO: Handle this in cleaner way.

    returns: filtered point_cloud (B, N, C) where points outside fov are removed (set to zero)
    """
    if (field_of_view == 360.0).all():
        return point_cloud
    
    pos = point_cloud[..., XYZ_SLICE]

    # limit points to field of view
    hor_fov, vert_fov = field_of_view[:, 0].unsqueeze(1), field_of_view[:, 1].unsqueeze(1) # B, 
    x, y, z = pos[:, :, 0], pos[:, :, 1], pos[:, :, 2] # B,N
    azimuth = torch.atan2(x, y) * 180 / torch.pi # B,N
    elevation = torch.atan2(z, torch.sqrt(x**2 + y**2)) * 180 / torch.pi # B,N
    in_fov_mask = (azimuth >= -hor_fov / 2) & (azimuth <= hor_fov / 2) & (elevation >= -vert_fov / 2) & (elevation <= vert_fov / 2) # B,N
    #apply mask to point cloud, points outside fov are set to zero and will be filtered out later
    point_cloud[~in_fov_mask] = 0.0

    return point_cloud
