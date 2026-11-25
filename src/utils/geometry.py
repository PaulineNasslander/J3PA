import torch


def quaternion_to_yaw(quat: torch.Tensor) -> torch.Tensor:
    """
    Extract yaw angle (rotation around z-axis) from quaternion.
    Args:
        quat: Tensor of shape (..., 4), format [w, x, y, z]
    Returns:
        yaw: Tensor of shape (...,)
    """
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    # Yaw formula for quaternion [w, x, y, z]
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    return yaw


def yaw_to_quaternion(yaw):
    """Yaw to Quaternion.

    Args:
        yaw: tensor of shape (N, 1)

    Returns:
        quat: tensor of shape (N, 4)
    """
    w = torch.cos(yaw / 2)
    x = torch.zeros_like(yaw)
    y = torch.zeros_like(yaw)
    z = torch.sin(yaw / 2)
    return torch.cat([w, x, y, z], dim=-1)


def apply_transform(transform: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
    """
    Apply a coordinate transformation on the input data.

    Args:
        transform: (..., M, M) transformation matrix
        data: (..., D, M) data matrix to be transformed

    Return:
        transformed data (..., D, M)
    """
    return torch.matmul(transform, data.unsqueeze(-1)).squeeze(-1)


def invert_4x4_transform(matrix: torch.Tensor) -> torch.Tensor:
    """Inverts a 4x4 rotation+translation matrix.

    Args:
        matrix: of shape (..., 4, 4)

    Return:
        Inverted matrix (..., 4, 4)
    """
    # pylint: disable=protected-access
    rotation = matrix[..., 0:3, 0:3]
    translation = matrix[..., 0:3, 3]
    rotation_inverse = rotation.transpose(-1, -2)
    translation_for_inverse = apply_transform(-1 * rotation_inverse, translation)  # - R^T t
    bottom = matrix[..., 3:4, 0:4]
    assert (bottom[..., :, 0:3].abs() < 1e-5).all()
    assert ((bottom[..., :, 3:4] - 1.0).abs() < 1e-5).all()
    inverted_matrix = torch.cat(
        [
            torch.cat([rotation_inverse, translation_for_inverse.unsqueeze(-1)], dim=-1),
            bottom,
        ],
        dim=-2,
    )
    return inverted_matrix


def apply_affine_4x4_transform(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """Apply a 4x4 affine transformation to 3D points.

    Args:
        points: Tensor of shape (..., 3) with last dim as (x,y,z)
        transform: Tensor of shape (..., 4, 4) representing the affine transformation matrix.
    Returns:
        transformed_points: Transformed points, same shape as input points.
    """
    assert (transform[..., 3, :3] == torch.zeros(3, device=transform.device)).all()
    assert (transform[..., 3, 3] == torch.ones(1, device=transform.device)).all()

    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]
    transformed_points = torch.matmul(rotation, points.unsqueeze(-1)).squeeze(-1) + translation
    return transformed_points
