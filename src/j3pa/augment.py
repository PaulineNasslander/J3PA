import torch
from torch.nn import functional as F

from constants import FULL_ROTATION_AUG_RANGE, POINT_FEATURE_DIM, XYZ_SLICE, REFL_INDEX, MAX_TRANSLATION, MAX_TIME_SHIFT, USE_TIME, REFL_DROP_PROB, REFL_NOISE_STD, USE_SCALE_AUG, SCALE_AUG_RANGE

@torch.autocast("cuda", enabled=False)
def sample_translations(batch_shape, device):
    return (2*torch.rand(*batch_shape, 3, device=device, dtype=torch.float32) - 1) * MAX_TRANSLATION

@torch.autocast("cuda", enabled=False)
def sample_time_shifts(batch_shape, device):
    return (2 * torch.rand(*batch_shape, 1, device=device, dtype=torch.float32) - 1) * MAX_TIME_SHIFT

@torch.autocast("cuda", enabled=False)
def sample_gaussian_noise(xyz: torch.Tensor, std: float = 0.02):
    return torch.randn_like(xyz, dtype=torch.float32) * std

@torch.autocast("cuda", enabled=False)
def sample_rotation_matrices(batch_shape, device):
    """Sample random rotation matrices with arbitrary batch shape.

    Args:
        batch_shape: (X, Y, Z, ...)
        device: obvious

    Returns:
        (*batch_shape, 3, 3)
    """

    # Sample rotation angles: shape (*batch_shape, 3)
    angles = torch.stack(
        [
            torch.empty(batch_shape, device=device, dtype=torch.float32).uniform_(*FULL_ROTATION_AUG_RANGE[0]),
            torch.empty(batch_shape, device=device, dtype=torch.float32).uniform_(*FULL_ROTATION_AUG_RANGE[1]),
            torch.empty(batch_shape, device=device, dtype=torch.float32).uniform_(*FULL_ROTATION_AUG_RANGE[2]),
        ],
        dim=-1,
    )

    cos_x, cos_y, cos_z = torch.cos(angles[..., 0]), torch.cos(angles[..., 1]), torch.cos(angles[..., 2])
    sin_x, sin_y, sin_z = torch.sin(angles[..., 0]), torch.sin(angles[..., 1]), torch.sin(angles[..., 2])

    # Allocate rotation matrices
    rot_shape = (*batch_shape, 3, 3)

    Rx = torch.zeros(rot_shape, device=device, dtype=torch.float32)
    Ry = torch.zeros(rot_shape, device=device, dtype=torch.float32)
    Rz = torch.zeros(rot_shape, device=device, dtype=torch.float32)

    # Fill Rx
    Rx[..., 0, 0] = 1
    Rx[..., 1, 1] = cos_x
    Rx[..., 1, 2] = -sin_x
    Rx[..., 2, 1] = sin_x
    Rx[..., 2, 2] = cos_x

    # Fill Ry
    Ry[..., 0, 0] = cos_y
    Ry[..., 0, 2] = sin_y
    Ry[..., 1, 1] = 1
    Ry[..., 2, 0] = -sin_y
    Ry[..., 2, 2] = cos_y

    # Fill Rz
    Rz[..., 0, 0] = cos_z
    Rz[..., 0, 1] = -sin_z
    Rz[..., 1, 0] = sin_z
    Rz[..., 1, 1] = cos_z
    Rz[..., 2, 2] = 1

    # Batched matrix multiply (broadcasted)
    R = Rz @ Ry @ Rx

    return R

@torch.autocast("cuda", enabled=False)
def rotate_and_translate(
    point_cloud: torch.Tensor,
    batch_indices: torch.Tensor,
    rotation_matrices: torch.Tensor = None,
    translations: torch.Tensor = None,
    time_shifts: torch.Tensor = None,
) -> torch.Tensor:
    """Rotate point cloud for data augmentation.

    Args:
        point_cloud: (N, 3) tensor of point cloud coordinates
            or (N, POINT_FEATURE_DIM) tensor with intensity and optional time
        batch_indices: (N,) tensor of batch indices for each point in the point cloud
        rotation_matrices: (B, 3, 3) rotation matrices to apply to the point cloud coordinates
        translations: (B, 3) translation vectors to apply to the point cloud coordinates
        time_shifts: (B, 1) time shift values to apply to the point cloud timestamps
    Returns:
        rotated and translated point cloud of same shape.
    """
    assert point_cloud.shape[-1] in (3, POINT_FEATURE_DIM)
    C = point_cloud.shape[-1]
    xyz_slice = slice(0, 3) if C == 3 else XYZ_SLICE
    ptcl = point_cloud.clone()
    xyz = ptcl[..., xyz_slice]

    if rotation_matrices is not None:
        # (B, 3, 3) -> (N, 3, 3)
        rotation_matrices = rotation_matrices[batch_indices]

        # (N, 3) @ (N, 3, 3) -> (N, 3)
        xyz = (rotation_matrices @ xyz.unsqueeze(-1)).squeeze(-1)

    if translations is not None:
        # (B, 3) -> (N, 3)
        translations = translations[batch_indices]

        # (N, 3) + (N, 3) -> (N, 3)
        xyz += translations

    ptcl[..., xyz_slice] = xyz

    if time_shifts is not None:
        assert C > 3, "Only 3 point feature dims, XYZ-only inputs (e.g. query points) carry no time column."
        assert USE_TIME, "USE_TIME must be True if time_shifts are provided."
        times = ptcl[..., 0]
        times += time_shifts[batch_indices, 0]
        ptcl[..., 0] = times

    return ptcl

@torch.autocast("cuda", enabled=False)
def elastic_warp(xyz: torch.Tensor, batch_indices: torch.Tensor, cell_size: float, std: float = 0.05, smoothening_steps: int = 2, batch_size: int | None = None):
    """Apply elastic deformation to the point cloud.
    Args:
        xyz: (N, 3) tensor of point cloud coordinates
        batch_indices: (N,) tensor of batch indices for each point
        cell_size: Size of the cells in the noise grid.
        std: Standard deviation of the noise to apply to each point.
        smoothening_steps: Number of times to apply average pooling to the noise field for smoothening.
    Returns:
        (N, 3) tensor of warped point cloud coordinates
    """
    # TODO Global min/max leads to a large grid and inefficiencies.l This function is generally quite bad for large batch sizes.
    min_xyz = xyz.amin(dim=0)
    max_xyz = xyz.amax(dim=0)
    # The extra cells are added to ensure a smooth interpolation of the noise field at the edges.
    # We will smoothen with kernel = 3 smoothening_steps times and we will lose a cell at each edge
    # ever time 
    # We add an additional + 1 so that we can define the min/max corners as lying in the middle of the edge
    # cells of the interpolation grid.
    # 1e-5 adds safety in the degenerate case where we have all points at the same x, y, or z and could get 0.
    # Note that we execute augmentations in fp32 normally and not bfloat16 so this value is OK.
    ncells_xyz = ((max_xyz - min_xyz + 1e-5) / cell_size).ceil().int() + 1
    padded_ncells_xyz = ncells_xyz + 2 * smoothening_steps

    # (B, 3, X, Y, Z) channels-first and batch dimension for using the torch pooling
    # functionality.
    B = batch_size if batch_size is not None else batch_indices.max().item() + 1
    # each sample in the batch gets its own noise field, so that they are deformed differently
    # even though the grids are the same size.
    warp_field = torch.randn(B, 3, *padded_ncells_xyz, device=xyz.device, dtype=xyz.dtype)

    for _ in range(smoothening_steps):
        # Important to use padding = 0 to line up with n_cells_xyz above.
        warp_field = F.avg_pool3d(warp_field, kernel_size=3, stride=1, padding=0)

    # Compute coordinates of each lidar point in [-1, 1] on the noise grid for interpolation.
    # Count number of cells. Falls in range [0, n_cells_xyz - 2] with n_cells_xyz - 1 options (because of the +1) in the latter.
    ptcl_coord = (xyz - min_xyz) / cell_size
    ptcl_coord /= (ncells_xyz - 1)
    ptcl_coord *= 2
    ptcl_coord -= 1

    # grid_sample funktion defines the point sampling coordinates 
    # opposite from the axis order. It's like they want us to have bugs  >:-|
    ptcl_coord = ptcl_coord.flip(-1)
    

    # Sample from warp field for all batches at once: (B, 3, N, 1, 1)
    N = xyz.shape[0]

    # Expand ptcl_coord to (B, N, 1, 1, 3) - all batches see all points
    ptcl_coord_batched = ptcl_coord.unsqueeze(0).expand(B, -1, -1).unsqueeze(2).unsqueeze(2)


    # Sample from warp field: (B, 3, N, 1, 1)
    point_warp = F.grid_sample(warp_field, ptcl_coord_batched, align_corners=True, mode="bilinear")

    # Reshape to (B, 3, N) and select the correct batch for each point.
    # Do not use squeeze() without dims because B=1 would drop the batch axis.
    point_warp = point_warp.squeeze(-1).squeeze(-1)  # (B, 3, N)
    # Use batch_indices to gather the correct warp for each point: (N, 3)
    point_warp = point_warp[batch_indices, :, torch.arange(N, device=xyz.device)]
    return xyz + point_warp * std


@torch.no_grad()
@torch.autocast("cuda", enabled=False)
def sample_xy_flip_signs(batch_shape, device):

    # 2 Because we only consider flipping x or y
    flips = torch.rand(*batch_shape, 2, device=device, dtype=torch.float32)

    # Flip chance is 50% on each axis. This means 25% chance of no flip, 25% of flipping both,
    # and 25% chance of only flipping x or y.
    flips = torch.where(flips <= 0.5, 1.0, -1.0)

    # Z always stays identical.
    return torch.cat([flips, torch.ones(*batch_shape, 1, device=device, dtype=torch.float32)], dim=-1)


@torch.no_grad()
@torch.autocast("cuda", enabled=False)
def apply_common_augmentations(ptcl, batch_indices, batch_size: int | None = None):
    xyz = ptcl[..., XYZ_SLICE]

    # Elastic deformation
    xyz = elastic_warp(xyz, batch_indices, cell_size=0.5, std=0.2, smoothening_steps=2, batch_size=batch_size)

    B = batch_size if batch_size is not None else batch_indices.max().item() + 1
    # Scale the reflectivity in range [0.9, 1.1]
    random_refl_scale = torch.rand(B, device=ptcl.device, dtype=torch.float32) * 0.2 + 0.9
    ptcl[..., REFL_INDEX] *= random_refl_scale[batch_indices]
    if REFL_NOISE_STD > 0:
        ptcl[..., REFL_INDEX] += torch.randn_like(ptcl[..., REFL_INDEX]) * REFL_NOISE_STD
    ptcl[..., REFL_INDEX] = ptcl[..., REFL_INDEX].clamp(0.0, 255.0)
    if REFL_DROP_PROB > 0:
        drop = torch.rand(B, device=ptcl.device) < REFL_DROP_PROB
        ptcl[..., REFL_INDEX] = ptcl[..., REFL_INDEX].masked_fill(drop[batch_indices], 0.0)

    if USE_SCALE_AUG:
        # Rescale the entire point clouds in [0.95, 1.05]
        min_scale, max_scale = SCALE_AUG_RANGE
        random_space_scale = torch.rand(B, device=xyz.device, dtype=torch.float32) * (max_scale - min_scale) + min_scale
        xyz *= random_space_scale[batch_indices].unsqueeze(-1)

    # Noise with a 2cm standard deviation. Is applied pointwise so no need to worry about flattened batch.
    xyz += sample_gaussian_noise(xyz, std=0.02)

    #flip_signs = sample_xy_flip_signs((B,), xyz.device)[batch_indices]
    #xyz *= flip_signs

    # Update the point cloud with the augmented coordinates
    ptcl[..., XYZ_SLICE] = xyz

    return ptcl
