import torch

from j3pa.pt_transf import get_grid_idx
from j3pa.zorder import encode_zorder_fast_3d
from constants import MASK_INFO_WEIGHTING_VALID, POINT_CASCADE, FINE_VOXEL_CASCADE, ONE_STAGE_HOP, IN_STAGE_COUNT

@torch.autocast("cuda", enabled=False)
def random_so3_matrix(B, device: torch.device):
    # Unit Gaussians have no preferred direction.
    a = torch.randn(B, 3, 3, device=device, dtype=torch.float32)

    # QR decomposition gives us an orthonormal matrix Q (So that Q.T @ Q = Q.T @ Q = 1).
    # This is what we need for rotations.
    q, r = torch.linalg.qr(a)

    # The sign convention is easy to mishandle. The point of a uniform rotation is that for the
    # random rotation rot and a fixed rotation U  then U @ rot must also be a random SO(3) rotation. One
    # can show that if the sign of R in the QR decomposition is unique then UQ has the same distribution
    # as Q because UA has the same distribution as A above. Q is orthonormal and uniform under rotation
    # so it's an SO(3) rotation matrix.
    d = torch.diagonal(r, dim1=-2, dim2=-1)
    signs = torch.where(d < 0, -torch.ones_like(d), torch.ones_like(d))
    q = q * signs.unsqueeze(-2)

    # Exclude reflections by enforcinng enforcing a positive determinant.
    q[..., :, -1] *= torch.linalg.det(q).sign().unsqueeze(-1)
    return q

@torch.autocast("cuda", enabled=False)
def random_yaw_matrix(B, device: torch.device):
    angles = torch.rand(B, device=device, dtype=torch.float32) * (2 * torch.pi)
    R = torch.zeros(B, 3, 3, device=device, dtype=torch.float32)
    R[:, 0, 0] = angles.cos()
    R[:, 0, 1] = -angles.sin()
    R[:, 1, 0] = angles.sin()
    R[:, 1, 1] = angles.cos()
    R[:, 2, 2] = 1.0
    return R


def _batched_randperm(batch_idx):
    """Different from the randperm we are doing in the point transformer."""
    assert len(batch_idx.shape) == 1
    sort_keys = torch.rand(batch_idx.shape[0], device=batch_idx.device)
    
    # Since the sort_keys are always in [0, 1) the argsort will only operate within each sample.
    # Note that this numerically breaks when the batch size becomes about 10^23 because the added
    # small number will disappear. Would be impressive if we manage to make that a problem :D
    perm = torch.argsort(batch_idx.float() + sort_keys)

    # Compute the inverse of the permutation.
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.shape[0], device=perm.device)

    return perm, inv


def _sample_mask_information_weighted(
    linear_idx: torch.Tensor,
    batch_idx: torch.Tensor,
    drop_p: float,
    batch_size: int,
    point_count: torch.Tensor | None = None,
):
    """
        Args:
            linear_idx: (N,)
            batch_idx: (N,)
            drop_p: Dropped information content.
            batch_size: Precomputed batch size.
            point_count: Count of points or fine voxels in each voxel at this stage. This can not be recomputed here since
                         we don't have information about finer stages. The in-stage voxel count is used if nothing is provided.
    """
    assert point_count is None or torch.all(point_count >= 1)
    assert point_count is None or point_count.shape == batch_idx.shape
    assert linear_idx.shape == batch_idx.shape

    # (M,)
    unique_coarse_idx, up_idx = torch.unique(linear_idx, sorted=False, return_inverse=True)
    unique_batch_idx = torch.empty_like(unique_coarse_idx, dtype=batch_idx.dtype)
    # Nondeterministic but every location gets hit.
    unique_batch_idx[up_idx] = batch_idx

    # Count the total number of points or voxels at this stage and the number of points or finest
    # stage voxels at each subvoxel.
    # We can use empty in all the scatters below because we know we will hit every index.
    device = batch_idx.device
    sample_counts = torch.empty((batch_size,), device=device, dtype=torch.float32)
    block_count = torch.empty((unique_coarse_idx.shape[0],), device=device, dtype=torch.float32)
    if point_count is None:
        # Count the number of fine voxels in each coarse voxel.
        # (M,)
        sample_counts.scatter_reduce_(0, batch_idx, torch.ones_like(batch_idx, dtype=torch.float32), reduce="sum", include_self=False)
        block_count.scatter_reduce_(0, up_idx, torch.ones_like(up_idx, dtype=torch.float32), reduce="sum", include_self=False)
    else:
        sample_counts.scatter_reduce_(0, batch_idx, point_count, reduce="sum", include_self=False)
        block_count.scatter_reduce_(0, up_idx, point_count, reduce="sum", include_self=False)
        
    # Randomly permute the coarse voxels for each sample.
    # (M,)
    perm, inv_perm = _batched_randperm(unique_batch_idx)
    block_count = block_count[perm]

    # TODO Is this operation even needed given that each sample in the batch has the same index?
    unique_batch_idx = unique_batch_idx[perm]

    # Warning: This is only correct if each sample is contiguous (so samples acrosss the batch are not interleaved but appear in chunks).
    # (B,) - scalar  -> (B,)
    batch_offsets = sample_counts.cumsum(0) - sample_counts

    # Compute a running count up to sample_counts over all the blocks.
    # Should we do the cumsum exclusively or inclusively here? Currently in the insane case where
    # the first voxel exceeds the thresholds nothing gets masked.
    per_sample_cumsum = block_count.cumsum(0) - batch_offsets[unique_batch_idx]
    cutoffs = drop_p * sample_counts[unique_batch_idx]

    # In the extreme case of just 1 point or voxel we will not mask anything here.
    coarse_mask = per_sample_cumsum >= cutoffs

    # Undo the permutation
    coarse_mask = coarse_mask[inv_perm]

    return coarse_mask[up_idx]


def _grid_to_linear_idx(grid_idx: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
    num_bits = grid_idx.max().item().bit_length()
    z_sf_idx = encode_zorder_fast_3d(grid_idx, num_bits)
    max_sf_idx = z_sf_idx.max() + 1
    return max_sf_idx * batch_idx + z_sf_idx


def _cascade_information_counts(stage_up_idx: list, stage_sizes: list, per_point: bool = False, one_hop: bool = False) -> torch.Tensor:
    """Count the number of stage 0 voxels or points in each deeper voxel.
    
    Args:
        up_idx_stages: list of up idx tensors connecting the stages in the model graph.
        per_point: Whether to report the final counts in number of points (True) or number of voxels
                   in the finest stage (False).
        one_hop: Whether to only cascade one hope between each two stages instead of doing a full cascade. For the first stage
                 each voxel counts as 1.
    """
    assert len(stage_up_idx) == len(stage_sizes)
    assert not (one_hop and per_point), "Per point information contents can only be computed when cascading."

    # Use float32 since it will be easier to do fraction calculations later.
    device = stage_up_idx[0].device
    dtype = torch.float32
    if per_point:
        svc = torch.empty(stage_sizes[0], device=device, dtype=dtype)
        svc.scatter_reduce_(0, stage_up_idx[0], torch.ones_like(stage_up_idx[0], dtype=dtype), reduce="sum", include_self=False)
        stage_voxel_counts = [svc]
    else:
        stage_voxel_counts = [torch.ones(stage_sizes[0], device=device, dtype=dtype)]
    for s in range(1, len(stage_up_idx)):
        sc = torch.empty(stage_sizes[s], device=device, dtype=dtype)

        # We just count subvoxels and don't care about how much info these contain.
        if one_hop:
            sc.scatter_reduce_(0, stage_up_idx[s], torch.ones_like(stage_up_idx[s], dtype=dtype), reduce="sum", include_self=False)
        # We cascade the count all the way from the finest stage or the original voxels.
        else:
            sc.scatter_reduce_(0, stage_up_idx[s], stage_voxel_counts[-1], reduce="sum", include_self=False)


        stage_voxel_counts.append(sc)
    return stage_voxel_counts


@torch.no_grad()
def sample_cascaded_masks(
    stage_pos,
    stage_batch_idx,
    stage_up_idx,
    voxel_size,
    first_pred_stage,
    min_block_sizes,
    max_block_sizes,
    stage_drop_probs,
    batch_size,
    mask_info_weighting,
    rotate_sampling,
    shift_sampling,
):
    assert mask_info_weighting in MASK_INFO_WEIGHTING_VALID
    assert all(mi <= ma for mi, ma in zip(min_block_sizes, max_block_sizes))
    assert first_pred_stage >= 0
    num_stages = len(stage_pos)
    assert num_stages == len(stage_batch_idx) == len(stage_up_idx)
    assert all(sui.shape == sbi.shape for sui, sbi in zip(stage_up_idx[1:], stage_batch_idx[:-1]))
    assert len(stage_drop_probs) == (num_stages - first_pred_stage)
    assert batch_size is not None
    device = stage_batch_idx[-1].device

    # Compute information mass of every voxel across stages.
    # We need this for all stages since the cascade starts from the bottom.
    stage_sizes = [sp.shape[0] for sp in stage_pos]

    # How to count the voxel information.
    # There are many different options so there is some branching here.
    if mask_info_weighting == IN_STAGE_COUNT:
        # Each voxel counts equally towards the weight of blocks in the stage.
        stage_voxel_counts = [torch.ones(s, device=device, dtype=torch.float32) for s in stage_sizes]
    else:
        per_point = (mask_info_weighting == POINT_CASCADE)
        if not per_point:
            assert mask_info_weighting in (FINE_VOXEL_CASCADE, ONE_STAGE_HOP)
        one_hop = (mask_info_weighting == ONE_STAGE_HOP)
        stage_voxel_counts = _cascade_information_counts(stage_up_idx=stage_up_idx, stage_sizes=stage_sizes, per_point=per_point, one_hop=one_hop)
    # TODO: In principle we could do a microoptimization here to reduce peak memory by deleting the tensors for the stages we are not predicting at once
    # we completed the cascade. But it's probably unecessary complexity for now.

    # We will return the sizes used in the masking so they can be used to condition the predictor and optionally modulate the encoder or predictor output.
    # Make voxel_size (B, 1) so that it broadcasts with the (B, 3) block sizes. This line supports both a float and a (B,) tensor of sampelwise voxel sizes.
    voxel_size = torch.as_tensor(voxel_size, device=device, dtype=torch.float32).expand(batch_size).reshape(batch_size, 1)
    stage_voxel_sizes = [voxel_size * (2**(s + 1 + first_pred_stage)) for s, _ in enumerate(stage_drop_probs)]
    stage_block_sizes = [torch.randint(low=min_bs, high=max_bs + 1, size=(batch_size, 3), device=device, dtype=torch.float32) for _, min_bs, max_bs in zip(stage_drop_probs, min_block_sizes, max_block_sizes)]

    # Rotate the positions in each stage for the mask sampling randomly. This way the teacher rotation can not leak into
    # the student by realizing the masks are aligned to the teacher grid.
    if rotate_sampling:
        rot_stage_pos = []
        with torch.autocast("cuda", enabled=False):
            for sp, bi in zip(stage_pos, stage_batch_idx):
                so3 = random_yaw_matrix(batch_size, device=sp.device)
                sp = (so3[bi] @ sp.unsqueeze(-1)).squeeze(-1)
                rot_stage_pos.append(sp)
        stage_pos = rot_stage_pos

    primary_masks = []
    num_stages = len(stage_pos)
    # Start with a mask for the deepest stage. All ones
    total_mask = torch.ones(stage_batch_idx[-1].shape, device=device, dtype=torch.bool)
    for s in reversed(range(num_stages)):
        ms = s - first_pred_stage

        # Sample a new primary mask for predictive stages.
        if ms >= 0:
            # Must be computed before masking to align grids for all stages.
            # TODO This could be sped up by computing the minimum before masking to anchor the grid, then mask and then compute the indices.
            # TODO There can still be misalignment given we compute grid cells on stage_pos which are already aggregated. But does that matter given we
            # use the correct cascade anyway? If misalignment is OK do we have to precompute it or can we just mask first? Maybe more randomness is good.
            bi = stage_batch_idx[s]
            stage_block_size_metric = (stage_block_sizes[ms] * stage_voxel_sizes[ms])

            # Appaned the positions by the minimum - a random number up to half the mask block size so that mask blocks are never aligned if you
            # see the same scene twice.
            if shift_sampling:
                shifted_min = torch.empty(batch_size, 3, device=stage_pos[s].device, dtype=stage_pos[s].dtype)
                shifted_min.scatter_reduce_(0, bi.unsqueeze(-1).expand(bi.shape[0], 3), stage_pos[s], reduce="amin", include_self=False)
                shifted_min -= (stage_block_size_metric * torch.rand(*shifted_min.shape, device=shifted_min.device) / 2.0)
            else:
                shifted_min = None
            stage_block_size_metric = stage_block_size_metric[bi]
            stage_block_idx, _ = get_grid_idx(stage_pos[s], bi, stage_block_size_metric, batch_size, min_pos=shifted_min)


            masked_bi = bi[total_mask]
            stage_block_idx = stage_block_idx[total_mask]
            linear_idx = _grid_to_linear_idx(stage_block_idx, masked_bi)
            block_masks = _sample_mask_information_weighted(linear_idx, masked_bi, stage_drop_probs[ms], batch_size, stage_voxel_counts[s][total_mask])
            # TODO Micro-optimization here would be to release the stage_voxel_count memory after no longer needing it.
            # Need to use 1s here because we will predict at the 0s (masked) of the primary masks.
            primary_mask = torch.ones_like(bi, dtype=torch.bool)
            primary_mask[total_mask] = block_masks
            primary_masks.append(primary_mask)
            total_mask = total_mask & primary_mask

        # Cascade the total mask up to the next stage.
        total_mask = total_mask[stage_up_idx[s]]
    return total_mask, primary_masks[::-1], stage_block_sizes


# TODO Should this be deterministic?
# TODO Today we silently get far more queries for dense samples. Is that good or bad?
def clip_mask(mask: torch.Tensor, batch_idx: torch.Tensor, expected_per_sample: int, batch_size: int, weight_by_query: bool = False) -> torch.Tensor:
    """For limiting the number of predicted queries.

        Args:
            batch_idx: (N)
            mask: (N,)
            expected_per_sample: The expected maximum number of queries per sample.
            weight_by_queries: If True then samples with more masked positions at a given stage will get
                               a proportionally larger fraction of the queries.
    
        Returns:
            new mask with only max_per_sample True positions for each sample.
    """
    queries = ~mask

    # Now we need to permute the query locations so we don't bias where to sample.
    perm, inv_perm = _batched_randperm(batch_idx)
    queries = queries[perm]

    # We need a running count of samples that can be a query. We start by computing offsets of a total count
    # over the batch to get counts per sample.
    # Crucial to use torch.zeros here since it could have happened that a sample has no queries! (I.e. when a single block
    # already exceeded the information threshold above.)
    queries_per_sample = torch.zeros((batch_size,), device=batch_idx.device, dtype=torch.float32)
    queries_per_sample.scatter_reduce_(0, batch_idx, queries.float(), reduce="sum", include_self=False)
    csum = queries.cumsum(0)
    offsets = (queries_per_sample.cumsum(0) - queries_per_sample)[batch_idx]

    # Now the count within each sample is just the total running count, compensated by the batch offsets.
    running_count = csum - offsets

    # Keep queries only up to the cutoff and map them back onto the original mask.
    if weight_by_query:
        keep_frac = expected_per_sample * batch_size / queries_per_sample.sum()
        cutoffs = keep_frac * queries_per_sample
        cutoffs = cutoffs[batch_idx]
        cutoffs = torch.where(cutoffs < 1, 1, cutoffs)
        query_keep_mask = (running_count <= cutoffs)
    else:
        query_keep_mask = (running_count <= expected_per_sample)
    query_keep_mask = query_keep_mask[inv_perm]

    # 0 locations in original primary mask are potential queries, we limit the amount of them now
    return mask | (~query_keep_mask)


def nonempty_queries(mask: torch.Tensor, batch_idx: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
        Args:
            mask: (N,)
            batch_idx: (N,)
            batch_size: Obvious.
    """
    assert mask.shape == batch_idx.shape

    # Important not to use torch.empty here because we could have ended up with zero queries.
    query_count = torch.zeros((batch_size,), device=batch_idx.device, dtype=torch.float32)
    query_count.scatter_reduce_(0, batch_idx, (~mask).float(), reduce="sum")

    # Every batch is guaranteed to have at least one point.
    batch_count = torch.empty((batch_size,), device=batch_idx.device, dtype=batch_idx.dtype)
    batch_count.scatter_reduce_(0, batch_idx, torch.ones_like(batch_idx, dtype=batch_idx.dtype), reduce="sum", include_self=False)

    # Add at least one query for samples that have none (which will be in-context but that is OK).
    no_queries = (query_count < 0.5)

    # We could do random sampling or unmask the first entry.
    batch_offsets = batch_count.cumsum(0) - batch_count
    random_entries = (torch.rand(batch_size, device=batch_idx.device, dtype=torch.float32)*batch_count).floor().long()
    random_entries = torch.minimum(random_entries, (batch_count - 1).clamp(min=0))
    query_locs = batch_offsets + random_entries

    return mask.scatter(0, query_locs, torch.where(no_queries, torch.zeros_like(no_queries), mask[query_locs]))