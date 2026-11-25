import torch
from torch.nn import functional as F
import itertools

from constants import (
    XYZ_SLICE,
    MASK_PROB,
    MIN_MASK_SIZES,
    MAX_MASK_SIZES,
    MASK_INFO_WEIGHTING,
    WEIGHT_QUERY_COUNT_BY_MASKED_COUNT,
    EXP_NUM_SUPERVISION_QUERIES_PER_SAMPLE,
    SAME_GRID,
    NORM_TEACHER,
    USE_CONTEXT_WINDOW,
    CONTEXT_MIN_FRAC,
    CONTEXT_MAX_FRAC,
    USE_TIME,
    ENC_RETURN_DOWN_PATH,
    DEEPEST_MID_TARGET,
    ROTATE_CONTEXT_WINDOW,
    ROTATE_MASK_SAMPLING,
    SHIFT_MASK_SAMPLING,
)
from j3pa.augment import rotate_and_translate, sample_rotation_matrices, sample_translations, sample_time_shifts
from j3pa.mask_sampler import sample_cascaded_masks, clip_mask, nonempty_queries, random_yaw_matrix
from j3pa.predictor import Predictor
from j3pa.layers import ContextCombiner
from j3pa.mask_modulator import MaskEmbedding
from j3pa.pt_transf import PointTransformer, get_grid_idx

# Per sample in batch.
# These are not guarantees on point content but rather a fallback for undoing the context window.
# If the number of queries or points falls below this threshold. If these are triggered w 
MIN_POINT_SAFEGUARD = 500
MIN_QUERY_SAFEGUARD = 10

# We try to make sure our representation has 
MIN_DEEPEST_VOXELS = 2


@torch.autocast("cuda", enabled=False)
def _context_window(x, bi, min_frac=CONTEXT_MIN_FRAC, max_frac=CONTEXT_MAX_FRAC, batch_size=None):
    """
        x : (N, 3)
        bi: (N)

    Returns:
        window_min: (B, 3)
        window_max: (B, 3)
    """

    assert 0 < min_frac <= 1.0
    B = batch_size if batch_size is not None else bi.max().item() + 1
    # Randomly rotate the point cloud in which we sample the crops so that the teacher orientation can not leak from here.
    # TODO Perhaps this should be a yaw-uniform rotation.
    if ROTATE_CONTEXT_WINDOW:
        R = random_yaw_matrix(B, device=x.device)
        x = (R[bi] @ x.unsqueeze(-1)).squeeze(-1)
    else:
        R = None
    min_pos = torch.empty((B, 3), device=x.device, dtype=x.dtype)
    min_pos.scatter_reduce_(0, bi.unsqueeze(-1).expand(-1,3), x, reduce="amin", include_self=False)
    max_pos = torch.empty((B, 3), device=x.device, dtype=x.dtype)
    max_pos.scatter_reduce_(0, bi.unsqueeze(-1).expand(-1,3), x, reduce="amax", include_self=False)

    # Fraction of the range to keep. Will be from min_frac to max_frac.
    fracs = torch.rand(B, 3, device=x.device, dtype=x.dtype)*(max_frac - min_frac) + min_frac
    
    # From bottom corner to upper corner up to the aspect ratio. We add offsets below.
    window_min = min_pos
    window_max = min_pos + (max_pos - min_pos)*fracs

    # Uniformly drawn offsets. Always positive since 0 draws a box at the minimum.
    # Center range from (max - min)*(1 - fracs)
    offsets = torch.rand(B, 3, device=x.device, dtype=x.dtype)*(max_pos - min_pos)*(1 - fracs)

    # Note that min_pos is now implicitly changed but we don't use it below anyway.
    window_min += offsets
    window_max += offsets

    return window_min, window_max, R

def loss_scale(dim, max_dim, opt: str):
    if opt == "sqrt":
        scale = (float(dim)**0.5)/(max_dim**0.5)
    elif opt == "lin":
        scale = float(dim)/max_dim
    elif opt == "equal":
        scale = 1
    else:
        assert False, "Option doesn't exist."
    return scale

class PointJEPA:
    def __init__(
        self,
        student_encoder: PointTransformer,
        teacher_encoder: PointTransformer,
        predictor: Predictor,
        student_mask_embedding: MaskEmbedding | None = None,
        exp_num_queries_per_sample=EXP_NUM_SUPERVISION_QUERIES_PER_SAMPLE,
        context_combiner: ContextCombiner | None = None,
        fixed_teacher_voxel_size: bool = False,
    ):
        """fixed_teacher_voxel_size: The teacher always encodes at its default voxel size (VOXEL_SIZE)."""
        assert student_encoder.num_stages == teacher_encoder.num_stages
        assert abs(student_encoder.voxel_size - teacher_encoder.voxel_size) < 1e-12
        assert abs(student_encoder.voxel_size - predictor.voxel_size) < 1e-12
        if predictor.condition_prediction:
            assert student_mask_embedding is not None
        assert not (SAME_GRID and fixed_teacher_voxel_size), "Teacher and student grids differ with a fixed teacher voxel size."
        self._fixed_teacher_voxel_size = fixed_teacher_voxel_size

        # Initialize with prebuilt modules to enable DDP only on student and predictor
        self._student = student_encoder
        self._teacher = teacher_encoder
        self._predictor = predictor
        self._student_mask_embedding = student_mask_embedding
        self._context_combiner = context_combiner
        if ENC_RETURN_DOWN_PATH:
            assert context_combiner is not None

        # We make the assumption that for every stage we are trying to predict we will sample masks.
        # Masks from deeper stages are cascaded up to prevent information leakage but not predicted
        # at shallower stages.
        pred_stage_offset = student_encoder.num_stages - predictor.num_stages
        self._stages_to_predict = [pred_stage_offset + i for i in range(predictor.num_stages)]

        for param in self._teacher.parameters():
            param.requires_grad = False

        self._exp_num_queries = exp_num_queries_per_sample

    @property
    def fixed_teacher_voxel_size(self):
        return self._fixed_teacher_voxel_size

    def __call__(self, ptcl, batch_idx, validation=False, batch_size=None, num_sweeps=None, voxel_size=None):
        """voxel_size: Optional (B,) encoder input voxel size for each sample. Defaults to the encoder voxel size.

        Keep default encoder scales implicit so their fixed-scale divisions use Python floats.
        """
        B = batch_size if batch_size is not None else batch_idx.max().item() + 1
        student_voxel_size = voxel_size
        if voxel_size is None:
            voxel_size = torch.full((B,), self._student.voxel_size, device=ptcl.device, dtype=torch.float32)
        teacher_voxel_size = None if self._fixed_teacher_voxel_size else student_voxel_size
        # Teacher
        with torch.no_grad():
            # Sample separete random rotation matrices for student and teachers data augmentation
            # WIP: Sample different rotations and translations for each sample in the batch
            R_t = sample_rotation_matrices((B,), device=ptcl.device) if not validation else None
            T_t = sample_translations((B,), device=ptcl.device) if not validation else None
            S_t = sample_time_shifts((B,), device=ptcl.device) if (not validation and USE_TIME) else None

            teach_ptcl = rotate_and_translate(ptcl, batch_idx, R_t, T_t, S_t)
            teach_xyz = teach_ptcl[..., XYZ_SLICE]
            if SAME_GRID:
                grid_divisor = self._teacher.voxel_size if teacher_voxel_size is None else teacher_voxel_size[batch_idx]
                grid_idx_teacher, _ = get_grid_idx(teach_xyz, batch_idx, cell_size=grid_divisor, batch_size=B)
            else:
                grid_idx_teacher = None
            if ENC_RETURN_DOWN_PATH:
                teach_feats, teach_pos, teach_batch_indices, teach_up_indices, teach_regs, teach_down_feats = self._teacher(
                    teach_ptcl, batch_idx, grid_idx_teacher, batch_size=B, return_down_path=True, voxel_size=teacher_voxel_size,
                )
            else:
                teach_feats, teach_pos, teach_batch_indices, teach_up_indices, teach_regs = self._teacher(
                    teach_ptcl, batch_idx, grid_idx_teacher, batch_size=B, voxel_size=teacher_voxel_size,
                )
                teach_down_feats = None
            
        # We have the option to return the entire embedding of the teacher for analysis purposes. Such analisys can also require the corresponding batch indices.
        all_teacher_feats = (teach_feats, teach_batch_indices, teach_pos, teach_up_indices) if validation else None

        # Now mask the og point cloud:
        # As the point order is not shuffled we can generate the mask in the teacher position space
        # and apply the same mask to the student point cloud in the student position space.
        # WIP Pass teacher batch indices here
        # Mask block sizes are in (stage) voxels of the teacher's voxel size.
        student_input_mask, query_points, teach_feat_supervision, teach_batch_indices_supervision, mask_sizes, teach_feat_down_supervision = self._mask_and_sample(
            teach_feats, teach_pos, teach_up_indices, teach_batch_indices, batch_size=B, teach_feats_down=teach_down_feats, voxel_size=teacher_voxel_size,
        )

        # Release all the teacher states we wil no longer use before we run the student forward and backward.
        if not validation:
            del teach_feats
            del teach_down_feats
            del teach_pos
            del teach_up_indices
            del teach_batch_indices

        #mask_sizes_supervision = [ms[bi] for ms, bi in zip(mask_sizes, teach_batch_indices_supervision)]
        query_batch_indices = teach_batch_indices_supervision

        mask_sizes = [ms.long() for ms in mask_sizes]  # Needed for embedding tables.
        teach_regs_supervision = [teach_regs[stage] for stage in self._stages_to_predict] if teach_regs else None

        if USE_CONTEXT_WINDOW:
            # Masking to the context window needs to happen here otherwise up index logic breaks.
            # The downside of doing it here is that we can lose some of our sampled queries.
            window_min, window_max, R_window = _context_window(teach_xyz, batch_idx, CONTEXT_MIN_FRAC, CONTEXT_MAX_FRAC, batch_size=B)
            with torch.autocast("cuda", enabled=False):
                if R_window is not None:
                    teach_xyz_window_rot = (R_window[batch_idx] @ teach_xyz.unsqueeze(-1)).squeeze(-1)
                else:
                    teach_xyz_window_rot = teach_xyz
            point_mask = torch.all((teach_xyz_window_rot >= window_min[batch_idx]) & (teach_xyz_window_rot <= window_max[batch_idx]), dim=-1)
            if R_window is not None:
                del teach_xyz_window_rot
            combined_mask = point_mask & student_input_mask
            
            per_batch_count = torch.zeros(B, dtype=batch_idx.dtype, device=batch_idx.device)
            per_batch_count.scatter_reduce_(0, batch_idx, combined_mask.to(dtype=batch_idx.dtype), reduce="sum", include_self=True)
            content_safeguard = (per_batch_count < MIN_POINT_SAFEGUARD)

            # TODO This is bad because we unmask the full pointcloud.
            # Doing combined_mask | safeguard would unmask a full sample if the safeguard is truggerd (i.e
            # the whole batch index satisfies it. We combined this with the student_input_mask so that for
            # those clouds triggering the safeguard, any value that was true in the student_input_mask will be
            # true again. The combined_mask is never true where student_input_mask is false so at the triggered
            # indices we effectively get student_input_mask back.
            combined_mask = combined_mask | (content_safeguard[batch_idx] & student_input_mask)

            # TODO Deep-stage voxels may be under-resolved; consider constraining deformable offsets.
            # We could assert the maximum distance between points exceeding the voxel size of the deepest stage.
            # Fallback if not the case. We only care about K=2 case because we're too dumb to fix the general one.

            # We need to filter the query points to only those that are within the context window.
            # First we'll need to compute a collective safeguard across all stages: If any stage has too
            # few queries then we will need to trigger the safeguard everywhere. In the second loop we
            # will do the actual masking. If we do everything in one loop we can get situations where a later
            # stage would trigger a safeguard but we already masked queries in earlier stages.

            # (B,) A sample either trips the safeguard or not
            b_safeguard = torch.zeros(B, dtype=torch.bool, device=batch_idx.device)
            q_masks = []
            for i, (qf, qp, qbi) in enumerate(zip(teach_feat_supervision, query_points, query_batch_indices)):
                with torch.autocast("cuda", enabled=False):
                    if R_window is not None:
                        qp_window_rot = (R_window[qbi] @ qp.unsqueeze(-1)).squeeze(-1)
                    else:
                        qp_window_rot = qp
                q_mask = torch.all((qp_window_rot >= window_min[qbi]) & (qp_window_rot <= window_max[qbi]), dim=-1)
                if R_window is not None:
                    del qp_window_rot

                # If we already undid the context window we should also undo it for the queries.
                q_mask = q_mask | content_safeguard[qbi]

                # Important that this is zeros and not empty. When we mask out there is no longer a guarantee
                # that every sample gets a hit in the scatter below.
                per_batch_count = torch.zeros(B, dtype=qbi.dtype, device=qbi.device)
                per_batch_count.scatter_reduce_(0, qbi, q_mask.long(), reduce="sum", include_self=True)
                b_safeguard |= (per_batch_count < MIN_QUERY_SAFEGUARD)
                q_masks.append(q_mask)

            # Now we can apply the cascaded safeguard to all stages and their query masks.
            for i, (q_mask, qf, qp, qbi) in enumerate(zip(q_masks, teach_feat_supervision, query_points, query_batch_indices)):

                # We have to be careful unmasking queries. We can not simply apply the safeguard, because that
                # re-enables all queries which could now fall outside of the student context. To make this safe
                # we remove the context crop for cases where we had too few queries.                
                q_safeguard = b_safeguard[qbi]
                p_safeguard = b_safeguard[batch_idx]
                q_mask = q_mask | q_safeguard
                combined_mask = combined_mask | (p_safeguard & student_input_mask)

                teach_feat_supervision[i] = qf[q_mask]
                if teach_feat_down_supervision[i] is not None:
                    teach_feat_down_supervision[i] = teach_feat_down_supervision[i][q_mask]
                query_points[i] = qp[q_mask]
                query_batch_indices[i] = qbi[q_mask]

        else:
            combined_mask = student_input_mask

        # We should do ths here because otherwise we can not recover the full point cloud.
        del teach_xyz
        grid_idx_teacher = grid_idx_teacher[combined_mask] if SAME_GRID else None
        ptcl = ptcl[combined_mask]
        batch_idx = batch_idx[combined_mask]

        # Student
        R_s = sample_rotation_matrices((B,), device=ptcl.device) if not validation else None
        T_s = sample_translations((B,), device=ptcl.device) if not validation else None
        S_s = sample_time_shifts((B,), device=ptcl.device) if (not validation and USE_TIME) else None

        # WIP: Sample different rotations and translations for each sample in the batch
        ptcl = rotate_and_translate(ptcl, batch_idx, R_s, T_s, S_s)

        # Note that grid_idx_teacher will be None if SAME_GRID is False. This will result in the student computing its own grid indices
        # which means that the grids are not aligned between teacher and student.
        if ENC_RETURN_DOWN_PATH:
            student_feats, student_pos, student_batch_indices, _, student_regs, student_down_feats = self._student(
                ptcl, batch_idx, grid_idx_teacher, B, return_down_path=True, voxel_size=student_voxel_size,
            )
        else:
            student_feats, student_pos, student_batch_indices, _, student_regs = self._student(
                ptcl, batch_idx, grid_idx_teacher, B, voxel_size=student_voxel_size,
            )

        # filter student_feats and student_pos to only those at stages we predict at
        # We also need to apply the same layernorm to the student features at stages we pass to predictor.
        context_feats = [student_feats[stage] for stage in self._stages_to_predict]
        if ENC_RETURN_DOWN_PATH:
            # The deepest stage has no decoder, so without a mid target its down-path features are its output features.
            num_combined = len(context_feats) if DEEPEST_MID_TARGET else len(context_feats) - 1
            student_down_feats = [student_down_feats[stage] for stage in self._stages_to_predict[:num_combined]]
            context_feats = self._context_combiner(context_feats[:num_combined], student_down_feats) + context_feats[num_combined:]
        student_pos = [student_pos[stage] for stage in self._stages_to_predict]
        student_batch_indices = [student_batch_indices[stage] for stage in self._stages_to_predict]
        student_regs = [student_regs[stage] for stage in self._stages_to_predict] if student_regs else None
        assert len(mask_sizes) == len(student_batch_indices)
        if self._predictor.condition_prediction:
            student_mask_emb = self._student_mask_embedding(mask_sizes)
        else:
            student_mask_emb = None
        student_supervision_feats, student_supervision_regs = context_feats, student_regs
        #del student_feats
        #del student_regs

        # We need to undo the teacher augmentations first and then apply the student augmentations
        # so that we are making predictions in the student coordinate system.
        # For efficiency we can both undo the teacher rotation and apply the student rotation at the
        # same time. The translations have to be done separately
        # We need to do this under a super important disable autocast so that any transformations on the point coordinats keep percision float32.
        with torch.autocast("cuda", enabled=False):
            R_t_to_s = R_s @ (R_t.transpose(-2, -1)) if not validation else None
            neg_T_t = -T_t if not validation else None
            query_points = [
                rotate_and_translate(rotate_and_translate(qp, qbi, None, neg_T_t), qbi, R_t_to_s, T_s)
                for qp, qbi in zip(query_points, query_batch_indices) # noqa: B905
            ]

        # TODO mask teacher batch indices
        pred_feats = self._predictor(
            student_supervision_feats,
            student_pos,
            query_points,
            student_batch_indices,
            query_batch_indices,
            student_supervision_regs,
            condition_embeddings=student_mask_emb,
            num_sweeps=num_sweeps,
            voxel_size=voxel_size,
        )

        assert len(pred_feats) == len(teach_feat_supervision)
        assert all(pf.shape[0] == tf.shape[0] for pf, tf in zip(pred_feats, teach_feat_supervision))  # noqa: B905

        return (
            (pred_feats, teach_feat_supervision, teach_feat_down_supervision, all_teacher_feats, student_supervision_regs, teach_regs_supervision)
            if all_teacher_feats is not None
            else (pred_feats, teach_feat_supervision, teach_feat_down_supervision, student_supervision_regs, teach_regs_supervision)
        )

    @torch.autocast("cuda", enabled=False)
    def mask_loss(
        self,
        predict_feat: list[torch.Tensor],
        teacher_feat: list[torch.Tensor],
        dim_scaling: str = "sqrt",
        num_q_weight: bool = False,
        teacher_feat_down: list[torch.Tensor | None] | None = None,
        batch_size: int | None = None,
    ):
        """Stages with a second target (down-path features, or the mid-stage features of the deepest stage) predict
        [output, second target] and sum both L1 losses.

        With num_q_weight every stage loss is scaled by the fraction of the expected batch query budget that was kept,
        which needs the batch size.
        """
        assert not num_q_weight or batch_size is not None, "Weighting by the number of queries needs the batch size."
        assert len(predict_feat) == len(teacher_feat)
        teacher_feat_down = teacher_feat_down if teacher_feat_down is not None else [None] * len(teacher_feat)
        assert len(teacher_feat_down) == len(teacher_feat)

        # Targets: the outputs of all stages first, then the
        # second targets of the stages that have them.
        preds, targets, second_stages = [], [], []
        for sf, tf, tfd in zip(predict_feat, teacher_feat, teacher_feat_down, strict=True):
            C = tf.shape[-1]
            assert sf.shape[-1] == (2 * C if tfd is not None else C)
            preds.append(sf[..., :C].float())
            targets.append(tf)
        for stage, (sf, tf, tfd) in enumerate(zip(predict_feat, teacher_feat, teacher_feat_down, strict=True)):
            if tfd is not None:
                preds.append(sf[..., tf.shape[-1] :].float())
                targets.append(tfd)
                second_stages.append(stage)

        # Non-affine layernorm is used in V-JEPA and I-JEPA but not documented in their
        # papers. As in V-JEPA 2.1 every target level is normalized separately.
        targets = [F.layer_norm(t, normalized_shape=t.shape[-1:]) if NORM_TEACHER else t for t in targets]
        target_losses = [F.l1_loss(p, t, reduction="mean") for p, t in zip(preds, targets, strict=True)]

        num_stages = len(teacher_feat)
        stage_losses = list(target_losses[:num_stages])
        for stage, loss in zip(second_stages, target_losses[num_stages:], strict=True):
            stage_losses[stage] = stage_losses[stage] + loss

        total_loss = 0
        max_C = float(max(t.shape[-1] for t in targets[:num_stages]))
        for stage, sf in enumerate(predict_feat):
            loss = stage_losses[stage]
            if num_q_weight:
                # The query budget is per sample while the predictions hold the queries of the whole batch.
                frac_survived = sf.shape[0] / (EXP_NUM_SUPERVISION_QUERIES_PER_SAMPLE * batch_size)
                loss = loss * frac_survived

            loss = loss * loss_scale(targets[stage].shape[-1], max_C, dim_scaling)
            stage_losses[stage] = loss
            total_loss += loss
        return total_loss, stage_losses

    @torch.autocast("cuda", enabled=False)
    def reg_loss(self, student_regs, teach_regs, dim_scaling: str = "sqrt"):
        assert len(student_regs) == len(teach_regs)
        stage_losses = []
        total_loss = 0
        max_C = float(max(s.shape[-1] for s in student_regs))
        for sr, tr in zip(student_regs, teach_regs):  # noqa: B905
            # We tried applying a nonaffine norm here like in the main masking loss but it did not seem to help.
            # If anything, results slightly degrade.
            loss = F.l1_loss(sr, tr, reduction="mean") 
            loss = loss * loss_scale(sr.shape[-1], max_C, dim_scaling)
            stage_losses.append(loss)
            total_loss += stage_losses[-1]
        return total_loss, stage_losses

    def update_teacher(self, ema: float):
        """Teacher is EMA of student."""
        student = self._student.module if hasattr(self._student, "module") else self._student
        student_params, teacher_params = list(student.parameters()), list(self._teacher.parameters())
        assert len(student_params) == len(teacher_params)
        with torch.no_grad():
            # t <- t + (1 - ema)(s - t) == ema * t + (1 - ema) * s, in one fused multi-tensor kernel.
            torch._foreach_lerp_(teacher_params, student_params, 1 - ema)

    def _mask_and_sample(
        self, teach_feats, teach_pos, teach_up_indices, teach_batch_indices, batch_size=None, teach_feats_down=None, voxel_size=None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor | None]]:
        """Sample masks at each stage and get the masked voxel centers for supervision.

        We cascade the masks from deeper stages to shallower stages to prevent information leakage
        but only use the primary mask at each stage for supervision. We also sample the centers of
        the masked voxels that contain points (query_points) and the centers of the masked
        voxels that do not contain points (free_query_points) for supervision.

        Sampled query points are used as the queries for the predictor and we limit the number
        of them to max_num_supervision_queries.

        Args:
            voxel_size: Optional (B,) encoder input voxel size of each sample. Defaults to the encoder voxel size.

        Returns: 
            total_pointwise_mask: (N,) boolean mask indicating which of the N points in the
                original point cloud.
            query_points: list of (K_i, 3) centers of masked voxels that contain points
                for each predictive stage i.
            teach_feat_supervision: list of (K_i, D) teacher features for masked voxels that
                contain points for each predictive stage i.
            teach_batch_indices_supervision: list of (K_i) batch indices indicating which sample
                each of the masked teacher features belong to.
            stage_mask_block_sizes: mask block sizes per sample for each predictive stage i.
            query_feat_down: list of (K_i, D) teacher down-path features like teach_feat_supervision,
                None for stages without them or if teach_feats_down is not given.
        """

        first_pred_stage = self._student.num_stages - len(self._stages_to_predict)
        assert len(MASK_PROB) == len(self._stages_to_predict)
        total_pointwise_mask, stage_primary_masks, stage_mask_block_sizes = sample_cascaded_masks(
            teach_pos,
            teach_batch_indices,
            teach_up_indices,
            self._student.voxel_size if voxel_size is None else voxel_size,
            first_pred_stage,
            MIN_MASK_SIZES,
            MAX_MASK_SIZES,
            MASK_PROB,
            batch_size,
            mask_info_weighting=MASK_INFO_WEIGHTING,
            rotate_sampling=ROTATE_MASK_SAMPLING,
            shift_sampling=SHIFT_MASK_SAMPLING,
        )

        # We need to limit the number of primary masks to max_num_supervision_queries to avoid OOM.
        # Limit the number of zeros in each tensor independently to max_num_supervision_queries
        teach_batch_idx = [teach_batch_indices[i] for i in self._stages_to_predict]
        primary_masks = [
            # If no positions on the primary mask are true, we should turn one on. This could happen in case a single block
            # exceeded the information threshold when we were building masks.
            nonempty_queries(
                clip_mask(pm, bi, self._exp_num_queries, batch_size, weight_by_query=WEIGHT_QUERY_COUNT_BY_MASKED_COUNT),
                bi,
                batch_size,
            )
            for pm, bi in zip(stage_primary_masks, teach_batch_idx)
        ]


        # Filter query positions, GT features and batch indices to only those at stages we predict at.
        teach_pos = [teach_pos[i] for i in self._stages_to_predict]
        query_points = [tp[~pm] for tp, pm in zip(teach_pos, primary_masks)]

        teach_feats = [teach_feats[i] for i in self._stages_to_predict]
        query_feat = [tf[~pm] for tf, pm in zip(teach_feats, primary_masks)]

        teach_feats_down = [teach_feats_down[i] for i in self._stages_to_predict] if teach_feats_down is not None else [None] * len(primary_masks)
        query_feat_down = [None if tfd is None else tfd[~pm] for tfd, pm in zip(teach_feats_down, primary_masks)]

        query_batch_idx = [tbi[~pm] for tbi, pm in zip(teach_batch_idx, primary_masks)]

        # Keep one mask block size per sample; the caller expands them per query where needed.
        return total_pointwise_mask, query_points, query_feat, query_batch_idx, stage_mask_block_sizes, query_feat_down


    def trainable_named_parameters(self):
        return itertools.chain(
        # Initialize with prebuilt modules to enable DDP only on student and predictor
            self._student.named_parameters(),
            self._predictor.named_parameters(),
            self._student_mask_embedding.named_parameters() if self._student_mask_embedding is not None else [],
            self._context_combiner.named_parameters() if self._context_combiner is not None else [],
        )

    def trainable_parameters(self):
        return itertools.chain(
        # Initialize with prebuilt modules to enable DDP only on student and predictor
            self._student.parameters(),
            self._predictor.parameters(),
            self._student_mask_embedding.parameters() if self._student_mask_embedding is not None else [],
            self._context_combiner.parameters() if self._context_combiner is not None else [],
        )
