"""Distill a pre-trained JEPA model into a smaller student model.

This script follows the same structure as the original JEPA training script with some differences:
1. It loads a pre-trained teacher model that is then kept frozen throughout distillation (no EMA used).
2. It randomly initializes a smaller student model i.e. student != teacher at initialization.
"""

import argparse
import gc
import os
import shutil
import tempfile
import time
from contextlib import ExitStack

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

import constants
from chadamw import ChadamW
from constants import (
    ACCUMULATED_STEPS,
    ADALN_ALPHA_PLUS_ONE,
    ADALN_ATTN_ALPHA,
    ADALN_FFN_ALPHA,
    ADALN_WEIGHT_DECAY,
    ALIGN_REGISTERS,
    BATCH_SIZE,
    CONDITION_ENCODER_ON_VOXEL_SIZE,
    CONDITION_PRED_ON_MASK_SIZE,
    CONDITION_PRED_ON_NUM_SWEEPS,
    CONDITION_PRED_ON_VOXEL_SIZE,
    DDP_BF16_GRAD_SYNC,
    DEEPEST_MID_TARGET,
    ENC_POSTNORM_POOL,
    ENC_QK_LAYERNORM,
    ENC_QK_LAYERNORM_IMPL,
    ENC_QKV_BIAS,
    ENC_RETURN_DOWN_PATH,
    ENC_SDPA_GATE,
    ENC_XSA,
    ENCODER_GROUP_SIZE,
    ENCODER_NUM_DOWN_LAYERS,
    ENCODER_NUM_HEADS,
    ENCODER_NUM_UP_LAYERS,
    ENCODER_STAGE_DIMS,
    EPOCHS,
    EPOCHS_DEBUG,
    FIXED_POS_NORM_VOXEL_SIZE,
    FIXED_TEACHER_VOXEL_SIZE,
    LOSS_DIM_SCALING,
    LOSS_NUM_Q_WEIGHT,
    MASK_EMBEDDING_INIT_STD,
    MASK_EMBEDDING_TABLE,
    MAX_LEARNING_RATE,
    MAX_MASK_SIZES,
    MAX_NUM_SWEEPS_EMBED,
    MIN_LEARNING_RATE,
    MIN_MASK_SIZES,
    NUM_PRED_LAYERS,
    NUM_REGISTERS,
    NUM_SWEEPS_EMBEDDING_TABLE,
    NUM_SWEEPS_QUERY_INPUT,
    NUM_WORKERS,
    PRED_CONDITIONING,
    PRED_HEAD_DIM,
    PRED_LEARNED_INTERPOLATION,
    PRED_NUM_NEIGHBORS,
    PRED_NUM_OFFSETS,
    PRED_OFFSET_SCALING,
    PRED_QK_ATTN,
    PRED_STAGE_DIMS,
    PREFETCH_FACTOR,
    REFERENCE_VOXEL_SIZE,
    REGISTER_LOSS_WEIGHT,
    REGISTER_REDUCTION,
    SEP_REG_VOXEL_NORMS,
    USE_ATTENTION_BIAS,
    USE_TIME,
    VOXEL_SIZE,
    VOXEL_SIZE_COND_INIT_STD,
    VOXEL_SIZE_POINT_FEATURE,
    VOXEL_SIZE_QUERY_INPUT,
    VOXEL_SIZE_RANGE,
    WARMUP_STEPS,
)
from evaluation.constants_loader import get_constant, load_constants_file
from j3pa.augment import apply_common_augmentations
from j3pa.deform_predictor import DeformPredictor
from j3pa.layers import ContextCombiner
from j3pa.mask_modulator import MaskEmbedding
from j3pa.point_jepa import PointJEPA
from j3pa.pt_transf import PointSelfAttn, PointTransformer
from online_segmntation.online_segm_eval import pointwise_cls_eval
from utils.multi_gpu_utils import (
    get_gpu_device,
    get_gpu_world_rank,
    init_multigpu_via_torch_ddp,
    is_main_process,
    torch_ddp_active,
    torch_ddp_run_on_master_only,
)
from utils.utils import EntropyMetrics, filter_and_flatten, get_dataloader

# Saved run constants may predate the teacher backend override.
DISTILL_TEACHER_QK_LAYERNORM_IMPL = getattr(constants, "DISTILL_TEACHER_QK_LAYERNORM_IMPL", "triton_fused_rope")

# Probes need a labelled dataset, which a Combined run takes from these in turn.
COMBINED_PROBE_DATASETS = ("NuScenes",)
# Combined runs have no labels, so JEPA validation uses this dataset's val split.
COMBINED_VAL_DATASET = "ZOD"


class HackyDDP(DDP):
    """Super annoying that you can't use class properties on your modules in DDP..."""

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def load_teacher_encoder(teacher_checkpoint_path, teacher_constants_path, *, qk_layernorm_impl=None):
    constants = load_constants_file(teacher_constants_path)
    checkpoint_use_time = constants.get("USE_TIME", True)
    if checkpoint_use_time != USE_TIME:
        raise ValueError(
            f"Checkpoint USE_TIME={checkpoint_use_time} does not match "
            f"src/constants.py USE_TIME={USE_TIME}. Match the setting before evaluation."
        )

    stage_dims = list(get_constant(constants, "ENCODER_STAGE_DIMS"))
    pred_stage_dims = list(get_constant(constants, "PRED_STAGE_DIMS"))
    if len(pred_stage_dims) != len(PRED_STAGE_DIMS):
        raise ValueError(
            f"The teacher supervised {len(pred_stage_dims)} stages but the student predicts {len(PRED_STAGE_DIMS)}; "
            f"only the supervised stages of the teacher are normalised."
        )
    # Mirrors train_jepa.py: the encoder only normalises (and upsamples to)
    # the stages the predictor actually supervises.
    out_norm_stages = list(
        reversed([len(stage_dims) - 1 - i for i in range(len(pred_stage_dims))])
    )
    # Runs predating these options were trained without them.
    return_down_path = constants.get("ENC_RETURN_DOWN_PATH", False)
    deepest_mid_target = constants.get("DEEPEST_MID_TARGET", False) and return_down_path
    if ENC_RETURN_DOWN_PATH and (not return_down_path or deepest_mid_target != DEEPEST_MID_TARGET):
        raise ValueError(
            f"The down-path targets need a teacher trained with them: teacher ENC_RETURN_DOWN_PATH={return_down_path}, "
            f"DEEPEST_MID_TARGET={deepest_mid_target} vs student {ENC_RETURN_DOWN_PATH}, {DEEPEST_MID_TARGET}."
        )
    if qk_layernorm_impl is None:
        qk_layernorm_impl = DISTILL_TEACHER_QK_LAYERNORM_IMPL
    if qk_layernorm_impl in (None, "saved"):
        qk_layernorm_impl = constants.get("ENC_QK_LAYERNORM_IMPL", "triton")
    qk_layernorm = get_constant(constants, "ENC_QK_LAYERNORM")

    depth = {
        "num_down_layers_per_stage": list(get_constant(constants, "ENCODER_NUM_DOWN_LAYERS")),
        "num_up_layers_per_stage": list(get_constant(constants, "ENCODER_NUM_UP_LAYERS")),
    }

    encoder = PointTransformer(
        stage_dims=stage_dims,
        **depth,
        num_heads_per_stage=list(get_constant(constants, "ENCODER_NUM_HEADS")),
        group_size=get_constant(constants, "ENCODER_GROUP_SIZE"),
        downsample_only=False,
        voxel_size=get_constant(constants, "VOXEL_SIZE"),
        num_registers=get_constant(constants, "NUM_REGISTERS"),
        register_reduction=get_constant(constants, "REGISTER_REDUCTION"),
        out_norm_stages=out_norm_stages,
        sep_reg_voxel_norms=get_constant(constants, "SEP_REG_VOXEL_NORMS"),
        use_attention_bias=get_constant(constants, "USE_ATTENTION_BIAS"),
        qk_layernorm=qk_layernorm,
        qk_layernorm_impl=qk_layernorm_impl,
        # Runs predating these options were trained without them.
        condition_voxel_size=constants.get("CONDITION_ENCODER_ON_VOXEL_SIZE", False),
        voxel_size_point_feature=constants.get("VOXEL_SIZE_POINT_FEATURE", False),
        pos_norm_voxel_size=constants.get("REFERENCE_VOXEL_SIZE") if constants.get("FIXED_POS_NORM_VOXEL_SIZE", False) else None,
        voxel_size_range=constants.get("VOXEL_SIZE_RANGE"),
        # Runs from before these options existed had neither, whatever the current defaults are. XSA has no
        # weights, so building it wrong would not be caught by loading the checkpoint.
        xsa=constants.get("ENC_XSA", False),
        pool_postnorm=constants.get("ENC_POSTNORM_POOL", False),
        qkv_bias=constants.get("ENC_QKV_BIAS", False),
        sdpa_gate=constants.get("ENC_SDPA_GATE", False),
        return_down_path=return_down_path,
        deepest_mid_target=deepest_mid_target,
    )

    state_dict = torch.load(teacher_checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    state_dict, pretrain_epoch = state_dict["teacher_encoder"], state_dict["epoch"]

    if state_dict["_init_proj.0.weight"].shape != encoder._init_proj[0].weight.shape:
        raise ValueError(
            f"{teacher_checkpoint_path} was trained with encoder input features that no longer exist (ENC_INPUT_VERSION="
            f"'new', or a run from before ENC_INPUT_VERSION existed). Only the raw point features + PE inputs remain."
        )
    encoder.load_state_dict(state_dict)
    print(
        f"Loaded teacher encoder from {teacher_checkpoint_path} "
        f"(pretraining epoch {pretrain_epoch})"
    )
    attention_layers = [layer for layer in encoder.modules() if isinstance(layer, PointSelfAttn)]
    fused_layers = sum(
        layer._q_ln is not None and layer._rope is not None and layer._qk_layernorm_impl == "triton_fused_rope"
        for layer in attention_layers
    )
    print(
        f"Teacher QK-LayerNorm: enabled={qk_layernorm}, backend={qk_layernorm_impl}, "
        f"fused_rope_layers={fused_layers}/{len(attention_layers)}"
    )
    return encoder


def main(args):
    writer = SummaryWriter(log_dir=os.path.join(args.output_directory, "tensorboard"))

    version = "mini" if args.debug else "full"
    epochs = EPOCHS_DEBUG if args.debug else EPOCHS
    encoder_stage_dims = ENCODER_STAGE_DIMS if not args.mini_model else [32, 64]
    # With its single predictor stage, the mini model has no up connections.
    encoder_num_down_layers = ENCODER_NUM_DOWN_LAYERS if not args.mini_model else [1, 2]
    encoder_num_up_layers = ENCODER_NUM_UP_LAYERS if not args.mini_model else [0]
    encoder_num_heads_per_stage = ENCODER_NUM_HEADS if not args.mini_model else [1, 2]
    pred_stage_dims = PRED_STAGE_DIMS if not args.mini_model else [64]
    encoder_group_size = ENCODER_GROUP_SIZE if not args.mini_model else 256
    voxel_size = VOXEL_SIZE if not args.mini_model else 1
    voxel_size_range = VOXEL_SIZE_RANGE if not args.mini_model else None
    reference_voxel_size = REFERENCE_VOXEL_SIZE if not args.mini_model else voxel_size
    pos_norm_voxel_size = reference_voxel_size if FIXED_POS_NORM_VOXEL_SIZE else None
    batch_size, num_workers, prefetch_factor = (
        BATCH_SIZE if not args.debug else 2,
        NUM_WORKERS if not args.debug else 4,
        PREFETCH_FACTOR if not args.debug else 1,
    )
    load_num_sweeps = CONDITION_PRED_ON_NUM_SWEEPS or NUM_SWEEPS_QUERY_INPUT
    train_loader, train_sampler = get_dataloader(
        train_flag=True,
        version=version,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        random_crop=True,
        shuffle=True,
        drop_last=True,
        dataset_type=args.dataset,
        load_frame_id=True,
        load_num_sweeps=load_num_sweeps,
        aggregate_sweeps=True,
    )
    val_dataset = COMBINED_VAL_DATASET if args.dataset == "Combined" else args.dataset
    val_loader, _ = get_dataloader(
        train_flag=False,
        version=version,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        random_crop=False,
        shuffle=False,
        drop_last=False,
        # Annotations are unused, but loading them pins validation to the key sweep instead of a random reference.
        load_annotations=True,
        load_frame_id=False,
        dataset_type=val_dataset,
        load_num_sweeps=load_num_sweeps,
    )
    # Keep metadata alive for the run; some datasets have millions of metadata entries and frequent collection is costly.
    gc.collect()
    gc.freeze()

    out_norm_stages = list(reversed([len(encoder_stage_dims) - 1 - i for i in range(len(pred_stage_dims))]))
    print(f"{out_norm_stages=}")
    student_enc = PointTransformer(
        stage_dims=encoder_stage_dims,
        num_down_layers_per_stage=encoder_num_down_layers,
        num_up_layers_per_stage=encoder_num_up_layers,
        num_heads_per_stage=encoder_num_heads_per_stage,
        group_size=encoder_group_size,
        downsample_only=False,
        voxel_size=voxel_size,
        num_registers=NUM_REGISTERS,
        register_reduction=REGISTER_REDUCTION,
        out_norm_stages=out_norm_stages,
        sep_reg_voxel_norms = SEP_REG_VOXEL_NORMS,
        use_attention_bias=USE_ATTENTION_BIAS,
        qk_layernorm=ENC_QK_LAYERNORM,
        qk_layernorm_impl=ENC_QK_LAYERNORM_IMPL,
        xsa=ENC_XSA,
        sdpa_gate=ENC_SDPA_GATE,
        pool_postnorm=ENC_POSTNORM_POOL,
        qkv_bias=ENC_QKV_BIAS,
        return_down_path=ENC_RETURN_DOWN_PATH,
        condition_voxel_size=CONDITION_ENCODER_ON_VOXEL_SIZE,
        voxel_size_point_feature=VOXEL_SIZE_POINT_FEATURE,
        pos_norm_voxel_size=pos_norm_voxel_size,
        voxel_size_range=voxel_size_range,
        voxel_size_cond_init_std=VOXEL_SIZE_COND_INIT_STD,
        deepest_mid_target=DEEPEST_MID_TARGET,
    )
   
    encoder_dim_pred_stages = encoder_stage_dims[-len(pred_stage_dims) :] 
    if CONDITION_PRED_ON_MASK_SIZE:
        student_mask_embed = MaskEmbedding(
            stage_dims=pred_stage_dims,
            min_sizes=MIN_MASK_SIZES,
            max_sizes=MAX_MASK_SIZES,
            embedding_table=MASK_EMBEDDING_TABLE,
            init_std=MASK_EMBEDDING_INIT_STD,
        )
    else:
        student_mask_embed = None
    # Predicted stages with a second target: the decoded ones (down path) and, with a mid target, the deepest one.
    num_second_targets = (len(pred_stage_dims) if DEEPEST_MID_TARGET else len(pred_stage_dims) - 1) if ENC_RETURN_DOWN_PATH else 0

    context_combiner = (
        ContextCombiner(encoder_dim_pred_stages[:num_second_targets], checkpoint=True) if ENC_RETURN_DOWN_PATH else None
    )

    teacher_enc = load_teacher_encoder(
        args.teacher_checkpoint, args.teacher_constants, qk_layernorm_impl=args.teacher_qk_layernorm_impl
    )
    assert teacher_enc.num_stages == student_enc.num_stages, (
        f"Distillation requires equal encoder stage counts: teacher has {teacher_enc.num_stages}, "
        f"student has {student_enc.num_stages}."
    )
    # The predictor regresses teacher features, so its outputs have the teacher's widths.
    teacher_dim_pred_stages = list(teacher_enc._stage_dims[-len(pred_stage_dims) :])
    assert not ALIGN_REGISTERS or teacher_dim_pred_stages == encoder_dim_pred_stages, "Register alignment needs equal student and teacher widths."

    # Parameters and gradient norm tag of the predictor's conditioning layers, for logging.
    pred_cond_key, pred_cond_tag = (
        ("_ms_film", "Grads/film_grad_norm") if PRED_CONDITIONING == "film" else ("_adaln_zero_pred", "Grads/adaln_grad_norm")
    )
    predictor = DeformPredictor(
        encoder_stage_dims=encoder_dim_pred_stages,
        predictor_stage_dims=pred_stage_dims,
        voxel_size=voxel_size,
        layers_per_stage=NUM_PRED_LAYERS,
        mask_sizes=MIN_MASK_SIZES,
        first_pred_stage=(len(encoder_stage_dims) - len(pred_stage_dims)),
        predictor_head_dim=PRED_HEAD_DIM,
        num_neighbors=PRED_NUM_NEIGHBORS,
        num_points=PRED_NUM_OFFSETS,
        condition_prediction=CONDITION_PRED_ON_MASK_SIZE,
        qk_attn=PRED_QK_ATTN,
        learnable_interpolation=PRED_LEARNED_INTERPOLATION,
        condition_num_sweeps=CONDITION_PRED_ON_NUM_SWEEPS,
        num_sweeps_embedding_table=NUM_SWEEPS_EMBEDDING_TABLE,
        max_num_sweeps=MAX_NUM_SWEEPS_EMBED,
        num_sweeps_embedding_init_std=MASK_EMBEDDING_INIT_STD,
        num_sweeps_query_input=NUM_SWEEPS_QUERY_INPUT,
        adaln_attn_alpha=ADALN_ATTN_ALPHA,
        adaln_ffn_alpha=ADALN_FFN_ALPHA,
        adaln_alpha_plus_one=ADALN_ALPHA_PLUS_ONE,
        pred_before_unet=ENC_RETURN_DOWN_PATH,
        condition_voxel_size=CONDITION_PRED_ON_VOXEL_SIZE,
        voxel_size_query_input=VOXEL_SIZE_QUERY_INPUT,
        offset_scaling=PRED_OFFSET_SCALING,
        reference_voxel_size=reference_voxel_size,
        voxel_size_range=voxel_size_range,
        voxel_size_embedding_init_std=VOXEL_SIZE_COND_INIT_STD,
        pred_deepest_mid=DEEPEST_MID_TARGET,
        conditioning=PRED_CONDITIONING,
        target_stage_dims=teacher_dim_pred_stages,
    )

    # Count and print the number of trainable parameters in the model
    print()
    print("=" * 10, "Model Summary", "=" * 10)
    print(f"Teacher Encoder Parameters (frozen): {sum(p.numel() for p in teacher_enc.parameters()) / 1e6:.2f}M")
    print(f"Encoder Parameters: {sum(p.numel() for p in student_enc.parameters() if p.requires_grad) / 1e6:.2f}M")
    print(f"Predictor Parameters: {sum(p.numel() for p in predictor.parameters() if p.requires_grad) / 1e6:.2f}M")
    print()

    # Move to GPU first, then wrap with DDP
    student_enc = student_enc.to(device=get_gpu_device())
    teacher_enc = teacher_enc.to(device=get_gpu_device())
    predictor = predictor.to(device=get_gpu_device())

    student_enc = HackyDDP(student_enc, gradient_as_bucket_view=True) if torch_ddp_active() else student_enc
    predictor = HackyDDP(predictor, gradient_as_bucket_view=True) if torch_ddp_active() else predictor

    if student_mask_embed is not None:
        student_mask_embed.to(device=get_gpu_device())
        student_mask_embed = HackyDDP(student_mask_embed, gradient_as_bucket_view=True) if torch_ddp_active() else student_mask_embed

    if context_combiner is not None:
        context_combiner.to(device=get_gpu_device())
        context_combiner = HackyDDP(context_combiner, gradient_as_bucket_view=True) if torch_ddp_active() else context_combiner

    jepa = PointJEPA(
        student_enc, teacher_enc, predictor, student_mask_embed,
        context_combiner=context_combiner, fixed_teacher_voxel_size=FIXED_TEACHER_VOXEL_SIZE,
    )
    ddp_modules = [
        m for m in (student_enc, predictor, student_mask_embed, context_combiner)
        if isinstance(m, DDP)
    ]

    if DDP_BF16_GRAD_SYNC and ddp_modules:
        for module in ddp_modules:
            module.register_comm_hook(
                state=module.process_group,
                hook=default_hooks.bf16_compress_hook,
            )
        if is_main_process():
            print("DDP: BF16 gradient communication enabled.")

    no_decay_keywords = ["_embedding_table", "_p_offsets", "film"]
    if NUM_SWEEPS_EMBEDDING_TABLE:
        no_decay_keywords.append("_num_sweeps_embed")
    if not ADALN_WEIGHT_DECAY:
        no_decay_keywords.append("_adaln_zero_pred")
    optimizer = ChadamW(
        jepa.trainable_named_parameters(),
        lr=MAX_LEARNING_RATE,
        no_decay_keywords=no_decay_keywords,
        weight_decay=5e-2,
        fused=True,
        foreach=False,
    )

    # Warmup to prevent instability by fast teacher runaway at the start
    # Convert the configured warmup duration to optimizer steps using the effective batch size.
    warmup_steps = int(WARMUP_STEPS / torch.distributed.get_world_size() if torch_ddp_active() else WARMUP_STEPS)
    warmup_steps = warmup_steps // (ACCUMULATED_STEPS*batch_size) 
    linear_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-8, total_iters=warmup_steps)

    # For this calculation to be exactly correct we need to floor divide the number of iterations
    # per epoch first before multiplying by the epoch count since we can lose some steps each
    # epoch rather than once if we try to accumulate each epoch.
    total_train_iter = len(train_loader) // ACCUMULATED_STEPS * epochs
    cosine_annealer = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_train_iter - warmup_steps, eta_min=MIN_LEARNING_RATE)

    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[linear_warmup, cosine_annealer], milestones=[warmup_steps])
    if args.student_checkpoint:
        last_epoch = _load_student_encoder_from_checkpoint(
            jepa,
            optimizer,
            scheduler,
            args.student_checkpoint,
            args.output_directory,
        )
        start_epoch = last_epoch + 1
        print(f"Resuming training from epoch {start_epoch}.")
    else:
        start_epoch = 1


    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    prev_loss = None
    prev_reg_loss = None
    prev_stage_losses = [None] * len(pred_stage_dims)

    frame_id_log_path = os.path.join(args.output_directory, f"frame_ids_rank{get_gpu_world_rank()}.csv")
    # Keep the stream open across epochs so rank-local frame IDs are appended incrementally.
    _frame_id_log = open(frame_id_log_path, "a")  # noqa: SIM115
    if _frame_id_log.tell() == 0:
        _frame_id_log.write("step,epoch,frame_ids\n")

    # Gradients set to None are re-allocated outside the DDP buckets under no_sync, holding a second full copy of
    # the gradients. With DDP we therefore zero them in place, and only skip syncs once two synced backwards have made
    # every gradient a bucket view (DDP rebuilds its buckets once after the first iteration).
    num_synced_backwards = 0
    set_grads_to_none = not ddp_modules

    for epoch in range(start_epoch, epochs + 1):
        # TODO: We could make the accumulation logic depend on this to not lose a potential step
        # each epoch.
        global_step = (epoch - 1) * len(train_loader)  # step in terms of batches, not epochs
        print("~" * 100)
        print(f"Starting Epoch {epoch}")
        print("Training")

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for batch_idx, batch in enumerate(train_loader):
            point_cloud, frame_ids = batch[0], batch[1]
            start_time = time.perf_counter()
            batch_step = global_step + batch_idx
            _frame_id_log.write(f"{batch_step},{epoch},{';'.join(frame_ids)}\n")
            point_cloud = point_cloud.to(device=get_gpu_device())
            actual_batch_size = point_cloud.shape[0]
            num_sweeps = batch[2].to(device=get_gpu_device(), dtype=torch.long) if load_num_sweeps else None
            point_cloud, batch_indices = filter_and_flatten(point_cloud)  # (N, C), (N,)
            if voxel_size_range is not None:
                sample_voxel_size = torch.empty(actual_batch_size, device=get_gpu_device()).uniform_(*voxel_size_range)
            else:
                sample_voxel_size = None

            # Do common augmentations shared between teacher and student.
            point_cloud = apply_common_augmentations(point_cloud, batch_indices, batch_size=actual_batch_size)

            is_accum_step = (batch_idx + 1) % ACCUMULATED_STEPS == 0
            skip_sync = not is_accum_step and num_synced_backwards >= 2
            with ExitStack() as grad_sync:
                # Only the last micro-step all-reduces: the mean over ranks of the summed local gradients equals the sum of
                # per-micro-step means. no_sync must cover the forward too, where DDP prepares the backward reduction.
                if skip_sync:
                    for module in ddp_modules:
                        grad_sync.enter_context(module.no_sync())

                # Disable the autocast bfloat16 cache to save memory.
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                    pred, teach, teach_down, student_regs, teacher_regs = jepa(
                        point_cloud, batch_indices, batch_size=actual_batch_size, num_sweeps=num_sweeps, voxel_size=sample_voxel_size,
                    )
                    loss, stage_losses = jepa.mask_loss(pred, teach, LOSS_DIM_SCALING, LOSS_NUM_Q_WEIGHT, teach_down, batch_size=actual_batch_size)
                    if ALIGN_REGISTERS:
                        reg_loss, _ = jepa.reg_loss(student_regs, teacher_regs, LOSS_DIM_SCALING)
                        loss += REGISTER_LOSS_WEIGHT * reg_loss

                    # Does not matter because of scaling in AdamW, but gradient clipping will become messy without this.
                    loss /= ACCUMULATED_STEPS

                loss.backward()
            num_synced_backwards += not skip_sync
            # Step counted in optimizer steps rather than batches, matching the LR/EMA schedules.
            steps_per_epoch = len(train_loader) // ACCUMULATED_STEPS
            total_training_step = steps_per_epoch * (epoch - 1) + batch_idx // ACCUMULATED_STEPS
            if is_accum_step:
                # Compute gradient norms before clipping for collapse diagnostics
                enc_grad_norm = _compute_grad_norm(student_enc.parameters())
                pred_grad_norm = _compute_grad_norm(predictor.parameters())
                pred_cond_grad_norm = (
                    _compute_grad_norm(p for n, p in predictor.named_parameters() if pred_cond_key in n)
                    if (CONDITION_PRED_ON_MASK_SIZE or CONDITION_PRED_ON_NUM_SWEEPS or CONDITION_PRED_ON_VOXEL_SIZE)
                    else None
                )
                mask_embed_grad_norm = _compute_grad_norm(student_mask_embed.parameters()) if student_mask_embed is not None else None

                # We only clip once we have the final accumulated gradients to be mathematically equivalent.
                total_grad_norm = nn.utils.clip_grad_norm_(jepa.trainable_parameters(), max_norm=1.0)

                optimizer.step()
                optimizer.zero_grad(set_to_none=set_grads_to_none)
                scheduler.step()
        
                current_lr = scheduler.get_last_lr()[0]
                writer.add_scalar("Training/lr", current_lr, total_training_step)
                writer.add_scalar("Grads/total_grad_norm", total_grad_norm, total_training_step)
                writer.add_scalar("Grads/encoder_grad_norm", enc_grad_norm, total_training_step)
                writer.add_scalar("Grads/predictor_grad_norm", pred_grad_norm, total_training_step)
                if pred_cond_grad_norm is not None:
                    writer.add_scalar(pred_cond_tag, pred_cond_grad_norm, total_training_step)
                if mask_embed_grad_norm is not None:
                    writer.add_scalar("Grads/mask_embed_grad_norm", mask_embed_grad_norm, total_training_step)
                    with torch.no_grad():
                        mask_embed_rms = _scalars_to_cpu([
                            p.float().pow(2).mean().sqrt() for p in student_mask_embed.parameters()
                        ])
                    for i, rms in enumerate(mask_embed_rms):
                        writer.add_scalar(f"Params/mask_embed_rms_{i}", rms, total_training_step)

            loss_value, *stage_loss_values = _scalars_to_cpu([loss, *stage_losses])
            step_label = "STEP" if is_accum_step else "ACC"
            print(f"[{step_label}] Batch {batch_idx} \t | \t Loss: {loss_value:.4f}")

            # The moving averages are updated every batch, but only logged once per optimizer step.
            if ALIGN_REGISTERS:
                mavg_reg_loss = calc_mavg_loss(prev_reg_loss, reg_loss.item())
                prev_reg_loss = mavg_reg_loss

            # add average loss over accumulated steps
            mavg_loss = calc_mavg_loss(prev_loss, loss_value)
            prev_loss = mavg_loss

            for stage_idx, stage_loss in enumerate(reversed(stage_loss_values)):
                prev_stage_losses[stage_idx] = calc_mavg_loss(prev_stage_losses[stage_idx], stage_loss)

            if is_accum_step:
                if ALIGN_REGISTERS:
                    writer.add_scalar("Loss/reg_loss", mavg_reg_loss, total_training_step)
                writer.add_scalar("Loss/train", mavg_loss, total_training_step)
                for stage_idx in range(len(prev_stage_losses)):
                    writer.add_scalar(
                        f"Loss/train_stage_{len(encoder_stage_dims) - (stage_idx + 1)}",
                        prev_stage_losses[stage_idx],
                        total_training_step,
                    )

            # log time:
            batch_time = time.perf_counter() - start_time
            print(f"[TIME] Batch {batch_idx} processing time: {batch_time:.4f} seconds")
            writer.add_scalar("Timing/batch_processing_time", batch_time, batch_step)

        _save_checkpoint(jepa, optimizer, scheduler, epoch, args.output_directory)

        # This is needed to prevent dangling gradients from the previous epoch because of accumulation.
        optimizer.zero_grad(set_to_none=set_grads_to_none)

        if args.eval_every > 0 and epoch % args.eval_every == 0:
            print("~" * 100)
            print("Evaluating")

            # Create a temporary directory for intermediate files in the working directory
            tmp_work_dir = tempfile.mkdtemp(dir=args.output_directory)
            losses = []
            stage_losses = [[] for _ in range(len(pred_stage_dims))]  # separate list for each stage
            # The teacher is frozen, so the entropy tracks the student.
            student_module = _get_module(jepa._student)
            num_stages = len(student_module._stage_dims)
            entropy_metrics = EntropyMetrics(num_stages, tmp_work_dir)

            for eval_idx, batch in enumerate(val_loader):
                if load_num_sweeps:
                    point_cloud, num_sweeps, _, _ = batch
                    num_sweeps = num_sweeps.to(device=get_gpu_device(), dtype=torch.long)
                else:
                    (point_cloud, _, _), num_sweeps = batch, None
                point_cloud = point_cloud.to(device=get_gpu_device())
                actual_batch_size = point_cloud.shape[0]
                point_cloud, batch_indices = filter_and_flatten(point_cloud)  # (N, C)

                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    pred, teach, teach_down, _, student_regs, teacher_regs = jepa(
                        point_cloud,
                        batch_indices,
                        validation=True,
                        batch_size=actual_batch_size,
                        num_sweeps=num_sweeps,
                    )
                    val_loss, val_stage_losses = jepa.mask_loss(
                        pred, teach, LOSS_DIM_SCALING, LOSS_NUM_Q_WEIGHT, teach_down, batch_size=actual_batch_size
                    )
                    if ALIGN_REGISTERS:
                        val_reg_los, _ = jepa.reg_loss(student_regs, teacher_regs, LOSS_DIM_SCALING)
                        val_loss += REGISTER_LOSS_WEIGHT * val_reg_los
                    val_loss /= ACCUMULATED_STEPS

                val_loss_value, *val_stage_loss_values = _scalars_to_cpu([val_loss, *val_stage_losses])
                losses.append(val_loss_value)
                for stage_idx, stage_loss in enumerate(val_stage_loss_values):
                    stage_losses[stage_idx].append(stage_loss)

                with torch.no_grad():
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        # Unmasked student, like the teacher features in train_jepa.py.
                        stud_feats, _, stud_batch_inds, _, stud_regs = student_module(
                            point_cloud, batch_indices, batch_size=actual_batch_size,
                        )
                    entropy_metrics.accumulate(stud_feats, stud_batch_inds, stud_regs, eval_idx, actual_batch_size, batch_size)

                print(f"[VAL] Batch {eval_idx} \t | \t Loss: {val_loss_value:.4f}")

            entropy_metrics.merge_local()
            entropy_metrics.gather_ddp()

            # Gather JEPA losses across GPUs
            if torch_ddp_active():
                gathered_data = [None] * dist.get_world_size()
                dist.all_gather_object(gathered_data, [losses, stage_losses])
                losses       = _flatten([d[0] for d in gathered_data])
                stage_losses = [_flatten([d[1][si] for d in gathered_data]) for si in range(len(stage_losses))]

            if is_main_process():
                writer.add_scalar("Loss/val", sum(losses) / len(losses), epoch)
                for stage_idx, stage_loss in enumerate(reversed(stage_losses)):
                    writer.add_scalar(f"Loss/val_stage_{len(encoder_stage_dims) - (stage_idx + 1)}", sum(stage_loss) / len(stage_loss), epoch)
                entropy_metrics.log(writer, epoch, args.output_directory)

            # Important to have barrier here, make sure all workers finished using the
            # temp dirs before we try to delete them
            dist.barrier() if torch_ddp_active() else None
            entropy_metrics.cleanup()

        if args.pointwise_cls_eval_every > 0 and epoch % args.pointwise_cls_eval_every == 0:
            print("~" * 100)
            probe_dataset = _probe_dataset(args.dataset, epoch, args.pointwise_cls_eval_every)
            probe_writer = _probe_writer(writer, args.dataset, probe_dataset)
            print(f"Running Pointwise CLS Evaluation on {probe_dataset} for epoch {epoch}")
            miou, macc, all_acc = pointwise_cls_eval(
                encoder=_get_module(jepa._student),
                version=version,
                epoch=epoch,
                num_train_steps=5000,
                writer=probe_writer,
                dataset_type=probe_dataset,
            )
            # only on non-main processes since only main process will be not none.
            if is_main_process():
                probe_writer.add_scalar("CLS Seg/mIoU", miou, epoch)
                probe_writer.add_scalar("CLS Seg/mAcc", macc, epoch)
                probe_writer.add_scalar("CLS Seg/allAcc", all_acc, epoch)

        print(f"Epoch {epoch} completed.")
        print("=" * 100)

    _frame_id_log.close()

    print("~" * 100)
    print("Training completed.")


def _flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


def _probe_dataset(dataset: str, epoch: int, every: int) -> str:
    """Labelled dataset a probe trains and evaluates on: the pre-training one, or NuScenes for Combined."""
    if dataset != "Combined":
        return dataset
    return COMBINED_PROBE_DATASETS[(epoch // every - 1) % len(COMBINED_PROBE_DATASETS)]


def _probe_writer(writer: SummaryWriter, dataset: str, probe_dataset: str):
    """The writer, for Combined adding the probe dataset to every tag section so each dataset gets its own plots."""
    return writer if dataset != "Combined" else _SectionSuffixWriter(writer, probe_dataset)


class _SectionSuffixWriter:
    """Forwards to a SummaryWriter, turning tag "Section/name" into "Section <suffix>/name"."""

    def __init__(self, writer: SummaryWriter, suffix: str):
        self._writer = writer
        self._suffix = suffix

    def __getattr__(self, name):
        method = getattr(self._writer, name)
        if not name.startswith("add_"):
            return method

        def add(tag, *args, **kwargs):
            section, slash, rest = tag.partition("/")
            return method(f"{section} {self._suffix}{slash}{rest}", *args, **kwargs)

        return add


def calc_mavg_loss(prev_loss, new_loss, mavg_factor=0.9):
    """Calculate moving average of loss."""
    if prev_loss is None:
        return new_loss
    return mavg_factor * prev_loss + (1 - mavg_factor) * new_loss


def _scalars_to_cpu(values: list[torch.Tensor]) -> list[float]:
    """Transfer a group of detached scalar diagnostics to the CPU together."""
    if not values:
        return []
    return torch.stack([value.detach() for value in values]).cpu().tolist()


def _aggregate_norms(norms: list[torch.Tensor]) -> float:
    """Accumulate scalar norms in FP64 on-device, then extract one host scalar."""
    if not norms:
        return 0.0
    return torch.stack(norms).double().square().sum().sqrt().item()


def _compute_grad_norm(parameters) -> float:
    """Compute the pre-clip L2 gradient norm with one host transfer, not one per parameter."""
    return _aggregate_norms([
        p.grad.detach().float().norm(2) for p in parameters if p.grad is not None
    ])


def _get_module(mod):
    if torch_ddp_active():
        return mod.module
    return mod

@torch_ddp_run_on_master_only
def _save_checkpoint(jepa, optimizer, scheduler, epoch, output_path) -> None:
    """Save a training checkpoint."""

    extra_modules = {}
    if jepa._student_mask_embedding is not None:
        extra_modules["student_mask_embed"] = _get_module(jepa._student_mask_embedding).state_dict()
    if jepa._context_combiner is not None:
        extra_modules["context_combiner"] = _get_module(jepa._context_combiner).state_dict()

    student = _get_module(jepa._student)
    predictor = _get_module(jepa._predictor)
    checkpoint_data = {
        "epoch": epoch,
        "student_encoder": student.state_dict(),
        "predictor": predictor.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        **extra_modules
    }
    out_dir = os.path.join(output_path, "checkpoints")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(output_path, "checkpoints", f"epoch_{epoch}.pt")
    torch.save(checkpoint_data, path)


def _load_student_encoder_from_checkpoint(jepa, optimizer, scheduler, checkpoint, output_path):
    """Load a student encoder from a training checkpoint."""
    chkpt_dir = os.path.join(output_path, "checkpoints", checkpoint)
    checkpoint_data = torch.load(chkpt_dir, map_location=get_gpu_device())

    # only student and predictor may be DDP wrapped
    _get_module(jepa._student).load_state_dict(checkpoint_data["student_encoder"])
    _get_module(jepa._predictor).load_state_dict(checkpoint_data["predictor"])
    optimizer.load_state_dict(checkpoint_data["optimizer"])
    scheduler.load_state_dict(checkpoint_data["scheduler"])

    if jepa._student_mask_embedding is not None:
        # Checkpoints from before the mask modulator was removed also hold an unused "teacher_mask_embed".
        _get_module(jepa._student_mask_embedding).load_state_dict(checkpoint_data["student_mask_embed"])
    if jepa._context_combiner is not None:
        _get_module(jepa._context_combiner).load_state_dict(checkpoint_data["context_combiner"])

    epoch = checkpoint_data["epoch"]
    print(f"Loaded checkpoint from {chkpt_dir} at epoch {epoch}.")
    return epoch


@torch_ddp_run_on_master_only
def save_initial_metadata(cmd_args):
    # save file with constants used (training configuration)
    os.makedirs(cmd_args.output_directory, exist_ok=True)
    constants_copy = os.path.join(cmd_args.output_directory, "constants.py")
    shutil.copyfile(constants.__file__, constants_copy)
    with open(constants_copy, "a") as f:
        f.write(f"\n# Ran as debug mode = {cmd_args.debug}\n")
        teacher_backend = None if cmd_args.teacher_qk_layernorm_impl == "saved" else cmd_args.teacher_qk_layernorm_impl
        f.write(f"DISTILL_TEACHER_QK_LAYERNORM_IMPL = {teacher_backend!r}\n")
    print(f"Output directory: {cmd_args.output_directory}")


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="Train JEPA model")

    # Required argument for experiment directory

    parser.add_argument(
        "--output_directory",
        "-o",
        type=str,
        required=True,
        help="Path to super directory with experiments. These experiments will be restored.",
    )
    parser.add_argument(
        "--student_checkpoint",
        "-c",
        type=str,
        required=False,
        help=("Path to the student checkpoint to load if restarting a training run from checkpoint"),
    )
    parser.add_argument(
        "--teacher_checkpoint",
        type=str,
        required=True,
        help=("name of checkpoint to load teacher from, this needs to be a pre-trained teacher checkpoint full path."),
    )
    parser.add_argument(
        "--teacher_constants",
        type=str,
        required=True,
        help=("path to the teacher constants file, this needs to be a pre-trained teacher constants full path."),
    )
    parser.add_argument(
        "--teacher_qk_layernorm_impl",
        choices=["torch", "triton", "triton_fused_rope", "saved"],
        default=DISTILL_TEACHER_QK_LAYERNORM_IMPL if DISTILL_TEACHER_QK_LAYERNORM_IMPL is not None else "saved",
        help=(
            "Override the frozen teacher's QK-LayerNorm backend on new or resumed runs. "
            "Defaults to DISTILL_TEACHER_QK_LAYERNORM_IMPL (triton_fused_rope if absent). "
            "Use 'saved' to inherit the teacher constants, falling back to triton if absent."
        ),
    )
    parser.add_argument(
        "--torch_ddp",
        action="store_true",
        help="Enable torch distributed data parallel training. This is useful for multi-GPU training.",
    )
    parser.add_argument(
        "--pointwise_cls_eval_every",
        "-pe",
        type=int,
        default=3,
        help="How many epochs between pointwise classification evaluations. -1 to disable entirely.",
    )
    parser.add_argument(
        "--eval_every",
        "-ee",
        type=int,
        default=3,
        help="How many epochs between evaluations. -1 to disable entirely.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode. Will use a small subset of the data.",
    )
    parser.add_argument(
        "--mini_model",
        action="store_true",
        help="Enable mini model mode. Will use a smaller model for faster experimentation.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ZOD",
        choices=["ZOD", "NuScenes", "Argoverse2", "TruckScenes", "PandaSet", "KITTI", "Cirrus", "Combined"],
        help="Dataset to pre-train and evaluate on. 'Combined' pre-trains on all three and evaluates on NuScenes.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cmd_args = parse_args()
    if cmd_args.torch_ddp:
        init_multigpu_via_torch_ddp()

    torch.multiprocessing.set_sharing_strategy("file_system")
    save_initial_metadata(cmd_args)

    main(cmd_args)
