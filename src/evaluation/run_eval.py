"""Single-view probe evaluation for a pretrained point-cloud encoder.

Loads a J3PA checkpoint, attaches a small task head, and reports
semantic segmentation, panoptic segmentation and object detection under three
protocols:

    linear_probe      backbone frozen, head trained
    nl_probe          backbone frozen, Linear-ReLU-Linear head trained
    full_finetune     head first trained as a linear probe, then everything trained

Segmentation is evaluated at the *input* point resolution against the official
nuScenes-lidarseg / nuScenes-panoptic labels, so the numbers are directly
comparable to published results rather than to the voxel-resolution online probe.
Evaluation runs over the full validation split on single key frames with no sweep
aggregation, which is the setting the official labels are defined in and the
setting the official pointcept repo uses as well.

Every experiment is evaluated at several training budgets (1k/5k/20k/50k/100k steps by
default).

The training recipe follows run_eval_bellsandwhistels.py (Pointcept's nuScenes linear / PointGroup
probes), without any of its test-time augmentation or train-mode backbone tricks:

    - CE + Lovasz for semantic and panoptic, plus PointGroup's offset L1 + direction losses for panoptic;
      detection is the same instance-segmentation recipe with targets from the box annotations alone
      (box class per point, box centre offset / size / yaw), decoded by clustering the centre votes into one
      box per cluster and scored with the nuScenes devkit's mAP/NDS
    - classifier linear for every protocol but nl_probe; panoptic offset / detection box branch is always
      Linear-BatchNorm-ReLU-Linear
    - no xyz in the head input, ignore class never predicted
    - every point kept for training and evaluation (no PC_RANGE crop, as Pointcept); detection scores only
      boxes and predictions within the nuScenes class ranges
    - AdamW + OneCycle over the longest budget, total batch of 64 (gradient accumulation)
    - full_finetune starts from a linear probe trained for --finetune_probe_steps (5k) steps on the frozen backbone,
      so a random head does not push arbitrary gradients into the pretrained encoder, then warms its learning rates up
      over --finetune_warmup_steps (2k) steps; its budgets count the fine-tuning steps after the probe
    - backbone in eval mode during both probe training and evaluation; one forward pass per sample
    - panoptic: cluster cell 0.3 m, official PQ with 15-point minimum for FP/FN, plus PQ_dagger

If no task flags are set, all three tasks are evaluated.
If no protocol flags are set, all protocols are evaluated.

Example:
    PYTHONPATH=src python src/evaluation/run_eval.py \
        --checkpoint experiments/tst_nusc/checkpoints/epoch_100.pt \
        --constants experiments/tst_nusc \
        --output_directory experiments/tst_nusc/ \
        --semantic_segmentation --linear_probe
"""

import argparse
import contextlib
import importlib
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.backbones import FROZEN, FULL, build_backbone
from evaluation.heads import DetectionHead, PanopticSegmentationHead, SemanticSegmentationHead
from evaluation.panoptic_metrics import PanopticQuality, cluster_instances
from evaluation.qualitative import log_detection_comparison, log_instance_comparison, log_semantic_comparison
from evaluation.reporting import ResultsRecorder
from evaluation.upcast import upcast_to_input_resolution
from evaluation.probe_utils import (
    _calc_mavg_loss,
    _cleanup_dataloader,
    _flatten,
    limit_pc_to_fov,
    rotate_cloud,
)
from evaluation.run_eval_bellsandwhistels import (
    DEFAULT_CLUSTER_CELL_SIZE,
    LABELLED_DATASET_MODULES,
    NUSCENES_MIN_INST_POINTS,
    POINTCEPT_FT_BLOCK_LR,
    POINTCEPT_FT_LR,
    POINTCEPT_FT_ONECYCLE,
    POINTCEPT_FT_WEIGHT_DECAY,
    POINTCEPT_INSSEG_ONECYCLE,
    POINTCEPT_INSSEG_WEIGHT_DECAY,
    POINTCEPT_LOVASZ_WEIGHT,
    POINTCEPT_LR,
    POINTCEPT_ONECYCLE,
    POINTCEPT_TOTAL_BATCH_SIZE,
    POINTCEPT_WEIGHT_DECAY,
    DEFAULT_DETECTION_SCORE_THRESHOLD,
    DEFAULT_NMS_IOU,
    decode_detections,
    detection_loss,
    lovasz_softmax_loss,
    nuscenes_detection_metrics,
    scored_ground_truth,
)
from online_segmntation.online_segm_eval import (
    accumulate_seg_counts,
    compute_seg_metrics_from_counts,
)
from utils.multi_gpu_utils import (
    get_gpu_device,
    init_multigpu_via_torch_ddp,
    is_main_process,
    torch_ddp_active,
)
from utils.utils import filter_and_flatten, get_dataloader
from constants import (
    PROBE_BATCH_SIZE,
    PROBE_HIDDEN_DIM,
    PROBE_NUM_WORKERS,
    PROBE_SEED,
)

SEMANTIC = "semantic"
PANOPTIC = "panoptic"
DETECTION = "detection"
TASKS = (SEMANTIC, PANOPTIC, DETECTION)
NL_PROBE = "nl_probe"
# The linear probe reuses the FROZEN backbone mode as its protocol id; nl_probe is
# a distinct protocol that happens to share that mode.
LINEAR_PROBE = FROZEN
PROTOCOLS = (LINEAR_PROBE, NL_PROBE, FULL)
PROTOCOL_NAMES = {LINEAR_PROBE: "linear_probe", NL_PROBE: "nl_probe", FULL: "full_finetune"}
# How much of the backbone each protocol trains.
BACKBONE_MODE = {LINEAR_PROBE: FROZEN, NL_PROBE: FROZEN, FULL: FULL}


class ProbeModel(nn.Module):
    """Backbone plus task head as one module, so DDP can wrap them together."""

    def __init__(self, backbone, head, task: str, ignore_logit: int | None = None):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.task = task
        # Logit of the ignore class, pinned to -inf so it is never predicted.
        self.ignore_logit = ignore_logit

    def forward(self, point_cloud, batch_indices):
        """Encode and apply the head.

        Args:
            point_cloud: (N, C) flattened points.
            batch_indices: (N,) sample index per point.

        Returns:
            ``(head_output, positions)``. The head output is a logits tensor for semantic, a
            ``(logits, offsets)`` pair for panoptic and a ``(logits, boxes)`` pair for detection.
        """
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            features, _, batch_indices, up_indices, positions = self.backbone(point_cloud, batch_indices) #Already here the features are the upcasted and agregated features. 
            # The head sees only the backbone features, not the xyz the upcast appends.
            features = upcast_to_input_resolution(features, up_indices, positions)[:, :-3]
            outputs = self.head(features)
        if self.ignore_logit is not None:
            logits, offsets = outputs if self.task == PANOPTIC else (outputs, None)
            logits = logits.index_fill(-1, torch.tensor([self.ignore_logit], device=logits.device), float("-inf"))
            outputs = logits if offsets is None else (logits, offsets)
        return outputs, positions


def _unwrap(model):
    """The underlying module, whether or not DDP wrapped it."""
    return model.module if isinstance(model, DDP) else model


def _flatten_batch(point_cloud, seg_labels, device):
    """Drop padding rows from a batch, keeping labels aligned.

    `filter_and_flatten` decides what is padding by looking for all-zero rows, so
    the labels have to be filtered with that exact same mask instead of being
    flattened independently.

    Args:
        point_cloud: (B, N, C) padded clouds.
        seg_labels: dict with "semantic"/"instance" of shape (B, N), or None.
        device: Device to move the results to.

    Returns:
        ``(points, batch_indices, semantic, instance)`` with the last two None
        when ``seg_labels`` is None.
    """
    padding_mask = torch.any(point_cloud != 0, dim=-1)  # (B, N)
    points, batch_indices = filter_and_flatten(point_cloud)
 
    if seg_labels is None:
        return points, batch_indices, None, None
    # Labels arrive on the CPU while point_cloud is already on the GPU, so move
    # them before indexing with the mask.
    semantic = seg_labels["semantic"].to(device=device)[padding_mask]
    instance = seg_labels["instance"].to(device=device)[padding_mask]
    return points, batch_indices, semantic, instance


def _unpack(batch, load_segmentation: bool):
    """Split a dataloader batch into its named parts."""
    if load_segmentation:
        point_cloud, annotations, field_of_view, seg_labels = batch
        return point_cloud, annotations, field_of_view, seg_labels
    point_cloud, annotations, field_of_view = batch
    return point_cloud, annotations, field_of_view, None


def _cycle(loader):
    """Yield batches forever, restarting the loader when it runs out."""
    while True:
        yield from loader


def _num_eval_batches(loader, args) -> int:
    """Batches to evaluate: the whole split unless --limit_eval_batches is set."""
    if args.limit_eval_batches is None:
        return len(loader)
    return min(args.limit_eval_batches, len(loader))


def _timed_forward(model, points, batch_indices, device):
    """``model(points, batch_indices)`` plus its wall time in seconds, GPU work included."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    outputs, positions = model(points, batch_indices)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return outputs, positions, time.perf_counter() - start


def _inference_ms_per_frame(seconds, frames) -> float:
    """Mean model forward time per frame over all ranks, in milliseconds."""
    if torch_ddp_active():
        for tensor in (seconds, frames):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return 1000.0 * (seconds / frames.clamp_min(1)).item()


def _build_head(task: str, feature_dim: int, args, hidden_dim: int | None = 512) -> nn.Module:
    """Instantiate the head for ``task`` in the dataset's class space.

    ``hidden_dim=None`` gives a single linear layer, i.e. a true linear probe.
    """
    if task == SEMANTIC:
        return SemanticSegmentationHead(
            in_feat_dim=feature_dim,
            num_classes=args.num_seg_classes,
            ignore_id=args.seg_ignore_id,
            hidden_dim=hidden_dim,
        )
    if task == PANOPTIC:
        return PanopticSegmentationHead(
            in_feat_dim=feature_dim,
            num_classes=args.num_seg_classes,
            thing_ids=args.seg_thing_ids,
            hidden_dim=hidden_dim,
            ignore_id=args.seg_ignore_id,
        )
    if task == DETECTION:
        return DetectionHead(
            in_feat_dim=feature_dim,
            num_classes=len(args.class_name_to_id),
            hidden_dim=hidden_dim,
        )
    raise ValueError(f"Unknown task {task!r}")


def _compute_loss(task, model, outputs, positions, batch_indices, annotations, semantic, instance, ignore_id):
    """Task-specific loss (same terms as run_eval_bellsandwhistels.py), returning ``(loss, extra_terms)``."""
    head = _unwrap(model).head
    if task == DETECTION:
        return detection_loss(head, outputs[0], outputs[1], positions, annotations, batch_indices)

    logits = outputs[0] if task == PANOPTIC else outputs
    lovasz = POINTCEPT_LOVASZ_WEIGHT * lovasz_softmax_loss(logits, semantic, ignore_id)
    if task == SEMANTIC:
        cross_entropy = head.loss(logits, semantic)
        return cross_entropy + lovasz, {"ce": cross_entropy.item(), "lovasz": lovasz.item()}
    # Keywords rather than position: the head takes six same-typed tensors
    # and silently mis-ordering them only shows up as a missing gradient.
    total, extra = head.loss(
        logits=logits,
        offsets=outputs[1],
        semantic_labels=semantic,
        instance_labels=instance,
        positions=positions,
        batch_indices=batch_indices,
    )
    return total + lovasz, {**extra, "lovasz": lovasz.item()}


def probe_feature_dim(backbone, loader, args, device) -> int:
    """Run one batch through the backbone just to read off its output width.

    The feature width depends on the run's stage dimensions.
    """
    batch = next(iter(loader))
    point_cloud, _, field_of_view, _ = _unpack(batch, args.load_segmentation)
    point_cloud = limit_pc_to_fov(point_cloud.to(device), field_of_view.to(device))
    points, batch_indices, _, _ = _flatten_batch(point_cloud, None, device)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        features, _, _, up_indices, positions = backbone(points, batch_indices)

    # Minus the xyz the upcast appends, which the head does not get.
    upcasted_feat_dim = upcast_to_input_resolution(features, up_indices, positions).shape[-1] - 3
    stage_dims = [f.shape[-1] for f in features]
    return upcasted_feat_dim, stage_dims

def train_head(model, data_iter, optimizer, scheduler, args, task, device, writer, tag,
               start_step, end_step, running_loss):
    """Train from ``start_step`` up to ``end_step`` optimizer steps.

    Training is resumed in segments so the same run can be evaluated at several
    budgets without retraining from scratch for each one. Each optimizer step
    accumulates ``args.accumulation_steps`` micro-batches.

    Returns:
        The moving-average loss, to be threaded into the next segment.
    """
    model.train()
    # The backbone stays in eval mode under every protocol (no drop_path, RoPE jitter or random curve
    # orders), so the probe trains on exactly the features it is evaluated on. requires_grad was settled
    # before DDP was built and must not change here.
    _unwrap(model).backbone.eval()

    start_time = time.perf_counter()
    for step in range(start_step, end_step):
        for micro_step in range(args.accumulation_steps):
            batch = next(data_iter)

            point_cloud, annotations, field_of_view, seg_labels = _unpack(batch, args.load_segmentation)
            point_cloud = point_cloud.to(device)
            field_of_view = field_of_view.to(device)
            annotations = {k: v.to(device=device) for k, v in annotations.items()}
            point_cloud = limit_pc_to_fov(point_cloud, field_of_view)
            point_cloud, annotations, _ = rotate_cloud(point_cloud, annotations)
            points, batch_indices, semantic, instance = _flatten_batch(point_cloud, seg_labels, device)

            # Gradients are only all-reduced across ranks on the last micro-batch.
            last_micro_step = micro_step == args.accumulation_steps - 1
            with contextlib.nullcontext() if last_micro_step or not isinstance(model, DDP) else model.no_sync():
                outputs, positions = model(points, batch_indices)
                loss, extra = _compute_loss(
                    task, model, outputs, positions, batch_indices, annotations, semantic, instance,
                    args.seg_ignore_id,
                )
                (loss / args.accumulation_steps).backward()
            running_loss = _calc_mavg_loss(running_loss, loss.item())

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        if writer is not None and is_main_process():
            writer.add_scalar(f"{tag}/train_loss", running_loss, step)
            for name, value in extra.items():
                writer.add_scalar(f"{tag}/train_loss_{name}", value, step)
        print(f"  [{tag}] train step {step + 1}/{end_step} loss {running_loss:.4f}", end="\r")

    print()
    seconds = time.perf_counter() - start_time
    print(f"  [{tag}] {end_step - start_step} train steps in {seconds:.0f}s ({seconds / max(1, end_step - start_step):.2f} s/step)")
    return running_loss


def evaluate_segmentation(model, loader, args, task, device, writer=None, tag="", step=0) -> dict:
    """mIoU/mAcc/allAcc, plus PQ/SQ/RQ when running the panoptic task.

    ``writer``/``tag``/``step`` are only used for the qualitative renders, which
    are filed under ``step`` so the same frame can be compared across milestones.
    """
    model.eval()
    num_classes = args.num_seg_classes
    tp_counts = torch.zeros(num_classes, dtype=torch.long, device=device)
    fp_counts = torch.zeros(num_classes, dtype=torch.long, device=device)
    fn_counts = torch.zeros(num_classes, dtype=torch.long, device=device)
    correct_fg = torch.zeros(1, dtype=torch.long, device=device)
    total_fg = torch.zeros(1, dtype=torch.long, device=device)
    panoptic_quality = PanopticQuality(
        args.seg_class_name_to_id, args.seg_thing_ids, args.seg_ignore_id, device=device,
        min_points=NUSCENES_MIN_INST_POINTS,
    ) if task == PANOPTIC else None
    num_batches = _num_eval_batches(loader, args)
    loss_sum = torch.zeros(1, dtype=torch.float64, device=device)
    loss_batches = torch.zeros(1, dtype=torch.float64, device=device)
    inference_seconds = torch.zeros(1, dtype=torch.float64, device=device)
    inference_frames = torch.zeros(1, dtype=torch.float64, device=device)

    for eval_idx, batch in enumerate(loader):
        if eval_idx >= num_batches:
            break
        point_cloud, _, field_of_view, seg_labels = _unpack(batch, True)
        point_cloud = point_cloud.to(device)
        field_of_view = field_of_view.to(device)
        point_cloud = limit_pc_to_fov(point_cloud, field_of_view)
        points, batch_indices, semantic, instance = _flatten_batch(point_cloud, seg_labels, device) #The segmentation labels are mover to device in here.

        with torch.no_grad():
            outputs, positions, seconds = _timed_forward(model, points, batch_indices, device)
            # The first batch is a warm-up (allocator, kernel autotuning), unless it is the only one.
            if eval_idx > 0 or num_batches == 1:
                inference_seconds += seconds
                inference_frames += int(batch_indices.max()) + 1
            # Annotations are only read by the detection loss, which is not used here.
            loss, _ = _compute_loss(
                task, model, outputs, positions, batch_indices, None, semantic, instance, args.seg_ignore_id
            )
        loss_sum += loss.item()
        loss_batches += 1

        logits = outputs[0] if task == PANOPTIC else outputs
        # Ignore-labelled points must not contribute to IoU at all, otherwise a
        # prediction on noise would be charged as a false positive.
        labelled = torch.ones_like(semantic, dtype=torch.bool) if args.seg_ignore_id is None else semantic != args.seg_ignore_id
        accumulate_seg_counts(
            tp_counts, fp_counts, fn_counts, correct_fg, total_fg,
            logits[labelled].float(), semantic[labelled],
        )
        predicted_semantic = logits.argmax(dim=-1)
        predicted_instance = None

        if panoptic_quality is not None:
            offsets = outputs[1]
            predicted_instance = cluster_instances(
                positions.float() + offsets.float(),
                predicted_semantic,
                batch_indices,
                thing_ids=args.seg_thing_ids,
                voxel_size=DEFAULT_CLUSTER_CELL_SIZE,
            )
            panoptic_quality.update(
                predicted_semantic, predicted_instance, semantic, instance, batch_indices
            )

        if eval_idx < args.num_vis_samples and writer is not None and is_main_process():
            # First sample of the batch only: the loader is unshuffled, so this is
            # the same frame at every milestone.
            frame = batch_indices == 0
            log_semantic_comparison(
                writer, f"qualitative/{tag}/frame_{eval_idx}/semantic", step,
                positions[frame], semantic[frame], predicted_semantic[frame],
                args.seg_class_name_to_id, args.seg_ignore_id,
            )
            if predicted_instance is not None:
                log_instance_comparison(
                    writer, f"qualitative/{tag}/frame_{eval_idx}/instances", step,
                    positions[frame], instance[frame], semantic[frame],
                    predicted_instance[frame], predicted_semantic[frame],
                    args.seg_ignore_id,
                )

        print(f"  eval step {eval_idx + 1}/{num_batches}", end="\r")
    print()

    inference_ms = _inference_ms_per_frame(inference_seconds, inference_frames)
    if torch_ddp_active():
        for tensor in (tp_counts, fp_counts, fn_counts, correct_fg, total_fg, loss_sum, loss_batches):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        if panoptic_quality is not None:
            panoptic_quality.all_reduce()

    if not is_main_process():
        return {}

    if writer is not None:
        writer.add_scalar(f"{tag}/val_loss", (loss_sum / loss_batches).item(), step)

    metrics = compute_seg_metrics_from_counts(
        tp_counts, fp_counts, fn_counts, correct_fg, total_fg,
        num_classes, class_name_to_id=args.seg_class_name_to_id,
    )
    results = {
        "mIoU": metrics["mIoU"],
        "mAcc": metrics["mAcc"],
        "allAcc": metrics["allAcc"],
        "per_class_iou": metrics["per_class_iou"],
        "per_class_acc": metrics["per_class_acc"],
        "inference_ms_per_frame": inference_ms,
    }
    if panoptic_quality is not None:
        panoptic = panoptic_quality.compute(metrics["per_class_iou"])
        panoptic["per_class_pq"] = panoptic.pop("per_class")
        results.update(panoptic)
    return results


def evaluate_detection(model, loader, args, device, writer=None, tag="", step=0) -> dict:
    """nuScenes devkit mAP, TP errors and NDS of the clustered boxes over the validation split."""
    model.eval()
    local_predictions, local_ground_truths = [], []
    num_batches = _num_eval_batches(loader, args)
    loss_sum = torch.zeros(1, dtype=torch.float64, device=device)
    loss_batches = torch.zeros(1, dtype=torch.float64, device=device)
    inference_seconds = torch.zeros(1, dtype=torch.float64, device=device)
    inference_frames = torch.zeros(1, dtype=torch.float64, device=device)

    for eval_idx, batch in enumerate(loader):
        if eval_idx >= num_batches:
            break
        point_cloud, annotations, field_of_view, seg_labels = _unpack(batch, args.load_segmentation)
        point_cloud = point_cloud.to(device)
        field_of_view = field_of_view.to(device)
        annotations = {k: v.to(device=device) for k, v in annotations.items()}
        point_cloud = limit_pc_to_fov(point_cloud, field_of_view)
        points, batch_indices, _, _ = _flatten_batch(point_cloud, None, device)

        with torch.no_grad():
            outputs, positions, seconds = _timed_forward(model, points, batch_indices, device)
            if eval_idx > 0 or num_batches == 1:
                inference_seconds += seconds
                inference_frames += int(batch_indices.max()) + 1
            loss, _ = _compute_loss(
                DETECTION, model, outputs, positions, batch_indices, annotations, None, None, None
            )
        loss_sum += loss.item()
        loss_batches += 1

        detected_boxes = decode_detections(
            positions, outputs[0].float().softmax(dim=-1), outputs[1].float(), batch_indices,
            args.class_name_to_id, DEFAULT_DETECTION_SCORE_THRESHOLD, DEFAULT_NMS_IOU,
        )
        truth_boxes = scored_ground_truth(annotations, args.class_name_to_id)
        local_predictions.extend(detected_boxes)
        local_ground_truths.extend(truth_boxes)

        if eval_idx < args.num_vis_samples and writer is not None and is_main_process():
            # First sample of the batch only: the loader is unshuffled, so this is the same frame at every milestone.
            frame = batch_indices == 0
            log_detection_comparison(
                writer, f"qualitative/{tag}/frame_{eval_idx}/boxes", step,
                positions[frame], 1.0 - outputs[0][frame].float().softmax(dim=-1)[:, 0],
                truth_boxes[0], detected_boxes[0], args.class_name_to_id,
            )

        print(f"  eval step {eval_idx + 1}/{num_batches}", end="\r")
    print()

    inference_ms = _inference_ms_per_frame(inference_seconds, inference_frames)
    if torch_ddp_active():
        world_size = dist.get_world_size()
        gathered_predictions = [None] * world_size
        gathered_ground_truths = [None] * world_size
        dist.all_gather_object(gathered_predictions, local_predictions)
        dist.all_gather_object(gathered_ground_truths, local_ground_truths)
        local_predictions = _flatten(gathered_predictions)
        local_ground_truths = _flatten(gathered_ground_truths)
        for tensor in (loss_sum, loss_batches):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    if not is_main_process():
        return {}

    if writer is not None:
        writer.add_scalar(f"{tag}/val_loss", (loss_sum / loss_batches).item(), step)

    metrics = nuscenes_detection_metrics(local_ground_truths, local_predictions, args.class_name_to_id)
    metrics["inference_ms_per_frame"] = inference_ms
    return metrics


def _wrap_for_ddp(model, args, device):
    """``model`` in DDP when it is active. DDP only all-reduces the parameters that require grad when it is built."""
    if not torch_ddp_active():
        return model
    return DDP(model, device_ids=[device.index] if device.type == "cuda" else None, find_unused_parameters=False)


def _build_optimizer(trainable, task, backbone_mode, block_modules, total_steps, warmup_steps=None):
    """AdamW plus a OneCycle schedule over ``total_steps``, with Pointcept's recipe for ``backbone_mode``.

    A frozen backbone gets the linear-probe recipe; otherwise the parameters of ``block_modules`` train at the
    lower fine-tuning rate. ``warmup_steps`` replaces the recipe's warm-up fraction with a number of steps.
    """
    if backbone_mode == FROZEN:
        param_groups = [dict(params=trainable, lr=POINTCEPT_LR)]
        if task in (PANOPTIC, DETECTION):
            weight_decay, onecycle = POINTCEPT_INSSEG_WEIGHT_DECAY, POINTCEPT_INSSEG_ONECYCLE
        else:
            weight_decay, onecycle = POINTCEPT_WEIGHT_DECAY, POINTCEPT_ONECYCLE
    else:
        block_ids = {id(p) for module in block_modules for p in module.parameters()}
        param_groups = [
            dict(params=[p for p in trainable if id(p) not in block_ids], lr=POINTCEPT_FT_LR),
            dict(params=[p for p in trainable if id(p) in block_ids], lr=POINTCEPT_FT_BLOCK_LR),
        ]
        weight_decay, onecycle = POINTCEPT_FT_WEIGHT_DECAY, POINTCEPT_FT_ONECYCLE
    if warmup_steps is not None:
        onecycle = {**onecycle, "pct_start": warmup_steps / total_steps}
    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[group["lr"] for group in param_groups], total_steps=total_steps, **onecycle
    )
    return optimizer, scheduler


def run_experiment(args, task, protocol, train_loader, val_loader, device, writer, recorder) -> dict:
    """Train one (task, protocol) combination, evaluating at each budget.

    Each milestone is handed to ``recorder`` as soon as it is measured, so the
    scalars reach TensorBoard during the run instead of only at the very end.

    Returns:
        ``{train_steps: metrics}`` for every milestone in ``args.eval_at_steps``.
    """
    tag = f"{task}/{PROTOCOL_NAMES[protocol]}"
    print("~" * 100)
    print(f"Running {tag}")

    # Rebuilt per experiment so fine-tuning never leaks weights into the next run.
    backbone = build_backbone(args).to(device=device)
    # Only nl_probe has a hidden classifier layer; the offset/box branches are MLPs under every protocol.
    head = _build_head(
        task,
        args.feature_dim,
        args,
        hidden_dim=PROBE_HIDDEN_DIM if protocol == NL_PROBE else None,
    ).to(device=device)
    ignore_logit = None if task == DETECTION else args.seg_ignore_id
    model = ProbeModel(backbone=backbone, head=head, task=task, ignore_logit=ignore_logit)
    # Pointcept's fine-tuning "block" group: the encoder trains at the lower rate.
    block_modules = [backbone]
    data_iter = _cycle(train_loader)
    running_loss = None

    if protocol == FULL and args.finetune_probe_steps > 0:
        # A random head would push arbitrary gradients into the pretrained encoder, so the head is first trained
        # as a linear probe on the frozen backbone, with the linear_probe protocol's recipe.
        backbone.set_trainable(FROZEN)
        probe = _wrap_for_ddp(model, args, device)
        trainable = [p for p in probe.parameters() if p.requires_grad]
        print(
            f"  Linear probe ({sum(p.numel() for p in trainable) / 1e6:.2f}M trainable parameters) "
            f"for {args.finetune_probe_steps} steps before fine-tuning"
        )
        optimizer, scheduler = _build_optimizer(trainable, task, FROZEN, block_modules, args.finetune_probe_steps)
        running_loss = train_head(
            probe, data_iter, optimizer, scheduler, args, task, device, writer, f"{tag}/linear_probe_init",
            0, args.finetune_probe_steps, running_loss,
        )
        # The probe's DDP wrapper does not all-reduce the backbone, so it is rebuilt once the backbone is unfrozen.
        del probe, optimizer, scheduler

    backbone.set_trainable(BACKBONE_MODE[protocol])
    model = _wrap_for_ddp(model, args, device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"  {sum(p.numel() for p in trainable) / 1e6:.2f}M trainable parameters")

    # Spans the longest budget, so earlier budgets see a prefix of the same schedule.
    optimizer, scheduler = _build_optimizer(
        trainable, task, BACKBONE_MODE[protocol], block_modules, args.eval_at_steps[-1],
        warmup_steps=args.finetune_warmup_steps if protocol == FULL else None,
    )

    results, trained_steps = {}, 0
    for milestone in args.eval_at_steps:
        running_loss = train_head(
            model, data_iter, optimizer, scheduler, args, task, device, writer, tag,
            trained_steps, milestone, running_loss,
        )
        trained_steps = milestone

        if task == DETECTION:
            metrics = evaluate_detection(model, val_loader, args, device, writer, tag, milestone)
        else:
            metrics = evaluate_segmentation(model, val_loader, args, task, device, writer, tag, milestone)
        results[milestone] = metrics
        if is_main_process() and metrics:
            recorder.add(tag, metrics, milestone, writer=writer)
            summary = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items() if not isinstance(v, dict))
            print(f"  [{tag}] after {milestone} steps: {summary}")

    del model, backbone, head, optimizer
    torch.cuda.empty_cache()
    return results


def main(args):
    if args.torch_ddp:
        init_multigpu_via_torch_ddp()
    device = torch.device(get_gpu_device())

    tasks = [t for t in TASKS if getattr(args, f"{t}_enabled")]
    protocols = [p for p in PROTOCOLS if getattr(args, f"{PROTOCOL_NAMES[p]}_enabled")]
    if not tasks or not protocols:
        raise ValueError("Nothing to run: enable at least one task and one protocol.")
    print(f"Tasks: {tasks}  Protocols: {[PROTOCOL_NAMES[p] for p in protocols]}")

    args.load_segmentation = SEMANTIC in tasks or PANOPTIC in tasks

    world_size = dist.get_world_size() if torch_ddp_active() else 1
    # Every backbone and protocol trains with the same total batch; what does not fit per step is accumulated.
    if POINTCEPT_TOTAL_BATCH_SIZE % (args.batch_size * world_size):
        raise ValueError(
            f"--batch_size {args.batch_size} x {world_size} GPUs must divide the total batch of {POINTCEPT_TOTAL_BATCH_SIZE}."
        )
    args.accumulation_steps = POINTCEPT_TOTAL_BATCH_SIZE // (args.batch_size * world_size)
    print(f"Total batch {POINTCEPT_TOTAL_BATCH_SIZE} = {args.batch_size} per GPU x {world_size} GPUs x {args.accumulation_steps} accumulation steps")

    dataset_module = importlib.import_module(LABELLED_DATASET_MODULES[args.dataset])
    if not args.pc_range_filter:
        # The dataset crops to its module's PC_RANGE at load time and forked loader workers inherit the override.
        dataset_module.PC_RANGE = [-math.inf, math.inf] * 3
    # Keep sparse frames: the pre-training MIN_NUM_POINTS filter would silently swap them for random others.
    dataset_module.MIN_NUM_POINTS = 1

    loader_kwargs = dict(
        version=args.version,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        load_annotations=True,  # rotate_cloud and the OD head both need them
        dataset_type=args.dataset,
        load_segmentation=args.load_segmentation,
    )
    train_loader, _ = get_dataloader(
        train_flag=True,
        shuffle=True,
        drop_last=True,
        seed=PROBE_SEED,
        **loader_kwargs,
    )
    # Test time is key frames only: the lidarseg/panoptic labels are defined on
    # the key frame, so aggregating sweeps would only add unlabelled points and
    # make the reported numbers depend on the sweep count.
    val_loader, _ = get_dataloader(
        train_flag=False,
        shuffle=False,
        drop_last=False,
        seed=PROBE_SEED,
        **loader_kwargs,
    )
    print(f"Evaluating on {_num_eval_batches(val_loader, args)} of {len(val_loader)} validation batches per rank")

    # Class spaces come from the dataset: nuScenes scores the lidarseg classes and
    # ignores id 0, while the ZOD/Argoverse labels are its own object classes with
    # id 0 meaning background.
    args.class_name_to_id = train_loader.dataset.get_class_name_to_id()
    args.seg_class_name_to_id = train_loader.dataset.get_seg_class_name_to_id()
    args.seg_thing_ids = train_loader.dataset.get_seg_thing_ids()
    args.seg_ignore_id = train_loader.dataset.get_seg_ignore_id()
    args.num_seg_classes = len(args.seg_class_name_to_id) + 1  # + background/ignore at id 0

    probe_backbone = build_backbone(args).to(device=device)
    args.feature_dim, args.stage_dims = probe_feature_dim(probe_backbone, val_loader, args, device)
    print(f"Backbone emits {args.feature_dim} features per input point")
    del probe_backbone
    torch.cuda.empty_cache()

    writer = (
        SummaryWriter(log_dir=os.path.join(args.output_directory, "j3pa_final_eval"))
        if is_main_process()
        else None
    )
    recorder = ResultsRecorder()

    # Could maybe have made this more efficient by combining the tasks into one run of backbone many of 
    # head but that will get complicated quickly for fine-tuning case so lets do this way:
    for task in tasks:
        for protocol in protocols:
            run_experiment(args, task, protocol, train_loader, val_loader, device, writer, recorder)

    if is_main_process():
        recorder.print_summary()
        recorder.publish(writer)
        writer.close()

    _cleanup_dataloader(train_loader)
    _cleanup_dataloader(val_loader)
    if torch_ddp_active():
        dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--checkpoint", type=str, required=True, help="J3PA checkpoint to evaluate.")
    parser.add_argument("--constants", type=str, required=True, help="constants.py (or legacy constants_values.txt) of the j3pa run, or its directory.")
    parser.add_argument("--use_student", action="store_true", help="Use the student weights instead of the teacher/EMA weights.")
    parser.add_argument("--output_directory", type=str, required=True)
    parser.add_argument("--dataset",  choices=list(LABELLED_DATASET_MODULES), default="NuScenes")

    parser.add_argument("--semantic_segmentation", dest="semantic_enabled", action="store_true")
    parser.add_argument("--panoptic_segmentation", dest="panoptic_enabled", action="store_true")
    parser.add_argument("--detection", dest="detection_enabled", action="store_true")

    parser.add_argument("--linear_probe", dest="linear_probe_enabled", action="store_true")
    parser.add_argument("--nl_probe", dest="nl_probe_enabled", action="store_true")
    parser.add_argument("--full_finetune", dest="full_finetune_enabled", action="store_true")

    parser.add_argument("--version", choices=["mini", "full"], default="full")
    parser.add_argument("--batch_size", type=int, default=PROBE_BATCH_SIZE, help="Per GPU and micro-step. Gradients are accumulated up to a total batch of 64.")
    parser.add_argument("--num_workers", type=int, default=PROBE_NUM_WORKERS)
    parser.add_argument("--prefetch_factor", type=int, default=1)
    parser.add_argument("--eval_at_steps", type=int, nargs="+", default=[500, 1000, 5000, 20000, 50000], help="Training budgets to evaluate at. Training is resumed between them, so the largest value is the total number of steps.")
    parser.add_argument("--finetune_probe_steps", type=int, default=5000, help="Steps --full_finetune first trains the head as a linear probe on the frozen backbone before unfreezing it; 0 skips the probe. --eval_at_steps counts the fine-tuning steps after it.")
    parser.add_argument("--finetune_warmup_steps", type=int, default=2000, help="Learning-rate warm-up steps of --full_finetune once the backbone is unfrozen.")
    parser.add_argument("--limit_eval_batches", type=int, default=None, help="Evaluate only this many validation batches. Default is the whole split; use it to shorten debug runs.")
    parser.add_argument("--num_vis_samples", type=int, default=4, help="Validation frames rendered to TensorBoard at each milestone. 0 disables the qualitative figures.")
    parser.add_argument("--pc_range_filter", action="store_true", help="Crop clouds and boxes to PC_RANGE as in pre-training. Off by default: every point is kept.")
    parser.add_argument("--voxel_size", type=float, default=None, help="Input voxel grid size in metres the backbone runs at. Default: the run's VOXEL_SIZE.")
    parser.add_argument("--torch_ddp", action="store_true")

    args = parser.parse_args()

    args.eval_at_steps = sorted(set(args.eval_at_steps))
    if args.eval_at_steps[0] <= 0:
        parser.error("--eval_at_steps must be positive")
    if args.voxel_size is not None and args.voxel_size <= 0:
        parser.error("--voxel_size must be positive")

    if not any([args.semantic_enabled, args.panoptic_enabled, args.detection_enabled]):
        args.semantic_enabled = args.panoptic_enabled = True
        args.detection_enabled = args.dataset not in ("SemanticKITTI", "Waymo")
    if args.dataset in ("SemanticKITTI", "Waymo") and args.detection_enabled:
        parser.error(f"{args.dataset} has no detection boxes")
    if not any([args.linear_probe_enabled, args.nl_probe_enabled, args.full_finetune_enabled]):
        args.linear_probe_enabled = args.nl_probe_enabled = args.full_finetune_enabled = True

    if args.finetune_probe_steps < 0:
        parser.error("--finetune_probe_steps must be non-negative")
    # OneCycleLR divides by zero on a one-step warm-up.
    if args.finetune_warmup_steps < 0 or args.finetune_warmup_steps == 1:
        parser.error("--finetune_warmup_steps must be 0 or at least 2")
    if args.full_finetune_enabled and args.finetune_warmup_steps >= args.eval_at_steps[-1]:
        parser.error(
            f"--finetune_warmup_steps ({args.finetune_warmup_steps}) must be shorter than the longest "
            f"--eval_at_steps budget ({args.eval_at_steps[-1]}) when running --full_finetune"
        )

    return args


if __name__ == "__main__":
    main(parse_args())
