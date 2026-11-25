"""Probe evaluation with test-time augmentation for a pretrained point-cloud encoder.

Uses the nuScenes settings below for all supported datasets.

Loads a J3PA checkpoint, freezes it and trains a single
linear layer on its concatenated multi-stage features, following Pointcept's
nuScenes linear-probe recipe:

    - CE + Lovasz loss, AdamW 2e-3 with OneCycle over the longest budget
    - rotate/scale/flip/jitter + Mix3D train augmentation
    - frozen backbone kept in train mode while training (random curve orders)
    - no xyz in the head input, no ignore-class prediction
    - every point kept (no PC_RANGE crop)

Added test-time augmentation with grid shifts and curve order repetitions.

Tasks: semantic segmentation (default) and, with --panoptic_segmentation, panoptic
segmentation as in PointGroup / Pointcept's PointGroup probes: semantic logits plus an MLP offset
branch trained with offset L1 + direction losses, offsets clustered by linking the shifted
points of each predicted class, and PQ computed as in the official nuScenes panoptic
evaluator (min 15 points for FP/FN, PQ-dagger). With --detection, 3D object detection per stage-0 voxel: every
voxel gets the multi-stage features of its voxels at all stages plus its own position, is classified into the class
of the box it lies in (or background) and, if in a box, regresses that box's centre offset, size, sin/cos yaw and BEV
velocity. Every voxel proposes its box (centre = voxel position + offset, score = class probability) and 3D NMS
(pytorch3d IoU) keeps the best of overlapping boxes, scored with the nuScenes devkit's mAP/NDS. 

Protocols: linear probe (default); --nl_probe, a Linear-ReLU-Linear head on the same frozen features;
--full_finetune, encoder and linear head trained with Pointcept's fine-tune recipe
without its decoder. It first trains the head on the frozen encoder for --finetune_probe_steps (linear probe recipe),
then unfreezes the encoder with a --finetune_warmup_steps learning-rate warm-up; its budgets count the fine-tuning
steps after that first stage.
Every requested task x protocol pair is trained and evaluated with the same TTA.

At each training budget (5k/10k/20k/50k/100k steps by default) the probe is evaluated
with test-time augmentation over the full validation split, at input point resolution, on single key frames. 

Example:
    PYTHONPATH=src python src/evaluation/run_eval_bellsandwhistels.py \
        --dataset NuScenes \
        --checkpoint experiments/tst_nusc/checkpoints/epoch_100.pt \
        --constants experiments/tst_nusc \
        --output_directory experiments/tst_nusc/
"""

import argparse
import contextlib
import importlib
import itertools
import math
import os
import sys
import time
from pathlib import Path

import torch
import numpy as np
import torch.distributed as dist
from pytorch3d.ops import box3d_overlap
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.backbones import FROZEN, FULL, build_backbone
from evaluation.heads import DetectionHead, PanopticSegmentationHead, SemanticSegmentationHead
from evaluation.nuscenes_od_metrics import evaluate_nuscenes_detection
from evaluation.od_post_processing import Detection, get_gt_boxes
from evaluation.panoptic_metrics import PanopticQuality, cluster_instances
from evaluation.qualitative import log_detection_comparison, log_instance_comparison, log_semantic_comparison
from evaluation.reporting import ResultsRecorder
from evaluation.upcast import upcast_to_input_resolution
from j3pa.pt_transf import get_grid_idx
from evaluation.probe_utils import _calc_mavg_loss, _cleanup_dataloader, _flatten, limit_pc_to_fov
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
from utils.geometry import quaternion_to_yaw, yaw_to_quaternion
from constants import PROBE_HIDDEN_DIM, PROBE_NUM_WORKERS, PROBE_SEED, XYZ_SLICE

SEMANTIC = "semantic"
PANOPTIC = "panoptic"
DETECTION = "detection"
LINEAR_PROBE = "linear_probe"
NL_PROBE = "nl_probe"
FULL_FINETUNE = "full_finetune"
BACKBONE_MODE = {LINEAR_PROBE: FROZEN, NL_PROBE: FROZEN, FULL_FINETUNE: FULL}
# Official nuScenes panoptic eval: unmatched segments below this size are neither FP nor FN.
NUSCENES_MIN_INST_POINTS = 15
DEFAULT_CLUSTER_CELL_SIZE = 0.3
# The nuScenes devkit scores at most this many boxes per frame.
DETECTION_MAX_BOXES = 500
# Detection decoding: per-point candidates under this score are dropped, one per class and centre cell (m) kept,
# at most this many per frame enter NMS, and boxes are made at least this big so pytorch3d's IoU stays defined.
DEFAULT_DETECTION_SCORE_THRESHOLD = 0.05
DEFAULT_NMS_IOU = 0.0
DETECTION_DEDUP_CELL = 0.1
DETECTION_NMS_PRE_MAX = 1000
DETECTION_MIN_SIZE = 0.05
# nuScenes detection-eval class ranges (m, BEV distance from the ego), applied to boxes and detections alike.
DETECTION_CLASS_RANGE = {
    "car": 50.0, "truck": 50.0, "construction_vehicle": 50.0, "bus": 50.0, "trailer": 50.0,
    "barrier": 30.0, "motorcycle": 40.0, "bicycle": 40.0, "pedestrian": 40.0, "traffic_cone": 30.0,
}
# Datasets with per-point labels and boxes (OctoSense and Combined are pre-training only), and the
# module whose PC_RANGE global each one crops to.
LABELLED_DATASET_MODULES = {
    "NuScenes": "data.nuscenes_dataset",
    "ZOD": "data.zod_dataset",
    "Argoverse2": "data.argoverse2_dataset",
    "TruckScenes": "data.truckscenes_dataset",
    "PandaSet": "data.pandaset_dataset",
    "KITTI": "data.kitti_dataset",
    "SemanticKITTI": "data.semantickitti_dataset",
    "Waymo": "data.waymo_dataset",
    "Cirrus": "data.cirrus_dataset",
}

# Pointcept's nuScenes linear-probe recipe.
POINTCEPT_LR = 2e-3
POINTCEPT_WEIGHT_DECAY = 5e-3
POINTCEPT_ONECYCLE = dict(pct_start=0.04, anneal_strategy="cos", div_factor=10.0, final_div_factor=100.0)
POINTCEPT_LOVASZ_WEIGHT = 1.0
POINTCEPT_MIX_PROB = 0.0  # Probability of applying Mix3D augmentation during probe training.
POINTCEPT_TOTAL_BATCH_SIZE = 64
# Pointcept's PointGroup linear probe, used for panoptic.
POINTCEPT_INSSEG_WEIGHT_DECAY = 0.05
POINTCEPT_INSSEG_ONECYCLE = dict(pct_start=0.05, anneal_strategy="cos", div_factor=10.0, final_div_factor=1000.0)
# Measured on a 40 GB A100 (full nuScenes, linear probe): J3PA 6 GiB; 4 GPUs give the total of 64.
# Full fine-tuning needs --batch_size 4; the rest of the 64 comes from gradient accumulation.
BATCH_SIZE_PER_GPU = 16
# Pointcept works in metres * 0.2; these are its values converted back to metres.
POINTCEPT_JITTER_SIGMA = 0.0025 / 0.2
POINTCEPT_JITTER_CLIP = 0.01 / 0.2
POINTCEPT_TTA_SCALES = (0.9, 0.95, 1.0, 1.05, 1.1)
# Pointcept's nuScenes fine-tune recipe; its "block" param group (the encoder) trains at the lower rate.
POINTCEPT_FT_LR = 1e-3
POINTCEPT_FT_BLOCK_LR = 2e-5
POINTCEPT_FT_WEIGHT_DECAY = 0.05
POINTCEPT_FT_ONECYCLE = dict(pct_start=0.05, anneal_strategy="cos", div_factor=10.0, final_div_factor=1000.0)
# Forwards per sample at test time; larger view grids are randomly subsampled down to this.
MAX_TTA_FORWARDS = 96


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Gradient of the Lovasz extension w.r.t. sorted errors (Alg. 1 of Berman et al. 2018)."""
    # Adapted from LovaszSoftmax (https://github.com/bermanmaxim/LovaszSoftmax), Copyright (c) 2018 Maxim Berman.
    # Licensed under the MIT License; see ../../NOTICE. Modified: type hints, no separate single-element case.
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1 - gt_sorted).cumsum(0)
    jaccard = 1.0 - intersection / union
    jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def lovasz_softmax_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int | None = None) -> torch.Tensor:
    """Pointcept's ``LovaszLoss(mode="multiclass")``: over the whole batch, averaged over the classes present."""
    probas = logits.float().softmax(dim=-1)
    if ignore_index is not None:
        valid = labels != ignore_index
        probas, labels = probas[valid], labels[valid]
    if labels.numel() == 0:
        return probas.sum() * 0.0
    losses = []
    for class_id in labels.unique():
        foreground = (labels == class_id).float()
        errors = (foreground - probas[:, class_id]).abs()
        errors_sorted, permutation = torch.sort(errors, descending=True)
        losses.append(torch.dot(errors_sorted, _lovasz_grad(foreground[permutation])))
    return torch.stack(losses).mean()


def _probe_loss(head, logits, offsets, semantic, instance, positions, batch_indices, ignore_id):
    """Pointcept's CE + Lovasz criteria, plus the panoptic head's PointGroup offset losses."""
    lovasz = POINTCEPT_LOVASZ_WEIGHT * lovasz_softmax_loss(logits, semantic, ignore_id)
    if offsets is None:
        return head.loss(logits, semantic) + lovasz
    total, _ = head.loss(
        logits=logits,
        offsets=offsets,
        semantic_labels=semantic,
        instance_labels=instance,
        positions=positions,
        batch_indices=batch_indices,
    )
    return total + lovasz


def detection_loss(head, logits, boxes, positions, annotations, batch_indices):
    """CE + Lovasz on the box classes plus the box regression losses, with targets from the boxes alone.

    Returns:
        ``(total, {term: value})``.
    """
    labels, box_targets = head.targets(positions, annotations, batch_indices)
    total, extra = head.loss(logits, boxes, labels, box_targets)
    lovasz = POINTCEPT_LOVASZ_WEIGHT * lovasz_softmax_loss(logits, labels)
    return total + lovasz, {**extra, "lovasz": lovasz.item()}


class ProbeModel(nn.Module):
    """Backbone and head as one module, so DDP can wrap them together."""

    def __init__(self, backbone, head, ignore_logit: int | None = None):
        super().__init__()
        self.backbone = backbone
        self.head = head
        # Logit of the ignore class, pinned to -inf so it is never predicted (Pointcept has no such output).
        self.ignore_logit = ignore_logit

    def forward(self, point_cloud, batch_indices, grid_shift: float = 0.0):
        """Encode and classify each input point, or each stage-0 voxel for the detection head.

        Args:
            point_cloud: (N, C) flattened points.
            batch_indices: (N,) sample index per point.
            grid_shift: Offset in metres, along every axis, of the backbone's voxel grid.

        Returns:
            ``(logits, offsets or boxes, positions, batch_indices, point_to_voxel)``, one row per input point, or per
            stage-0 voxel for detection (``point_to_voxel`` (N,) then maps each input point to its voxel, else
            None); offsets towards the instance centre come only from the panoptic head.
        """
        point_to_voxel = None
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            features, _, stage_batch, up_indices, positions = self._encode(point_cloud, batch_indices, grid_shift)
            if isinstance(self.head, DetectionHead):
                # All points of a stage-0 voxel share its features, so detection runs per voxel, on the upcast
                # features plus the voxel position (mean of its input points, unaffected by the grid shift).
                point_to_voxel, batch_indices = up_indices[0], stage_batch[0]
                counts = torch.bincount(point_to_voxel, minlength=batch_indices.shape[0]).clamp_min(1)
                positions = torch.zeros(batch_indices.shape[0], 3, device=positions.device).index_add_(
                    0, point_to_voxel, positions.float()
                ) / counts[:, None]
                identity = torch.arange(positions.shape[0], device=positions.device)
                features = upcast_to_input_resolution(features, [identity, *up_indices[1:]], positions)
            else:
                # Pointcept's head sees only the backbone features, not the xyz the upcast appends.
                features = upcast_to_input_resolution(features, up_indices, positions)[:, :-3]
            # TODO: try normalising the features (e.g. non-affine BatchNorm) before the linear layer;
            # j3pa's 1024-dim deepest stage otherwise dominates the 96..1024-dim concat.
            outputs = self.head(features)
        logits, offsets = outputs if isinstance(outputs, tuple) else (outputs, None)
        if self.ignore_logit is not None:
            logits = logits.index_fill(
                -1, torch.tensor([self.ignore_logit], device=logits.device), float("-inf")
            )
        return logits, offsets, positions, batch_indices, point_to_voxel

    def _encode(self, point_cloud, batch_indices, grid_shift):
        """Backbone forward with its voxel grid moved by ``grid_shift`` metres along every axis."""
        if grid_shift == 0:
            return self.backbone(point_cloud, batch_indices)
        # J3PA anchors its grid at each sample's lowest point, so translating the cloud would move nothing.
        xyz = point_cloud[:, XYZ_SLICE]
        # The backbone's, not the encoder's: --voxel_size overrides it.
        voxel_size = self.backbone.voxel_size
        _, min_pos = get_grid_idx(xyz, batch_indices, voxel_size)
        grid_idx = torch.div(xyz - min_pos[batch_indices] + grid_shift, voxel_size, rounding_mode="trunc").long()
        return self.backbone(point_cloud, batch_indices, grid_idx=grid_idx)


def _coarsest_voxel_size(backbone) -> float:
    """Edge in metres of the backbone's deepest voxels; shifts within it move every stage's boundaries."""
    # The encoder pools by 2 into its first stage and between stages.
    return backbone.voxel_size * 2 ** len(backbone.num_heads_per_stage)


def _tta_views(args, coarsest_voxel_size: float) -> list[tuple[bool, float, float, int]]:
    """``(flip, scale, grid_shift, curve_draw)`` per test-time forward.

    The full grid of combinations is used when it fits in ``MAX_TTA_FORWARDS``. Otherwise
    combinations are picked greedily so that every value of every setting is used, each about
    equally often, without repeating a combination. Ties are broken by a seeded shuffle, so
    every budget is evaluated on the same views.
    """
    flips = (False, True) if args.flip_aug else (False,)
    # Golden-ratio offsets: evenly spread but never a multiple of a finer stage's voxel, which would leave it unmoved.
    shifts = [(k * (math.sqrt(5) - 1) / 2) % 1 * coarsest_voxel_size for k in range(args.num_grid_shifts)]
    settings = (flips, POINTCEPT_TTA_SCALES, shifts, range(args.num_curve_orders))
    combinations = list(itertools.product(*(range(len(values)) for values in settings)))
    if len(combinations) > MAX_TTA_FORWARDS:
        shuffle = torch.randperm(len(combinations), generator=torch.Generator().manual_seed(PROBE_SEED))
        remaining = [combinations[i] for i in shuffle.tolist()]
        uses = [[0] * len(values) for values in settings]
        combinations = []
        for _ in range(MAX_TTA_FORWARDS):
            pick = min(remaining, key=lambda combo: sum(uses[s][v] for s, v in enumerate(combo)))
            remaining.remove(pick)
            combinations.append(pick)
            for s, v in enumerate(pick):
                uses[s][v] += 1
    return [tuple(values[v] for values, v in zip(settings, combo)) for combo in combinations]


def _unwrap(model):
    """The underlying module, whether or not DDP wrapped it."""
    return model.module if isinstance(model, DDP) else model


def _wrap_for_ddp(model, args, device):
    """``model`` in DDP when it is active. DDP only all-reduces the parameters that require grad when it is built."""
    if not torch_ddp_active():
        return model
    return DDP(model, device_ids=[device.index] if device.type == "cuda" else None, find_unused_parameters=False)


def _build_optimizer(trainable, task, block_ids, total_steps, warmup_steps=None):
    """AdamW plus OneCycle over ``total_steps``: the linear-probe recipe for a frozen encoder (``block_ids`` None),
    else the fine-tune recipe with the ``block_ids`` parameters at the lower rate. ``warmup_steps`` replaces the
    recipe's warm-up fraction.
    """
    if block_ids is None:
        param_groups = [dict(params=trainable, lr=POINTCEPT_LR)]
        if task in (PANOPTIC, DETECTION):
            weight_decay, onecycle = POINTCEPT_INSSEG_WEIGHT_DECAY, POINTCEPT_INSSEG_ONECYCLE
        else:
            weight_decay, onecycle = POINTCEPT_WEIGHT_DECAY, POINTCEPT_ONECYCLE
    else:
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


def _flatten_batch(point_cloud, seg_labels, device):
    """Drop padding rows from a batch, keeping labels aligned.

    `filter_and_flatten` decides what is padding by looking for all-zero rows, so
    the labels have to be filtered with that exact same mask instead of being
    flattened independently.

    Args:
        point_cloud: (B, N, C) padded clouds, already on ``device``.
        seg_labels: dict with "semantic" and "instance" of shape (B, N).
        device: Device to move the labels to.

    Returns:
        ``(points, batch_indices, semantic, instance)``.
    """
    padding_mask = torch.any(point_cloud != 0, dim=-1)  # (B, N)
    points, batch_indices = filter_and_flatten(point_cloud)
    # Labels arrive on the CPU, so move them before indexing with the GPU mask.
    semantic = seg_labels["semantic"].to(device=device)[padding_mask]
    instance = seg_labels["instance"].to(device=device)[padding_mask]
    return points, batch_indices, semantic, instance


def _load_batch(batch, device):
    """Points, batch indices, semantic and instance labels and box annotations of a loader batch, cropped to the camera FOV."""
    point_cloud, annotations, field_of_view, seg_labels = batch
    point_cloud = limit_pc_to_fov(point_cloud.to(device), field_of_view.to(device))
    annotations = {
        key: annotations[key].to(device)
        for key in ("class", "center", "size", "quats", "velocity", "num_points")
        if key in annotations
    }
    return (*_flatten_batch(point_cloud, seg_labels, device), annotations)


def _cycle(loader):
    """Yield batches forever, restarting the loader when it runs out."""
    while True:
        yield from loader


def _pointcept_train_augment(points, batch_indices, flip=True):
    """Pointcept's nuScenes semseg train augmentations, drawn per sample, on xyz in metres.

    RandomRotate(z, [-pi, pi], p=0.5) -> RandomScale([0.9, 1.1]) -> RandomFlip(p=0.5, x and y
    independently, skipped when ``flip`` is False) -> RandomJitter.

    Returns:
        ``(augmented points, angle (B,), scale (B,), signs (B, 2))`` so boxes can follow the points.
    """
    num_samples = int(batch_indices.max()) + 1
    device = points.device
    xyz = points[:, XYZ_SLICE].float()

    angle = torch.empty(num_samples, device=device).uniform_(-math.pi, math.pi)
    angle = torch.where(torch.rand(num_samples, device=device) < 0.5, angle, torch.zeros_like(angle))
    cos, sin = angle.cos()[batch_indices], angle.sin()[batch_indices]
    xyz = torch.stack([cos * xyz[:, 0] - sin * xyz[:, 1], sin * xyz[:, 0] + cos * xyz[:, 1], xyz[:, 2]], dim=-1)

    scale = torch.empty(num_samples, device=device).uniform_(0.9, 1.1)
    xyz = xyz * scale[batch_indices].unsqueeze(-1)

    signs = torch.ones(num_samples, 2, device=device)
    if flip:
        signs = torch.where(torch.rand(num_samples, 2, device=device) < 0.5, -1.0, 1.0)
        xyz[:, :2] = xyz[:, :2] * signs[batch_indices]

    xyz = xyz + (POINTCEPT_JITTER_SIGMA * torch.randn_like(xyz)).clamp(-POINTCEPT_JITTER_CLIP, POINTCEPT_JITTER_CLIP)

    augmented = points.clone()
    augmented[:, XYZ_SLICE] = xyz.to(points.dtype)
    return augmented, angle, scale, signs


def _augment_boxes(annotations, angle, scale, signs):
    """Apply the point augmentation (z rotation, then scale, then x/y sign flips) to the boxes of each sample.

    Args:
        annotations: dict with "class" (B, M), "center" (B, M, 3), "size" (B, M, 3) and "quats" (B, M, 4).
        angle, scale: (B,) rotation in radians and scale factor.
        signs: (B, 2) +-1 for the x and y flips.
    """
    cos, sin = angle.cos()[:, None], angle.sin()[:, None]
    x, y, z = annotations["center"].unbind(-1)
    center = torch.stack([cos * x - sin * y, sin * x + cos * y, z], dim=-1) * scale[:, None, None]
    center[..., :2] = center[..., :2] * signs[:, None, :]
    yaw = quaternion_to_yaw(annotations["quats"]) + angle[:, None]
    # Mirroring x maps heading (cos, sin) to (-cos, sin), i.e. pi - yaw; mirroring y gives -yaw.
    yaw = torch.where(signs[:, None, 0] < 0, math.pi - yaw, yaw)
    yaw = torch.where(signs[:, None, 1] < 0, -yaw, yaw)
    # Padding rows (class 0) keep their all-zero quaternion.
    present = (annotations["class"] != 0).unsqueeze(-1)
    quats = torch.where(present, yaw_to_quaternion(yaw.unsqueeze(-1)), annotations["quats"])
    augmented = {
        "class": annotations["class"],
        "center": center,
        "size": annotations["size"] * scale[:, None, None],
        "quats": quats,
    }
    if "velocity" in annotations:
        vx, vy = annotations["velocity"].unbind(-1)
        velocity = torch.stack([cos * vx - sin * vy, sin * vx + cos * vy], dim=-1) * scale[:, None, None]
        augmented["velocity"] = velocity * signs[:, None, :]
    return augmented


def _tta_predictions(model, points, batch_indices, semantic, instance, args):
    """Test-time augmentation: softmax summed and instance offsets averaged over ``args.tta_views``.

    Each view combines a scale, an optional x/y flip, a voxel-grid shift and a fresh draw
    of random space-filling-curve orders. Offsets are mapped back to the unaugmented frame
    before averaging; the grid shift is a translation and leaves them unchanged.

    Returns:
        ``(probabilities, offsets or None, loss)`` with ``loss`` the mean of the per-forward losses.
    """
    model = _unwrap(model)
    probabilities = torch.zeros(points.shape[0], args.num_seg_classes, device=points.device)
    offset_sum = None
    losses = []
    xy = slice(XYZ_SLICE.start, XYZ_SLICE.start + 2)
    for flip, scale, grid_shift, _ in args.tta_views:
        augmented = points.clone()
        augmented[:, XYZ_SLICE] *= scale
        if flip:
            augmented[:, xy] *= -1
        with torch.no_grad():
            logits, offsets, positions, _, _ = model(augmented, batch_indices, grid_shift=grid_shift)
            loss = _probe_loss(
                model.head, logits, offsets, semantic, instance, positions, batch_indices, args.seg_ignore_id
            )
        probabilities += logits.float().softmax(dim=-1)
        losses.append(loss)
        if offsets is not None:
            offsets = offsets.float() / scale
            if flip:
                offsets[:, :2] *= -1
            offset_sum = offsets if offset_sum is None else offset_sum + offsets
    offsets = None if offset_sum is None else offset_sum / len(args.tta_views)
    return probabilities, offsets, torch.stack(losses).mean()


def _tta_detections(model, points, batch_indices, annotations, args):
    """Detection TTA: class softmax and boxes averaged per input point over ``args.tta_views``.

    Each view's voxel predictions go to the points in the voxel. Centres (voxel position + offset), sizes and
    velocities are divided by the view's scale, and a flip (x and y mirrored, i.e. a half turn) negates their x/y
    and both sin and cos of the yaw, so all views agree on the input frame.

    Returns:
        ``(probabilities (N, C+1), boxes (N, 10), loss)``, box offsets relative to the input points, with ``loss``
        that of the fused prediction against the unaugmented boxes.
    """
    model = _unwrap(model)
    head = DetectionHead
    probabilities, boxes = 0.0, 0.0
    xy = slice(XYZ_SLICE.start, XYZ_SLICE.start + 2)
    for flip, scale, grid_shift, _ in args.tta_views:
        augmented = points.clone()
        augmented[:, XYZ_SLICE] *= scale
        if flip:
            augmented[:, xy] *= -1
        with torch.no_grad():
            logits, view_boxes, positions, _, point_to_voxel = model(augmented, batch_indices, grid_shift=grid_shift)
        view_boxes = view_boxes.float().clone()
        view_boxes[:, head.OFFSET] += positions
        view_boxes = view_boxes[point_to_voxel]
        for metric in (head.OFFSET, head.SIZE, head.VELOCITY):
            view_boxes[:, metric] /= scale
        if flip:
            view_boxes[:, [0, 1, 6, 7, 8, 9]] *= -1
        probabilities = probabilities + logits.float().softmax(dim=-1)[point_to_voxel]
        boxes = boxes + view_boxes
    num_views = len(args.tta_views)
    probabilities, boxes = probabilities / num_views, boxes / num_views
    positions = points[:, XYZ_SLICE].float()
    boxes[:, head.OFFSET] -= positions
    with torch.no_grad():
        loss, _ = detection_loss(
            model.head, probabilities.clamp_min(1e-6).log(), boxes, positions, annotations, batch_indices
        )
    return probabilities, boxes, loss


# pytorch3d's box3d_overlap vertex order: unit-cube corners (0,0,0), (1,0,0), (1,1,0), (0,1,0), then z = 1.
_UNIT_BOX_CORNERS = torch.tensor(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=torch.float32
) - 0.5


def _box_corners(centers, sizes, yaws) -> torch.Tensor:
    """(K, 8, 3) corners of yawed boxes, in pytorch3d's order."""
    local = _UNIT_BOX_CORNERS.to(centers.device)[None] * sizes[:, None]
    cos, sin = yaws.cos()[:, None], yaws.sin()[:, None]
    x, y = local[..., 0], local[..., 1]
    return torch.stack([cos * x - sin * y, sin * x + cos * y, local[..., 2]], dim=-1) + centers[:, None]


def _greedy_nms(iou: np.ndarray, threshold: float) -> list[int]:
    """Indices kept by greedy NMS over score-sorted boxes, given their pairwise IoU."""
    suppressed = np.zeros(len(iou), dtype=bool)
    keep = []
    for i in range(len(iou)):
        if suppressed[i]:
            continue
        keep.append(i)
        suppressed |= iou[i] > threshold
    return keep


def decode_detections(
    positions, probabilities, boxes, batch_indices, class_name_to_id, score_threshold, nms_iou
) -> list[np.ndarray]:
    """One candidate box per point (or voxel) predicted as an object, then class-agnostic 3D NMS per frame.

    A candidate takes its object class with that probability as score, centre = position + offset, and the
    regressed size, yaw (from sin/cos) and velocity. Per frame, candidates under ``score_threshold`` are
    dropped, the best per class and `DETECTION_DEDUP_CELL` centre cell kept (the points of one object propose
    near-identical boxes), the best `DETECTION_NMS_PRE_MAX` of those go through NMS with pytorch3d's exact 3D IoU,
    which drops every box overlapping a better one by more than ``nms_iou``, whatever the classes.

    Returns:
        Per frame, an array of the best ``DETECTION_MAX_BOXES`` detections, best first, then cut to the
        nuScenes class ranges.
    """
    head = DetectionHead
    class_names = list(class_name_to_id)
    scores, labels = probabilities[:, 1:].max(dim=-1)
    labels = labels + 1
    proposes = (probabilities.argmax(dim=-1) > 0) & (scores >= score_threshold)
    centers = positions.float() + boxes[:, head.OFFSET].float()
    sizes = boxes[:, head.SIZE].float().clamp_min(DETECTION_MIN_SIZE)
    yaws = torch.atan2(boxes[:, 6].float(), boxes[:, 7].float())
    velocities = boxes[:, head.VELOCITY].float()
    if positions.is_cuda:
        # pytorch3d's IoU kernel needs memory outside PyTorch's cache, which the TTA forwards can fill completely.
        torch.cuda.empty_cache()
    frames = []
    for frame in range(int(batch_indices.max()) + 1):
        candidates = ((batch_indices == frame) & proposes).nonzero().squeeze(-1)
        detections = []
        if len(candidates):
            candidates = candidates[scores[candidates].argsort(descending=True)]
            cells = torch.floor(centers[candidates] / DETECTION_DEDUP_CELL).long()
            keys = torch.cat([labels[candidates, None], cells], dim=-1)
            _, group = torch.unique(keys, dim=0, return_inverse=True)
            # Candidates are best first, so the smallest index of each group is its best box.
            first = torch.full((int(group.max()) + 1,), len(group), device=group.device).scatter_reduce_(
                0, group, torch.arange(len(group), device=group.device), reduce="amin"
            )
            candidates = candidates[first.sort().values][:DETECTION_NMS_PRE_MAX]
            corners = _box_corners(centers[candidates], sizes[candidates], yaws[candidates])
            _, iou = box3d_overlap(corners, corners)
            kept = candidates[_greedy_nms(iou.cpu().numpy(), nms_iou)[:DETECTION_MAX_BOXES]]
            quats = yaw_to_quaternion(yaws[kept].unsqueeze(-1))
            box_params = torch.cat([centers[kept], sizes[kept], quats], dim=-1).cpu().numpy()
            for box, label, score, velocity in zip(
                box_params, labels[kept].tolist(), scores[kept].tolist(), velocities[kept].cpu().numpy(), strict=True
            ):
                detection = Detection(box, np.array(class_names[label - 1]), np.array(score))
                detection.velocity = velocity
                detections.append(detection)
        frames.append(_in_class_range(np.array(detections, dtype=object)))
    return frames


def _num_eval_batches(loader, args) -> int:
    """Batches to evaluate: the whole split unless --limit_eval_batches is set."""
    if args.limit_eval_batches is None:
        return len(loader)
    return min(args.limit_eval_batches, len(loader))


def probe_feature_dim(backbone, loader, device) -> tuple[int, list[int]]:
    """Width of the per-point feature the head sees, and of each stage, read off one throwaway forward.

    The feature width depends on the run's stage dimensions.
    """
    points, batch_indices, _, _, _ = _load_batch(next(iter(loader)), device)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        features, _, _, up_indices, positions = backbone(points, batch_indices)
    # Minus the xyz the upcast appends, which the head does not get.
    feature_dim = upcast_to_input_resolution(features, up_indices, positions).shape[-1] - 3
    return feature_dim, [f.shape[-1] for f in features]


def train_head(model, data_iter, optimizer, scheduler, args, device, writer, tag, start_step, end_step, running_loss):
    """Train from ``start_step`` up to ``end_step`` optimizer steps.

    Training is resumed in segments so the same run can be evaluated at several
    budgets without retraining from scratch for each one.

    Returns:
        The moving-average loss, to be threaded into the next segment.
    """
    model.train()
    # Pointcept keeps the frozen backbone in train mode; J3PA has no BatchNorm statistics to drift.
    _unwrap(model).backbone.train(True)
    detection = isinstance(_unwrap(model).head, DetectionHead)

    start_time = time.perf_counter()
    for step in range(start_step, end_step):
        for micro_step in range(args.accumulation_steps):
            points, batch_indices, semantic, instance, annotations = _load_batch(next(data_iter), device)
            points, angle, scale, signs = _pointcept_train_augment(points, batch_indices, flip=args.flip_aug)
            # Mix3D as in Pointcept's point_collate_fn: overlay consecutive pairs of scenes (not for boxes).
            #Off by default here anyway. 
            if not detection and torch.rand(()) < POINTCEPT_MIX_PROB:
                # Keep the two scenes' instance ids apart once they share a batch index.
                instance = torch.where(instance > 0, instance + (batch_indices % 2) * (instance.max() + 1), instance)
                batch_indices = torch.div(batch_indices, 2, rounding_mode="floor")

            # Gradients are only all-reduced across ranks on the last micro-batch.
            last_micro_step = micro_step == args.accumulation_steps - 1
            with contextlib.nullcontext() if last_micro_step or not isinstance(model, DDP) else model.no_sync():
                logits, offsets, positions, output_batch, _ = model(points, batch_indices)
                if detection:
                    loss, _ = detection_loss(
                        _unwrap(model).head, logits, offsets, positions,
                        _augment_boxes(annotations, angle, scale, signs), output_batch,
                    )
                else:
                    loss = _probe_loss(
                        _unwrap(model).head, logits, offsets, semantic, instance, positions, batch_indices, args.seg_ignore_id
                    )
                (loss / args.accumulation_steps).backward()
            running_loss = _calc_mavg_loss(running_loss, loss.item())

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        if writer is not None:
            writer.add_scalar(f"{tag}/train_loss", running_loss, step)
        print(f"  [{tag}] train step {step + 1}/{end_step} loss {running_loss:.4f}", end="\r")

    print()
    seconds = time.perf_counter() - start_time
    print(f"  [{tag}] {end_step - start_step} train steps in {seconds:.0f}s ({seconds / max(1, end_step - start_step):.2f} s/step)")
    return running_loss


def _set_eval_mode(model, args):
    """Eval mode, except where train mode is what draws the random curve orders TTA averages over."""
    model.eval()
    # J3PA only draws random curve orders in train mode (it has no dropout/BatchNorm).
    _unwrap(model).backbone.train(args.num_curve_orders > 1)


def _in_class_range(objects) -> np.ndarray:
    """Keep boxes within their class's nuScenes evaluation range (BEV distance from the ego)."""
    keep = [
        np.hypot(obj.box[0], obj.box[1]) <= DETECTION_CLASS_RANGE.get(str(obj.class_name), math.inf) for obj in objects
    ]
    return objects[np.array(keep, dtype=bool)] if len(objects) else objects


def scored_ground_truth(annotations, class_name_to_id) -> list[np.ndarray]:
    """Per frame, the boxes the nuScenes eval scores: hit by a lidar or radar point and within class range.

    Boxes carry their ``velocity`` when the dataset has it.
    """
    if "num_points" in annotations:
        annotations = {**annotations, "class": torch.where(annotations["num_points"] > 0, annotations["class"], 0)}
    frames = get_gt_boxes(annotations, class_name_to_id)
    if "velocity" in annotations:
        for frame, boxes in enumerate(frames):
            # get_gt_boxes keeps the non-padding rows in order.
            rows = (annotations["class"][frame] != 0).nonzero().squeeze(-1)
            for box, velocity in zip(boxes, annotations["velocity"][frame][rows].cpu().numpy(), strict=True):
                box.velocity = velocity
    return [_in_class_range(frame) for frame in frames]


def nuscenes_detection_metrics(ground_truths, predictions, class_name_to_id) -> dict:
    """nuScenes devkit mAP, TP errors and NDS, printed per class and flattened for the per-class charts."""
    metrics = evaluate_nuscenes_detection(ground_truths, predictions, list(class_name_to_id))
    per_class_tp = metrics.pop("per_class_tp")
    for name, ap in metrics["per_class_ap"].items():
        errors = "  ".join(f"{k}={v:.3f}" for k, v in per_class_tp[name].items())
        print(f"    {name:<22} AP={ap:.3f}  {errors}")
    for error, short in (("trans_err", "ate"), ("scale_err", "ase"), ("orient_err", "aoe")):
        metrics[f"per_class_{short}"] = {name: errors[error] for name, errors in per_class_tp.items()}
    return metrics


def evaluate_detection(model, loader, args, device, writer=None, tag="", step=0) -> dict:
    """nuScenes devkit mAP, TP errors and NDS of the TTA-averaged boxes; val loss is the mean over views."""
    _set_eval_mode(model, args)
    num_batches = _num_eval_batches(loader, args)
    loss_sum = torch.zeros(1, dtype=torch.float64, device=device)
    loss_batches = torch.zeros(1, dtype=torch.float64, device=device)
    local_predictions, local_ground_truths = [], []

    for eval_idx, batch in enumerate(loader):
        if eval_idx >= num_batches:
            break
        points, batch_indices, _, _, annotations = _load_batch(batch, device)
        probabilities, boxes, loss = _tta_detections(model, points, batch_indices, annotations, args)
        loss_sum += loss.item()
        loss_batches += 1

        detected = decode_detections(
            points[:, XYZ_SLICE], probabilities, boxes, batch_indices,
            args.class_name_to_id, args.detection_score_threshold, args.nms_iou,
        )
        truth = scored_ground_truth(annotations, args.class_name_to_id)
        local_predictions.extend(detected)
        local_ground_truths.extend(truth)

        if eval_idx < args.num_vis_samples and writer is not None:
            # First sample of the batch only: the loader is unshuffled, so this is the same frame at every budget.
            frame = batch_indices == 0
            log_detection_comparison(
                writer, f"qualitative/{tag}/frame_{eval_idx}/boxes", step,
                points[frame][:, XYZ_SLICE], 1.0 - probabilities[frame, 0],
                truth[0], detected[0], args.class_name_to_id,
            )

        print(f"  eval step {eval_idx + 1}/{num_batches}", end="\r")
    print()

    if torch_ddp_active():
        world_size = dist.get_world_size()
        gathered_predictions, gathered_ground_truths = [None] * world_size, [None] * world_size
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

    return nuscenes_detection_metrics(local_ground_truths, local_predictions, args.class_name_to_id)


def evaluate(model, loader, args, device, writer=None, tag="", step=0) -> dict:
    """mIoU/mAcc/allAcc under TTA, plus PQ/SQ/RQ for the panoptic head; val loss is the mean over views.

    ``writer``/``step`` file the qualitative renders under ``step`` so the same
    frame can be compared across budgets.
    """
    _set_eval_mode(model, args)
    num_classes = args.num_seg_classes
    tp_counts = torch.zeros(num_classes, dtype=torch.long, device=device)
    fp_counts = torch.zeros(num_classes, dtype=torch.long, device=device)
    fn_counts = torch.zeros(num_classes, dtype=torch.long, device=device)
    correct_fg = torch.zeros(1, dtype=torch.long, device=device)
    total_fg = torch.zeros(1, dtype=torch.long, device=device)
    panoptic_quality = PanopticQuality(
        args.seg_class_name_to_id, args.seg_thing_ids, args.seg_ignore_id, device=device, min_points=NUSCENES_MIN_INST_POINTS
    ) if isinstance(_unwrap(model).head, PanopticSegmentationHead) else None
    num_batches = _num_eval_batches(loader, args)
    loss_sum = torch.zeros(1, dtype=torch.float64, device=device)
    loss_batches = torch.zeros(1, dtype=torch.float64, device=device)

    for eval_idx, batch in enumerate(loader):
        if eval_idx >= num_batches:
            break
        points, batch_indices, semantic, instance, _ = _load_batch(batch, device)
        probabilities, offsets, loss = _tta_predictions(model, points, batch_indices, semantic, instance, args)
        loss_sum += loss.item()
        loss_batches += 1

        # Ignore-labelled points must not contribute to IoU at all, otherwise a
        # prediction on noise would be charged as a false positive.
        labelled = torch.ones_like(semantic, dtype=torch.bool) if args.seg_ignore_id is None else semantic != args.seg_ignore_id
        accumulate_seg_counts(
            tp_counts, fp_counts, fn_counts, correct_fg, total_fg,
            probabilities[labelled], semantic[labelled],
        )
        positions = points[:, XYZ_SLICE]
        predicted_semantic = probabilities.argmax(dim=-1)
        predicted_instance = None
        if panoptic_quality is not None:
            predicted_instance = cluster_instances(
                positions.float() + offsets,
                predicted_semantic,
                batch_indices,
                thing_ids=args.seg_thing_ids,
                voxel_size=args.cluster_cell_size,
                min_points=args.cluster_min_points,
            )
            panoptic_quality.update(predicted_semantic, predicted_instance, semantic, instance, batch_indices)

        if eval_idx < args.num_vis_samples and writer is not None:
            # First sample of the batch only: the loader is unshuffled, so this is
            # the same frame at every budget.
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
    }
    if panoptic_quality is not None:
        panoptic = panoptic_quality.compute(metrics["per_class_iou"])
        panoptic["per_class_pq"] = panoptic.pop("per_class")
        results.update(panoptic)
    return results


def train_and_evaluate(args, task, protocol, train_loader, val_loader, device, writer, recorder) -> dict:
    """Train one (task, protocol) probe, evaluating at each budget in ``args.eval_at_steps``.

    Each budget is handed to ``recorder`` as soon as it is measured, so the
    scalars reach TensorBoard during the run instead of only at the very end.

    Returns:
        ``{train_steps: metrics}``.
    """
    tag = f"{task}/{protocol}"
    print("~" * 100)
    print(f"Running {tag}")

    # Rebuilt per probe, so one probe's train-mode settings never leak into the next.
    backbone = build_backbone(args).to(device=device)
    args.tta_views = _tta_views(args, _coarsest_voxel_size(backbone))
    print(f"  {len(args.tta_views)} test-time forwards per sample")
    head_input_dim = args.feature_dim
    if task == DETECTION:
        head_input_dim += 3  # the stage-0 voxel position
    hidden_dim = PROBE_HIDDEN_DIM if protocol == NL_PROBE else None
    if task == PANOPTIC:
        head = PanopticSegmentationHead(
            in_feat_dim=head_input_dim,
            num_classes=args.num_seg_classes,
            thing_ids=args.seg_thing_ids,
            hidden_dim=hidden_dim,
            ignore_id=args.seg_ignore_id,
        )
    elif task == DETECTION:
        head = DetectionHead(
            in_feat_dim=head_input_dim, num_classes=len(args.class_name_to_id), hidden_dim=hidden_dim
        )
    else:
        head = SemanticSegmentationHead(
            in_feat_dim=head_input_dim,
            num_classes=args.num_seg_classes,
            ignore_id=args.seg_ignore_id,
            hidden_dim=hidden_dim,
        )
    ignore_logit = None if task == DETECTION else args.seg_ignore_id
    model = ProbeModel(backbone=backbone, head=head.to(device=device), ignore_logit=ignore_logit)
    # Pointcept's fine-tuning "block" group: the encoder trains at the lower rate.
    block_ids = None if BACKBONE_MODE[protocol] == FROZEN else {id(p) for p in backbone.parameters()}
    data_iter = _cycle(train_loader)
    running_loss = None

    if protocol == FULL_FINETUNE and args.finetune_probe_steps > 0:
        # A random head would push arbitrary gradients into the pretrained encoder.
        backbone.set_trainable(FROZEN)
        probe = _wrap_for_ddp(model, args, device)
        trainable = [p for p in probe.parameters() if p.requires_grad]
        print(
            f"  Linear probe ({sum(p.numel() for p in trainable) / 1e6:.3f}M trainable parameters) "
            f"for {args.finetune_probe_steps} steps before fine-tuning"
        )
        optimizer, scheduler = _build_optimizer(trainable, task, None, args.finetune_probe_steps)
        running_loss = train_head(
            probe, data_iter, optimizer, scheduler, args, device, writer, f"{tag}/linear_probe_init",
            0, args.finetune_probe_steps, running_loss,
        )
        # The probe's DDP wrapper does not all-reduce the encoder, so it is rebuilt once the encoder is unfrozen.
        del probe, optimizer, scheduler

    backbone.set_trainable(BACKBONE_MODE[protocol])
    model = _wrap_for_ddp(model, args, device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"  {sum(p.numel() for p in trainable) / 1e6:.3f}M trainable parameters")

    # Spans the longest budget, so earlier budgets see a prefix of the same schedule.
    optimizer, scheduler = _build_optimizer(
        trainable, task, block_ids, args.eval_at_steps[-1],
        warmup_steps=args.finetune_warmup_steps if protocol == FULL_FINETUNE else None,
    )

    results, trained_steps = {}, 0
    for budget in args.eval_at_steps:
        running_loss = train_head(
            model, data_iter, optimizer, scheduler, args, device, writer, tag, trained_steps, budget, running_loss,
        )
        trained_steps = budget

        metrics = (evaluate_detection if task == DETECTION else evaluate)(model, val_loader, args, device, writer, tag, budget)
        results[budget] = metrics
        if is_main_process() and metrics:
            recorder.add(tag, metrics, budget, writer=writer)
            summary = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items() if not isinstance(v, dict))
            print(f"  [{tag}] after {budget} steps: {summary}")

    del model, backbone, head, optimizer
    torch.cuda.empty_cache()
    return results


def main(args):
    # Ragged point batches fragment the CUDA cache; must be set before the first allocation.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # Under torchrun without DDP every rank would land on cuda:0 and train its own probe.
    if args.torch_ddp or int(os.environ.get("WORLD_SIZE", "1")) > 1:
        init_multigpu_via_torch_ddp()
    device = torch.device(get_gpu_device())

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
        # Pointcept scores every point; the dataset crops to its module's PC_RANGE at load time
        # and forked loader workers inherit the override.
        dataset_module.PC_RANGE = [-math.inf, math.inf] * 3
    # Keep sparse frames: the pre-training MIN_NUM_POINTS filter would silently swap them for random others.
    dataset_module.MIN_NUM_POINTS = 1

    loader_kwargs = dict(
        version=args.version,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        load_annotations=True,  # the field of view only comes with the annotations
        dataset_type=args.dataset,
        load_segmentation=True,
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
    print(f"Evaluating after {args.eval_at_steps} steps ({len(train_loader) // args.accumulation_steps} steps per epoch per rank)")

    # Class space comes from the dataset: nuScenes scores the lidarseg classes and
    # ignores id 0, while for ZOD/Argoverse id 0 means background.
    args.seg_class_name_to_id = train_loader.dataset.get_seg_class_name_to_id()
    args.seg_ignore_id = train_loader.dataset.get_seg_ignore_id()
    args.seg_thing_ids = train_loader.dataset.get_seg_thing_ids()
    args.num_seg_classes = len(args.seg_class_name_to_id) + 1  # + background/ignore at id 0
    args.class_name_to_id = train_loader.dataset.get_class_name_to_id()  # detection classes

    probe_backbone = build_backbone(args).to(device=device)
    args.feature_dim, args.stage_dims = probe_feature_dim(probe_backbone, val_loader, device)
    print(f"Head sees {args.feature_dim} features per input point")
    del probe_backbone
    torch.cuda.empty_cache()

    writer = (
        SummaryWriter(log_dir=os.path.join(args.output_directory, "j3pa_final_eval_withTTA"))
        if is_main_process()
        else None
    )
    recorder = ResultsRecorder()
    for protocol in args.protocols:
        for task in args.tasks:
            train_and_evaluate(args, task, protocol, train_loader, val_loader, device, writer, recorder)

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

    parser.add_argument("--semantic_segmentation", dest="semantic_enabled", action="store_true", help="Default when no task flag is given.")
    parser.add_argument("--panoptic_segmentation", dest="panoptic_enabled", action="store_true", help="On NuScenes this needs the nuScenes-panoptic add-on for real instance ids.")
    parser.add_argument("--detection", dest="detection_enabled", action="store_true", help="3D object detection on the dataset's box classes, scored with the nuScenes devkit's mAP/NDS.")
    parser.add_argument("--linear_probe", dest="linear_probe_enabled", action="store_true", help="Default when no protocol flag is given.")
    parser.add_argument("--nl_probe", dest="nl_probe_enabled", action="store_true", help=f"Linear-ReLU-Linear head with {PROBE_HIDDEN_DIM} hidden units.")
    parser.add_argument("--full_finetune", dest="full_finetune_enabled", action="store_true", help="Encoder and linear head trained, after --finetune_probe_steps of head-only training.")

    parser.add_argument("--version", choices=["mini", "full"], default="full")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE_PER_GPU, help="Per GPU and step. Gradients are accumulated up to a total batch of 64 whatever the backbone; use 4 for full fine-tuning.")
    parser.add_argument("--num_workers", type=int, default=PROBE_NUM_WORKERS)
    parser.add_argument("--prefetch_factor", type=int, default=1)
    parser.add_argument("--eval_at_steps", type=int, nargs="+", default=[5000, 10000, 20000, 50000, 100000], help="Training budgets to evaluate at. Training is resumed between them, so the largest value is the total number of steps.")
    parser.add_argument("--finetune_probe_steps", type=int, default=5000, help="Steps --full_finetune first trains the head as a linear probe on the frozen encoder before unfreezing it; 0 skips them. --eval_at_steps counts the fine-tuning steps after them.")
    parser.add_argument("--finetune_warmup_steps", type=int, default=2000, help="Learning-rate warm-up steps of --full_finetune once the encoder is unfrozen.")
    parser.add_argument("--no_flip_aug", dest="flip_aug", action="store_false", help="Drop the x/y flips from the train augmentation and the TTA views.")
    parser.add_argument("--num_curve_orders", type=int, default=4, help="Random space-filling-curve draws per TTA view. 1 keeps j3pa's fixed eval-mode orders.")
    parser.add_argument("--num_grid_shifts", type=int, default=4, help="Voxel-grid offsets per TTA view, spread across the backbone's coarsest voxel so every stage's boundaries move. 1 disables the shift.")
    parser.add_argument("--cluster_cell_size", type=float, default=DEFAULT_CLUSTER_CELL_SIZE, help="Panoptic: side in metres of the cells the predicted instance centres are linked in; touching occupied cells of one predicted class form one instance.")
    parser.add_argument("--cluster_min_points", type=int, default=1, help="Panoptic: drop predicted instances with fewer points. The official PQ never counts segments under 15 points as false positives, but they can still match small objects.")
    parser.add_argument("--detection_score_threshold", type=float, default=DEFAULT_DETECTION_SCORE_THRESHOLD, help="Detection: drop per-voxel box candidates whose class probability is lower.")
    parser.add_argument("--nms_iou", type=float, default=DEFAULT_NMS_IOU, help="Detection: NMS drops boxes whose 3D IoU with a better box (any class) exceeds this; 0 leaves no overlapping boxes.")
    parser.add_argument("--limit_eval_batches", type=int, default=None, help="Evaluate only this many validation batches. Default is the whole split; use it to shorten debug runs.")
    parser.add_argument("--num_vis_samples", type=int, default=4, help="Validation frames rendered to TensorBoard at each budget. 0 disables the qualitative figures.")
    parser.add_argument("--pc_range_filter", action="store_true", help="Crop clouds to PC_RANGE as in pre-training. Off by default: Pointcept scores every point.")
    parser.add_argument("--voxel_size", type=float, default=None, help="Input voxel grid size in metres the backbone runs at. Default: the run's VOXEL_SIZE.")
    parser.add_argument("--torch_ddp", action="store_true")

    args = parser.parse_args()

    args.eval_at_steps = sorted(set(args.eval_at_steps))
    if args.eval_at_steps[0] <= 0:
        parser.error("--eval_at_steps must be positive")
    if args.voxel_size is not None and args.voxel_size <= 0:
        parser.error("--voxel_size must be positive")
    args.tasks = [
        task for task, on in (
            (SEMANTIC, args.semantic_enabled), (PANOPTIC, args.panoptic_enabled), (DETECTION, args.detection_enabled)
        ) if on
    ] or [SEMANTIC]
    args.protocols = [
        p for p, on in (
            (LINEAR_PROBE, args.linear_probe_enabled),
            (NL_PROBE, args.nl_probe_enabled),
            (FULL_FINETUNE, args.full_finetune_enabled),
        ) if on
    ] or [LINEAR_PROBE]
    if args.num_curve_orders < 1:
        parser.error("--num_curve_orders must be at least 1")
    if args.finetune_probe_steps < 0:
        parser.error("--finetune_probe_steps must be non-negative")
    if args.finetune_warmup_steps < 0 or args.finetune_warmup_steps == 1:
        parser.error("--finetune_warmup_steps must be 0 or at least 2")
    if FULL_FINETUNE in args.protocols and args.finetune_warmup_steps >= args.eval_at_steps[-1]:
        parser.error("--finetune_warmup_steps must be shorter than the longest --eval_at_steps budget")
    if args.num_grid_shifts < 1:
        parser.error("--num_grid_shifts must be at least 1")
    if max(args.num_grid_shifts, args.num_curve_orders) > MAX_TTA_FORWARDS:
        parser.error(f"--num_grid_shifts and --num_curve_orders must fit in MAX_TTA_FORWARDS={MAX_TTA_FORWARDS}")

    if args.dataset in ("SemanticKITTI", "Waymo") and DETECTION in args.tasks:
        parser.error(f"{args.dataset} has no detection boxes")
    return args


if __name__ == "__main__":
    main(parse_args())
