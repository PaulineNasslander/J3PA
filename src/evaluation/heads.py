"""Task heads used for the final NuScenes benchmarks.

Every head takes ``hidden_dim``: ``None`` makes it a single linear layer for the
linear probe, an int makes it Linear-ReLU-Linear for every other protocol. They
stay small so the probe measures the backbone rather than the head, and run in
fp32 even when the backbone is in bf16, matching the online probes in
`online_segmntation/`.
"""

import torch
import torch.nn.functional as F
from torch import nn

from evaluation.od_head import ORIENTATION_DIM, POSITION_OFFSET_DIM, SIZE_DIM
from utils.geometry import quaternion_to_yaw


class SemanticSegmentationHead(nn.Module):
    """Per-point classifier over a dataset's segmentation classes.

    Linear by default; with ``hidden_dim`` set it is Linear-ReLU-Linear (NL probe).
    Output index 0 is the ignore/background class, so logits line up one-to-one
    with the label ids the dataset produces.
    """

    def __init__(self, in_feat_dim: int, num_classes: int, ignore_id: int | None = None,
                 hidden_dim: int | None = None):
        super().__init__()
        self.num_outputs = num_classes
        self._ignore_index = -100 if ignore_id is None else ignore_id  # -100 disables it
        self._trunk = None if hidden_dim is None else nn.Sequential(
            nn.Linear(in_feat_dim, hidden_dim),
            nn.ReLU(),
        )
        self._layer = nn.Linear(in_feat_dim if hidden_dim is None else hidden_dim, self.num_outputs)

    def forward(self, point_feats: torch.Tensor) -> torch.Tensor:
        """Classify each point.

        Args:
            point_feats: (N, in_feat_dim) features at input resolution.

        Returns:
            (N, num_outputs) logits.
        """
        if self._trunk is not None:
            # First layer in bf16 for memory with wide concat features, as in the other heads.
            point_feats = self._trunk(point_feats)
        with torch.autocast("cuda", enabled=False):  # fp32 head for stability
            return self._layer(point_feats.float())

    def loss(self, logits: torch.Tensor, semantic_labels: torch.Tensor) -> torch.Tensor:
        """Cross entropy over labelled points only."""
        return F.cross_entropy(logits.float(), semantic_labels, ignore_index=self._ignore_index)


class PanopticSegmentationHead(nn.Module):
    """PointGroup-style heads: semantic logits plus a per-point offset towards the instance centroid.

    The classifier follows ``hidden_dim`` like `SemanticSegmentationHead`. The offset ("bias")
    branch is Linear-BatchNorm-ReLU-Linear at the input width whatever ``hidden_dim``, as in
    Pointcept's PointGroup probes, so a linear probe need not regress offsets linearly.
    """

    def __init__(self, in_feat_dim: int, num_classes: int, thing_ids: tuple[int, ...],
                 hidden_dim: int | None = 512, ignore_id: int | None = None):
        super().__init__()
        self.num_outputs = num_classes
        self._thing_ids = thing_ids
        self._semantic = SemanticSegmentationHead(in_feat_dim, num_classes, ignore_id=ignore_id, hidden_dim=hidden_dim)
        self._offset_trunk = nn.Sequential(
            nn.Linear(in_feat_dim, in_feat_dim),
            nn.BatchNorm1d(in_feat_dim, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )
        self._offset_layer = nn.Linear(in_feat_dim, 3)

    def forward(self, point_feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict semantics and instance-centre offsets.

        Args:
            point_feats: (N, in_feat_dim) features at input resolution.

        Returns:
            ``(logits (N, C+1), offsets (N, 3))`` where offsets are in metres.
        """
        logits = self._semantic(point_feats)
        # Trunk in bf16 for memory with the wide concat features, output layer in fp32 for stability.
        hidden = self._offset_trunk(point_feats)
        with torch.autocast("cuda", enabled=False):
            return logits, self._offset_layer(hidden.float())

    def loss(
        self,
        logits: torch.Tensor,
        offsets: torch.Tensor,
        semantic_labels: torch.Tensor,
        instance_labels: torch.Tensor,
        positions: torch.Tensor,
        batch_indices: torch.Tensor,
        offset_weight: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Semantic CE plus PointGroup's offset losses.

        The offset losses are L1 summed over xyz and the negative cosine between predicted
        and true offset, both averaged over the points of thing instances.

        Args:
            logits: (N, C+1) semantic logits.
            offsets: (N, 3) predicted offsets.
            semantic_labels: (N,) ground-truth class ids.
            instance_labels: (N,) ground-truth instance ids (0 = no instance).
            positions: (N, 3) point xyz.
            batch_indices: (N,)
            offset_weight: Scale of the offset terms relative to the CE term.

        Returns:
            ``(total_loss, {"semantic": ..., "offset_l1": ..., "offset_direction": ...})``.
        """
        semantic_loss = self._semantic.loss(logits, semantic_labels)

        thing_mask = torch.isin(
            semantic_labels, torch.tensor(self._thing_ids, device=semantic_labels.device)
        ) & (instance_labels > 0)
        if thing_mask.any():
            targets = _instance_centre_offsets(
                positions[thing_mask], instance_labels[thing_mask], batch_indices[thing_mask]
            )
            predicted = offsets[thing_mask].float()
            l1_loss = (predicted - targets).abs().sum(dim=-1).mean()
            direction_loss = -(
                F.normalize(predicted, dim=-1, eps=1e-8) * F.normalize(targets, dim=-1, eps=1e-8)
            ).sum(dim=-1).mean()
        else:
            # A detached zero would leave the offset branch out of the autograd
            # graph, and DDP then aborts on the ranks whose batch happens to hold
            # no thing points ("parameters that were not used in producing loss").
            l1_loss = direction_loss = offsets.float().sum() * 0.0

        total = semantic_loss + offset_weight * (l1_loss + direction_loss)
        return total, {
            "semantic": semantic_loss.item(),
            "offset_l1": l1_loss.item(),
            "offset_direction": direction_loss.item(),
        }


def _instance_centre_offsets(
    positions: torch.Tensor, instance_labels: torch.Tensor, batch_indices: torch.Tensor
) -> torch.Tensor:
    """Vector from each point to the centroid of its instance.

    Args:
        positions: (M, 3) xyz of points that belong to some instance.
        instance_labels: (M,) instance id per point, all non-zero.
        batch_indices: (M,) Needed to disambiguate instances between 

    Returns:
        (M, 3) target offsets.
    """
    linear_ids = (instance_labels.amax() + 1)*batch_indices + instance_labels
    unique_linear_ids, inverse = torch.unique(linear_ids, return_inverse=True)
    sums = torch.zeros(len(unique_linear_ids), 3, device=positions.device, dtype=torch.float32)
    sums.index_add_(0, inverse, positions.float())
    counts = torch.zeros(len(unique_linear_ids), device=positions.device, dtype=torch.float32)
    counts.index_add_(0, inverse, torch.ones_like(inverse, dtype=torch.float32))
    centres = sums / counts.unsqueeze(-1)
    return centres[inverse] - positions.float()



def assign_points_to_boxes(
    positions: torch.Tensor, centers: torch.Tensor, sizes: torch.Tensor, yaws: torch.Tensor
) -> torch.Tensor:
    """Index of the oriented box each point lies in, the nearest centre where boxes overlap, -1 outside all.

    Args:
        positions: (N, 3) point xyz.
        centers: (M, 3) box centres.
        sizes: (M, 3) extent along the box's heading, its side and z.
        yaws: (M,) heading in radians.

    Returns:
        (N,) long box index per point.
    """
    offsets = positions[:, None] - centers[None]  # (N, M, 3)
    cos, sin = yaws.cos(), yaws.sin()
    local = torch.stack(
        [cos * offsets[..., 0] + sin * offsets[..., 1], -sin * offsets[..., 0] + cos * offsets[..., 1], offsets[..., 2]],
        dim=-1,
    )
    inside = (local.abs() <= sizes / 2).all(dim=-1)  # (N, M)
    distances = offsets.norm(dim=-1).masked_fill(~inside, float("inf"))
    return torch.where(inside.any(dim=-1), distances.argmin(dim=-1), -1)


class DetectionHead(nn.Module):
    """Per-point (or per-voxel) detection, trained on the box annotations alone.

    The classifier gives each point's box class (0 = in no box) and follows ``hidden_dim`` like
    `SemanticSegmentationHead`. The box branch is Linear-BatchNorm-ReLU-Linear at the input width, like
    `PanopticSegmentationHead`'s offset branch, and regresses for points in a box the offset to the box
    centre (3), its size (3), sin/cos of its yaw (2) and its BEV velocity (2). Every point proposes its own box;
    NMS keeps the best of overlapping ones.
    """

    OFFSET, SIZE, YAW, VELOCITY = slice(0, 3), slice(3, 6), slice(6, 8), slice(8, 10)
    BOX_OUTPUTS = POSITION_OFFSET_DIM + SIZE_DIM + ORIENTATION_DIM + 2

    def __init__(self, in_feat_dim: int, num_classes: int, hidden_dim: int | None = None):
        super().__init__()
        self.num_outputs = num_classes + 1
        self._semantic = SemanticSegmentationHead(in_feat_dim, self.num_outputs, hidden_dim=hidden_dim)
        self._box_trunk = nn.Sequential(
            nn.Linear(in_feat_dim, in_feat_dim),
            nn.BatchNorm1d(in_feat_dim, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )
        self._box_layer = nn.Linear(in_feat_dim, self.BOX_OUTPUTS)

    def forward(self, point_feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict box classes and boxes.

        Args:
            point_feats: (N, in_feat_dim) features at input resolution.

        Returns:
            ``(logits (N, C+1), boxes (N, 10))``, boxes as [centre offset, size, sin yaw, cos yaw, velocity] in
            metres and m/s.
        """
        logits = self._semantic(point_feats)
        # Trunk in bf16 for memory with the wide concat features, output layer in fp32 for stability.
        hidden = self._box_trunk(point_feats)
        with torch.autocast("cuda", enabled=False):
            return logits, self._box_layer(hidden.float())

    @staticmethod
    def targets(
        positions: torch.Tensor, annotations: dict[str, torch.Tensor], batch_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-point targets from the boxes: the class of the box a point lies in (0 outside all) and that box.

        Args:
            positions: (N, 3) point xyz.
            annotations: "class" (B, M), 0 for padding, "center" and "size" (B, M, 3), "quats" (B, M, 4) and,
                if the dataset has it, "velocity" (B, M, 2), NaN where unknown.
            batch_indices: (N,)

        Returns:
            ``(labels (N,), box_targets (N, 10))``; box targets are zero outside boxes, velocity NaN where unknown.
        """
        labels = torch.zeros(positions.shape[0], dtype=torch.long, device=positions.device)
        box_targets = torch.zeros(positions.shape[0], DetectionHead.BOX_OUTPUTS, device=positions.device)
        box_targets[:, DetectionHead.VELOCITY] = float("nan")
        for sample in range(annotations["class"].shape[0]):
            present = annotations["class"][sample] != 0
            if not present.any():
                continue
            centers = annotations["center"][sample][present].float()
            sizes = annotations["size"][sample][present].float()
            yaws = quaternion_to_yaw(annotations["quats"][sample][present].float())
            velocities = (
                annotations["velocity"][sample][present].float() if "velocity" in annotations
                else torch.full((centers.shape[0], 2), float("nan"), device=centers.device)
            )
            points = (batch_indices == sample).nonzero().squeeze(-1)
            box = assign_points_to_boxes(positions[points].float(), centers, sizes, yaws)
            points, box = points[box >= 0], box[box >= 0]
            labels[points] = annotations["class"][sample][present][box].long()
            box_targets[points] = torch.cat(
                [centers[box] - positions[points].float(), sizes[box], yaws[box].sin()[:, None], yaws[box].cos()[:, None],
                 velocities[box]],
                dim=-1,
            )
        return labels, box_targets

    def loss(
        self, logits: torch.Tensor, boxes: torch.Tensor, labels: torch.Tensor, box_targets: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Class CE plus, over points in a box, PointGroup's offset L1 + direction losses and L1 on size, yaw and
        (where known) velocity.

        Args:
            logits: (N, C+1) class logits.
            boxes: (N, 10) predicted boxes.
            labels, box_targets: From `targets`.
        """
        semantic_loss = self._semantic.loss(logits, labels)
        inside = labels > 0
        if inside.any():
            predicted, target = boxes[inside].float(), box_targets[inside]
            offset_loss = (predicted[:, self.OFFSET] - target[:, self.OFFSET]).abs().sum(dim=-1).mean()
            direction_loss = -(
                F.normalize(predicted[:, self.OFFSET], dim=-1, eps=1e-8) * F.normalize(target[:, self.OFFSET], dim=-1, eps=1e-8)
            ).sum(dim=-1).mean()
            size_loss = (predicted[:, self.SIZE] - target[:, self.SIZE]).abs().sum(dim=-1).mean()
            yaw_loss = (predicted[:, self.YAW] - target[:, self.YAW]).abs().sum(dim=-1).mean()
            known = ~target[:, self.VELOCITY].isnan().any(dim=-1)
            velocity_loss = (
                (predicted[known, self.VELOCITY] - target[known, self.VELOCITY]).abs().sum(dim=-1).mean()
                if known.any() else predicted.sum() * 0.0
            )
        else:
            # Keeps the box branch in the autograd graph, or DDP aborts on ranks without boxes.
            offset_loss = direction_loss = size_loss = yaw_loss = velocity_loss = boxes.float().sum() * 0.0
        total = semantic_loss + offset_loss + direction_loss + size_loss + yaw_loss + velocity_loss
        return total, {
            "semantic": semantic_loss.item(),
            "offset_l1": offset_loss.item(),
            "offset_direction": direction_loss.item(),
            "size_l1": size_loss.item(),
            "yaw_l1": yaw_loss.item(),
            "velocity_l1": velocity_loss.item(),
        }


class ObjectDetectionHead(nn.Module):
    """Per-point detection head: objectness per class plus a box (centre offset, size, yaw as sin/cos).

    Output per point: ``num_classes`` objectness logits, then offset to the box centre (3), size (3) and
    sin/cos of the yaw (2). The classifier follows ``hidden_dim`` like `SemanticSegmentationHead`; the box
    branch is Linear-BatchNorm-ReLU-Linear at the input width whatever ``hidden_dim``, like the panoptic
    offset branch, so a linear probe need not regress boxes linearly. As in panoptic segmentation, only the
    points inside a box are positives for it and regress it.
    """

    def __init__(self, in_feat_dim, num_classes, hidden_dim: int | None = 512):
        """Initialize."""
        super().__init__()
        self._box_out_dims = POSITION_OFFSET_DIM + SIZE_DIM + ORIENTATION_DIM  # = 8
        self._obj_out_dims = num_classes

        self._trunk = None if hidden_dim is None else nn.Sequential(
            nn.Linear(in_feat_dim, hidden_dim),
            nn.ReLU(),
        )
        self._obj_layer = nn.Linear(in_feat_dim if hidden_dim is None else hidden_dim, self._obj_out_dims)
        self._box_trunk = nn.Sequential(
            nn.Linear(in_feat_dim, in_feat_dim),
            nn.BatchNorm1d(in_feat_dim, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )
        self._box_layer = nn.Linear(in_feat_dim, self._box_out_dims)
    
    def forward(self, point_feats: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            point_feats: (N, in_feat_dim) tensor of point features, where N is total points across batches
            
        Returns:
            (N, obj_out_dims + box_out_dims) tensor of predictions where N is the number of points in the original point cloud
        """
        # Trunks in bf16 for memory with the wide concat features, output layers in fp32 for stability.
        obj_feats = point_feats if self._trunk is None else self._trunk(point_feats)
        box_feats = self._box_trunk(point_feats)
        with torch.autocast("cuda", enabled=False):
            return torch.cat((self._obj_layer(obj_feats.float()), self._box_layer(box_feats.float())), dim=-1)
    
    
    def loss(self, pos, preds, annotations, batch_indices):
        """Compute loss."""
        centers = annotations["center"]  # (B, M, 3)
        sizes = torch.relu(annotations["size"]) + 1e-7  # (B, M, 3)
        quats = annotations["quats"]  # (B, M, 4)
        classes = annotations["class"]  # (B, M)

        # go from position offset to absolute position in preds
        abs_preds = preds.clone()
        abs_preds[:, self._obj_out_dims : self._obj_out_dims + POSITION_OFFSET_DIM] += pos  # (N, 3)
        abs_preds = abs_preds.to(torch.float32)

        # split predictions
        obj_p = abs_preds[:, : self._obj_out_dims]  # (N, obj_out_dims)
        box_p = abs_preds[:, self._obj_out_dims :]  # (N, box_out_dims)

        obj_loss = torch.tensor(0.0, device=obj_p.device, dtype=torch.float32)
        box_loss = torch.tensor(0.0, device=obj_p.device, dtype=torch.float32)

        for b_idx in range(centers.shape[0]):
            # Process each batch separately
            unpad_mask = (centers[b_idx] != 0).any(dim=-1)  # (M,)
            centers_b = centers[b_idx][unpad_mask]  # (M', 3)
            sizes_b = sizes[b_idx][unpad_mask]  # (M', 3)
            quats_b = quats[b_idx][unpad_mask]  # (M', 4)
            classes_b = classes[b_idx][unpad_mask]  # (M',)

            batch_mask = batch_indices == b_idx
            obj_p_b = obj_p[batch_mask]  # (N_b, obj_out_dims)
            box_p_b = box_p[batch_mask]  # (N_b, box_out_dims)
            pos_b = pos[batch_mask]  # (N_b, 3)

            if centers_b.shape[0] == 0:
                #print("No objects in this frame.")
                obj_t_b = torch.zeros_like(obj_p_b, device=obj_p_b.device, dtype=torch.float32)
                obj_loss += F.binary_cross_entropy_with_logits(
                    obj_p_b,
                    obj_t_b,
                    reduction="mean",
                )
                continue

            # get targets:
            yaw_b = quaternion_to_yaw(quats_b)  # (M',)
            box_t_b = torch.cat(
                [centers_b, sizes_b, torch.sin(yaw_b).unsqueeze(-1), torch.cos(yaw_b).unsqueeze(-1)], dim=-1
            )  # (M', 8)
            closest_target_idx = assign_points_to_boxes(pos_b.float(), centers_b, sizes_b, yaw_b)  # (N_b,)
            obj_t_b = torch.zeros_like(obj_p_b)  # (N_b, num_classes)
            inside = closest_target_idx >= 0
            obj_t_b[inside, classes_b[closest_target_idx[inside]].long() - 1] = 1.0

            # === Object Classification Loss ===
            pos_weight = torch.ones(obj_p_b.shape[-1], device=obj_p_b.device) * 2.0
            obj_loss += F.binary_cross_entropy_with_logits(
                obj_p_b,
                obj_t_b,
                pos_weight=pos_weight,
                reduction="mean",
            )

            # === Box Regression Loss ===
            fg_mask = closest_target_idx >= 0
            if fg_mask.any():
                box_p_fg = box_p_b[fg_mask]  # (K, 8)
                box_t_fg = box_t_b[closest_target_idx[fg_mask]]  # (K, 8)
                box_loss += F.l1_loss(box_p_fg, box_t_fg, reduction="mean")
            else:
                print(f"No foreground points for box regression. Although there are GTs in batch {b_idx}.")

        return obj_loss, box_loss
