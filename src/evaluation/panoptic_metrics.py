"""Panoptic segmentation for point clouds: instance clustering and Panoptic Quality.

The metric is a port of ``PanopticEval`` from the SemanticKITTI / nuScenes devkits
(nuscenes-devkit eval/panoptic/panoptic_seg_evaluator.py), the standard LiDAR panoptic
benchmark code. Per scan and class, a predicted and a ground-truth segment match when
their IoU exceeds 0.5, and

    PQ = sum(IoU of matches) / (TP + 0.5 * FP + 0.5 * FN) = SQ * RQ

Ground-truth ignore points are dropped first. Unmatched segments only count as FP/FN
when they have at least ``min_points`` points (nuScenes: 15, SemanticKITTI: 50, 0 counts
every segment). Means run over every scored class, also those absent from the data.

Panoptic Quality evaluation adapted from the SemanticKITTI API
(https://github.com/PRBonn/semantic-kitti-api, auxiliary/eval_np.py), as distributed in the nuScenes devkit:
Copyright (c) 2019, University of Bonn. Authors: Andres Milioto and Jens Behley. Licensed under the MIT
License; see ../../NOTICE for the license text.
Modified: per-scan PyTorch inputs, vectorised match bookkeeping, DDP reduction, and thing/stuff and PQ-dagger
summaries.
"""

import itertools

import numpy as np
import torch

# The devkit packs (prediction, ground truth) instance pairs into one int64 with this stride.
_PAIR_OFFSET = 2**32


def cluster_instances(
    shifted_positions: torch.Tensor,
    semantic_preds: torch.Tensor,
    batch_indices: torch.Tensor,
    thing_ids: tuple[int, ...],
    voxel_size: float = 0.3,
    min_points: int = 1,
) -> torch.Tensor:
    """PointGroup-style clustering of the predicted instance centres (point + offset).

    PointGroup links shifted points of the same predicted class within a radius (ball query)
    and takes the connected components (BFS). Without its CUDA op, this links occupied cells
    of side ``voxel_size`` to their 26 neighbours instead, per sample and class. Components
    with fewer than ``min_points`` points are dropped, like PointGroup's cluster_min_points.

    Args:
        shifted_positions: (N, 3) point positions plus predicted offsets.
        semantic_preds: (N,) predicted class ids.
        batch_indices: (N,) sample index per point.
        thing_ids: Class ids that have instances.
        voxel_size: Side in metres of the linking cells.
        min_points: Smallest instance kept.

    Returns:
        (N,) instance ids, unique within the batch; 0 for stuff, ignore and dropped points.
    """
    device = shifted_positions.device
    instance_ids = torch.zeros(shifted_positions.shape[0], dtype=torch.long, device=device)
    thing = torch.isin(semantic_preds, torch.tensor(thing_ids, device=device))
    if not thing.any():
        return instance_ids

    cells = torch.floor(shifted_positions[thing] / voxel_size).long()
    # Offset so every neighbour of an occupied cell still has non-negative coordinates.
    cells = cells - cells.amin(dim=0) + 1
    extent = cells.amax(dim=0) + 2
    group = batch_indices[thing].long() * (int(semantic_preds.max()) + 1) + semantic_preds[thing]

    def cell_key(group, cells):
        return ((group * extent[0] + cells[:, 0]) * extent[1] + cells[:, 1]) * extent[2] + cells[:, 2]

    keys, point_cell = torch.unique(cell_key(group, cells), return_inverse=True)
    num_cells = keys.shape[0]
    cell_coords = torch.empty(num_cells, 3, dtype=torch.long, device=device)
    cell_coords[point_cell] = cells
    cell_group = torch.empty(num_cells, dtype=torch.long, device=device)
    cell_group[point_cell] = group

    sources, targets = [], []
    for step in itertools.product((-1, 0, 1), repeat=3):
        if step == (0, 0, 0):
            continue
        neighbour = cell_key(cell_group, cell_coords + torch.tensor(step, device=device))
        position = torch.searchsorted(keys, neighbour).clamp(max=num_cells - 1)
        found = keys[position] == neighbour
        sources.append(found.nonzero().squeeze(-1))
        targets.append(position[found])
    sources, targets = torch.cat(sources), torch.cat(targets)

    # Connected components: propagate the smallest cell index, with pointer jumping.
    labels = torch.arange(num_cells, device=device)
    while True:
        updated = labels.scatter_reduce(0, sources, labels[targets], reduce="amin")
        updated = updated[updated]
        if torch.equal(updated, labels):
            break
        labels = updated
    _, component = torch.unique(labels, return_inverse=True)
    point_component = component[point_cell]
    kept = torch.bincount(point_component)[point_component] >= min_points
    instance_ids[thing] = torch.where(kept, point_component + 1, 0)
    return instance_ids


class PanopticQuality:
    """Running PQ/SQ/RQ accumulator over a whole evaluation set, as in the LiDAR panoptic devkits."""

    def __init__(self, seg_class_name_to_id: dict[str, int], thing_ids: tuple[int, ...],
                 ignore_id: int | None = None, device: torch.device | str = "cpu", min_points: int = 0):
        # Scored classes are the dataset's named ones; id 0 is ignore or background and never scored.
        self._id_to_name = {i: name for name, i in seg_class_name_to_id.items() if i != ignore_id}
        self._ignore_id = ignore_id
        self._device = device
        self._min_points = min_points
        self._include = sorted(self._id_to_name)
        self._things = [c for c in self._include if c in thing_ids]
        self._stuff = [c for c in self._include if c not in thing_ids]
        # Rows: true positives, false positives, false negatives, summed IoU of the true positives.
        self._counts = np.zeros((4, max(self._include) + 1))

    def update(
        self,
        pred_semantic: torch.Tensor,
        pred_instance: torch.Tensor,
        gt_semantic: torch.Tensor,
        gt_instance: torch.Tensor,
        batch_indices: torch.Tensor,
    ) -> None:
        """Accumulate one batch, scan by scan.

        Args:
            pred_semantic: (N,) predicted class ids.
            pred_instance: (N,) predicted instance ids, 0 for stuff.
            gt_semantic: (N,) ground-truth class ids.
            gt_instance: (N,) ground-truth instance ids, 0 for stuff.
            batch_indices: (N,) sample index per point.
        """
        for sample in torch.unique(batch_indices):
            mask = batch_indices == sample
            self._update_scan(*(t[mask].cpu().numpy() for t in (pred_semantic, pred_instance, gt_semantic, gt_instance)))

    def _update_scan(self, x_sem, x_inst, y_sem, y_inst) -> None:
        """The devkit's ``addBatchPanoptic`` for one scan (x = prediction, y = ground truth)."""
        x_inst, y_inst = x_inst.astype(np.int64) + 1, y_inst.astype(np.int64) + 1
        if self._ignore_id is not None:
            labelled = y_sem != self._ignore_id
            x_sem, x_inst, y_sem, y_inst = x_sem[labelled], x_inst[labelled], y_sem[labelled], y_inst[labelled]
        for cl in self._include:
            x_in = x_inst * (x_sem == cl)
            y_in = y_inst * (y_sem == cl)
            unique_pred, counts_pred = np.unique(x_in[x_in > 0], return_counts=True)
            unique_gt, counts_gt = np.unique(y_in[y_in > 0], return_counts=True)
            valid = (x_in > 0) & (y_in > 0)
            unique_pair, counts_pair = np.unique(x_in[valid] + _PAIR_OFFSET * y_in[valid], return_counts=True)
            gt_labels = unique_pair // _PAIR_OFFSET
            pred_labels = unique_pair % _PAIR_OFFSET
            gt_areas = counts_gt[np.searchsorted(unique_gt, gt_labels)]
            pred_areas = counts_pred[np.searchsorted(unique_pred, pred_labels)]
            ious = counts_pair / (gt_areas + pred_areas - counts_pair)
            matches = ious > 0.5
            self._counts[0, cl] += matches.sum()
            self._counts[3, cl] += ious[matches].sum()
            matched_gt = np.isin(unique_gt, gt_labels[matches])
            matched_pred = np.isin(unique_pred, pred_labels[matches])
            self._counts[2, cl] += np.sum((counts_gt >= self._min_points) & ~matched_gt)
            self._counts[1, cl] += np.sum((counts_pred >= self._min_points) & ~matched_pred)

    def all_reduce(self) -> None:
        """Sum the counters across DDP ranks."""
        import torch.distributed as dist

        counts = torch.from_numpy(self._counts).to(self._device)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        self._counts = counts.cpu().numpy()

    def compute(self, per_class_iou: dict[str, float] | None = None) -> dict:
        """PQ/SQ/RQ over all, thing and stuff classes.

        With the semantic ``per_class_iou`` (class name -> IoU) it also returns PQ_dagger,
        the mean of thing-class PQ and stuff-class IoU.
        """
        tp, fp, fn, iou_sum = self._counts
        sq = iou_sum / np.maximum(tp, 1e-15)
        rq = tp / np.maximum(tp + 0.5 * fp + 0.5 * fn, 1e-15)
        pq = sq * rq
        results = {
            "PQ": float(pq[self._include].mean()),
            "SQ": float(sq[self._include].mean()),
            "RQ": float(rq[self._include].mean()),
            "PQ_things": float(pq[self._things].mean()) if self._things else 0.0,
            "PQ_stuff": float(pq[self._stuff].mean()) if self._stuff else 0.0,
        }
        if per_class_iou is not None:
            stuff_iou = [per_class_iou.get(self._id_to_name[c], 0.0) for c in self._stuff]
            results["PQ_dagger"] = float(np.mean(list(pq[self._things]) + stuff_iou))
        results["per_class"] = {
            self._id_to_name[c]: {"PQ": float(pq[c]), "SQ": float(sq[c]), "RQ": float(rq[c])} for c in self._include
        }
        return results
