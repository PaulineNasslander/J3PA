"""Postprocess centerpoint output.

This file includes code adapted from [MMDetection3D] (https://github.com/atonderski/mmdetection3d/)
Copyright (c) OpenMMLab. Licensed under Apache-2.0; see ../../LICENSE.

Modifications:
- Combined functions from [mmdet3d/models/task_modules/coders/centerpoint_bbox_coders.py],
 [mmdet3d/models/dense_heads/centerpoint_head.py]
- Renamed and adapted functions for use in [Distilling Representations into Points]

"""

import numpy as np
import torch

from constants import (
    MAX_NUM_OBJECTS,
    MIN_RADIUS,
    SCORE_THRESHOLD,
)
from data.zod_dataset import CLASS_NAME_TO_ID as ZOD_CLASS_NAME_TO_ID
from utils.geometry import yaw_to_quaternion


class BoxObject:
    """Box object for annotation."""

    def __init__(self, box, class_name):
        """Initialize.

        Args:
            box (tensor): [x, y, z, sx, sy, sz, q1, q2, q3, q4]
            class_name (str): class of the object
        """
        if len(box) != 10:
            raise ValueError(f"Box must have 10 elements: [x, y, z, sx, sy, sz, q1, q2, q3, q4], got {box.shape=}")

        self.box = box
        self.class_name = class_name

    def __str__(self):
        """Call by print() and str() - user-friendly representation."""
        return f"Box ({self.box}) label {self.class_name}"

    def __repr__(self):
        """Call by repr() and when displaying in interactive shells."""
        return f"Box ({self.box}) label {self.class_name}"


class Detection(BoxObject):
    """Box object for detection."""

    def __init__(self, box, class_name, score):
        """Initialize.

        Args:
            box (tensor): same as BoxObject
            class_name (str): object class
            score (float): confidence score of the detection
        """
        super().__init__(box, class_name)
        self.score = score


# pylint: disable=protected-access
def get_gt_boxes(annotations, class_name_to_id=None):
    """Get list over frames with gt boxes (BoxObject).

    Params:
        annotations: dictionary with ground truth boxes and classes (B, M, ...)
        where B is batch size, M is max number of boxes (padded)
        class_name_to_id: dict mapping class name to int ID (defaults to ZOD)
    Returns:
        gts: list of len B of ndarrays of BoxObject for each frame
    """
    if class_name_to_id is None:
        class_name_to_id = ZOD_CLASS_NAME_TO_ID
    class_names_array = np.array(list(class_name_to_id.keys()))
    gts = []
    for batch_idx in range(annotations["class"].shape[0]):
        class_names = annotations["class"][batch_idx].to(torch.long).cpu()
        no_padded_mask = class_names != 0
        if no_padded_mask.sum() == 0:
            gts.append(np.array([], dtype=object))
            continue

        class_names = class_names[no_padded_mask]
        class_names = class_names_array[class_names.numpy() - 1]
        boxes = torch.cat(
            (
                annotations["center"][batch_idx][no_padded_mask],
                annotations["size"][batch_idx][no_padded_mask],
                annotations["quats"][batch_idx][no_padded_mask],
            ),
            dim=-1,
        )
        box_objects = [BoxObject(np.array(box.cpu()), np.array(cls)) for box, cls in zip(boxes, class_names, strict=False)]
        gts.append(np.array(box_objects, dtype=object))
    return gts


def get_dt_boxes(pos, preds, batch_indices=None, class_name_to_id=None):
    """Get list over frames with pred boxes (Detection).

    params:
        pos: (N, 3) tensor of point positions
        preds: (N, 9) tensor of predictions
        batch_indices: (N,) tensor of batch indices for each point
        class_name_to_id: dict mapping class name to int ID (defaults to ZOD)
    returns:
        dts: list of len B of ndarrays of Detection objects for each frame
    """
    if class_name_to_id is None:
        class_name_to_id = ZOD_CLASS_NAME_TO_ID
    dts = []
    if batch_indices is None:
        dts.append(_get_single_dt_boxes(pos, preds, class_name_to_id))
        return dts

    for batch_idx in torch.unique(batch_indices):
        mask = batch_indices == batch_idx
        pos_batch = pos[mask]
        preds_batch = preds[mask]
        dts_batch = _get_single_dt_boxes(pos_batch, preds_batch, class_name_to_id)

        dts.append(dts_batch)

    return dts


def _get_single_dt_boxes(pos, preds, class_name_to_id=None):
    """Get pred boxes for a single frame (Detection)."""
    if class_name_to_id is None:
        class_name_to_id = ZOD_CLASS_NAME_TO_ID
    class_names_array = np.array(list(class_name_to_id.keys()))
    boxes3d, scores, labels = decode_output(pos, preds)

    centers2d = boxes3d[:, [0, 1]]
    boxes_bev = torch.cat([centers2d, scores.view(-1, 1)], dim=1)
    keep = torch.tensor(
        circle_nms(boxes_bev.detach().cpu().numpy(), MIN_RADIUS, post_max_size=MAX_NUM_OBJECTS),
        dtype=torch.long,
        device=boxes_bev.device,
    )

    boxes3d = boxes3d[keep].cpu()
    scores = scores[keep].cpu()
    labels = labels[keep].cpu()

    labels = class_names_array[labels.numpy() - 1]

    dts = [
        Detection(np.array(box.cpu()), np.array(label.item()), np.array(score.item()))
        for box, score, label in zip(boxes3d, scores, labels, strict=False)
    ]

    return np.array(dts, dtype=object)


def circle_nms(dets, thresh: float, post_max_size: int = 83):
    """Circular NMS.

    An object is only counted as positive if no other center with a higher
    confidence exists within a radius r using a bird-eye view distance metric.

    Args:
        dets (Tensor): Detection results with the shape of [N, 3].
        thresh (float): Value of threshold.
        post_max_size (int): Max number of prediction to be kept.
            Defaults to 83.

    Returns:
        list: Indexes of the detections to be kept.
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    scores = dets[:, 2]
    order = scores.argsort()[::-1].astype(np.int32)  # highest->lowest
    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int32)
    keep = []
    for _i in range(ndets):
        i = order[_i]  # start with highest score box
        if suppressed[i] == 1:  # if any box have enough iou with this, remove it
            continue
        keep.append(i)
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            # calculate center distance between i and j box
            dist = (x1[i] - x1[j]) ** 2 + (y1[i] - y1[j]) ** 2

            # ovr = inter / areas[j]
            if dist <= thresh:
                suppressed[j] = 1

    if post_max_size < len(keep):
        return keep[:post_max_size]

    return keep


# pylint: disable=too-many-locals
def decode_output(pos, preds):
    """Decode centernet output to boxes.

    params:
        pos: (N, 3) tensor of point positions
        preds: (N, 9) tensor of predictions
    returns:
        predictions_dicts: list over frames in batch of dicts with keys: bboxes, scores, labels"""

    abs_preds = preds.clone()
    abs_preds[:, 1:4] += pos  # (N, 3)
    abs_preds[:, 0] = torch.sigmoid(abs_preds[:, 0])  # objectness score between 0 and 1

    # use score threshold
    if SCORE_THRESHOLD is not None:
        thresh_mask = abs_preds[:, 0] > SCORE_THRESHOLD

    abs_preds = abs_preds[thresh_mask]
    abs_preds[:, 4:7] = torch.nn.functional.relu(abs_preds[:, 4:7]) + 1e-7  # size should be positive
    yaw = torch.atan2(abs_preds[..., 7], abs_preds[..., 8]).unsqueeze(-1)  # (M, 1)
    quat = yaw_to_quaternion(yaw)

    boxes3d = torch.cat([abs_preds[..., 1:7], quat], dim=-1)  # (M, 10) x,y,z,l,h,w,q1,q2,q3,q4
    scores = abs_preds[..., 0]  # (M,)
    labels = torch.ones_like(scores, dtype=torch.long)  # (M,) all class 1 for now as we only need vehicle performance in online eval

    return boxes3d, scores, labels
