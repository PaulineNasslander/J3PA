"""Evaluation of object detection.


This file includes code adapted from the Argoverse project (https://github.com/argoverse/argoverse-api),
originally licensed under the MIT License:

Copyright (c) 2019, 2020 Argo AI, LLC

Modifications:
- Combined functions from [argoverse/evaluation/detection/eval.py],
[argoverse/evaluation/detection/utils.py], [argoverse/utils/transform.py]
- Copied, renamed and adapted functions for use in [Distilling Representations into Points]

See ../../NOTICE for the MIT license text.

"""

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R

from constants import (
    AFFINITY_THRESH,
    MAX_NUM_BOXES,
    N_REC_SAMPLES,
    N_TP_ERRORS,
    SIGNIFICANT_DIGITS,
    STATISTIC_NAMES,
    SUMMARY_DEFAULT_VALS,
    TP_NORMALIZTION_TERMS,
    TP_THRESH,
)
from data.zod_dataset import CLASS_NAME_TO_ID
from utils.multi_gpu_utils import (
    torch_ddp_run_on_master_only,
)


@torch_ddp_run_on_master_only
def evaluate_argoverse_style(ground_truths, detections, class_name_to_id=None):
    """Evaluate object detection.

    Params:
        ground_truths: list of ndarrays of BoxObjects
        detections: list of ndarrays of Detections
        OBS: must be the same length
        class_name_to_id: dict mapping class name to int ID (defaults to ZOD).
            Must match the map used to build the boxes, otherwise the summary
            rows are all defaults and the averages come out as zeros.
    Returns:
        Evaluation metrics of shape (C + 1, K)
        where C + 1 is the number of classes.
        plus a row for their means.
        K refers to the number of evaluation metrics.
    """
    if class_name_to_id is None:
        class_name_to_id = CLASS_NAME_TO_ID
    data: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    num_cls_instances: defaultdict[str, int] = defaultdict(int)
    accum = [get_detection_stats(dts, gts, class_name_to_id) for dts, gts in zip(detections, ground_truths, strict=False)]
    for frame_stats, frame_num_cls in accum:
        for cls_name, cls_stats in frame_stats.items():
            data[cls_name].append(cls_stats)
        for class_name, num_instances in frame_num_cls.items():
            num_cls_instances[class_name] += num_instances

    aggregated_data = {k: np.vstack(v) for k, v in data.items()}

    init_data = {dt_cls: SUMMARY_DEFAULT_VALS for dt_cls, _ in class_name_to_id.items()}
    summary = pd.DataFrame.from_dict(init_data, orient="index", columns=STATISTIC_NAMES)
    summary_update = pd.DataFrame.from_dict(
        calculate(aggregated_data, num_cls_instances),
        orient="index",
        columns=STATISTIC_NAMES,
    )
    summary.update(summary_update)

    summary = summary.round(SIGNIFICANT_DIGITS)
    summary.index = summary.index.str.title()

    summary.loc["Average Metrics"] = summary.mean().round(SIGNIFICANT_DIGITS)

    return summary


# pylint: disable=too-many-locals
def calculate(data, num_cls_instances):
    """Calculate the detection metrics.

    Args:
        data: The aggregated data used for summarization.
        num_cls_instances: Map of classes to number of instances.

    Returns:
        summary: The summary statistics.
    """
    summary: defaultdict[str, list[float]] = defaultdict(list)
    recalls_interp = np.linspace(0, 1, N_REC_SAMPLES)
    num_ths = len(AFFINITY_THRESH)

    for cls_name, cls_stats in data.items():
        ninst = num_cls_instances[cls_name]
        ranks = cls_stats[:, -1].argsort()[::-1]  # sort by last column, i.e. confidences
        cls_stats = cls_stats[ranks]

        for i, _ in enumerate(AFFINITY_THRESH):
            tp = cls_stats[:, i].astype(bool)
            ap_th, precisions_interp = calc_ap(tp, recalls_interp, ninst)
            summary[cls_name] += [ap_th]

        # AP Metric.
        ap = np.array(summary[cls_name][:num_ths]).mean()

        # Select only the true positives for each instance.
        tp_metrics_mask = ~np.isnan(cls_stats[:, num_ths : num_ths + N_TP_ERRORS]).all(axis=1)

        # If there are no true positives set tp errors to
        # their maximum values due to normalization below).
        if ~tp_metrics_mask.any():
            tp_metrics = TP_NORMALIZTION_TERMS
        else:
            # Calculate TP metrics.
            tp_metrics = np.mean(
                cls_stats[:, num_ths : num_ths + N_TP_ERRORS][tp_metrics_mask],
                axis=0,
            )

        # Convert errors to scores.
        tp_scores = 1 - (tp_metrics / TP_NORMALIZTION_TERMS)

        # Compute Composite Detection Score (CDS).
        cds = ap * tp_scores.sum()  # mean()

        summary[cls_name] = [ap, *tp_metrics, cds]

    return summary


# pylint: disable=too-many-locals
def get_metrics(dts, gts):
    """Get metrics."""
    n_threshs = len(AFFINITY_THRESH)
    metrics = np.zeros(shape=(dts.shape[0], n_threshs + N_TP_ERRORS))

    # Set the true positive metrics to np.nan since error is undefined on false positives.
    metrics[:, n_threshs : n_threshs + N_TP_ERRORS] = np.nan
    if gts.shape[0] == 0:
        return metrics

    affinity_matrix = compute_affinity_matrix(dts, gts)
    gt_matches, unique_gt_matches, unique_dt_matches = get_tp_matches(affinity_matrix)

    # The affinity matrix is an N by M matrix of the detections
    # and ground truth labels respectively.
    # We want to take the corresponding affinity
    # for each of the initial assignments using `gt_matches`.
    # The following line grabs the max affinity for each detection to a ground truth label.
    affinities = np.take_along_axis(affinity_matrix.T, gt_matches, axis=0).squeeze(0)

    for i, thresh in enumerate(AFFINITY_THRESH):
        # `tp_mask` may need to be defined differently with other affinities.
        tp_mask = affinities[unique_dt_matches] > -thresh
        metrics[unique_dt_matches, i] = tp_mask

        # Only compute true positive error when `thresh` is equal to the tp threshold.
        is_tp_thresh = thresh == TP_THRESH
        # Ensure that there are true positives of the respective class in the frame.
        has_true_positives = np.count_nonzero(tp_mask) > 0
        if is_tp_thresh and has_true_positives:
            dt_tp_indices = unique_dt_matches[tp_mask]
            gt_tp_indices = unique_gt_matches[tp_mask]

            dts_m = dts[dt_tp_indices]
            gts_m = gts[gt_tp_indices]

            trans_error = get_ate(dts_m, gts_m)
            scale_error = get_ase(dts_m, gts_m)
            orient_error = get_aoe(dts_m, gts_m)

            metrics[dt_tp_indices, n_threshs : n_threshs + N_TP_ERRORS] = np.vstack((trans_error, scale_error, orient_error)).T

    return metrics


def get_tp_matches(affinity_matrix):
    """Get unique indices for gt boxes with corresponding prediction.

    Params:
    affinity_matrix: shape (N,M), ndarray
    """
    # Get the GT label for each max-affinity GT label, detection pair.
    gt_matches = affinity_matrix.argmax(axis=1)[np.newaxis, :]

    # Find the indices of the "first" detection assigned to each GT.
    unique_gt_matches, unique_dt_matches = np.unique(gt_matches, return_index=True)
    return gt_matches, unique_gt_matches, unique_dt_matches


def get_aoe(dts, gts):
    """Calculate the orientation error between gt and preds.

    Params:
    dts: annotation shape (N, 10)
    gts: predicitions corresponding to gt shape (N, 10)
    """
    dt_quats = np.array([dt.box[-4:] for dt in dts])
    gt_quats = np.array([gt.box[-4:] for gt in gts])
    dt_yaws = R.from_quat(quat_argo2scipy_vectorized(dt_quats)).as_euler("xyz")[:, 2]

    gt_yaws = R.from_quat(quat_argo2scipy_vectorized(gt_quats)).as_euler("xyz")[:, 2]

    orientation_errors = wrap_angle(dt_yaws - gt_yaws)

    return orientation_errors


def get_ase(dts, gts):
    """Calculate the scale error between gt and preds.

    Params:
    dts: annotation shape (N, 10)
    gts: predicitions corresponding to gt shape (N, 10)
    """
    dt_dims = np.array([dt.box[3:6] for dt in dts])
    gt_dims = np.array([gt.box[3:6] for gt in gts])
    scale_errors = 1 - iou_aligned_3d(dt_dims, gt_dims)

    return scale_errors


def get_ate(dts, gts):
    """Calculate the translation error between gt and preds.

    Params:
    dts: annotation shape (N,) type Detection
    gts: predicitions corresponding to gt shape (N,) type BoxObject
    """
    dt_centers = np.array([dt.box[0:3] for dt in dts])
    gt_centers = np.array([gt.box[0:3] for gt in gts])
    trans_errors = np.linalg.norm(dt_centers - gt_centers, axis=1)
    return trans_errors


def compute_affinity_matrix(dts, gts):
    """Calculate the affinity matrix.

    Args:
        dts: Detections (N,10) ndarray.
        gts: Ground truth labels (M,10) ndarray.

    Returns:
        sims: Affinity scores between detections and ground truth annotations (N, M).
    """
    dt_centers = np.array([dt.box[0:3] for dt in dts])
    gt_centers = np.array([gt.box[0:3] for gt in gts])
    sims = -cdist(dt_centers, gt_centers)

    return sims


def quat_argo2scipy_vectorized(q: np.ndarray) -> np.ndarray:
    """Re-order Argoverse's [w,x,y,z] quaternion to Scipy's [x,y,z,w]."""
    return q[..., [1, 2, 3, 0]]


def wrap_angle(angles: np.ndarray, period: float = np.pi) -> np.ndarray:
    """Map angles (in radians) from domain [-∞, ∞] to [0, π).

    Returns:
        Angles (in radians) mapped to the interval [0, π).
    """
    # Map angles to [0, ∞].
    angles = np.abs(angles)

    # Calculate floor division and remainder simultaneously.
    divs, mods = np.divmod(angles, period)

    # Select angles which exceed specified period.
    angle_complement_mask = np.nonzero(divs)

    # Take set complement of `mods` w.r.t. the set [0, π].
    # `mods` must be nonzero, thus the image is the interval [0, π).
    angles[angle_complement_mask] = period - mods[angle_complement_mask]
    return angles


def iou_aligned_3d(dt_dims: np.ndarray, gt_dims: np.ndarray) -> np.ndarray:
    """Calculate the 3d, axis-aligned (vertical axis alignment) IoU.

    Both objects are aligned to their
    +x axis and their centroids are placed at the origin before computation of the IoU.

    Args:
        dt_dims: Detections (N, 3).
        gt_dims: Ground truth labels (N, 3).

    Returns:
        Intersection-over-union between the detections and their assigned ground
        truth labels (N,).

    """
    inter = np.minimum(dt_dims, gt_dims).prod(axis=1)
    union = np.maximum(dt_dims, gt_dims).prod(axis=1)
    return inter / union  # .values


def get_detection_stats(dts, gts, class_name_to_id=CLASS_NAME_TO_ID):
    """Get statistics based on detections.

    Params:
    dts: detections ndarray of shape (N,) of detection objects
    gts: ground truths ndarray of shape (M,) of box objects
    class_name_to_id: classes to score (defaults to ZOD)

    Returns:
    cls_to_metrics: Class to accumulated statistics dictionary of shape |C| -> (N, K + S) where C
        is the number of detection classes, K is the number of true positive thresholds used for
        AP computation, and S is the number of true positive errors.
    num_cls_instances: Mapping of shape |C| -> (1,)
    the class names to the number of instances in the ground
        truth dataset.
    """

    cls_to_metrics = defaultdict(list)
    num_cls_instances = defaultdict(int)
    for class_name in class_name_to_id:
        dt_filtered = filter_instances(dts, class_name)
        gt_filtered = filter_instances(gts, class_name)

        if dt_filtered.shape[0] > 0:
            ranked_dts, ranked_scores = rank(dt_filtered)
            metrics = get_metrics(ranked_dts, gt_filtered)

            cls_to_metrics[class_name] = np.hstack((metrics, ranked_scores[:, None]))

        num_cls_instances[class_name] = gt_filtered.shape[0]

    return cls_to_metrics, num_cls_instances


def rank(dts) -> tuple[np.ndarray, np.ndarray]:
    """Rank detections in descending order according to score (detector confidence).

    Args:
        dts: Array of `ODetection` objects. (N,).

    Returns:
        ranked_dts: Array of `Detection` objects ranked by score (N,) where N <= MAX_NUM_BOXES.
        ranked_scores: Array of floats sorted in descending order (N,) where N <= MAX_NUM_BOXES.
    """
    scores = np.array([dt.score for dt in dts.tolist()])
    ranks = scores.argsort()[::-1]

    ranked_dts = dts[ranks]
    ranked_scores = scores[ranks]

    # Ensure the number of boxes considered per class is at most `MAX_NUM_BOXES`.
    return ranked_dts[:MAX_NUM_BOXES], ranked_scores[:MAX_NUM_BOXES]


def filter_instances(
    instances,
    target_class_name: str,
) -> np.ndarray:
    """Filter object instances.

    Args:
        instances: Instances to be filtered (N,), either detections or ground truth object labels
        target_class_name: Name of the class of interest.

    Returns:
        Filtered instances.
    """
    instances = np.array([instance for instance in instances if instance.class_name == target_class_name])

    return instances


def calc_ap(gt_ranked: np.ndarray, recalls_interp: np.ndarray, ninst: int) -> tuple[float, np.ndarray]:
    """Compute precision and recall, interpolated over n fixed recall points.

    Args:
        gt_ranked: Ground truths, ranked by confidence.
        recalls_interp: Interpolated recall values.
        ninst: Number of instances of this class.
    Returns:
        avg_precision: Average precision.
        precisions_interp: Interpolated precision values.
    """
    tp = gt_ranked

    cumulative_tp = np.cumsum(tp, dtype=int)
    cumulative_fp = np.cumsum(~tp, dtype=int)
    cumulative_fn = ninst - cumulative_tp

    precisions = cumulative_tp / (cumulative_tp + cumulative_fp + np.finfo(float).eps)
    recalls = cumulative_tp / (cumulative_tp + cumulative_fn)
    precisions = np.maximum.accumulate(precisions[::-1])[::-1]  # interpolated
    precisions_interp = np.interp(recalls_interp, recalls, precisions, right=0)

    avg_precision = precisions_interp.mean()
    return avg_precision, precisions_interp
