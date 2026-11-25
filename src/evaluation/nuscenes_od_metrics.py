"""nuScenes detection metrics: mAP, mATE/mASE/mAOE/mAVE/mAAE and NDS.

A port of the nuScenes devkit's nuscenes/eval/detection/algo.py (`accumulate`, `calc_ap`, `calc_tp`) and the
NDS of nuscenes/eval/detection/data_classes.py (Apache 2.0, Copyright 2019 Motional), working on our in-memory
`BoxObject`/`Detection` arrays instead of a results json, with the devkit's detection_cvpr_2019 config.
See ../../LICENSE for the Apache-2.0 license text.

Differences with the devkit that come from the inputs, not from this code:
    - nothing predicts attributes, so AAE is 1 for every class, as the devkit scores boxes without them; nor
      velocity unless the detections carry a ``velocity`` (BEV, m/s), else AVE is 1 too;
    - boxes are in the lidar frame; the BEV distances, velocity and yaw differences scored here do not depend on it.
"""

import numpy as np

DIST_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
TP_DIST_THRESHOLD = 2.0
MIN_RECALL = 0.1
MIN_PRECISION = 0.1
MEAN_AP_WEIGHT = 5
NUM_RECALL_SAMPLES = 101
TP_METRICS = {"trans_err": "mATE", "scale_err": "mASE", "orient_err": "mAOE", "vel_err": "mAVE", "attr_err": "mAAE"}
# Errors computed from the matched pairs, in the column order of `_accumulate`'s error rows.
MEASURED_TP_METRICS = ("trans_err", "scale_err", "orient_err", "vel_err")
# Left out of the class means by the devkit: cones have no heading, barriers no velocity or attributes.
UNDEFINED_TP_METRICS = {"traffic_cone": ("orient_err", "vel_err", "attr_err"), "barrier": ("vel_err", "attr_err")}
# The recall index from which AP and the TP errors are averaged.
FIRST_RECALL_INDEX = round(100 * MIN_RECALL) + 1


def evaluate_nuscenes_detection(ground_truths, detections, class_names) -> dict:
    """mAP, TP errors and NDS as the nuScenes devkit computes them.

    Args:
        ground_truths: Per frame, an array of `BoxObject`s, already filtered as the devkit does (class range, no
            boxes without lidar or radar points).
        detections: Per frame, an array of `Detection`s; the same frames in the same order.
        class_names: The detection classes to score.

    Returns:
        "mAP", "NDS", the class-mean TP errors ("mATE", "mASE", "mAOE", "mAVE", "mAAE"), and "per_class_ap"
        and "per_class_tp" (class -> TP error name -> value, NaN where the devkit leaves it undefined).
    """
    per_class_ap, per_class_tp = {}, {}
    unpredicted = ("attr_err",) if any(hasattr(dt, "velocity") for frame in detections for dt in frame) else ("vel_err", "attr_err")
    for name in class_names:
        gts = [[obj for obj in frame if str(obj.class_name) == name] for frame in ground_truths]
        dts = [[obj for obj in frame if str(obj.class_name) == name] for frame in detections]
        aps = []
        for threshold in DIST_THRESHOLDS:
            curves = _accumulate(gts, dts, name, threshold)
            aps.append(_calc_ap(curves))
            if threshold == TP_DIST_THRESHOLD:
                per_class_tp[name] = {
                    metric: np.nan if metric in UNDEFINED_TP_METRICS.get(name, ())
                    else 1.0 if metric in unpredicted
                    else _calc_tp(curves, metric)
                    for metric in TP_METRICS
                }
        per_class_ap[name] = float(np.mean(aps))

    mean_ap = float(np.mean(list(per_class_ap.values())))
    mean_tp = {metric: float(np.nanmean([per_class_tp[name][metric] for name in class_names])) for metric in TP_METRICS}
    tp_scores = [max(0.0, 1.0 - error) for error in mean_tp.values()]
    nds = (MEAN_AP_WEIGHT * mean_ap + sum(tp_scores)) / (MEAN_AP_WEIGHT + len(tp_scores))
    return {
        "mAP": mean_ap,
        "NDS": nds,
        **{TP_METRICS[metric]: error for metric, error in mean_tp.items()},
        "per_class_ap": per_class_ap,
        "per_class_tp": per_class_tp,
    }


def _accumulate(gts, dts, class_name, threshold) -> dict | None:
    """Precision, confidence and the TP errors at the recall samples; None when the devkit has no matches.

    Each frame's detections claim, in score order, the nearest unclaimed box within ``threshold`` (BEV centre
    distance). Matching never crosses frames, so doing it frame by frame equals the devkit's global score order.
    """
    num_gt = sum(len(frame) for frame in gts)
    if num_gt == 0:
        return None
    period = np.pi if class_name == "barrier" else 2 * np.pi
    scores, matched, errors = [], [], []
    for frame_gts, frame_dts in zip(gts, dts, strict=True):
        if not frame_dts:
            continue
        frame_scores = np.array([float(dt.score) for dt in frame_dts])
        order = np.argsort(-frame_scores, kind="stable")
        dt_boxes = np.stack([np.asarray(frame_dts[i].box, dtype=np.float64) for i in order])
        gt_boxes = np.stack([np.asarray(gt.box, dtype=np.float64) for gt in frame_gts]) if frame_gts else np.zeros((0, 10))
        distances = np.linalg.norm(dt_boxes[:, None, :2] - gt_boxes[None, :, :2], axis=-1)
        taken = np.zeros(len(gt_boxes), dtype=bool)
        for d, score in enumerate(frame_scores[order]):
            scores.append(score)
            candidates = np.where(taken, np.inf, distances[d])
            g = int(np.argmin(candidates)) if len(candidates) else -1
            if g >= 0 and candidates[g] < threshold:
                taken[g] = True
                matched.append(True)
                dt, gt = frame_dts[order[d]], frame_gts[g]
                errors.append((
                    candidates[g],
                    1.0 - _aligned_iou(dt_boxes[d, 3:6], gt_boxes[g, 3:6]),
                    _yaw_difference(dt_boxes[d, 6:10], gt_boxes[g, 6:10], period),
                    # NaN when the box's velocity is unknown, which the running mean skips.
                    float(np.linalg.norm(np.asarray(dt.velocity)[:2] - np.asarray(gt.velocity)[:2]))
                    if hasattr(dt, "velocity") and hasattr(gt, "velocity") else np.nan,
                ))
            else:
                matched.append(False)
                errors.append((np.nan,) * len(MEASURED_TP_METRICS))
    if not any(matched):
        return None

    order = np.argsort(-np.array(scores), kind="stable")
    scores = np.array(scores)[order]
    matched = np.array(matched)[order]
    errors = np.array(errors)[order][matched]
    true_positives = np.cumsum(matched)
    false_positives = np.cumsum(~matched)
    recall_samples = np.linspace(0, 1, NUM_RECALL_SAMPLES)
    recall = true_positives / num_gt
    confidence = np.interp(recall_samples, recall, scores, right=0)
    curves = {
        "precision": np.interp(recall_samples, recall, true_positives / (true_positives + false_positives), right=0),
        "confidence": confidence,
    }
    matched_scores = scores[matched]
    for column, metric in enumerate(MEASURED_TP_METRICS):
        curves[metric] = np.interp(confidence[::-1], matched_scores[::-1], _cummean(errors[:, column])[::-1])[::-1]
    return curves


def _cummean(values) -> np.ndarray:
    """The devkit's running mean that skips NaNs (all ones when everything is NaN)."""
    if np.isnan(values).all():
        return np.ones(len(values))
    sums = np.nancumsum(values.astype(float))
    counts = np.cumsum(~np.isnan(values))
    return np.divide(sums, counts, out=np.zeros_like(sums), where=counts != 0)


def _calc_ap(curves) -> float:
    """Area under the precision curve above the minimum recall and precision, normalised to [0, 1]."""
    if curves is None:
        return 0.0
    precision = np.clip(curves["precision"][FIRST_RECALL_INDEX:] - MIN_PRECISION, 0, None)
    return float(np.mean(precision)) / (1.0 - MIN_PRECISION)


def _calc_tp(curves, metric) -> float:
    """A TP error averaged from the minimum recall up to the highest recall reached."""
    if curves is None:
        return 1.0
    reached = np.nonzero(curves["confidence"])[0]
    last = reached[-1] if len(reached) else 0
    if last < FIRST_RECALL_INDEX:
        return 1.0
    return float(np.mean(curves[metric][FIRST_RECALL_INDEX : last + 1]))


def _aligned_iou(dt_size, gt_size) -> float:
    """3D IoU of two boxes with the same centre and heading."""
    intersection = np.prod(np.minimum(dt_size, gt_size))
    return intersection / (np.prod(dt_size) + np.prod(gt_size) - intersection)


def _yaw(quaternion) -> float:
    """Heading of the box's x axis in the BEV plane, from a [w, x, y, z] quaternion."""
    w, x, y, z = quaternion
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))


def _yaw_difference(dt_quaternion, gt_quaternion, period) -> float:
    """Absolute heading error, wrapped to the class's symmetry period."""
    difference = (_yaw(gt_quaternion) - _yaw(dt_quaternion) + period / 2) % period - period / 2
    if difference > np.pi:
        difference -= 2 * np.pi
    return abs(difference)
