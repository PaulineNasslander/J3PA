"""Side-by-side ground truth vs prediction renders for the evaluation.

mIoU and PQ hide *where* a model fails, so a handful of validation frames are
rendered at every training milestone. TensorBoard keeps one image per step under
the same tag, so its step slider turns into a before/after comparison of the same
frame at 1k/5k/20k/50k training steps.

Views are bird's eye: a top-down scatter is far easier to read for dense
segmentation than the 3D scatter used elsewhere in the repo, and the ego frame is
roughly flat anyway.
"""

import matplotlib

matplotlib.use("Agg")  # no display in the container or on the cluster
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

IGNORE_COLOUR = (0.30, 0.30, 0.30)
CORRECT_COLOUR = (0.25, 0.80, 0.35)
WRONG_COLOUR = (0.90, 0.20, 0.20)
MAX_PLOT_POINTS = 40000
# Half-width in metres of the detection renders, the largest nuScenes class evaluation range.
DETECTION_PLOT_RANGE = 50.0
# Predicted boxes below this score are not drawn; they would hide the real ones under clutter.
DETECTION_PLOT_MIN_SCORE = 0.3


def _seg_palette(num_classes: int) -> np.ndarray:
    """(num_classes, 3) RGB lookup, row 0 being the ignore/background colour."""
    base = np.asarray(plt.get_cmap("tab20").colors)[:, :3]
    palette = np.empty((num_classes, 3))
    palette[0] = IGNORE_COLOUR
    for class_id in range(1, num_classes):
        palette[class_id] = base[(class_id - 1) % len(base)]
    return palette


def _subsample(num_points: int) -> np.ndarray:
    """Indices of at most `MAX_PLOT_POINTS` points, stable for a given count.

    A fixed seed keeps the same points in every panel and at every milestone, so
    the renders of one frame stay directly comparable.
    """
    if num_points <= MAX_PLOT_POINTS:
        return np.arange(num_points)
    return np.random.default_rng(0).choice(num_points, MAX_PLOT_POINTS, replace=False)


def _bev_panel(axis, positions, colours, title: str) -> None:
    """Top-down scatter of one point cloud."""
    axis.scatter(positions[:, 0], positions[:, 1], c=colours, s=0.7, linewidths=0)
    axis.set_facecolor("black")
    axis.set_aspect("equal")
    axis.set_title(title, fontsize=10)
    axis.set_xticks([])
    axis.set_yticks([])


def _class_legend(figure, class_ids, id_to_name: dict[int, str], palette: np.ndarray) -> None:
    """One swatch per class that actually occurs in the frame."""
    handles = [
        Line2D(
            [], [], marker="o", linestyle="", markersize=5,
            color=palette[class_id],
            label=id_to_name.get(class_id, "background"),
        )
        for class_id in class_ids
    ]
    figure.legend(handles=handles, loc="lower center", ncol=6, fontsize=7, frameon=False)
    figure.subplots_adjust(bottom=0.18)


def _to_numpy(tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy() if tensor.is_floating_point() else tensor.detach().cpu().numpy()


def log_semantic_comparison(writer, tag: str, step: int, positions, gt_semantic, pred_semantic,
                            seg_class_name_to_id: dict[str, int], ignore_id: int | None) -> None:
    """Log ground truth, prediction and an error map for one frame.

    Args:
        writer: An open `SummaryWriter`.
        tag: TensorBoard tag; the same tag across milestones gives a step slider.
        step: Training budget this prediction was made at.
        positions: (N, 3) point coordinates of a single sample.
        gt_semantic: (N,) ground truth class ids.
        pred_semantic: (N,) predicted class ids.
        seg_class_name_to_id: The dataset's segmentation classes, for the legend.
        ignore_id: Class id that is not scored, or None if every point is.
    """
    positions = _to_numpy(positions)
    ground_truth = _to_numpy(gt_semantic)
    prediction = _to_numpy(pred_semantic)
    id_to_name = {i: name for name, i in seg_class_name_to_id.items()}
    palette = _seg_palette(len(seg_class_name_to_id) + 1)

    index = _subsample(len(positions))
    positions, ground_truth, prediction = positions[index], ground_truth[index], prediction[index]
    # Ignore points are not scored, so grey them out in the prediction panel too,
    # otherwise the two panels differ on points nobody is graded on.
    if ignore_id is None:
        scored = np.ones(len(ground_truth), dtype=bool)
    else:
        scored = ground_truth != ignore_id
        prediction = np.where(scored, prediction, ignore_id)

    correct = (prediction == ground_truth) & scored
    error_colours = np.tile(IGNORE_COLOUR, (len(prediction), 1))
    error_colours[scored] = WRONG_COLOUR
    error_colours[correct] = CORRECT_COLOUR
    accuracy = correct.sum() / max(scored.sum(), 1)

    figure, axes = plt.subplots(1, 3, figsize=(16, 6))
    _bev_panel(axes[0], positions, palette[ground_truth], "ground truth")
    _bev_panel(axes[1], positions, palette[prediction], "prediction")
    _bev_panel(axes[2], positions, error_colours, f"errors in red ({accuracy:.1%} correct)")
    _class_legend(
        figure,
        sorted(set(np.unique(ground_truth)) | set(np.unique(prediction))),
        id_to_name,
        palette,
    )

    writer.add_figure(tag, figure, global_step=step)
    plt.close(figure)


def log_instance_comparison(writer, tag: str, step: int, positions, gt_instance, gt_semantic,
                            pred_instance, pred_semantic, ignore_id: int | None) -> None:
    """Log ground truth vs predicted instances for one frame.

    Colours are arbitrary per instance, so read these for whether objects are
    split or merged, not for whether the colours match between panels.
    """
    positions = _to_numpy(positions)
    index = _subsample(len(positions))
    positions = positions[index]
    gt_instance, gt_semantic = _to_numpy(gt_instance)[index], _to_numpy(gt_semantic)[index]
    pred_instance, pred_semantic = _to_numpy(pred_instance)[index], _to_numpy(pred_semantic)[index]

    figure, axes = plt.subplots(1, 2, figsize=(11, 6))
    _bev_panel(axes[0], positions,
               _instance_colours(gt_instance, gt_semantic, ignore_id), "ground truth instances")
    _bev_panel(axes[1], positions,
               _instance_colours(pred_instance, pred_semantic, ignore_id), "predicted instances")
    figure.tight_layout()

    writer.add_figure(tag, figure, global_step=step)
    plt.close(figure)


def log_detection_comparison(writer, tag: str, step: int, positions, point_scores, gt_objects, dt_objects,
                             class_name_to_id: dict[str, int]) -> None:
    """Log ground truth boxes, predicted boxes and the per-point objectness for one frame, top-down.

    Args:
        writer: An open `SummaryWriter`.
        tag: TensorBoard tag; the same tag across milestones gives a step slider.
        step: Training budget this prediction was made at.
        positions: (N, 3) point coordinates of a single sample.
        point_scores: (N,) each point's highest class probability, before decoding into boxes.
        gt_objects: `BoxObject`s of the frame.
        dt_objects: `Detection`s of the frame; those under `DETECTION_PLOT_MIN_SCORE` are not drawn.
        class_name_to_id: The dataset's detection classes, for colours and the legend.
    """
    positions = _to_numpy(positions)
    point_scores = _to_numpy(point_scores)
    index = _subsample(len(positions))
    positions, point_scores = positions[index], point_scores[index]
    palette = _seg_palette(len(class_name_to_id) + 1)
    shown = [obj for obj in dt_objects if float(obj.score) >= DETECTION_PLOT_MIN_SCORE]

    figure, axes = plt.subplots(1, 3, figsize=(20, 7))
    grey = np.tile(IGNORE_COLOUR, (len(positions), 1))
    _bev_panel(axes[0], positions, grey, f"ground truth ({len(gt_objects)} boxes)")
    _bev_panel(axes[1], positions, grey, f"predictions with score >= {DETECTION_PLOT_MIN_SCORE} ({len(shown)} boxes)")
    _bev_panel(axes[2], positions, plt.get_cmap("magma")(np.clip(point_scores, 0, 1))[:, :3],
               "per-point objectness, ground truth outlined")
    for obj in gt_objects:
        colour = palette[class_name_to_id[str(obj.class_name)]]
        axes[0].add_patch(_box_footprint(obj.box, edgecolor=colour, linewidth=1.2))
        axes[2].add_patch(_box_footprint(obj.box, edgecolor="white", linewidth=0.8))
    for obj in shown:
        colour = palette[class_name_to_id[str(obj.class_name)]]
        axes[1].add_patch(_box_footprint(obj.box, edgecolor=colour, linewidth=1.2, alpha=float(obj.score)))
    for axis in axes:
        axis.set_xlim(-DETECTION_PLOT_RANGE, DETECTION_PLOT_RANGE)
        axis.set_ylim(-DETECTION_PLOT_RANGE, DETECTION_PLOT_RANGE)

    present = {str(obj.class_name) for obj in gt_objects} | {str(obj.class_name) for obj in shown}
    _class_legend(
        figure,
        sorted(class_name_to_id[name] for name in present),
        {i: name for name, i in class_name_to_id.items()},
        palette,
    )

    writer.add_figure(tag, figure, global_step=step)
    plt.close(figure)


def _box_footprint(box, **style) -> Polygon:
    """Top-down outline of a box given as [x, y, z, sx, sy, sz, qw, qx, qy, qz] with a yaw-only quaternion."""
    x, y, _, length, width, _, qw, qx, qy, qz = (float(v) for v in box)
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
    corners = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]]) * [length / 2, width / 2]
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return Polygon(corners @ rotation.T + [x, y], closed=True, fill=False, **style)


def _instance_colours(instance: np.ndarray, semantic: np.ndarray,
                      ignore_id: int | None) -> np.ndarray:
    """A repeatable arbitrary colour per instance id; grey for stuff and ignore."""
    palette = np.asarray(plt.get_cmap("tab20").colors)[:, :3]
    colours = np.tile(IGNORE_COLOUR, (len(instance), 1))
    has_instance = instance > 0
    if ignore_id is not None:
        has_instance &= semantic != ignore_id
    # Knuth multiplicative hash, so neighbouring ids do not get similar colours.
    spread = (instance[has_instance].astype(np.uint64) * np.uint64(2654435761)) % len(palette)
    colours[has_instance] = palette[spread.astype(np.int64)]
    return colours
