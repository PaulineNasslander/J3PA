"""Waymo Open Dataset 1.4.3 labelled LiDAR frames with per-point semantic and instance labels, for the evaluation probes only.

Ported from the adam/add-semantickitti-waymo branch (data/outdoor_labels.py, data/outdoor_dataset.py), reading
the NPZ caches its src/scripts/prepare_waymo.py wrote directly instead of through a manifest:

    $WAYMO_DATASET_ROOT/{training,validation}/<sha256 of frame id>.npz
        points (N, 4) float32 [x, y, z, raw intensity], vehicle frame, all five LiDARs and both returns
        labels (N, 2) int32 [instance, semantic]; only TOP LiDAR points are labelled, the rest are semantic 0 (ignore)

Only the 23,691 / 5,976 labelled training / validation frames are cached. Intensity is fed as tanh(raw) * 255,
the adam/add-semantickitti-waymo convention.
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from constants import USE_TIME, X_INDEX, Y_INDEX, Z_INDEX
from data.base_dataset import BaseLidarDataset
from data.zod_dataset import _zero_pad_point_cloud_with_labels

DATASET_ROOT = os.environ.get("WAYMO_DATASET_ROOT", "/datasets/waymo/semanticwaymo/dataset")
# Pointcept's Waymo crop (x/y +-75.2 m, z -4..2 m); the evaluators only apply it with --pc_range_filter.
PC_RANGE = [-75.2, 75.2, -75.2, 75.2, -4.0, 2.0]

SPLIT_COUNTS = {True: ("training", 23691), False: ("validation", 5976)}
# --version mini keeps every MINI_STRIDE-th frame.
MINI_STRIDE = 100

CLASSES = (
    "car", "truck", "bus", "other-vehicle", "motorcyclist", "bicyclist", "pedestrian", "sign", "traffic-light",
    "pole", "construction-cone", "bicycle", "motorcycle", "building", "vegetation", "tree-trunk", "curb", "road",
    "lane-marker", "other-ground", "walkable", "sidewalk",
)
# Classes with consistent positive instance ids in the full label census.
THING_IDS = tuple(range(1, 8))


def decode_labels(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(semantic 0..22, instance) per point; positive (class, instance) pairs of thing classes get scan-local ids 1.., the rest 0."""
    semantic = labels[:, 1].astype(np.int64)
    if np.any((semantic < 0) | (semantic > len(CLASSES))):
        raise ValueError("Waymo semantic ids must be in 0..22")
    valid = np.isin(semantic, THING_IDS) & (labels[:, 0] > 0)
    instance = np.zeros(len(labels), dtype=np.int64)
    if valid.any():
        _, inverse = np.unique(np.stack((semantic[valid], labels[valid, 0]), axis=1), axis=0, return_inverse=True)
        instance[valid] = inverse.reshape(-1) + 1
    return semantic, instance


class WaymoDataset(Dataset, BaseLidarDataset):
    """Labelled single frames, full 360 degree field of view, no detection boxes."""

    def __init__(self, train_flag: bool = True, version: str = "full"):
        split, expected = SPLIT_COUNTS[train_flag]
        self.split_dir = os.path.join(DATASET_ROOT, split)
        self.frames = sorted(name for name in os.listdir(self.split_dir) if name.endswith(".npz"))
        if len(self.frames) != expected:
            raise ValueError(f"Waymo {split}: {len(self.frames)} frames under {self.split_dir}, expected {expected}")
        if version == "mini":
            self.frames = self.frames[::MINI_STRIDE]
        print(f"Waymo {split}: {len(self.frames)} frames from {DATASET_ROOT}")

    @classmethod
    def get_class_name_to_id(cls) -> dict[str, int]:
        return {}

    @classmethod
    def get_seg_class_name_to_id(cls) -> dict[str, int]:
        return {name: i + 1 for i, name in enumerate(CLASSES)}

    @classmethod
    def get_seg_thing_ids(cls) -> tuple[int, ...]:
        return THING_IDS

    @classmethod
    def get_seg_ignore_id(cls) -> int | None:
        return 0

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        with np.load(os.path.join(self.split_dir, self.frames[index]), allow_pickle=False) as data:
            points, labels = data["points"], data["labels"]
        if labels.shape != (len(points), 2):
            raise ValueError(f"Waymo {self.frames[index]}: {len(points)} points but labels of shape {labels.shape}")
        semantic, instance = decode_labels(labels)

        points = points.astype(np.float32)
        # Raw intensity is mostly below 1 with rare outliers up to ~4e5; tanh keeps that spread within 0..255.
        points[:, 3] = np.tanh(points[:, 3]) * 255.0
        if USE_TIME:
            points = np.column_stack((np.zeros(len(points), dtype=np.float32), points))
        cloud = torch.from_numpy(points)
        labels = torch.from_numpy(np.stack((semantic, instance), axis=1))

        x_min, x_max, y_min, y_max, z_min, z_max = PC_RANGE
        keep = (
            (cloud[:, X_INDEX] >= x_min) & (cloud[:, X_INDEX] <= x_max)
            & (cloud[:, Y_INDEX] >= y_min) & (cloud[:, Y_INDEX] <= y_max)
            & (cloud[:, Z_INDEX] >= z_min) & (cloud[:, Z_INDEX] <= z_max)
        )
        cloud, labels = _zero_pad_point_cloud_with_labels(cloud[keep], labels[keep])

        annotations = {
            "class": torch.zeros(0, dtype=torch.int32),
            "center": torch.zeros(0, 3),
            "size": torch.zeros(0, 3),
            "quats": torch.zeros(0, 4),
            "num_points": torch.zeros(0, dtype=torch.int32),
        }
        field_of_view = torch.tensor([360.0, 360.0], dtype=torch.float32)
        return cloud, annotations, field_of_view, {"semantic": labels[:, 0], "instance": labels[:, 1]}
