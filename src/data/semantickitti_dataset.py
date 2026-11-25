"""SemanticKITTI single scans with per-point semantic and instance labels, for the evaluation probes only.

Ported from the adam/add-semantickitti-waymo branch (data/outdoor_labels.py, data/outdoor_dataset.py),
reading the extracted odometry layout directly instead of through a manifest:

    $SEMANTICKITTI_DATASET_ROOT/sequences/NN/velodyne/*.bin   float32 [x, y, z, reflectance 0..1]
    $SEMANTICKITTI_DATASET_ROOT/sequences/NN/labels/*.label   uint32, semantic id in the low 16 bits, instance in the high 16

Standard split: train 00-07, 09, 10 (19,130 scans), validation 08 (4,071 scans).
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from constants import USE_TIME, X_INDEX, Y_INDEX, Z_INDEX
from data.base_dataset import BaseLidarDataset, normalize_intensity
from data.zod_dataset import _zero_pad_point_cloud_with_labels

DATASET_ROOT = os.environ.get("SEMANTICKITTI_DATASET_ROOT", "/datasets/semantickitti/dataset")
# Pointcept's SemanticKITTI crop (x/y +-35.2 m, z -4..2 m); the evaluators only apply it with --pc_range_filter.
PC_RANGE = [-35.2, 35.2, -35.2, 35.2, -4.0, 2.0]

SPLITS = {
    True: ("00", "01", "02", "03", "04", "05", "06", "07", "09", "10"),
    False: ("08",),
}
SPLIT_COUNTS = {"00": 4541, "01": 1101, "02": 4661, "03": 801, "04": 271, "05": 2761, "06": 1101, "07": 1101, "08": 4071, "09": 1591, "10": 1201}
# --version mini keeps every MINI_STRIDE-th scan.
MINI_STRIDE = 100

CLASSES = (
    "car", "bicycle", "motorcycle", "truck", "other-vehicle", "person", "bicyclist", "motorcyclist",
    "road", "parking", "sidewalk", "other-ground", "building", "fence", "vegetation", "trunk", "terrain",
    "pole", "traffic-sign",
)
# Raw SemanticKITTI ids -> 1..19 (0 = ignore); moving variants merge into their static class.
RAW_TO_SEG_ID = {
    0: 0, 1: 0, 10: 1, 11: 2, 13: 5, 15: 3, 16: 5, 18: 4, 20: 5, 30: 6, 31: 7, 32: 8, 40: 9, 44: 10, 48: 11,
    49: 12, 50: 13, 51: 14, 52: 0, 60: 9, 70: 15, 71: 16, 72: 17, 80: 18, 81: 19, 99: 0,
    252: 1, 253: 7, 254: 6, 255: 8, 256: 5, 257: 5, 258: 4, 259: 5,
}
THING_IDS = tuple(range(1, 9))

_LUT = np.zeros(1 << 16, dtype=np.int64)
for _raw, _seg in RAW_TO_SEG_ID.items():
    _LUT[_raw] = _seg


def decode_labels(packed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(semantic 0..19, instance) per point; instance ids are kept for thing classes only."""
    raw = packed & 0xFFFF
    unknown = set(np.unique(raw).tolist()) - RAW_TO_SEG_ID.keys()
    if unknown:
        raise ValueError(f"Unknown SemanticKITTI semantic ids: {sorted(unknown)}")
    semantic = _LUT[raw]
    instance = (packed >> 16).astype(np.int64)
    instance[~np.isin(semantic, THING_IDS)] = 0
    return semantic, instance


class SemanticKITTIDataset(Dataset, BaseLidarDataset):
    """Labelled single scans, full 360 degree field of view, no detection boxes."""

    def __init__(self, train_flag: bool = True, version: str = "full"):
        self.scans = []
        for sequence in SPLITS[train_flag]:
            velodyne_dir = os.path.join(DATASET_ROOT, "sequences", sequence, "velodyne")
            names = sorted(name[:-4] for name in os.listdir(velodyne_dir) if name.endswith(".bin"))
            if len(names) != SPLIT_COUNTS[sequence]:
                raise ValueError(f"SemanticKITTI sequence {sequence}: {len(names)} scans under {velodyne_dir}, expected {SPLIT_COUNTS[sequence]}")
            self.scans += [(sequence, name) for name in names]
        if version == "mini":
            self.scans = self.scans[::MINI_STRIDE]
        print(f"SemanticKITTI {'train' if train_flag else 'val'}: {len(self.scans)} scans from {DATASET_ROOT}")

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
        return len(self.scans)

    def __getitem__(self, index):
        sequence, name = self.scans[index]
        sequence_dir = os.path.join(DATASET_ROOT, "sequences", sequence)
        points = np.fromfile(os.path.join(sequence_dir, "velodyne", f"{name}.bin"), dtype="<f4").reshape(-1, 4)
        packed = np.fromfile(os.path.join(sequence_dir, "labels", f"{name}.label"), dtype="<u4")
        if len(packed) != len(points):
            raise ValueError(f"SemanticKITTI {sequence}/{name}: {len(points)} points but {len(packed)} labels")
        semantic, instance = decode_labels(packed)

        points = points.copy()
        points[:, 3] = normalize_intensity(points[:, 3], 1.0)
        if USE_TIME:
            points = np.column_stack((np.zeros(len(points), dtype=np.float32), points))
        cloud = torch.from_numpy(points).float()
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
