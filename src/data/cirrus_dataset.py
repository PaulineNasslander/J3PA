"""Cirrus: two forward-facing Luminar H2 LiDARs (120 x 30 degree field of view, 250 m range) on a car, with 3D boxes.

Data: the release (https://github.com/Cirrus-dataset/Cirrus, arXiv 2012.02938) unpacked under
``CIRRUS_DATASET_ROOT`` (default /datasets/cirrus) as ``Cirrus_dataset<k>/`` (k = 1..7) with
``dataset<k>_gaussian/<ns>.xyz`` and ``dataset<k>_uniform/<ns>.xyz`` (CSV x, y, z, intensity 0..4095, then unused
columns), ``dataset<k>_annotation/<ns>.json`` (3D boxes) and ``match_uniform_dataset<k>.txt`` (the uniform scan taken
with each gaussian one). The two sensors scan with a gaussian and a uniform vertical pattern and are calibrated into
one frame; the boxes are labelled on the gaussian scans. Cirrus has no per-point labels; those come from the boxes,
precomputed by `scripts/precompute_cirrus_labels.py` into ``CIRRUS_LABELS_DIR``.

Frames are released at 1 Hz without ego poses, so sweeps cannot be merged: every sample is one scan and the requested
number of sweeps is ignored (``num_sweeps`` is always 1). The two patterns are not merged either; their scans are up
to 70 ms apart, which is ~2 m at highway speed.

Splits: Cirrus publishes none. All frames come from one evening (2019-05-17), recorded in blocks of 5 to 15 minutes;
``Cirrus_dataset2`` holds one highway and one urban block, which become val. Train is every other sequence, without
the frames within VAL_BUFFER_S of a val frame (sequence 1 continues directly before both val blocks).

Points stay in the sensor frame: x forward, y left, z up, the road about 2.1 m below. Scans carry no per-point time,
so the age is 0.
"""

import json
import os
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from constants import CIRRUS_LABELS_DIR, MIN_NUM_POINTS, PC_RANGE, RANDOM_CROP_CENTER_RANGES, USE_TIME, X_INDEX, Y_INDEX, Z_INDEX
from data.base_dataset import BaseLidarDataset, normalize_intensity
from data.zod_dataset import (
    NoPointsException,
    TorchList,
    _reshape_annotations_dict,
    _zero_pad_point_cloud,
    _zero_pad_point_cloud_with_labels,
)

DATASET_ROOT = os.environ.get("CIRRUS_DATASET_ROOT", "/datasets/cirrus")
RANDOM_CROP_CENTER_RANGE = RANDOM_CROP_CENTER_RANGES["Cirrus"]
MAX_TRY = 10
SEQUENCES = tuple(range(1, 8))
VAL_SEQUENCES = (2,)
VAL_BUFFER_S = 10.0
PATTERNS = ("gaussian", "uniform")
# Intensity is 12-bit; 4095 is also the largest value found in the release.
INTENSITY_MAX = 4095.0

# The paper's classes plus "unknown", which labels real objects of no other class.
CLASSES = ("vehicle", "large_vehicle", "pedestrian", "bicycle", "animal", "wheeled_pedestrian", "motorcycle", "trailer", "unknown")
CLASS_NAMES = list(CLASSES)
CLASS_NAME_TO_ID = {name: idx + 1 for idx, name in enumerate(CLASSES)}

# The sensors look forward with x forward, which `limit_pc_to_fov` (y forward) does not handle; the boxes cover the
# whole sensor field of view, so there is nothing to cut.
FIELD_OF_VIEW = torch.tensor([360.0, 360.0], dtype=torch.float32)

# How many frames each split keeps in "mini" (debug) mode.
MINI_SIZES = {"pre-train": 64, "train": 64, "val": 32}


class CirrusDataset(Dataset, BaseLidarDataset):
    """Single LiDAR scans, optionally with 3D box annotations and box-derived per-point labels."""

    @classmethod
    def get_class_name_to_id(cls) -> dict[str, int]:
        return CLASS_NAME_TO_ID

    def __init__(
        self,
        split: Literal["pre-train", "train", "val"],
        version: Literal["full", "mini"] = "full",
        load_annotations: bool = False,
        load_frame_id: bool = False,
        random_crop: bool = False,
        load_segmentation: bool = False,
        min_num_sweeps_per_sample: int = 1,
        max_num_sweeps_per_sample: int = 1,
        load_num_sweeps: bool = False,
    ):
        """Initialize.

        Args:
            split: "pre-train" is every train frame, each __getitem__ loading its gaussian or uniform scan at random.
                "train" and "val" are the annotated gaussian scans of the train and val sequences.
            version: "mini" keeps MINI_SIZES frames.
            load_annotations: Whether to load the 3D boxes (labelled splits only).
            load_segmentation: Also return per-point labels precomputed from the boxes by
                `scripts/precompute_cirrus_labels.py`: points inside a box get its class and 1-based ordinal, all
                others are background (0).
            min_num_sweeps_per_sample, max_num_sweeps_per_sample: Accepted for a common interface; Cirrus has no
                poses, so every sample is one scan.
        """
        if split not in MINI_SIZES:
            raise ValueError(f"Unknown split {split!r}, expected one of {list(MINI_SIZES)}.")
        assert 1 <= min_num_sweeps_per_sample <= max_num_sweeps_per_sample
        if split == "pre-train" and (load_annotations or load_segmentation):
            raise ValueError("Pre-training items mix in unlabelled uniform scans, use the train split for labels.")

        self._labelled = split != "pre-train"
        self._load_annotations = load_annotations
        self._load_frame_id = load_frame_id
        self._load_num_sweeps = load_num_sweeps
        self._random_crop = random_crop
        self._load_segmentation = load_segmentation

        frames = _split_frames("val" if split == "val" else "train")
        if version == "mini":
            frames = [frames[i] for i in np.linspace(0, len(frames) - 1, MINI_SIZES[split]).round().astype(int)]
        # Serialized storage so the index is not copy-on-write duplicated by the dataloader workers.
        self._frames = TorchList(frames)
        note = f", always 1 sweep (asked for up to {max_num_sweeps_per_sample})" if max_num_sweeps_per_sample > 1 else ""
        print(f"Cirrus: {split} ({version}) -> {len(frames)} frames per epoch{note}")

    def __len__(self):
        """Get length."""
        return len(self._frames)

    def __getitem__(self, index):
        """Get item.

        Returns:
            - point_cloud: (MAX_NUM_POINTS, C), [x, y, z, intensity 0..255] in the sensor frame, with an age of 0
              prepended when USE_TIME is enabled.
            - frame_id (optional): "dataset<k>/<pattern>/<ns timestamp>".
            - num_sweeps (optional): always 1.
            - annotations (optional): dict of padded tensors with keys
              "class" (MAX_NUM_OBJECTS,), "center" (MAX_NUM_OBJECTS, 3),
              "size" (MAX_NUM_OBJECTS, 3) as [length, width, height] and
              "quats" (MAX_NUM_OBJECTS, 4) as [w, x, y, z].
            - field_of_view (optional): tensor of shape (2) in degrees.
            - seg labels (optional): {"semantic", "instance"} per point.
        """
        current_index = index
        no_points_tries = 0

        while True:
            try:
                return self._getitem(current_index)
            except NoPointsException:
                no_points_tries += 1

                if no_points_tries > MAX_TRY:
                    # If repeated random crops fail, fall back to another item.
                    if len(self._frames) <= 1:
                        raise
                    current_index = torch.randint(0, len(self._frames), size=(1,)).item()
                    no_points_tries = 0

    def _getitem(self, index):
        """Load one scan and, when requested, its labels."""
        frame = self._frames[index]
        pattern = "gaussian"
        if not self._labelled:
            # Always drawn, so the crop draws that follow do not depend on whether the frame has a uniform scan.
            use_uniform = torch.randint(0, 2, size=(1,)).item() == 1
            if use_uniform and frame["uniform"] is not None:
                pattern = "uniform"
        stamp = frame["stamp"] if pattern == "gaussian" else frame["uniform"]
        lidar_points_tensor = load_scan(frame["sequence"], pattern, stamp)

        if self._load_segmentation:
            # The cloud is the raw gaussian scan the labels were computed for, in the same order.
            path = label_path(frame["sequence"], frame["stamp"])
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Missing labels for {frame['sequence']}/{frame['stamp']} at {path}. Run scripts/precompute_cirrus_labels.py first.")
            labels = np.load(path)
            semantic = torch.from_numpy(labels["semantic"]).long()
            instance = torch.from_numpy(labels["instance"]).long()

        # Filter points within PC_RANGE
        x_min, x_max, y_min, y_max, z_min, z_max = PC_RANGE
        if self._random_crop:
            center = torch.tensor([torch.rand(1).item() * (RANDOM_CROP_CENTER_RANGE[0][1] - RANDOM_CROP_CENTER_RANGE[0][0]) + RANDOM_CROP_CENTER_RANGE[0][0],
                                   torch.rand(1).item() * (RANDOM_CROP_CENTER_RANGE[1][1] - RANDOM_CROP_CENTER_RANGE[1][0]) + RANDOM_CROP_CENTER_RANGE[1][0]])
            x_min += center[0]
            x_max += center[0]
            y_min += center[1]
            y_max += center[1]

        mask = (
            (lidar_points_tensor[:, X_INDEX] >= x_min)
            & (lidar_points_tensor[:, X_INDEX] <= x_max)
            & (lidar_points_tensor[:, Y_INDEX] >= y_min)
            & (lidar_points_tensor[:, Y_INDEX] <= y_max)
            & (lidar_points_tensor[:, Z_INDEX] >= z_min)
            & (lidar_points_tensor[:, Z_INDEX] <= z_max)
        )
        if mask.sum() < MIN_NUM_POINTS:
            raise NoPointsException(
                f"Too few points left after filtering for index {index}, only {mask.sum()} points remaining."
            )
        if self._load_segmentation:
            lidar_points_tensor, seg_labels = _zero_pad_point_cloud_with_labels(
                lidar_points_tensor[mask], torch.stack((semantic, instance), dim=1)[mask]
            )
        else:
            lidar_points_tensor = _zero_pad_point_cloud(lidar_points_tensor[mask])

        outs = (lidar_points_tensor,)
        if self._load_frame_id:
            outs += (f"dataset{frame['sequence']}/{pattern}/{stamp}",)
        if self._load_num_sweeps:
            outs += (1,)

        if self._load_annotations:
            annotations = read_boxes(frame["sequence"], frame["stamp"])
            # Filter annotations based on pc range as well
            centers = annotations["center"]
            if centers.shape[0] > 0:
                xs, ys, zs = centers[:, 0], centers[:, 1], centers[:, 2]
                mask = (xs >= x_min) & (xs <= x_max) & (ys >= y_min) & (ys <= y_max) & (zs >= z_min) & (zs <= z_max)
                for key in annotations:
                    annotations[key] = annotations[key][mask]

            annotations = _reshape_annotations_dict(annotations)  # pad to fixed size
            outs += (annotations, FIELD_OF_VIEW)
        if self._load_segmentation:
            outs += ({"semantic": seg_labels[:, 0], "instance": seg_labels[:, 1]},)

        return outs[0] if len(outs) == 1 else outs


def build_cirrus_dataset(
    train_flag: bool,
    version: str = "full",
    load_annotations: bool = False,
    load_frame_id: bool = False,
    random_crop: bool = False,
    load_segmentation: bool = False,
    min_num_sweeps_per_sample: int = 1,
    max_num_sweeps_per_sample: int = 1,
    load_num_sweeps: bool = False,
) -> CirrusDataset:
    """Pick the split each training stage needs.

    * Pre-training (``train_flag`` without labels): the train frames, gaussian or uniform scan at random.
    * Probe training (``train_flag`` with labels): the annotated gaussian scans of the train sequences.
    * Validation: the annotated gaussian scans of ``Cirrus_dataset2``.
    """
    if not train_flag:
        split = "val"
    elif load_annotations or load_segmentation:
        split = "train"
    else:
        split = "pre-train"
    return CirrusDataset(
        split,
        version=version,
        load_annotations=load_annotations,
        load_frame_id=load_frame_id,
        random_crop=random_crop,
        load_segmentation=load_segmentation,
        min_num_sweeps_per_sample=min_num_sweeps_per_sample,
        max_num_sweeps_per_sample=max_num_sweeps_per_sample,
        load_num_sweeps=load_num_sweeps,
    )


def label_path(sequence: int, stamp: str, labels_dir: str | None = None) -> str:
    """Per-point labels of one annotated gaussian scan, from `scripts/precompute_cirrus_labels.py`."""
    return os.path.join(labels_dir or CIRRUS_LABELS_DIR, f"dataset{sequence}", f"{stamp}.npz")


def _sequence_dir(sequence: int) -> str:
    return os.path.join(DATASET_ROOT, f"Cirrus_dataset{sequence}")


def _index_sequence(sequence: int) -> list[dict]:
    """The sequence's annotated frames in time order: "sequence", "stamp" (ns, as in the file names) and "uniform",
    the stamp of the uniform scan taken with it (None if there is none)."""
    directory = _sequence_dir(sequence)
    stamps = sorted(file[:-5] for file in os.listdir(os.path.join(directory, f"dataset{sequence}_annotation")) if file.endswith(".json"))
    uniform = {}
    match_file = os.path.join(directory, f"match_uniform_dataset{sequence}.txt")
    if os.path.isfile(match_file):
        with open(match_file) as f:
            uniform = {gaussian[:-4]: other[:-4] for gaussian, other in (line.split() for line in f if line.strip())}
    return [{"sequence": sequence, "stamp": stamp, "uniform": uniform.get(stamp)} for stamp in stamps]


def _split_frames(split: Literal["train", "val"]) -> list[dict]:
    """Frames of the val sequences, or of the others without those within VAL_BUFFER_S of a val frame."""
    val = [frame for sequence in VAL_SEQUENCES for frame in _index_sequence(sequence)]
    if split == "val":
        return val
    val_times = np.sort(np.array([int(frame["stamp"]) for frame in val], dtype=np.int64))
    buffer_ns = int(VAL_BUFFER_S * 1e9)
    frames = []
    for sequence in SEQUENCES:
        if sequence in VAL_SEQUENCES:
            continue
        for frame in _index_sequence(sequence):
            time = int(frame["stamp"])
            position = np.searchsorted(val_times, time)
            near = val_times[max(position - 1, 0) : position + 1]
            if near.size == 0 or np.abs(near - time).min() > buffer_ns:
                frames.append(frame)
    return frames


def read_scan(sequence: int, pattern: str, stamp: str) -> np.ndarray:
    """(N, 4) [x, y, z, intensity 0..4095] of one scan, in file order."""
    path = os.path.join(_sequence_dir(sequence), f"dataset{sequence}_{pattern}", f"{stamp}.xyz")
    return pd.read_csv(path, header=None, usecols=[0, 1, 2, 3], dtype=np.float32).to_numpy()


def load_scan(sequence: int, pattern: str, stamp: str) -> torch.Tensor:
    """One scan as an (N, C) tensor, intensity rescaled to 0..255 and the age (0) prepended when USE_TIME is set."""
    columns = read_scan(sequence, pattern, stamp)
    columns[:, 3] = normalize_intensity(columns[:, 3], INTENSITY_MAX)
    if USE_TIME:
        columns = np.column_stack((np.zeros(columns.shape[0], dtype=np.float32), columns))
    return torch.from_numpy(columns)


def read_boxes(sequence: int, stamp: str) -> dict[str, torch.Tensor]:
    """The frame's boxes: "class" (M,), "center" (M, 3), "size" (M, 3) as [length, width, height] and "quats" (M, 4)
    as [w, x, y, z]. The files store the geometric centre and scalar-last [x, y, z, w] quaternions."""
    with open(os.path.join(_sequence_dir(sequence), f"dataset{sequence}_annotation", f"{stamp}.json")) as f:
        labels = json.load(f)["labels"]
    unknown = {label["label_class"] for label in labels} - CLASS_NAME_TO_ID.keys()
    if unknown:
        raise ValueError(f"Unexpected Cirrus classes {sorted(unknown)} in dataset{sequence}/{stamp}.")
    quats = np.array([label["quaternion"] for label in labels], dtype=np.float32).reshape(-1, 4)
    return {
        "class": torch.tensor([CLASS_NAME_TO_ID[label["label_class"]] for label in labels], dtype=torch.int32),
        "center": torch.tensor([label["position"] for label in labels], dtype=torch.float32).reshape(-1, 3),
        "size": torch.tensor([label["size"] for label in labels], dtype=torch.float32).reshape(-1, 3),
        "quats": torch.from_numpy(quats[:, [3, 0, 1, 2]]),
    }
