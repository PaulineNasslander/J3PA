"""PandaSet: a Pandar64 on a car, with 3D boxes in every frame and per-point semantic labels in most sequences.

Data: the 103 public sequences from https://pandaset.org (the devkit is not needed), as ``<sequence>/`` under
``PANDASET_DATASET_ROOT`` (default /datasets/pandaset). Needed from each are ``lidar/`` (``NN.pkl.gz`` sweeps,
``poses.json``), ``annotations/cuboids/`` and, for segmentation, ``annotations/semseg/``; cameras are unused.

Every sequence is 8 s of 80 frames at 10 Hz and every frame is annotated:
  * cuboids in all 103 sequences, in world coordinates with a yaw about the world z axis,
  * semseg, one of 42 classes for every point, in 76 sequences.

A sweep stores both LiDARs' points in world coordinates, already motion compensated: the 360 degree Pandar64
(``d == 0``) and the forward-facing PandarGT (``d == 1``). Only the Pandar64 is used, so the boxes are the ones
annotated on it (``cuboids.sensor_id`` -1 or 0; the PandarGT ones duplicate them). Points are moved into the
reference frame's pose from ``poses.json``, turned from PandaSet's x right / y forward to x forward / y left,
z up; the ground is about 0.3 m below that origin. Returns on the ego car's roof are dropped.

PandaSet has no official split (none in the devkit or the release). We use the de-facto one of OpenPCDet
(tools/cfgs/dataset_configs/pandaset_dataset.yaml, a random 61/20/22 split, whose test list OpenPCDet never uses),
with its test folded into val, and every val or test sequence without semseg moved to train, so every val frame
can score detection and segmentation.
"""

# The sequence split is adapted from OpenPCDet (Apache-2.0; see ../../LICENSE).
# Modified by combining its test split with validation and moving sequences
# without semantic labels to training.

import json
import os
from typing import Literal

import numpy as np
import pandas as pd
import torch
from pyquaternion import Quaternion
from torch.utils.data import Dataset

from constants import MIN_NUM_POINTS, PC_RANGE, RANDOM_CROP_CENTER_RANGES, USE_TIME, X_INDEX, Y_INDEX, Z_INDEX
from data.base_dataset import BaseLidarDataset, centered_sweep_chunk, normalize_intensity, sample_sweep_chunk
from data.zod_dataset import (
    NoPointsException,
    TorchList,
    _reshape_annotations_dict,
    _zero_pad_point_cloud,
    _zero_pad_point_cloud_with_labels,
)

DATASET_ROOT = os.environ.get("PANDASET_DATASET_ROOT", "/datasets/pandaset")
RANDOM_CROP_CENTER_RANGE = RANDOM_CROP_CENTER_RANGES["PandaSet"]
MAX_TRY = 10
PANDAR64 = 0
# Points within this footprint (|x|, |y| in metres) around the pose origin hit the ego car.
EGO_HALF_EXTENT = (2.5, 1.0)
# PandaSet's LiDAR frame is x right, y forward, z up; the rest of the codebase uses x forward, y left.
OURS_FROM_PANDASET = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

# Semseg classes (annotations/semseg/classes.json), ids 1..42; 0 never occurs in the labels.
SEG_CLASSES = (
    "Smoke", "Exhaust", "Spray or rain", "Reflection", "Vegetation", "Ground", "Road", "Lane Line Marking",
    "Stop Line Marking", "Other Road Marking", "Sidewalk", "Driveway", "Car", "Pickup Truck", "Medium-sized Truck",
    "Semi-truck", "Towed Object", "Motorcycle", "Other Vehicle - Construction Vehicle", "Other Vehicle - Uncommon",
    "Other Vehicle - Pedicab", "Emergency Vehicle", "Bus", "Personal Mobility Device", "Motorized Scooter", "Bicycle",
    "Train", "Trolley", "Tram / Subway", "Pedestrian", "Pedestrian with Object", "Animals - Bird", "Animals - Other",
    "Pylons", "Road Barriers", "Signs", "Cones", "Construction Signs", "Temporary Construction Barriers",
    "Rolling Containers", "Building", "Other Static Object",
)
SEG_CLASS_NAME_TO_ID = {name: idx + 1 for idx, name in enumerate(SEG_CLASSES)}
IGNORE_ID = 0
# The object classes, "Car" to "Rolling Containers", are exactly the cuboid labels.
CLASSES = SEG_CLASSES[SEG_CLASSES.index("Car") : SEG_CLASSES.index("Rolling Containers") + 1]
CLASS_NAMES = list(CLASSES)
CLASS_NAME_TO_ID = {name: idx + 1 for idx, name in enumerate(CLASSES)}
THING_IDS = tuple(SEG_CLASS_NAME_TO_ID[name] for name in CLASSES)

# PandaSet annotates the full 360 degrees, so keep every point.
FIELD_OF_VIEW = torch.tensor([360.0, 360.0], dtype=torch.float32)

# OpenPCDet's train plus its val and test sequences without semseg.
TRAIN_SEQUENCES = frozenset({
    "002", "004", "005", "006", "008", "012", "014", "015", "016", "018", "019", "020", "023", "024", "028", "030",
    "033", "034", "035", "037", "038", "039", "040", "041", "044", "045", "046", "047", "048", "050", "051", "052",
    "055", "058", "059", "062", "063", "064", "066", "067", "068", "069", "071", "074", "077", "079", "084", "085",
    "086", "088", "089", "091", "092", "093", "095", "098", "099", "100", "102", "104", "105", "106", "113", "116",
    "117", "119", "120", "122", "123", "124", "139", "158",
})
# OpenPCDet's val and test sequences that have semseg.
VAL_SEQUENCES = frozenset({
    "001", "003", "011", "013", "017", "021", "027", "029", "032", "042", "043", "053", "054", "056", "057", "065",
    "070", "072", "073", "078", "080", "090", "094", "097", "101", "103", "109", "110", "112", "115", "149",
})
# How many sequences each split keeps in "mini" (debug) mode.
MINI_NUM_SEQUENCES = {"pre-train": 4, "train": 4, "val": 2}


class PandaSetDataset(Dataset, BaseLidarDataset):
    """Pandar64 sweeps, optionally with 3D boxes and per-point semantic and instance labels."""

    @classmethod
    def get_class_name_to_id(cls) -> dict[str, int]:
        return CLASS_NAME_TO_ID

    @classmethod
    def get_seg_class_name_to_id(cls) -> dict[str, int]:
        return SEG_CLASS_NAME_TO_ID

    @classmethod
    def get_seg_thing_ids(cls) -> tuple[int, ...]:
        return THING_IDS

    @classmethod
    def get_seg_ignore_id(cls) -> int | None:
        """No point is labelled 0, so it only marks padding."""
        return IGNORE_ID

    def __init__(
        self,
        split: Literal["pre-train", "train", "val"],
        version: Literal["full", "mini"] = "full",
        frame_sampling: Literal["random", "all"] = "random",
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
            split: "pre-train" and "train" are TRAIN_SEQUENCES, "val" is VAL_SEQUENCES (all with semseg).
            version: "mini" keeps MINI_NUM_SEQUENCES sequences.
            frame_sampling: "random" makes every sequence one item and draws a random frame of it per __getitem__,
                so an epoch is a pass over the sequences. "all" makes every frame one item.
            load_annotations: Whether to load the 3D boxes.
            load_segmentation: Also return per-point labels: "semantic" is the semseg class (SEG_CLASS_NAME_TO_ID),
                "instance" the 1-based ordinal of the box an object point lies in, 0 for stuff and box-less points.
                Only sequences with semseg are used then.
            min_num_sweeps_per_sample, max_num_sweeps_per_sample: Range the number of merged consecutive sweeps is
                drawn from. The reference frame they are merged into is random for "random" sampling, and the
                indexed frame, in the middle of the run, for "all".
        """
        if split not in MINI_NUM_SEQUENCES:
            raise ValueError(f"Unknown split {split!r}, expected one of {list(MINI_NUM_SEQUENCES)}.")
        if frame_sampling not in ("random", "all"):
            raise ValueError(f"Unknown frame_sampling {frame_sampling!r}")
        assert 1 <= min_num_sweeps_per_sample <= max_num_sweeps_per_sample
        if split == "pre-train" and (load_annotations or load_segmentation):
            raise ValueError("Pre-training is label free, use the train split for labels.")
        if load_segmentation and max_num_sweeps_per_sample > 1:
            raise ValueError("Labels are only loaded for the reference sweep, so segmentation needs max_num_sweeps_per_sample == 1.")

        self._random_reference = frame_sampling == "random"
        self._load_annotations = load_annotations
        self._load_frame_id = load_frame_id
        self._load_num_sweeps = load_num_sweeps
        self._random_crop = random_crop
        self._load_segmentation = load_segmentation
        self._min_nsweeps = min_num_sweeps_per_sample
        self._max_nsweeps = max_num_sweeps_per_sample

        names = _list_sequences(split, load_segmentation, MINI_NUM_SEQUENCES[split] if version == "mini" else None)
        sequences = [{"name": name, "poses": _load_poses(name)} for name in names]
        if frame_sampling == "random":
            entries = [(seq_idx, -1) for seq_idx in range(len(sequences))]
        else:
            entries = [(seq_idx, frame) for seq_idx, sequence in enumerate(sequences) for frame in range(len(sequence["poses"]))]
        self._entries = np.asarray(entries, dtype=np.int64).reshape(-1, 2)
        # Serialized storage so the index is not copy-on-write duplicated by the dataloader workers.
        self._sequences = TorchList(sequences)
        print(f"PandaSet: {len(sequences)} {split} sequences ({version}), {frame_sampling} sampling -> {len(self._entries)} samples per epoch")

    def __len__(self):
        """Get length."""
        return len(self._entries)

    def __getitem__(self, index):
        """Get item.

        Returns:
            - point_cloud: (MAX_NUM_POINTS, C), [x, y, z, intensity] in the reference frame (x forward, y left, z up),
              with the age in seconds from the newest point prepended when USE_TIME is enabled.
            - frame_id (optional): "<sequence>/<frame>".
            - num_sweeps (optional): number of merged sweeps.
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
                    # If repeated random crops fail, fall back to another frame.
                    if len(self._entries) <= 1:
                        raise
                    current_index = torch.randint(0, len(self._entries), size=(1,)).item()
                    no_points_tries = 0

    def _getitem(self, index):
        """Load one (possibly merged) sweep and, when requested, its labels."""
        seq_idx, keyframe = (int(v) for v in self._entries[index])
        sequence = self._sequences[seq_idx]
        num_frames = len(sequence["poses"])
        nsweeps = torch.randint(self._min_nsweeps, self._max_nsweeps + 1, size=(1,)).item()
        if self._random_reference:
            start, end, reference = sample_sweep_chunk(num_frames, nsweeps)
        else:
            start, end, reference = centered_sweep_chunk(num_frames, nsweeps, keyframe)
        ego_from_world = ego_from_world_matrix(sequence["poses"][reference])
        lidar_points_tensor, keep = load_points(sequence["name"], range(start, end), ego_from_world)

        if self._load_annotations or self._load_segmentation:
            boxes = frame_boxes(sequence["name"], reference, ego_from_world)
        if self._load_segmentation:
            # One sweep only, so ``keep`` is the reference sweep's mask.
            semantic = _read(sequence["name"], "annotations/semseg", reference)["class"].to_numpy()[keep]
            instance = instance_ids(lidar_points_tensor[:, X_INDEX : Z_INDEX + 1].numpy(), semantic, boxes)
            semantic = torch.from_numpy(semantic).long()
            instance = torch.from_numpy(instance).long()

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
            outs += (f"{sequence['name']}/{reference:02d}",)
        if self._load_num_sweeps:
            outs += (end - start,)

        if self._load_annotations:
            annotations = boxes_to_annotations(boxes)
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


def build_pandaset_dataset(
    train_flag: bool,
    version: str = "full",
    load_annotations: bool = False,
    load_frame_id: bool = False,
    random_crop: bool = False,
    load_segmentation: bool = False,
    min_num_sweeps_per_sample: int = 1,
    max_num_sweeps_per_sample: int = 1,
    load_num_sweeps: bool = False,
) -> PandaSetDataset:
    """Pick the split and sampling each training stage needs.

    * Pre-training (``train_flag`` without labels): the 72 train sequences, a random frame per sequence per item.
    * Probe training (``train_flag`` with labels): every frame of the train sequences (the 45 with semseg if requested).
    * Validation: every frame of the 31 val sequences.
    """
    common = dict(
        version=version,
        load_frame_id=load_frame_id,
        load_num_sweeps=load_num_sweeps,
        random_crop=random_crop,
        load_annotations=load_annotations,
        load_segmentation=load_segmentation,
        min_num_sweeps_per_sample=min_num_sweeps_per_sample,
        max_num_sweeps_per_sample=max_num_sweeps_per_sample,
    )
    if not train_flag:
        return PandaSetDataset("val", frame_sampling="all", **common)
    if load_annotations or load_segmentation:
        return PandaSetDataset("train", frame_sampling="all", **common)
    return PandaSetDataset("pre-train", frame_sampling="random", **common)


def _list_sequences(split: str, need_semseg: bool, num_sequences: int | None) -> list[str]:
    """Names of the split's sequences, optionally only those with semseg and thinned to a fixed count."""
    sequences = VAL_SEQUENCES if split == "val" else TRAIN_SEQUENCES
    names = sorted(
        name for name in sequences
        if not need_semseg or os.path.isdir(os.path.join(DATASET_ROOT, name, "annotations", "semseg"))
    )
    if num_sequences is not None and num_sequences < len(names):
        keep = np.linspace(0, len(names) - 1, num_sequences).round().astype(int)
        names = [names[i] for i in keep]
    return names


def _load_poses(name: str) -> np.ndarray:
    """(F, 4, 4) world-from-LiDAR pose of every frame."""
    with open(os.path.join(DATASET_ROOT, name, "lidar", "poses.json")) as f:
        poses = json.load(f)
    matrices = np.zeros((len(poses), 4, 4))
    for frame, pose in enumerate(poses):
        heading, position = pose["heading"], pose["position"]
        matrices[frame] = Quaternion(heading["w"], heading["x"], heading["y"], heading["z"]).transformation_matrix
        matrices[frame, :3, 3] = position["x"], position["y"], position["z"]
    return matrices


def _read(name: str, folder: str, frame: int) -> pd.DataFrame:
    return pd.read_pickle(os.path.join(DATASET_ROOT, name, folder, f"{frame:02d}.pkl.gz"))


def ego_from_world_matrix(world_from_lidar: np.ndarray) -> np.ndarray:
    """4x4 transform from world coordinates into our axes of the LiDAR frame."""
    matrix = np.linalg.inv(world_from_lidar)
    matrix[:3] = OURS_FROM_PANDASET @ matrix[:3]
    return matrix


def load_points(name: str, frames: range, ego_from_world: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
    """Stack the Pandar64 points of ``frames`` in the frame ``ego_from_world`` (4x4).

    Returns:
        ``(points, keep)``: (N, C) tensor and, of the last frame, the mask of its sweep's rows that were kept.
    """
    xyz, intensity, times = [], [], []
    for frame in frames:
        sweep = _read(name, "lidar", frame)
        world = sweep[["x", "y", "z"]].to_numpy()
        ego = world @ ego_from_world[:3, :3].T + ego_from_world[:3, 3]
        keep = (sweep["d"].to_numpy() == PANDAR64) & ~((np.abs(ego[:, 0]) < EGO_HALF_EXTENT[0]) & (np.abs(ego[:, 1]) < EGO_HALF_EXTENT[1]))
        xyz.append(ego[keep])
        intensity.append(normalize_intensity(sweep["i"].to_numpy()[keep].astype(np.float32), 255.0))
        times.append(sweep["t"].to_numpy()[keep])
    times = np.concatenate(times)
    columns = [np.concatenate(xyz), np.concatenate(intensity)]
    if USE_TIME:
        columns.insert(0, times.max() - times if times.size else times)
    return torch.from_numpy(np.column_stack(columns)).float(), keep


def frame_boxes(name: str, frame: int, ego_from_world: np.ndarray) -> dict[str, np.ndarray]:
    """The frame's Pandar64 boxes in the frame ``ego_from_world``.

    Returns "class" (M,), "center" (M, 3), "size" (M, 3) as [length, width, height] and "rotation" (M, 3, 3)
    box-to-ego, with the box's x axis along its length. PandaSet's ``dimensions.x`` is the width and ``yaw`` turns
    the box's y (length) axis, so the heading is ``yaw + pi / 2``.
    """
    cuboids = _read(name, "annotations/cuboids", frame)
    cuboids = cuboids[cuboids["cuboids.sensor_id"] != 1]
    heading = cuboids["yaw"].to_numpy() + np.pi / 2
    cos, sin = np.cos(heading), np.sin(heading)
    world_rotation = np.zeros((len(cuboids), 3, 3))
    world_rotation[:, 0, 0], world_rotation[:, 0, 1], world_rotation[:, 1, 0], world_rotation[:, 1, 1] = cos, -sin, sin, cos
    world_rotation[:, 2, 2] = 1.0
    centers = cuboids[["position.x", "position.y", "position.z"]].to_numpy()
    return {
        "class": cuboids["label"].map(CLASS_NAME_TO_ID).fillna(0).to_numpy().astype(np.int32),
        "center": centers @ ego_from_world[:3, :3].T + ego_from_world[:3, 3],
        "size": cuboids[["dimensions.y", "dimensions.x", "dimensions.z"]].to_numpy(),
        "rotation": ego_from_world[:3, :3] @ world_rotation,
    }


def boxes_to_annotations(boxes: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """The boxes of known classes as annotation tensors, with [w, x, y, z] quaternions."""
    keep = boxes["class"] > 0
    quats = np.array([Quaternion(matrix=rotation, atol=1e-6).elements for rotation in boxes["rotation"][keep]]).reshape(-1, 4)
    return {
        "class": torch.from_numpy(boxes["class"][keep]),
        "center": torch.from_numpy(boxes["center"][keep]).float(),
        "size": torch.from_numpy(boxes["size"][keep]).float(),
        "quats": torch.from_numpy(quats).float(),
    }


def instance_ids(xyz: np.ndarray, semantic: np.ndarray, boxes: dict[str, np.ndarray]) -> np.ndarray:
    """1-based ordinal of the box each object-class point lies in, 0 elsewhere; the last box wins overlaps."""
    instance = np.zeros(xyz.shape[0], dtype=np.int64)
    things = np.flatnonzero(np.isin(semantic, THING_IDS))
    for box in range(boxes["center"].shape[0]):
        local = (xyz[things] - boxes["center"][box]) @ boxes["rotation"][box]
        inside = np.all(np.abs(local) <= boxes["size"][box] / 2, axis=1)
        instance[things[inside]] = box + 1
    return instance
