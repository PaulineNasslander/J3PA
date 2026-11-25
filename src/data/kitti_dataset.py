"""KITTI: a Velodyne HDL-64E on a car, with 3D boxes for the 7481 object benchmark frames, in the front camera view.

Data: the raw recordings (``raw/<date>/<drive>_sync/`` with ``velodyne_points/`` and ``oxts/``, plus the per-date
``calib_imu_to_velo.txt``) and the 3D object benchmark (``object/training/`` with ``velodyne/``, ``label_2/``,
``calib/`` and ``image_2/`` for the image sizes) from https://www.cvlibs.net/datasets/kitti, under
``KITTI_DATASET_ROOT`` (default /datasets/kitti). KITTI has 3D boxes but no per-point labels; those come from the
boxes, precomputed by `scripts/precompute_kitti_labels.py` into ``KITTI_LABELS_DIR``.

Splits: the object benchmark only labels its training frames, so they are split into the standard 3DOP train/val
(Chen et al. 2015; 3712/3769 frames whose drives do not overlap). ``kitti_object_frames.txt`` holds that split and
each frame's raw drive and index, from the object devkit's mapping; the object scans are byte-identical to those raw
scans, so the raw drives provide the neighbouring sweeps. Pre-training uses every raw drive except the 45 that hold
val frames.

Points stay in the Velodyne frame of the reference sweep: x forward, y left, z up, the road about 1.73 m below.
Sweeps are merged with the drive's OXTS (GPS/IMU) poses; the scans are not motion compensated within a sweep and carry
no per-point time, so all points of a sweep share one age. Boxes are labelled only for objects visible in the left
colour camera, so labelled samples keep only the points inside its frustum.
"""

import os
from functools import lru_cache
from typing import Literal, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from constants import KITTI_LABELS_DIR, MIN_NUM_POINTS, PC_RANGE, RANDOM_CROP_CENTER_RANGES, USE_TIME, X_INDEX, Y_INDEX, Z_INDEX
from data.base_dataset import BaseLidarDataset, centered_sweep_chunk, normalize_intensity, sample_sweep_chunk
from data.zod_dataset import (
    NoPointsException,
    TorchList,
    _reshape_annotations_dict,
    _zero_pad_point_cloud,
    _zero_pad_point_cloud_with_labels,
)

DATASET_ROOT = os.environ.get("KITTI_DATASET_ROOT", "/datasets/kitti")
RANDOM_CROP_CENTER_RANGE = RANDOM_CROP_CENTER_RANGES["KITTI"]
OBJECT_FRAMES_FILE = os.path.join(os.path.dirname(__file__), "kitti_object_frames.txt")
MAX_TRY = 10
# Pre-training cuts the drives (22 to 5177 frames) into items of about this many frames.
SEGMENT_FRAMES = 100
EARTH_RADIUS = 6378137.0

# Every label type except DontCare, which marks unlabelled regions without a 3D box.
CLASSES = ("Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc")
CLASS_NAMES = list(CLASSES)
CLASS_NAME_TO_ID = {name: idx + 1 for idx, name in enumerate(CLASSES)}

# Labelled samples are already cut to the camera frustum, so the downstream FOV filter has nothing left to do.
FIELD_OF_VIEW = torch.tensor([360.0, 360.0], dtype=torch.float32)

# How many drives (pre-train) or labelled frames (train, val) each split keeps in "mini" (debug) mode.
MINI_SIZES = {"pre-train": 4, "train": 64, "val": 32}


class KITTIDataset(Dataset, BaseLidarDataset):
    """Velodyne sweeps, optionally with 3D box annotations and box-derived per-point labels."""

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
            split: "pre-train" is every segment of SEGMENT_FRAMES frames of the raw drives without val frames, a random
                frame of it per __getitem__ and the full 360 degree cloud. "train" and "val" are the labelled object
                frames of the 3DOP split, cut to the camera frustum.
            version: "mini" keeps MINI_SIZES drives or frames.
            load_annotations: Whether to load the 3D boxes (labelled splits only).
            load_segmentation: Also return per-point labels precomputed from the boxes by
                `scripts/precompute_kitti_labels.py`: points inside a box get its class and 1-based ordinal, all
                others are background (0).
            min_num_sweeps_per_sample, max_num_sweeps_per_sample: Range the number of merged consecutive sweeps is
                drawn from. The reference sweep they are merged into is random for "pre-train", and the labelled
                frame, in the middle of the run, otherwise.
        """
        if split not in MINI_SIZES:
            raise ValueError(f"Unknown split {split!r}, expected one of {list(MINI_SIZES)}.")
        assert 1 <= min_num_sweeps_per_sample <= max_num_sweeps_per_sample
        if split == "pre-train" and (load_annotations or load_segmentation):
            raise ValueError("Pre-training drives are unlabelled, use the train split for labels.")
        if load_segmentation and max_num_sweeps_per_sample > 1:
            raise ValueError("Labels are only computed for the reference sweep, so segmentation needs max_num_sweeps_per_sample == 1.")

        self._labelled = split != "pre-train"
        self._load_annotations = load_annotations
        self._load_frame_id = load_frame_id
        self._load_num_sweeps = load_num_sweeps
        self._random_crop = random_crop
        self._load_segmentation = load_segmentation
        self._min_nsweeps = min_num_sweeps_per_sample
        self._max_nsweeps = max_num_sweeps_per_sample

        frames = _object_frames()
        if self._labelled:
            frames = [frame for frame in frames if frame[1] == split]
            if version == "mini":
                frames = [frames[i] for i in np.linspace(0, len(frames) - 1, MINI_SIZES[split]).round().astype(int)]
            drive_names = sorted({drive for _, _, drive, _ in frames})
        else:
            val_drives = {drive for _, frame_split, drive, _ in frames if frame_split == "val"}
            drive_names = [drive for drive in _list_drives() if drive not in val_drives]
            if version == "mini":
                drive_names = [drive_names[i] for i in np.linspace(0, len(drive_names) - 1, MINI_SIZES[split]).round().astype(int)]
        drives = [_index_drive(name) for name in drive_names]

        if self._labelled:
            drive_of = {drive["name"]: idx for idx, drive in enumerate(drives)}
            entries = [
                (drive_of[drive], int(np.searchsorted(drives[drive_of[drive]]["index"], int(raw_index))), int(frame))
                for frame, _, drive, raw_index in frames
            ]
        else:
            entries = [
                (drive_idx, int(part[0]), int(part[-1]) + 1)
                for drive_idx, drive in enumerate(drives)
                for part in np.array_split(np.arange(len(drive["index"])), max(1, round(len(drive["index"]) / SEGMENT_FRAMES)))
            ]
        self._entries = np.asarray(entries, dtype=np.int64).reshape(-1, 3)
        # Serialized storage so the index is not copy-on-write duplicated by the dataloader workers.
        self._drives = TorchList(drives)
        print(f"KITTI: {len(drives)} {split} drives ({version}) -> {len(self._entries)} {'labelled frames' if self._labelled else 'segments'} per epoch")

    def __len__(self):
        """Get length."""
        return len(self._entries)

    def __getitem__(self, index):
        """Get item.

        Returns:
            - point_cloud: (MAX_NUM_POINTS, C), [x, y, z, intensity 0..255] in the reference sweep's Velodyne frame,
              with the age in seconds from the newest sweep prepended when USE_TIME is enabled.
            - frame_id (optional): "<drive>/<raw index>" for pre-training, "object/<frame>" for labelled frames.
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
                    # If repeated random crops fail, fall back to another item.
                    if len(self._entries) <= 1:
                        raise
                    current_index = torch.randint(0, len(self._entries), size=(1,)).item()
                    no_points_tries = 0

    def _getitem(self, index):
        """Load one (possibly merged) sweep and, when requested, its labels."""
        drive_idx, first, last = (int(v) for v in self._entries[index])
        drive = self._drives[drive_idx]
        nsweeps = torch.randint(self._min_nsweeps, self._max_nsweeps + 1, size=(1,)).item()
        if self._labelled:
            # ``first`` is the labelled frame's position in the drive, ``last`` its object frame number.
            start, end, reference = centered_sweep_chunk(len(drive["index"]), nsweeps, first)
            object_frame = last
        else:
            start, end, reference = sample_sweep_chunk(last - first, nsweeps)
            start, end, reference = start + first, end + first, reference + first
        lidar_points_tensor = load_sweeps(drive, range(start, end), reference)

        if self._labelled:
            calib = read_calib(object_frame)
            boxes = read_boxes(object_frame, calib)
        if self._load_segmentation:
            # One sweep only, so the cloud is the raw scan the labels were computed for, in the same order.
            path = label_path(object_frame)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Missing labels for object frame {object_frame} at {path}. Run scripts/precompute_kitti_labels.py first.")
            labels = np.load(path)
            in_view = torch.from_numpy(labels["in_cone"])
            semantic = torch.from_numpy(labels["semantic"]).long()[in_view]
            instance = torch.from_numpy(labels["instance"]).long()[in_view]
            lidar_points_tensor = lidar_points_tensor[in_view]
        elif self._labelled:
            lidar_points_tensor = lidar_points_tensor[in_camera_view(lidar_points_tensor[:, X_INDEX : Z_INDEX + 1].numpy(), calib, object_frame)]

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
        # The camera frustum holds only ~16-19k points of a sweep, so labelled frames get a lower bar.
        min_points = MIN_NUM_POINTS // 4 if self._labelled else MIN_NUM_POINTS
        if mask.sum() < min_points:
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
            outs += (f"object/{object_frame:06d}" if self._labelled else f"{drive['name']}/{drive['index'][reference]:010d}",)
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


def build_kitti_dataset(
    train_flag: bool,
    version: str = "full",
    load_annotations: bool = False,
    load_frame_id: bool = False,
    random_crop: bool = False,
    load_segmentation: bool = False,
    min_num_sweeps_per_sample: int = 1,
    max_num_sweeps_per_sample: int = 1,
    load_num_sweeps: bool = False,
) -> KITTIDataset:
    """Pick the split each training stage needs.

    * Pre-training (``train_flag`` without labels): the 106 raw drives without val frames, in ~10 s segments.
    * Probe training (``train_flag`` with labels): the 3712 3DOP train frames.
    * Validation: the 3769 3DOP val frames.
    """
    if not train_flag:
        split = "val"
    elif load_annotations or load_segmentation:
        split = "train"
    else:
        split = "pre-train"
    return KITTIDataset(
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


def label_path(object_frame: int, labels_dir: str | None = None) -> str:
    """Per-point labels of one object frame, from `scripts/precompute_kitti_labels.py`."""
    return os.path.join(labels_dir or KITTI_LABELS_DIR, f"{object_frame:06d}.npz")


def _object_frames() -> list[tuple[int, str, str, str]]:
    """``(object frame, split, raw drive, raw index)`` of every labelled object frame."""
    with open(OBJECT_FRAMES_FILE) as f:
        rows = [line.split() for line in f if not line.startswith("#")]
    return [(int(frame), split, drive, index) for frame, split, drive, index in rows]


def _drive_dir(name: str) -> str:
    return os.path.join(DATASET_ROOT, "raw", name[:10], name)


def _list_drives() -> list[str]:
    """Raw ``<date>_drive_<id>_sync`` drives that have Velodyne scans (5 of the release have none)."""
    raw = os.path.join(DATASET_ROOT, "raw")
    return sorted(
        name
        for date in os.listdir(raw) if os.path.isdir(os.path.join(raw, date))
        for name in os.listdir(os.path.join(raw, date))
        if name.endswith("_sync") and os.path.isdir(os.path.join(raw, date, name, "velodyne_points", "data"))
    )


def _index_drive(name: str) -> dict:
    """The drive's scans: "name", "index" (F,) file numbers in time order and "time" (F,) seconds of day.

    A few scans are missing (e.g. 4 of 2011_09_26_drive_0009), so positions and file numbers can differ.
    """
    directory = os.path.join(_drive_dir(name), "velodyne_points")
    index = np.array(sorted(int(file[:-4]) for file in os.listdir(os.path.join(directory, "data"))), dtype=np.int32)
    with open(os.path.join(directory, "timestamps.txt")) as f:
        stamps = f.read().splitlines()
    return {"name": name, "index": index, "time": np.array([_seconds(stamps[i]) for i in index])}


def _seconds(stamp: str) -> float:
    hours, minutes, seconds = stamp.split()[1].split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


@lru_cache(maxsize=8)
def _velo_from_imu(date: str) -> np.ndarray:
    """4x4 transform from the OXTS IMU frame into the Velodyne frame of a recording day."""
    with open(os.path.join(DATASET_ROOT, "raw", date, "calib_imu_to_velo.txt")) as f:
        values = {key: np.array([float(v) for v in rest.split()]) for key, rest in (line.split(":", 1) for line in f) if key in ("R", "T")}
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = values["R"].reshape(3, 3), values["T"]
    return matrix


def _world_from_velo(name: str, raw_index: int, mercator_scale: float) -> np.ndarray:
    """4x4 pose of the Velodyne from the OXTS packet of a scan (Mercator x/y, altitude z), as in the KITTI devkit."""
    with open(os.path.join(_drive_dir(name), "oxts", "data", f"{raw_index:010d}.txt")) as f:
        lat, lon, alt, roll, pitch, yaw = (float(v) for v in f.read().split()[:6])
    tx = mercator_scale * np.radians(lon) * EARTH_RADIUS
    ty = mercator_scale * EARTH_RADIUS * np.log(np.tan(np.radians(90 + lat) / 2))
    cx, sx, cy, sy, cz, sz = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    world_from_imu = np.eye(4)
    world_from_imu[:3, :3] = (
        np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    )
    world_from_imu[:3, 3] = tx, ty, alt
    return world_from_imu @ np.linalg.inv(_velo_from_imu(name[:10]))


def _read_scan(name: str, raw_index: int) -> np.ndarray:
    """(N, 4) [x, y, z, reflectance 0..1] of one raw scan."""
    return np.fromfile(os.path.join(_drive_dir(name), "velodyne_points", "data", f"{raw_index:010d}.bin"), dtype=np.float32).reshape(-1, 4)


def load_sweeps(drive: dict, positions: Sequence[int], reference: int) -> torch.Tensor:
    """Stack the scans at ``positions`` in the Velodyne frame of ``reference``, as an (N, C) tensor."""
    name, index = drive["name"], drive["index"]
    if len(positions) > 1:
        with open(os.path.join(_drive_dir(name), "oxts", "data", f"{index[reference]:010d}.txt")) as f:
            mercator_scale = np.cos(np.radians(float(f.read().split()[0])))
        velo_from_world = np.linalg.inv(_world_from_velo(name, int(index[reference]), mercator_scale))
    clouds, ages = [], []
    newest = drive["time"][positions[-1]]
    for position in positions:
        scan = _read_scan(name, int(index[position])).astype(np.float64)
        if position != reference:
            transform = velo_from_world @ _world_from_velo(name, int(index[position]), mercator_scale)
            scan[:, :3] = scan[:, :3] @ transform[:3, :3].T + transform[:3, 3]
        # Reflectance is stored as 0..1.
        scan[:, 3] = normalize_intensity(scan[:, 3], 1.0)
        clouds.append(scan)
        ages.append(np.full(scan.shape[0], newest - drive["time"][position]))
    columns = np.concatenate(clouds)
    if USE_TIME:
        columns = np.column_stack((np.concatenate(ages), columns))
    return torch.from_numpy(columns).float()


def read_calib(object_frame: int) -> dict[str, np.ndarray]:
    """The frame's "P2" (3, 4) projection and "rect_from_velo" (4, 4) into the rectified camera frame."""
    with open(os.path.join(DATASET_ROOT, "object", "training", "calib", f"{object_frame:06d}.txt")) as f:
        values = {key: np.array([float(v) for v in rest.split()]) for key, rest in (line.split(":", 1) for line in f if ":" in line)}
    rectify, cam_from_velo = np.eye(4), np.eye(4)
    rectify[:3, :3] = values["R0_rect"].reshape(3, 3)
    cam_from_velo[:3] = values["Tr_velo_to_cam"].reshape(3, 4)
    return {"P2": values["P2"].reshape(3, 4), "rect_from_velo": rectify @ cam_from_velo}


def _image_size(object_frame: int) -> tuple[int, int]:
    """(width, height) of the frame's left colour image, from its PNG header."""
    with open(os.path.join(DATASET_ROOT, "object", "training", "image_2", f"{object_frame:06d}.png"), "rb") as f:
        header = f.read(24)
    return int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big")


def in_camera_view(xyz: np.ndarray, calib: dict[str, np.ndarray], object_frame: int) -> np.ndarray:
    """Mask of the points that project into the left colour image, the only region with labels."""
    width, height = _image_size(object_frame)
    image = np.c_[xyz, np.ones(xyz.shape[0])] @ (calib["P2"] @ calib["rect_from_velo"]).T
    depth = image[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u, v = image[:, 0] / depth, image[:, 1] / depth
    return (depth > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)


def read_boxes(object_frame: int, calib: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The frame's boxes in the Velodyne frame.

    Returns "class" (M,), "center" (M, 3), "size" (M, 3) as [length, width, height] and "yaw" (M,) about z.
    Labels give the bottom centre in the rectified camera frame (y down) and ``rotation_y`` about its y axis, which
    turns the box's length axis from the camera x axis.
    """
    with open(os.path.join(DATASET_ROOT, "object", "training", "label_2", f"{object_frame:06d}.txt")) as f:
        rows = [line.split() for line in f if line.split() and line.split()[0] in CLASS_NAME_TO_ID]
    values = np.array([[float(v) for v in row[8:15]] for row in rows]).reshape(-1, 7)
    height, width, length, x, y, z, rotation_y = values.T
    velo_from_rect = np.linalg.inv(calib["rect_from_velo"])
    centers = np.c_[x, y - height / 2, z] @ velo_from_rect[:3, :3].T + velo_from_rect[:3, 3]
    headings = np.c_[np.cos(rotation_y), np.zeros_like(rotation_y), -np.sin(rotation_y)] @ velo_from_rect[:3, :3].T
    return {
        "class": np.array([CLASS_NAME_TO_ID[row[0]] for row in rows], dtype=np.int32),
        "center": centers,
        "size": np.c_[length, width, height],
        "yaw": np.arctan2(headings[:, 1], headings[:, 0]),
    }


def boxes_to_annotations(boxes: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """Annotation tensors with yaw-only [w, x, y, z] quaternions."""
    half = boxes["yaw"] / 2
    quats = np.c_[np.cos(half), np.zeros_like(half), np.zeros_like(half), np.sin(half)]
    return {
        "class": torch.from_numpy(boxes["class"]),
        "center": torch.from_numpy(boxes["center"]).float(),
        "size": torch.from_numpy(boxes["size"]).float(),
        "quats": torch.from_numpy(quats).float(),
    }
