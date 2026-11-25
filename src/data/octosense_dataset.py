"""OctoSense, car platform: one roof-mounted Ouster OS1-64 at 10 Hz, without any 3D labels.

Data: https://huggingface.co/datasets/anthonytec2/OctoSense. Only the ``data.h5`` of every car sequence is needed
(its download script's ``--platform car --modalities data``), as ``<session>/<bag_id>/data.h5`` under
``OCTOSENSE_DATASET_ROOT`` (default /datasets/octosense/car). The LiDAR arrays are blosc2 compressed, so reading
them needs ``hdf5plugin``.

The release has no 3D boxes and no per-point classes (its only semantics are 2D camera pseudo-labels), so this is a
pre-training dataset only: annotations and segmentation cannot be requested.

A sweep is ``ouster/range_pcl`` (64 x 2048 destaggered, mm) with ``ouster/refl_pcl`` (calibrated reflectivity 0..255)
as intensity, in the sensor frame: x forward, y left, z up, about 1.7 m above the road. RKO-LIO already deskewed
every sweep to its scan time ``ouster/t``, so all points of a sweep share one age. Sweeps are merged with the
LiDAR-odometry poses ``ouster/odom/map_T_lidart``.
"""

import os
import time
from typing import Literal, Sequence

import h5py
import hdf5plugin  # noqa: F401  Registers the blosc2 filter the LiDAR arrays need.
import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import Dataset

from constants import MIN_NUM_POINTS, PC_RANGE, RANDOM_CROP_CENTER_RANGES, USE_TIME, X_INDEX, Y_INDEX, Z_INDEX
from data.base_dataset import BaseLidarDataset, centered_sweep_chunk, normalize_intensity, sample_sweep_chunk
from data.zod_dataset import NoPointsException, TorchList, _zero_pad_point_cloud

DATASET_ROOT = os.environ.get("OCTOSENSE_DATASET_ROOT", "/datasets/octosense/car")
RANDOM_CROP_CENTER_RANGE = RANDOM_CROP_CENTER_RANGES["OctoSense"]
MAX_TRY = 10
# Every sequence (~10 min) is cut into items of this many frames, so a pass over the items covers each drive evenly.
SEGMENT_FRAMES = 200
# Frames further apart than this (s) start a new segment, so merged sweeps never span a recording gap.
MAX_FRAME_GAP = 0.25
# No-return pixels are deskewed to where the sensor was when they fired, at most speed * 0.1 s from it.
MAX_TRAIL_RANGE = 5.0
# Returns closer than this (m) to such a no-return pixel are dropped with them.
TRAIL_MARGIN = 0.3

# Official test split (metadata.parquet, split == "test", platform == "car"). It holds the 4 degraded sequences.
TEST_BAGS = frozenset({
    "rosbag2_2026_01_04-14_01_35", "rosbag2_2026_01_04-14_46_43", "rosbag2_2026_01_04-20_33_11",
    "rosbag2_2026_01_08-07_23_08", "rosbag2_2026_01_08-18_06_21", "rosbag2_2026_01_08-20_26_41",
    "rosbag2_2026_01_08-20_47_06", "rosbag2_2026_01_08-20_57_15", "rosbag2_2026_01_09-07_47_28",
    "rosbag2_2026_01_09-08_08_34", "rosbag2_2026_01_09-11_19_30", "rosbag2_2026_01_09-14_13_08",
    "rosbag2_2026_01_09-14_44_04", "rosbag2_2026_01_14-08_22_52", "rosbag2_2026_01_14-10_38_26",
    "rosbag2_2026_01_14-13_36_37", "rosbag2_2026_01_14-14_07_25", "rosbag2_2026_01_14-16_58_11",
    "rosbag2_2026_01_14-19_06_39", "rosbag2_2026_01_14-19_16_57", "rosbag2_2026_01_15-11_08_38",
    "rosbag2_2026_01_15-11_19_04", "rosbag2_2026_01_15-19_48_58", "rosbag2_2026_01_16-08_03_26",
    "rosbag2_2026_01_16-14_10_47", "rosbag2_2026_01_16-14_44_01", "rosbag2_2026_01_16-15_04_42",
    "rosbag2_2026_01_16-15_18_53", "rosbag2_2026_01_16-15_30_55", "rosbag2_2026_01_17-08_01_00",
    "rosbag2_2026_01_17-09_19_46", "rosbag2_2026_01_17-09_30_06", "rosbag2_2026_01_20-07_23_49",
    "rosbag2_2026_01_20-09_02_38", "rosbag2_2026_01_20-13_11_03", "rosbag2_2026_01_20-15_09_47",
    "rosbag2_2026_01_20-16_01_34", "rosbag2_2026_01_21-11_35_05", "rosbag2_2026_01_21-14_18_58",
    "rosbag2_2026_01_21-15_23_55", "rosbag2_2026_01_21-16_25_27", "rosbag2_2026_01_21-17_58_34",
    "rosbag2_2026_01_21-19_26_50", "rosbag2_2026_01_22-09_19_22", "rosbag2_2026_01_22-12_56_21",
    "rosbag2_2026_01_22-17_02_18", "rosbag2_2026_01_22-18_42_40", "rosbag2_2026_01_22-20_08_47",
    "rosbag2_2026_01_23-09_29_56", "rosbag2_2026_01_23-11_49_13", "rosbag2_2026_01_23-12_09_32",
    "rosbag2_2026_01_23-12_30_30", "rosbag2_2026_01_23-13_11_09", "rosbag2_2026_01_23-14_09_19",
    "rosbag2_2026_01_23-18_44_20", "rosbag2_2026_01_23-19_04_41", "rosbag2_2026_01_23-19_35_53",
    "rosbag2_2026_01_23-20_29_02", "rosbag2_2026_01_24-07_49_10", "rosbag2_2026_01_24-08_20_12",
    "rosbag2_2026_01_24-08_51_05", "rosbag2_2026_01_24-09_26_50", "rosbag2_2026_01_24-09_52_55",
    "rosbag2_2026_01_24-12_33_07", "rosbag2_2026_01_24-12_43_17", "rosbag2_2026_01_24-15_02_20",
    "rosbag2_2026_01_24-16_34_02", "rosbag2_2026_01_24-17_15_06", "rosbag2_2026_01_24-18_58_31",
    "rosbag2_2026_01_24-20_31_58", "rosbag2_2026_01_28-09_05_42", "rosbag2_2026_01_28-09_16_09",
    "rosbag2_2026_01_28-10_37_59", "rosbag2_2026_01_28-10_59_22", "rosbag2_2026_02_01-09_33_21",
    "rosbag2_2026_02_01-09_43_40", "rosbag2_2026_02_02-09_45_50", "rosbag2_2026_02_02-10_54_22",
})
# How many sequences each split keeps in "mini" (debug) mode.
MINI_NUM_SEQUENCES = {"pre-train": 4, "test": 2}


class OctoSenseDataset(Dataset, BaseLidarDataset):
    """Single or merged Ouster sweeps, without labels."""

    @classmethod
    def get_class_name_to_id(cls) -> dict[str, int]:
        return {}

    def __init__(
        self,
        split: Literal["pre-train", "test"],
        version: Literal["full", "mini"] = "full",
        load_frame_id: bool = False,
        random_crop: bool = False,
        min_num_sweeps_per_sample: int = 1,
        max_num_sweeps_per_sample: int = 1,
        load_num_sweeps: bool = False,
    ):
        """Initialize.

        Args:
            split: "pre-train" is the official train split, "test" the official test split.
            version: "mini" keeps MINI_NUM_SEQUENCES sequences.
            min_num_sweeps_per_sample, max_num_sweeps_per_sample: Range the number of merged consecutive sweeps is
                drawn from. Every item is a segment of SEGMENT_FRAMES frames; for "pre-train" the run of sweeps and
                the reference sweep they are merged into are random within it, for "test" the reference is the
                segment's middle frame, in the middle of the run.
        """
        if split not in MINI_NUM_SEQUENCES:
            raise ValueError(f"Unknown split {split!r}, expected one of {list(MINI_NUM_SEQUENCES)}.")
        assert 1 <= min_num_sweeps_per_sample <= max_num_sweeps_per_sample
        self._random_reference = split == "pre-train"
        self._load_frame_id = load_frame_id
        self._load_num_sweeps = load_num_sweeps
        self._random_crop = random_crop
        self._min_nsweeps = min_num_sweeps_per_sample
        self._max_nsweeps = max_num_sweeps_per_sample

        start = time.time()
        paths = _list_sequences(split, MINI_NUM_SEQUENCES[split] if version == "mini" else None)
        sequences = [_index_sequence(path) for path in paths]
        entries = [
            (seq_idx, segment_start, segment_end)
            for seq_idx, sequence in enumerate(sequences)
            for segment_start, segment_end in _segments(sequence["time"])
        ]
        self._entries = np.asarray(entries, dtype=np.int64).reshape(-1, 3)
        # Serialized storage so the index is not copy-on-write duplicated by the dataloader workers.
        self._sequences = TorchList(sequences)
        print(
            f"OctoSense: {len(sequences)} {split} sequences ({version}), indexed in {time.time() - start:.0f}s "
            f"-> {len(self._entries)} segments of up to {SEGMENT_FRAMES} frames per epoch"
        )

    def __len__(self):
        """Get length."""
        return len(self._entries)

    def __getitem__(self, index):
        """Get item.

        Returns:
            - point_cloud: (MAX_NUM_POINTS, C), [x, y, z, reflectivity] in the reference sweep's sensor frame, with the
              age in seconds from the newest sweep prepended when USE_TIME is enabled.
            - frame_id (optional): "<bag_id>/<reference sweep index in the sequence>".
            - num_sweeps (optional): number of merged sweeps.
        """
        current_index = index
        no_points_tries = 0

        while True:
            try:
                return self._getitem(current_index)
            except NoPointsException:
                no_points_tries += 1

                if no_points_tries > MAX_TRY:
                    # If repeated random crops fail, fall back to another segment.
                    if len(self._entries) <= 1:
                        raise
                    current_index = torch.randint(0, len(self._entries), size=(1,)).item()
                    no_points_tries = 0

    def _getitem(self, index):
        """Load one (possibly merged) sweep."""
        seq_idx, segment_start, segment_end = (int(v) for v in self._entries[index])
        sequence = self._sequences[seq_idx]
        nsweeps = torch.randint(self._min_nsweeps, self._max_nsweeps + 1, size=(1,)).item()
        segment_length = segment_end - segment_start
        if self._random_reference:
            start, end, reference = sample_sweep_chunk(segment_length, nsweeps)
        else:
            start, end, reference = centered_sweep_chunk(segment_length, nsweeps, segment_length // 2)
        positions = range(segment_start + start, segment_start + end)
        reference += segment_start
        lidar_points_tensor = load_sweeps(sequence, positions, reference)

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
        lidar_points_tensor = _zero_pad_point_cloud(lidar_points_tensor[mask])

        outs = (lidar_points_tensor,)
        if self._load_frame_id:
            outs += (f"{sequence['name']}/{sequence['lidar_index'][reference]}",)
        if self._load_num_sweeps:
            outs += (len(positions),)
        return outs[0] if len(outs) == 1 else outs


def build_octosense_dataset(
    train_flag: bool,
    version: str = "full",
    load_annotations: bool = False,
    load_frame_id: bool = False,
    random_crop: bool = False,
    load_segmentation: bool = False,
    min_num_sweeps_per_sample: int = 1,
    max_num_sweeps_per_sample: int = 1,
    load_num_sweeps: bool = False,
) -> OctoSenseDataset:
    """Pre-training (``train_flag``) uses the 293 train sequences, otherwise the 78 test ones."""
    if load_annotations or load_segmentation:
        raise ValueError("OctoSense has no 3D annotations or per-point labels, it can only be used for pre-training.")
    return OctoSenseDataset(
        "pre-train" if train_flag else "test",
        version=version,
        load_frame_id=load_frame_id,
        random_crop=random_crop,
        min_num_sweeps_per_sample=min_num_sweeps_per_sample,
        max_num_sweeps_per_sample=max_num_sweeps_per_sample,
        load_num_sweeps=load_num_sweeps,
    )


def _list_sequences(split: str, num_sequences: int | None) -> list[str]:
    """``<session>/<bag_id>`` of the split's sequences, optionally thinned to a fixed count."""
    sequences = sorted(
        f"{session}/{bag_id}"
        for session in os.listdir(DATASET_ROOT)
        for bag_id in os.listdir(os.path.join(DATASET_ROOT, session))
        if (bag_id in TEST_BAGS) == (split == "test")
    )
    if num_sequences is not None and num_sequences < len(sequences):
        keep = np.linspace(0, len(sequences) - 1, num_sequences).round().astype(int)
        sequences = [sequences[i] for i in keep]
    return sequences


def _index_sequence(path: str) -> dict:
    """The sequence's sweeps that have an odometry pose:

    name         bag id
    path         ``<session>/<bag_id>``
    lidar_index  (F,) row of each sweep in the ``ouster/*`` arrays
    odom_index   (F,) row of its pose in the ``ouster/odom/*`` arrays
    time         (F,) scan time in seconds
    """
    with h5py.File(os.path.join(DATASET_ROOT, path, "data.h5"), "r") as f:
        lidar_time = f["ouster/t"][:]
        odom_time = f["ouster/odom/t"][:]
    # Both share the scan times, but a few sequences have a sweep more or less in one of them.
    odom_index = np.searchsorted(odom_time, lidar_time).clip(0, len(odom_time) - 1)
    has_pose = np.abs(odom_time[odom_index] - lidar_time) < 1e-3
    return {
        "name": os.path.basename(path),
        "path": path,
        "lidar_index": np.flatnonzero(has_pose).astype(np.int32),
        "odom_index": odom_index[has_pose].astype(np.int32),
        "time": lidar_time[has_pose],
    }


def _segments(times: np.ndarray) -> list[tuple[int, int]]:
    """Split a sequence into ``(start, end)`` runs of about SEGMENT_FRAMES sweeps that span no recording gap."""
    breaks = np.flatnonzero(np.diff(times) > MAX_FRAME_GAP) + 1
    segments = []
    for run in np.split(np.arange(times.shape[0]), breaks):
        for part in np.array_split(run, max(1, round(run.shape[0] / SEGMENT_FRAMES))):
            if part.shape[0]:
                segments.append((int(part[0]), int(part[-1]) + 1))
    return segments


def load_sweeps(sequence: dict, positions: Sequence[int], reference: int) -> torch.Tensor:
    """Stack the sweeps at ``positions`` in the sensor frame of ``reference``, as an (N, C) tensor."""
    with h5py.File(os.path.join(DATASET_ROOT, sequence["path"], "data.h5"), "r") as f:
        if len(positions) > 1:
            first, last = sequence["odom_index"][positions[0]], sequence["odom_index"][positions[-1]]
            poses = f["ouster/odom/map_T_lidart"][first : last + 1]
            reference_from_map = np.linalg.inv(poses[sequence["odom_index"][reference] - first])
        clouds, ages = [], []
        newest = sequence["time"][positions[-1]]
        for position in positions:
            xyz, reflectivity = read_sweep(f, int(sequence["lidar_index"][position]))
            if position != reference:
                transform = reference_from_map @ poses[sequence["odom_index"][position] - first]
                xyz = xyz @ transform[:3, :3].T + transform[:3, 3]
            clouds.append(np.column_stack((xyz, normalize_intensity(reflectivity, 255.0))))
            ages.append(np.full(xyz.shape[0], newest - sequence["time"][position]))
    columns = np.concatenate(clouds)
    if USE_TIME:
        columns = np.column_stack((np.concatenate(ages), columns))
    return torch.from_numpy(columns).float()


def read_sweep(f: h5py.File, index: int) -> tuple[np.ndarray, np.ndarray]:
    """The returns of sweep ``index``: (N, 3) xyz in metres and (N,) reflectivity."""
    xyz = f["ouster/range_pcl"][index].astype(np.float64) / 1000.0
    reflectivity = f["ouster/refl_pcl"][index]
    keep = _returns(xyz, reflectivity)
    return xyz[keep], reflectivity[keep].astype(np.float32)


def _returns(xyz: np.ndarray, reflectivity: np.ndarray) -> np.ndarray:
    """Mask of the pixels that are real returns.

    No-return pixels are stored as [0, 0, 0] but the deskew moved them along with their column, onto the trail
    the sensor drove during the scan. They carry reflectivity 0, which real returns practically never do, apart
    from a few hundred per sweep, which are recognised by lying on that trail.
    """
    no_return = reflectivity == 0
    keep = ~no_return
    distance = np.linalg.norm(xyz, axis=1)
    trail = xyz[no_return & (distance < MAX_TRAIL_RANGE)]
    if trail.shape[0] == 0:
        return keep
    near = keep & (distance < np.linalg.norm(trail, axis=1).max() + TRAIL_MARGIN)
    if near.any():
        to_trail, _ = cKDTree(trail).query(xyz[near], distance_upper_bound=TRAIL_MARGIN)
        keep[near] = np.isinf(to_trail)
    return keep
