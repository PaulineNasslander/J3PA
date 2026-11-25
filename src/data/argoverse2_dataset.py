"""Argoverse 2 dataset for LiDAR sweeps with 3D object annotations."""

import os
from typing import Literal, Sequence

import numpy as np
import pandas as pd
import torch
from pyquaternion import Quaternion
from torch.utils.data import Dataset

from constants import ARGOVERSE2_LABELS_DIR, MIN_NUM_POINTS, PC_RANGE, RANDOM_CROP_CENTER_RANGES, USE_TIME, X_INDEX, Y_INDEX, Z_INDEX, XYZ_SLICE, POINT_FEATURE_DIM
from data.base_dataset import BaseLidarDataset, centered_sweep_chunk, sample_sweep_chunk
from data.zod_dataset import (
    NoPointsException,
    TorchList,
    _reshape_annotations_dict,
    _zero_pad_point_cloud,
    _zero_pad_point_cloud_with_labels,
)

DATASET_ROOT = os.environ.get("ARGOVERSE2_DATASET_ROOT", "/datasets/argoverse2")
RANDOM_CROP_CENTER_RANGE = RANDOM_CROP_CENTER_RANGES["Argoverse2"]
MAX_TRY = 10

# The full Argoverse 2 taxonomy, verified against every train and val log.
CLASSES = (
    "ANIMAL",
    "ARTICULATED_BUS",
    "BICYCLE",
    "BICYCLIST",
    "BOLLARD",
    "BOX_TRUCK",
    "BUS",
    "CONSTRUCTION_BARREL",
    "CONSTRUCTION_CONE",
    "DOG",
    "LARGE_VEHICLE",
    "MESSAGE_BOARD_TRAILER",
    "MOBILE_PEDESTRIAN_CROSSING_SIGN",
    "MOTORCYCLE",
    "MOTORCYCLIST",
    "OFFICIAL_SIGNALER",
    "PEDESTRIAN",
    "RAILED_VEHICLE",
    "REGULAR_VEHICLE",
    "SCHOOL_BUS",
    "SIGN",
    "STOP_SIGN",
    "STROLLER",
    "TRAFFIC_LIGHT_TRAILER",
    "TRUCK",
    "TRUCK_CAB",
    "VEHICULAR_TRAILER",
    "WHEELCHAIR",
    "WHEELED_DEVICE",
    "WHEELED_RIDER",
)
CLASS_NAMES = [cls for cls in CLASSES]
CLASS_NAME_TO_ID = {name: idx + 1 for idx, name in enumerate(CLASSES)}

# Split groups. Pre-training uses every log that exists apart from the annotated
# val/test logs, so nothing it sees leaks into evaluation.
PRETRAIN_SPLITS = ("train", "lidar/train", "lidar/val", "lidar/test")  # 20700 logs
PROBE_TRAIN_SPLITS = ("train",)  # the 700 annotated logs
VAL_SPLITS = ("val",)
TEST_SPLITS = ("test",)  # wired up but unused: no annotations are published

# Argoverse2 annotates the full 360 degrees, so keep every point (the same convention nuScenes uses here).
FIELD_OF_VIEW = torch.tensor([360.0, 360.0], dtype=torch.float32)

# How many logs each stage keeps in "mini" (debug) mode.
MINI_NUM_SEQUENCES = {"pretrain": 64, "probe": 8, "val": 2}


class Argoverse2Dataset(Dataset, BaseLidarDataset):
    """LiDAR sweeps, optionally with 3D cuboid annotations."""

    @classmethod
    def get_class_name_to_id(cls) -> dict[str, int]:
        return CLASS_NAME_TO_ID

    def __init__(
        self,
        split: Literal["pre-train", "train", "val", "test"],
        frame_sampling: Literal["random", "all"] = "random",
        load_annotations: bool = False,
        load_frame_id: bool = False,
        random_crop: bool = False,
        num_sequences: int | None = None,
        load_segmentation: bool = False,
        load_num_sweeps: bool = False,
        min_num_sweeps_per_sample: int = 1,
        max_num_sweeps_per_sample: int = 1,
    ):
        """Initialize.
        
        Args:
            split: Which set of logs to use. Pre-training uses all logs except the annotated val/test logs, so it sees no evaluation data.
            frame_sampling: Whether to return every sweep of every log, or one random sweep per log. 
                The former is used for probe training and validation, the latter for pre-training.
            load_annotations: Whether to load 3D OD annotations.
            load_segmentation: Also return per-point semantic and instance labels.
                As Argoverse2 only contains 3D object annotations these are provided by annotating points 
                inside a bounding box by the class of the object. Other points are all labeled with 
                the "background class"
            min_num_sweeps_per_sample, max_num_sweeps_per_sample: Range the number of aggregated
                sweeps is drawn from per sample. If max > 1 we sample a number of consecutive sweeps 
                and motion compensate them into the ego frame of a reference sweep among them.
                With "random" frame sampling (pre-training) the run and reference are random, with
                "all" the indexed sweep is the reference in the middle of the run.
            
            The panoptic and semantic labels are precomputed and saved directly in the ARGOVERSE2_LABELS_DIR
            as frame_id.npz, each containing {"in_cone", "semantic", "instance"} keys. 
            "in_cone" is not relevant for Argoverse2 since all points are annotated.
        """
        self._frame_sampling = frame_sampling
        self._load_annotations = load_annotations
        self._load_frame_id = load_frame_id
        self._load_num_sweeps = load_num_sweeps
        self._random_crop = random_crop
        self._load_segmentation = load_segmentation
        assert 1 <= min_num_sweeps_per_sample <= max_num_sweeps_per_sample
        self._min_nsweeps = min_num_sweeps_per_sample
        self._max_nsweeps = max_num_sweeps_per_sample
        if load_segmentation and max_num_sweeps_per_sample > 1:
            raise ValueError("Labels are only loaded for the reference sweep and 0 is background, not ignore, so segmentation needs max_num_sweeps_per_sample == 1.")

        if split == "pre-train":
            dataroots = PRETRAIN_SPLITS
        elif split == "train":
            dataroots = PROBE_TRAIN_SPLITS
        elif split == "val":
            dataroots = VAL_SPLITS
        elif split == "test":
            dataroots = TEST_SPLITS
        else:
            raise ValueError(f"Unknown split {split!r}, expected one of pre-train, train, val, test.")

        sequences = _list_sequences(dataroots, num_sequences)
        if load_annotations or load_segmentation:
            assert split != "pre-train", "pre-train logs are unannotated"
            missing = [seq for seq in sequences if not os.path.isfile(_annotation_path(seq))]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} of {len(sequences)} logs have no annotations.feather "
                    f"(first: {missing[0]}). The lidar/* and test splits are unannotated."
                )

        if frame_sampling == "random":
            entries = sequences
        else:
            entries = _enumerate_frames(sequences)

        # Serialized storage so the index is not copy-on-write duplicated by the
        # dataloader workers.
        self._entries = TorchList(entries)
        print(
            f"Argoverse2: {len(sequences)} logs from {list(dataroots)}, "
            f"{frame_sampling} sampling -> {len(self._entries)} samples per epoch"
        )

    def __len__(self):
        """Get length."""
        return len(self._entries)

    def __getitem__(self, index):
        """Get item.

        Returns:
            - point_cloud: (MAX_NUM_POINTS, C), [x, y, z, intensity] with time prepended when USE_TIME is enabled.
            - frame_id (optional): "<split>/<log_id>/<timestamp_ns>".
            - num_sweeps (optional): number of aggregated sweeps.
            - annotations (optional): dict of padded tensors with keys
              "class" (MAX_NUM_OBJECTS,), "center" (MAX_NUM_OBJECTS, 3),
              "size" (MAX_NUM_OBJECTS, 3) as [length, width, height] and
              "quats" (MAX_NUM_OBJECTS, 4) as [w, x, y, z].
            - field_of_view (optional): tensor of shape (2) in degrees.
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
            except Exception:
                raise

    def _getitem(self, index):
        """Load one sweep and, when requested, the cuboids recorded for it."""
        sequence, sweep_file = self._locate(index)
        if self._max_nsweeps > 1:
            nsweeps = torch.randint(self._min_nsweeps, self._max_nsweeps + 1, size=(1,)).item()
            keyframe = None if self._frame_sampling == "random" else sweep_file
            sweep_files, reference = _sweep_chunk(sequence, nsweeps, keyframe)
            sweep_file = sweep_files[reference]
        else:
            sweep_files, reference = [sweep_file], 0
        lidar_points_tensor = _load_sweeps(sequence, sweep_files, reference)

        if self._load_segmentation:
            frame_id = f"{sequence}/{sweep_file.removesuffix('.feather')}"
            labels_path = os.path.join(ARGOVERSE2_LABELS_DIR, frame_id + ".npz")
            if not os.path.isfile(labels_path):
                raise FileNotFoundError(f"Missing segmentation labels for {frame_id} at {labels_path}")
            labels = np.load(labels_path)
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
        timestamp_ns = int(sweep_file.removesuffix(".feather"))
        if self._load_frame_id:
            outs += (f"{sequence}/{timestamp_ns}",)
        if self._load_num_sweeps:
            outs += (len(sweep_files),)

        if self._load_annotations:
            annotations = _load_annotations(_annotation_path(sequence), timestamp_ns)
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

    def _locate(self, index) -> tuple[str, str]:
        """Resolve an index to its ``(sequence, sweep_file)``."""
        entry = self._entries[index]
        if self._frame_sampling != "random":
            return entry

        # Listing on demand keeps start-up cheap for the 20 700 pre-training logs
        # and costs one directory read per sample, far less than the sweep itself.
        sweep_files = os.listdir(os.path.join(DATASET_ROOT, entry, "sensors", "lidar"))
        pick = torch.randint(0, len(sweep_files), size=(1,)).item()
        return entry, sweep_files[pick]


def _list_sequences(splits: Sequence[str], num_sequences: int | None) -> list[str]:
    """Collect ``"<split>/<log_id>"`` paths, optionally thinned to a fixed count."""
    sequences = []
    for split in splits:
        split_dir = os.path.join(DATASET_ROOT, split)
        sequences.extend(f"{split}/{log_id}" for log_id in sorted(os.listdir(split_dir)))

    if num_sequences is not None and num_sequences < len(sequences):
        # Stride rather than slice so every split stays represented.
        keep = np.linspace(0, len(sequences) - 1, num_sequences).round().astype(int)
        sequences = [sequences[i] for i in keep]
    return sequences


def _enumerate_frames(sequences: Sequence[str]) -> list[tuple[str, str]]:
    """Build the fixed ``(sequence, sweep_file)`` list covering every sweep."""
    frames = []
    for sequence in sequences:
        sweep_files = sorted(os.listdir(os.path.join(DATASET_ROOT, sequence, "sensors", "lidar")))
        frames.extend((sequence, sweep_file) for sweep_file in sweep_files)
    return frames


def _annotation_path(sequence: str) -> str:
    """Path of a log's annotation table."""
    return os.path.join(DATASET_ROOT, sequence, "annotations.feather")


def _sweep_chunk(sequence: str, nsweeps: int, keyframe: str | None = None) -> tuple[list[str], int]:
    """Returns nsweeps consecutive sweeps files and the index of the reference among them.

    The selected sweeps and reference are random, or if keyframe is provided, the reference is in the middle.
    """
    # Equal-length nanosecond names, so lexical order is chronological.
    sweep_files = sorted(os.listdir(os.path.join(DATASET_ROOT, sequence, "sensors", "lidar")))
    if keyframe is None:
        start, end, reference = sample_sweep_chunk(len(sweep_files), nsweeps)
    else:
        start, end, reference = centered_sweep_chunk(len(sweep_files), nsweeps, sweep_files.index(keyframe))
    return sweep_files[start:end], reference - start


def _load_poses(sequence: str, timestamps_ns: Sequence[int]) -> list[np.ndarray]:
    """4x4 city-from-ego poses at the given sweep timestamps."""
    poses = pd.read_feather(
        os.path.join(DATASET_ROOT, sequence, "city_SE3_egovehicle.feather"), use_threads=False
    ).set_index("timestamp_ns")
    matrices = []
    for row in poses.loc[list(timestamps_ns)].itertuples():
        matrix = Quaternion(row.qw, row.qx, row.qy, row.qz).transformation_matrix
        matrix[:3, 3] = (row.tx_m, row.ty_m, row.tz_m)
        matrices.append(matrix)
    return matrices


def _load_sweeps(sequence: str, sweep_files: Sequence[str], reference: int = 0) -> torch.Tensor:
    """Stack sweeps in the ego frame of ``sweep_files[reference]``, as an (N, 5) tensor."""
    lidar_dir = os.path.join(DATASET_ROOT, sequence, "sensors", "lidar")
    if len(sweep_files) == 1:
        return _load_sweep(os.path.join(lidar_dir, sweep_files[0]))

    timestamps_ns = [int(sweep_file.removesuffix(".feather")) for sweep_file in sweep_files]
    poses = _load_poses(sequence, timestamps_ns) # only loads the poses for the sweeps in the range of interest
    reference_from_world = np.linalg.inv(poses[reference])

    clouds = []
    for sweep_file, world_from_sweep in zip(sweep_files, poses):
        cloud = _load_sweep(os.path.join(lidar_dir, sweep_file), reference_ns=timestamps_ns[reference])
        transform = torch.from_numpy(reference_from_world @ world_from_sweep).float()
        cloud[:, XYZ_SLICE] = cloud[:, XYZ_SLICE] @ transform[:3, :3].T + transform[:3, 3]
        clouds.append(cloud)
    points = torch.cat(clouds)
    if USE_TIME and len(points) > 0:
        points[:, 0] -= points[:, 0].min()  # newest return is age 0, as for a single sweep
    return points


def _load_sweep(path: str, reference_ns: int | None = None) -> torch.Tensor:
    """Read one sweep as an (N, 5) [timestamp, x, y, z, intensity] tensor.

    ``reference_ns`` is the time ages are measured from; by default the sweep's newest return.
    """
    sweep = pd.read_feather(path, columns=["x", "y", "z", "intensity", "offset_ns"], use_threads=False)
    if sweep.empty:
        # Some AV2 sweeps hold no returns; the caller's too-few-points check redraws the sample.
        return torch.zeros((0, POINT_FEATURE_DIM))

    # ``offset_ns`` is the nanosecond offset of each return within the ~100 ms
    # revolution. Flip it so the newest return is 0 and older ones grow positive,
    # matching the age-in-seconds convention of the ZOD and nuScenes datasets.
    offsets = sweep["offset_ns"].to_numpy().astype(np.int64)
    if reference_ns is None:
        timestamps = (offsets.max() - offsets) / 1e9
    else:
        sweep_ns = int(os.path.basename(path).removesuffix(".feather"))
        timestamps = (reference_ns - sweep_ns - offsets) / 1e9

    columns = [sweep[name].to_numpy() for name in ("x", "y", "z", "intensity")]
    if USE_TIME:
        columns.insert(0, timestamps)
    return torch.from_numpy(np.stack(columns, axis=1)).float()


def _load_annotations(path: str, timestamp_ns: int) -> dict[str, torch.Tensor]:
    """Read the cuboids recorded for one sweep.

    The table holds every sweep of the log, so it is filtered down to the one
    timestamp. Sizes and quaternions already use the ``[length, width, height]``
    and ``[w, x, y, z]`` orders the rest of the codebase expects, and the cuboids
    are stated in the ego-vehicle frame the sweep also uses.
    """
    cuboids = pd.read_feather(path, use_threads=False)
    return _cuboids_to_annotations(cuboids[cuboids["timestamp_ns"] == timestamp_ns])


def _cuboids_to_annotations(cuboids: pd.DataFrame) -> dict[str, torch.Tensor]:
    """Convert the cuboid rows of a single sweep into annotation tensors."""
    class_ids = cuboids["category"].map(CLASS_NAME_TO_ID).fillna(0).to_numpy()
    keep = class_ids > 0  # Returns 0 if class is not found
    cuboids = cuboids[keep]

    return {
        "class": torch.from_numpy(class_ids[keep].astype(np.int32)),
        "center": torch.from_numpy(cuboids[["tx_m", "ty_m", "tz_m"]].to_numpy()).float(),
        "size": torch.from_numpy(cuboids[["length_m", "width_m", "height_m"]].to_numpy()).float(),
        "quats": torch.from_numpy(cuboids[["qw", "qx", "qy", "qz"]].to_numpy()).float(),
    }


def build_argoverse2_dataset(
    train_flag: bool,
    version: str = "full",
    load_annotations: bool = False,
    load_frame_id: bool = False,
    random_crop: bool = False,
    load_segmentation: bool = False,
    load_num_sweeps: bool = False,
    min_num_sweeps_per_sample: int = 1,
    max_num_sweeps_per_sample: int = 1,
) -> Argoverse2Dataset:
    """Pick the split and sampling each training stage needs.

    * Pre-training (``train_flag`` without annotations): the 20 700 logs of
      ``train`` plus all of ``lidar/*``, one randomly drawn sweep per log, so an
      epoch is a pass over the logs.
    * Probe training (``train_flag`` with annotations or segmentation): only the 700 annotated
      ``train`` logs, every sweep of each, iterated in full.
    * Validation: every sweep of the 150 annotated ``val`` logs, iterated in full.

    The ``test`` logs are reachable through ``Argoverse2Dataset("test", ...)``
    but are not part of any stage here, since they ship without annotations.
    """
    mini = version == "mini"
    common = dict(
        load_frame_id=load_frame_id,
        random_crop=random_crop,
        load_segmentation=load_segmentation,
        load_num_sweeps=load_num_sweeps,
        min_num_sweeps_per_sample=min_num_sweeps_per_sample,
        max_num_sweeps_per_sample=max_num_sweeps_per_sample,
    )

    if not train_flag:
        return Argoverse2Dataset(
            "val",
            frame_sampling="all",
            load_annotations=load_annotations,
            num_sequences=MINI_NUM_SEQUENCES["val"] if mini else None,
            **common,
        )

    if load_annotations or load_segmentation:
        return Argoverse2Dataset(
            "train",
            frame_sampling="all",
            load_annotations=load_annotations,
            num_sequences=MINI_NUM_SEQUENCES["probe"] if mini else None,
            **common,
        )

    return Argoverse2Dataset(
        "pre-train",
        frame_sampling="random",
        load_annotations=False,
        num_sequences=MINI_NUM_SEQUENCES["pretrain"] if mini else None,
        **common,
    )
