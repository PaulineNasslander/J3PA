"""MAN TruckScenes: six LiDARs on a truck, 3D boxes at the 2 Hz key frames (no devkit needed).

Data: download v1.2 from https://github.com/TUMFTM/truckscenes-devkit (see its README) and set
`TRUCKSCENES_DATASET_ROOT` (default /datasets/truckscenes). Needed from it are the metadata folders
`v1.2-trainval/` and `v1.2-mini/` (for --version mini), and the LiDAR folders `samples/LIDAR_*/` and
`sweeps/LIDAR_*/`; cameras, radars and v1.2-test are unused. Per-point segmentation labels come from
`scripts/precompute_truckscenes_labels.py`.

A sample is the merged cloud of all six LiDARs, every point motion compensated into the ego frame (rear axle on
the ground) at the frame time; points within 1 m of their LiDAR are dropped, as in the devkit. Frames group the
phase-synchronised LiDARs' files and are timed like the release's annotations (see `_frames`).

The JSON tables are read once per release and process, in a child process (`_extract_scenes`), since that
peaks at several GB; the per-scene arrays it returns are documented there.
"""

import json
import multiprocessing
import os
import struct
from collections import defaultdict
from typing import Literal, Sequence

import numpy as np
import torch
from pyquaternion import Quaternion
from scipy.ndimage import median_filter
from torch.utils.data import Dataset

from constants import (
    MIN_NUM_POINTS,
    PC_RANGE,
    RANDOM_CROP_CENTER_RANGES,
    TRUCKSCENES_LABELS_DIR,
    USE_TIME,
    X_INDEX,
    Y_INDEX,
    Z_INDEX,
)
RANDOM_CROP_CENTER_RANGE = RANDOM_CROP_CENTER_RANGES["TruckScenes"]
from data.base_dataset import BaseLidarDataset, centered_sweep_chunk, normalize_intensity, sample_sweep_chunk
from data.zod_dataset import (
    NoPointsException,
    TorchList,
    _reshape_annotations_dict,
    _zero_pad_point_cloud,
    _zero_pad_point_cloud_with_labels,
)

try:
    import numba
except ImportError:  # Optional: only speeds up .pcd decompression.
    numba = None

DATASET_ROOT = os.environ.get("TRUCKSCENES_DATASET_ROOT", "/datasets/truckscenes")
MAX_TRY = 10
# Points within this many metres (in x and in y) of their own LiDAR are dropped, as the devkit does.
MIN_DISTANCE = 1.0
# A 10 ms ego pose step this much faster or slower (m/s) than its neighbours is a localisation jump.
POSE_JUMP_SPEED = 5.0
LIDAR_PERIOD_US = 100_000

# Every sample merges all of them; the first is only the clock that groups files into frames.
LIDAR_CHANNELS = (
    "LIDAR_LEFT",
    "LIDAR_RIGHT",
    "LIDAR_TOP_FRONT",
    "LIDAR_TOP_LEFT",
    "LIDAR_TOP_RIGHT",
    "LIDAR_REAR",
)

# The official detection classes and category mapping (devkit eval/detection/utils.py).
CATEGORY_TO_CLASS = {
    "animal": "animal",
    "movable_object.barrier": "barrier",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.construction": "other_vehicle",
    "vehicle.other": "other_vehicle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "movable_object.trafficcone": "traffic_cone",
    "static_object.traffic_sign": "traffic_sign",
    "vehicle.ego_trailer": "trailer",
    "vehicle.trailer": "trailer",
    "vehicle.truck": "truck",
}
CLASSES = (
    "animal",
    "barrier",
    "bicycle",
    "bus",
    "car",
    "motorcycle",
    "other_vehicle",
    "pedestrian",
    "traffic_cone",
    "traffic_sign",
    "trailer",
    "truck",
)
CLASS_NAMES = list(CLASSES)
CLASS_NAME_TO_ID = {name: idx + 1 for idx, name in enumerate(CLASSES)}

# TruckScenes annotates the full 360 degrees, so keep every point.
FIELD_OF_VIEW = torch.tensor([360.0, 360.0], dtype=torch.float32)

# Official splits (devkit utils/splits.py): train is every v1.2-trainval scene not in val.
VAL_SCENES = frozenset({
    "scene-0044384af3d8494e913fb8b14915239e-3",
    "scene-0afa5cb3c6d34a998a7da1c6206511c7-3",
    "scene-0afa5cb3c6d34a998a7da1c6206511c7-5",
    "scene-13f4b71b1bd04a9e88747ad8f58a3f67-13",
    "scene-13f4b71b1bd04a9e88747ad8f58a3f67-3",
    "scene-13f4b71b1bd04a9e88747ad8f58a3f67-4",
    "scene-13f4b71b1bd04a9e88747ad8f58a3f67-9",
    "scene-19d55ce5a6a44f109a6909f484d23f76-1",
    "scene-23160c4bff134cdd86b96ee822b4aca9-18",
    "scene-23160c4bff134cdd86b96ee822b4aca9-2",
    "scene-2a8ba56871df4cbd9ae75f13c3171af2-6",
    "scene-30b0fa9fa9ec4284961734504bd5ca1e-10",
    "scene-30b0fa9fa9ec4284961734504bd5ca1e-11",
    "scene-30b0fa9fa9ec4284961734504bd5ca1e-12",
    "scene-3195e3bfd1de4f9ba82f874790c7856c-13",
    "scene-3f542f89ec5241b6a4e30ca743adcf34-33",
    "scene-3f542f89ec5241b6a4e30ca743adcf34-35",
    "scene-3f542f89ec5241b6a4e30ca743adcf34-3",
    "scene-3f542f89ec5241b6a4e30ca743adcf34-5",
    "scene-41640f6466f34238b0c03f673abe9417-13",
    "scene-41640f6466f34238b0c03f673abe9417-14",
    "scene-41640f6466f34238b0c03f673abe9417-16",
    "scene-47d21ae92c444c698d3db575db1dc8b7-8",
    "scene-4e55ee3cceca4d1f898971a91bd4d484-2",
    "scene-538ab15b0996453584970e99a7d82d34-5",
    "scene-6be74229febb4aad9e7ae1a37217b231-3",
    "scene-6ef44746fd28480cb19b5067ae254496-4",
    "scene-7174d2619b1c40f0a150811fe83c98eb-17",
    "scene-7174d2619b1c40f0a150811fe83c98eb-19",
    "scene-7174d2619b1c40f0a150811fe83c98eb-2",
    "scene-7174d2619b1c40f0a150811fe83c98eb-30",
    "scene-7174d2619b1c40f0a150811fe83c98eb-4",
    "scene-7174d2619b1c40f0a150811fe83c98eb-8",
    "scene-7412f997e097401c93900b0770eeec4a-3",
    "scene-757adc42b1aa4436a62036e10e3999ce-3",
    "scene-7d6fe6a041cd4ee3bd400a747ce9d94d-2",
    "scene-8e4cb91735074f69aaad8824fc4a5ff8-17",
    "scene-9b7f8b017e8c48d9816c4afa150c60a2-1",
    "scene-9b7f8b017e8c48d9816c4afa150c60a2-6",
    "scene-9b7f8b017e8c48d9816c4afa150c60a2-8",
    "scene-9db2aaafda8c40efbb195b214311f4f2-2",
    "scene-a038db099ca34ec4b8a9422758969bb8-7",
    "scene-a038db099ca34ec4b8a9422758969bb8-9",
    "scene-a07c321013b94cf0a0c281f59d0455bc-3",
    "scene-a2fd0ce217b5420982d94bccc8e0614c-15",
    "scene-a2fd0ce217b5420982d94bccc8e0614c-1",
    "scene-a2fd0ce217b5420982d94bccc8e0614c-9",
    "scene-a47fe0701d2b40628c47322c9f991986-10",
    "scene-a47fe0701d2b40628c47322c9f991986-13",
    "scene-a47fe0701d2b40628c47322c9f991986-7",
    "scene-a47fe0701d2b40628c47322c9f991986-8",
    "scene-a6a87db5125846bda72ccfc9931ee153-2",
    "scene-a6a87db5125846bda72ccfc9931ee153-9",
    "scene-a85ecba64f744f68aa06ba82d721ad40-9",
    "scene-b201f6e737e34c2d87830fc1e2911516-14",
    "scene-b201f6e737e34c2d87830fc1e2911516-8",
    "scene-b662852bb66c43b49c0dfacfdeebd2a5-9",
    "scene-b8e7da1bd8d6430792a51b52fea484a5-2",
    "scene-bbae3a9a7249479785a7598bafc84217-7",
    "scene-c1b21af2e68e4ab28d8afd75d8b8f34c-13",
    "scene-c95d907aed7f472e9fd73f9aac273878-26",
    "scene-c95d907aed7f472e9fd73f9aac273878-28",
    "scene-c95d907aed7f472e9fd73f9aac273878-5",
    "scene-c95d907aed7f472e9fd73f9aac273878-6",
    "scene-ca5494b037de4f59b1cfa66e7c5d8ed3-2",
    "scene-cc3bdae2b0a244db821e609e2f31ac6b-3",
    "scene-d24a0af639f642b9a721437b6ac8a893-10",
    "scene-d24a0af639f642b9a721437b6ac8a893-2",
    "scene-e1dbd50ff4694a46b4997f4dec87f667-27",
    "scene-e1dbd50ff4694a46b4997f4dec87f667-29",
    "scene-f123519a7ffd4e85bb61763335edbb6b-9",
    "scene-f7f816a9689240f7aa2f20f77e67ecea-3",
    "scene-fb64d203d417452f830ca73efed80c41-5",
    "scene-fc798fa53f614e9d991c4d1c893eb6d5-2",
    "scene-fdac5b1b465e4affaae74ac616150676-1",
})
MINI_TRAIN_SCENES = frozenset({
    "scene-3f542f89ec5241b6a4e30ca743adcf34-29",
    "scene-b201f6e737e34c2d87830fc1e2911516-6",
    "scene-6be74229febb4aad9e7ae1a37217b231-4",
    "scene-a6a87db5125846bda72ccfc9931ee153-11",
    "scene-b13542304b244af585a646082c57a079-1",
    "scene-23160c4bff134cdd86b96ee822b4aca9-14",
    "scene-fb64d203d417452f830ca73efed80c41-11",
    "scene-3195e3bfd1de4f9ba82f874790c7856c-15",
})
MINI_VAL_SCENES = frozenset({
    "scene-13f4b71b1bd04a9e88747ad8f58a3f67-4",
    "scene-0044384af3d8494e913fb8b14915239e-3",
})

# (release, scene filter) pairs that make up each split. Pre-training never sees val.
SPLITS = {
    "full": {
        "pre-train": (("v1.2-trainval", lambda name: name not in VAL_SCENES),),
        "train": (("v1.2-trainval", lambda name: name not in VAL_SCENES),),
        "val": (("v1.2-trainval", lambda name: name in VAL_SCENES),),
    },
    "mini": {
        "pre-train": (("v1.2-mini", lambda name: name in MINI_TRAIN_SCENES),),
        "train": (("v1.2-mini", lambda name: name in MINI_TRAIN_SCENES),),
        "val": (("v1.2-mini", lambda name: name in MINI_VAL_SCENES),),
    },
}


class TruckScenesDataset(Dataset, BaseLidarDataset):
    """Multi-LiDAR frames, optionally with 3D box annotations and per-point labels."""

    @classmethod
    def get_class_name_to_id(cls) -> dict[str, int]:
        return CLASS_NAME_TO_ID

    def __init__(
        self,
        split: Literal["pre-train", "train", "val"],
        version: Literal["full", "mini"] = "full",
        frame_sampling: Literal["random", "keyframes"] = "random",
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
            split: Which scenes to use. "pre-train" and "train" are the train scenes, "val" is for
                evaluation only.
            version: "mini" uses the 10 scenes of v1.2-mini (8 train / 2 val).
            frame_sampling: "random" makes every scene one item and draws a random frame of it per
                __getitem__, so an epoch is a pass over the scenes. "keyframes" makes every annotated
                2 Hz key frame one item.
            load_annotations: Whether to load 3D OD annotations (needs key frames).
            load_segmentation: Also return per-point semantic and instance labels, precomputed from
                the boxes by `scripts/precompute_truckscenes_labels.py`. 0 is background.
            min_num_sweeps_per_sample, max_num_sweeps_per_sample: Range the number of aggregated frames
                is drawn from. If max > 1, consecutive frames are merged into the ego frame at the time
                of a reference frame among them: random for "random" sampling, the key frame in the
                middle for "keyframes".
        """
        if frame_sampling not in ("random", "keyframes"):
            raise ValueError(f"Unknown frame_sampling {frame_sampling!r}")
        assert 1 <= min_num_sweeps_per_sample <= max_num_sweeps_per_sample
        if (load_annotations or load_segmentation) and frame_sampling != "keyframes":
            raise ValueError("Annotations only exist for key frames, use frame_sampling='keyframes'.")
        if load_segmentation and max_num_sweeps_per_sample > 1:
            raise ValueError("Labels are only loaded for the reference frame and 0 is background, not ignore, so segmentation needs max_num_sweeps_per_sample == 1.")

        self._frame_sampling = frame_sampling
        self._load_annotations = load_annotations
        self._load_frame_id = load_frame_id
        self._load_num_sweeps = load_num_sweeps
        self._random_crop = random_crop
        self._load_segmentation = load_segmentation
        self._min_nsweeps = min_num_sweeps_per_sample
        self._max_nsweeps = max_num_sweeps_per_sample

        scenes = load_scenes(split, version)
        if frame_sampling == "random":
            entries = [(scene_idx, -1) for scene_idx in range(len(scenes))]
        else:
            entries = [
                (scene_idx, position)
                for scene_idx, scene in enumerate(scenes)
                for position in np.flatnonzero(scene["frames"]["sample"] >= 0)
            ]
        self._entries = np.asarray(entries, dtype=np.int64).reshape(-1, 2)
        # Serialized storage so the scenes are not copy-on-write duplicated by the dataloader workers.
        self._scenes = TorchList(scenes)
        print(
            f"TruckScenes: {len(scenes)} {split} scenes ({version}), "
            f"{frame_sampling} sampling -> {len(self._entries)} samples per epoch"
        )

    def __len__(self):
        """Get length."""
        return len(self._entries)

    def __getitem__(self, index):
        """Get item.

        Returns:
            - point_cloud: tensor of shape (MAX_NUM_POINTS, 5), [age, x, y, z, intensity] in the ego frame
              at the reference frame's time. Age in seconds from the newest point.
            - frame_id (optional): "<scene name>/<frame time us>".
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
        """Load one (possibly aggregated) frame and, when requested, its labels."""
        scene_idx, keyframe = (int(v) for v in self._entries[index])
        scene = self._scenes[scene_idx]
        num_frames = scene["frames"]["time"].shape[0]
        nsweeps = torch.randint(self._min_nsweeps, self._max_nsweeps + 1, size=(1,)).item()
        if self._frame_sampling == "random":
            start, end, reference = sample_sweep_chunk(num_frames, nsweeps)
        else:
            start, end, reference = centered_sweep_chunk(num_frames, nsweeps, keyframe)
        positions = list(range(start, end))
        reference = positions[reference - start]
        frame = frame_pose(scene, reference)
        frame_time, ego_from_world = frame[0], np.linalg.inv(frame[1])

        lidar_points_tensor, files = load_points(scene, positions, frame)

        if self._load_segmentation:
            # One frame only, so `files` are exactly the reference frame's, in point order.
            semantic, instance = [], []
            for _, filename, keep in files:
                labels = np.load(label_path(filename))
                semantic.append(labels["semantic"][keep])
                instance.append(labels["instance"][keep])
            semantic = torch.from_numpy(np.concatenate(semantic)).long()
            instance = torch.from_numpy(np.concatenate(instance)).long()

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
            outs += (f"{scene['name']}/{frame_time}",)
        if self._load_num_sweeps:
            outs += (len(positions),)

        if self._load_annotations:
            sample = int(scene["frames"]["sample"][reference])
            annotations = scene_annotations(scene, sample, ego_from_world)
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


def build_truckscenes_dataset(
    train_flag: bool,
    version: str = "full",
    load_annotations: bool = False,
    load_frame_id: bool = False,
    random_crop: bool = False,
    load_segmentation: bool = False,
    min_num_sweeps_per_sample: int = 1,
    max_num_sweeps_per_sample: int = 1,
    load_num_sweeps: bool = False,
) -> TruckScenesDataset:
    """Pick the split and sampling each training stage needs.

    * Pre-training (``train_flag`` without labels): the 523 train scenes, one random frame per scene per
      item, so an epoch is a pass over the scenes.
    * Probe training (``train_flag`` with labels): every key frame of the 523 train scenes.
    * Validation: every key frame of the 75 val scenes.
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
        return TruckScenesDataset("val", frame_sampling="keyframes", **common)
    if load_annotations or load_segmentation:
        return TruckScenesDataset("train", frame_sampling="keyframes", **common)
    return TruckScenesDataset("pre-train", frame_sampling="random", **common)


def label_path(filename: str, labels_dir: str = TRUCKSCENES_LABELS_DIR) -> str:
    """Per-point labels of one key-frame LiDAR file, given its .pcd path relative to the dataset root."""
    return os.path.join(labels_dir, filename.removesuffix(".pcd") + ".npz")


# Scenes already extracted in this process, per release, so train and val share one read of the tables.
_SCENES: dict[str, list[dict]] = {}


def load_scenes(split: str, version: str = "full") -> list[dict]:
    """The scenes of a split, sorted by name."""
    if split not in SPLITS[version]:
        raise ValueError(f"Unknown split {split!r} for version {version!r}, expected one of {list(SPLITS[version])}.")
    scenes = []
    for version_dir, keep in SPLITS[version][split]:
        if version_dir not in _SCENES:
            # In a child process, so the several GB the JSON parse peaks at go back to the OS before workers fork.
            with multiprocessing.get_context("spawn").Pool(1) as pool:
                _SCENES[version_dir] = pool.apply(_extract_scenes, (version_dir,))
        scenes.extend(scene for scene in _SCENES[version_dir] if keep(scene["name"]))
    return sorted(scenes, key=lambda scene: scene["name"])


def _extract_scenes(version_dir: str) -> list[dict]:
    """Read a release's JSON tables into one dict per scene:

    name, sample_tokens   scene name and its key-frame samples in time order
    lidar                 {channel: {"filename" (.pcd path from the root), "timestamp" (n,) us, "is_key_frame" (n,),
                          "sample" (n,) index into sample_tokens, "pose" (n, 7)}}, every file in time order
    ego_from_sensor       {channel: (7,)}
    frames                the six LiDARs' files grouped into frames, with frame times (see `_frames`)
    ego_timestamp, ego_pose  (p,), (p, 7): every (~100 Hz) ego pose of the scene
    ann_*                 boxes of sample s are rows ann_offsets[s]:ann_offsets[s + 1] of ann_class (0 = not
                          a detection class), ann_center (global), ann_size [w, l, h], ann_rotation (global),
                          ann_num_lidar_pts

    Poses are world-from-ego [tx, ty, tz, qw, qx, qy, qz].
    """
    def table(name):
        with open(os.path.join(DATASET_ROOT, version_dir, f"{name}.json")) as f:
            return json.load(f)

    channel_of_sensor = {sensor["token"]: sensor["channel"] for sensor in table("sensor")}
    ego_from_sensor, lidar_of_calibration = {}, {}
    for calibration in table("calibrated_sensor"):
        channel = channel_of_sensor[calibration["sensor_token"]]
        if channel in LIDAR_CHANNELS:
            ego_from_sensor[channel] = np.array(calibration["translation"] + calibration["rotation"])
            lidar_of_calibration[calibration["token"]] = channel

    samples = {sample["token"]: sample for sample in table("sample")}
    # Several trucks recorded at the same time, so ego poses are attributed to scenes through the
    # records (of every sensor, ~100 Hz together) that reference them, never by time.
    scene_of_pose = {sample["ego_pose_token"]: sample["scene_token"] for sample in samples.values()}
    lidar_records = defaultdict(list)
    for record in table("sample_data"):
        scene_token = samples[record["sample_token"]]["scene_token"]
        scene_of_pose[record["ego_pose_token"]] = scene_token
        channel = lidar_of_calibration.get(record["calibrated_sensor_token"])
        if channel is not None:
            lidar_records[(scene_token, channel)].append(
                (record["timestamp"], record["filename"], record["is_key_frame"], record["sample_token"], record["ego_pose_token"])
            )
    poses, scene_poses = {}, defaultdict(list)
    for pose in table("ego_pose"):
        poses[pose["token"]] = pose["translation"] + pose["rotation"]
        if pose["token"] in scene_of_pose:
            scene_poses[scene_of_pose[pose["token"]]].append((pose["timestamp"], *poses[pose["token"]]))
    del scene_of_pose

    category_of = {category["token"]: category["name"] for category in table("category")}
    class_of_instance = {
        instance["token"]: CLASS_NAME_TO_ID.get(CATEGORY_TO_CLASS.get(category_of[instance["category_token"]]), 0)
        for instance in table("instance")
    }
    annotations_of = defaultdict(list)
    for annotation in table("sample_annotation"):
        annotations_of[annotation["sample_token"]].append(annotation)

    scenes = []
    for scene in table("scene"):
        sample_tokens, token = [], scene["first_sample_token"]
        while token:
            sample_tokens.append(token)
            token = samples[token]["next"]
        sample_position = {token: i for i, token in enumerate(sample_tokens)}

        lidar = {}
        for channel in LIDAR_CHANNELS:
            records = sorted(lidar_records[(scene["token"], channel)])
            lidar[channel] = {
                "filename": [record[1] for record in records],
                "timestamp": np.array([record[0] for record in records], dtype=np.int64),
                "is_key_frame": np.array([record[2] for record in records], dtype=bool),
                "sample": np.array([sample_position[record[3]] for record in records], dtype=np.int32),
                "pose": np.array([poses[record[4]] for record in records], dtype=np.float64).reshape(-1, 7),
            }
        ego = np.array(sorted(scene_poses[scene["token"]]), dtype=np.float64).reshape(-1, 8)
        sample_times = np.array([samples[token]["timestamp"] for token in sample_tokens], dtype=np.int64)

        offsets, annotations = [0], []
        for token in sample_tokens:
            annotations.extend(annotations_of[token])
            offsets.append(len(annotations))
        scenes.append({
            "name": scene["name"],
            "sample_tokens": sample_tokens,
            "lidar": lidar,
            "frames": _frames(lidar, sample_times),
            "ego_from_sensor": ego_from_sensor,
            "ego_timestamp": ego[:, 0].astype(np.int64),
            "ego_pose": ego[:, 1:],
            "ego_translation_smooth": _remove_pose_jumps(ego[:, 0], ego[:, 1:4]),
            "ann_offsets": np.array(offsets, dtype=np.int64),
            "ann_class": np.array([class_of_instance[ann["instance_token"]] for ann in annotations], dtype=np.int32),
            "ann_center": np.array([ann["translation"] for ann in annotations], dtype=np.float64).reshape(-1, 3),
            "ann_size": np.array([ann["size"] for ann in annotations], dtype=np.float32).reshape(-1, 3),
            "ann_rotation": np.array([ann["rotation"] for ann in annotations], dtype=np.float64).reshape(-1, 4),
            "ann_num_lidar_pts": np.array([ann["num_lidar_pts"] for ann in annotations], dtype=np.int32),
        })
    return scenes


def _remove_pose_jumps(times_us: np.ndarray, translations: np.ndarray) -> np.ndarray:
    """The trajectory with localisation jumps taken out, for relative motion only.

    ~20% of the scenes' poses jump by up to a few metres within one 10 ms step and keep the offset afterwards.
    Steps whose velocity is more than POSE_JUMP_SPEED off the local median (normal noise is <= 1.5 m/s at p99)
    are replaced by that median velocity before integrating again.
    """
    if translations.shape[0] < 2:
        return translations.copy()
    dt = np.maximum(np.diff(times_us), 1)[:, None] / 1e6
    velocity = np.diff(translations, axis=0) / dt
    median = median_filter(velocity, size=(21, 1), mode="nearest")
    jump = np.linalg.norm(velocity - median, axis=1) > POSE_JUMP_SPEED
    velocity[jump] = median[jump]
    return np.concatenate([translations[:1], translations[0] + np.cumsum(velocity * dt, axis=0)])


def _frames(lidar: dict, sample_times: np.ndarray) -> dict[str, np.ndarray]:
    """Group the six LiDARs' files into frames, one per LIDAR_LEFT file.

    The LiDARs are phase-synchronised (paper A.1), so each has a fixed offset to LIDAR_LEFT; a file is matched
    to LEFT time + that offset, and frames where a LiDAR has no file within half a period are dropped (this
    happens at a few scene edges). At key frames every LiDAR's key-frame file of the sample is used, and the
    frame time is the sample timestamp: the annotation time, when all scans align in front of the truck.
    Elsewhere the frame time is the LEFT time plus the scene's median sample-to-LEFT offset.

    Returns:
        "files" (F, 6) file index per LiDAR_CHANNELS entry, "time" (F,) us, "sample" (F,) or -1.
    """
    reference = lidar[LIDAR_CHANNELS[0]]
    key = np.flatnonzero(reference["is_key_frame"])
    columns, valid = [np.arange(reference["timestamp"].shape[0])], reference["is_key_frame"].copy()
    matched = np.ones(reference["timestamp"].shape[0], dtype=bool)
    for channel in LIDAR_CHANNELS[1:]:
        record = lidar[channel]
        # One key-frame file per LiDAR and sample, in sample order, so key frames line up elementwise.
        channel_key = np.flatnonzero(record["is_key_frame"])
        target = reference["timestamp"] + np.median(record["timestamp"][channel_key] - reference["timestamp"][key])
        after = np.clip(np.searchsorted(record["timestamp"], target), 1, record["timestamp"].shape[0] - 1)
        closer_before = np.abs(record["timestamp"][after - 1] - target) <= np.abs(record["timestamp"][after] - target)
        index = np.where(closer_before, after - 1, after)
        matched &= np.abs(record["timestamp"][index] - target) <= LIDAR_PERIOD_US // 2
        index[key] = channel_key
        columns.append(index)
    valid |= matched
    sample = np.where(reference["is_key_frame"], reference["sample"], -1)
    to_sample = np.median(sample_times[reference["sample"][key]] - reference["timestamp"][key])
    time = np.where(reference["is_key_frame"], sample_times[reference["sample"]], reference["timestamp"] + to_sample)
    return {
        "files": np.stack(columns, axis=1)[valid].astype(np.int32),
        "time": time[valid].astype(np.int64),
        "sample": sample[valid].astype(np.int32),
    }


def frame_files(scene: dict, position: int) -> list[tuple[str, int]]:
    """``(channel, file index)`` of each LiDAR in frame ``position`` (see `_frames`)."""
    return list(zip(LIDAR_CHANNELS, (int(i) for i in scene["frames"]["files"][position])))


def frame_pose(scene: dict, position: int) -> tuple[int, np.ndarray]:
    """The frame's time (us) and the 4x4 world-from-ego pose then."""
    frame_time = int(scene["frames"]["time"][position])
    rotations, translations = ego_poses_at(scene, np.array([frame_time], dtype=np.int64))
    world_from_ego = np.eye(4)
    world_from_ego[:3, :3], world_from_ego[:3, 3] = rotations[0], translations[0]
    return frame_time, world_from_ego


def ego_poses_at(scene: dict, times_us: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World-from-ego rotations (N, 3, 3) and translations (N, 3) at the given times.

    Interpolated linearly (normalised quaternions) between the ~100 Hz ego poses of the scene.
    """
    pose_times, poses = scene["ego_timestamp"], scene["ego_pose"]
    after = np.clip(np.searchsorted(pose_times, times_us, side="right"), 1, pose_times.shape[0] - 1)
    before = after - 1
    weight = np.clip((times_us - pose_times[before]) / (pose_times[after] - pose_times[before]), 0.0, 1.0)[:, None]
    q0, q1 = poses[before, 3:], poses[after, 3:]
    q1 = q1 * np.sign(np.sum(q0 * q1, axis=1, keepdims=True))  # same hemisphere
    w, x, y, z = ((1 - weight) * q0 + weight * q1).T
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    rotations = np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=-1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=-1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=-1),
    ], axis=1)
    translations = (1 - weight) * poses[before, :3] + weight * poses[after, :3]
    return rotations, translations


def pose_matrix(pose: np.ndarray) -> np.ndarray:
    """4x4 transform from [tx, ty, tz, qw, qx, qy, qz]."""
    matrix = Quaternion(pose[3:]).transformation_matrix
    matrix[:3, 3] = pose[:3]
    return matrix


def _smooth_translation_at(scene: dict, times_us: np.ndarray) -> np.ndarray:
    smooth = scene["ego_translation_smooth"]
    return np.stack([np.interp(times_us, scene["ego_timestamp"], smooth[:, k]) for k in range(3)], axis=1)


def load_file(
    scene: dict, channel: str, idx: int, frame: tuple[int, np.ndarray], min_distance: float = MIN_DISTANCE
) -> dict[str, np.ndarray]:
    """One LiDAR file's points in the ego frame of ``frame`` = ``(time us, 4x4 world-from-ego)``.

    The .pcd points are not motion compensated: every point is relative to where the truck was when it
    was measured. So each point is moved by the truck's motion between its timestamp and the frame time,
    which keeps a sweep taken at highway speed from smearing by about a metre. That motion comes from the
    de-jumped trajectory, so a localisation jump inside the frame does not tear it apart.

    Returns:
        "xyz" (N, 3) float64, "intensity" (N,) 0..255, "timestamp" (N,) int64 us, and "keep", the mask of
        the raw .pcd points that survived ``min_distance`` (the others are already removed).
    """
    frame_time, world_from_ego = frame
    record = scene["lidar"][channel]
    cloud = read_pcd(os.path.join(DATASET_ROOT, record["filename"][idx]))
    points = np.stack([cloud["x"], cloud["y"], cloud["z"]], axis=1).astype(np.float64)
    keep = ~((np.abs(points[:, 0]) < min_distance) & (np.abs(points[:, 1]) < min_distance))
    ego_from_sensor = pose_matrix(scene["ego_from_sensor"][channel])
    points = points[keep] @ ego_from_sensor[:3, :3].T + ego_from_sensor[:3, 3]

    timestamps = cloud["timestamp"][keep].astype(np.int64)
    rotations, _ = ego_poses_at(scene, timestamps)
    motion = _smooth_translation_at(scene, timestamps) - _smooth_translation_at(scene, np.array([frame_time]))
    world = np.einsum("nij,nj->ni", rotations, points) + motion + world_from_ego[:3, 3]
    ego_from_world = np.linalg.inv(world_from_ego)
    return {
        "xyz": world @ ego_from_world[:3, :3].T + ego_from_world[:3, 3],
        # Stored as 0..1.
        "intensity": normalize_intensity(cloud["intensity"][keep], 1.0),
        "timestamp": timestamps,
        "keep": keep,
    }


def load_points(scene: dict, positions: Sequence[int], frame: tuple[int, np.ndarray]) -> tuple[torch.Tensor, list[tuple[str, str, np.ndarray]]]:
    """Stack the six-LiDAR frames at ``positions`` in the ego frame of ``frame`` (see `load_file`).

    Returns:
        ``(points, files)``: (N, 5) [age, x, y, z, intensity] tensor, the age in seconds from the newest
        point, and ``(channel, filename, keep)`` of every file in the order its points appear.
    """
    xyz, intensity, times, files = [], [], [], []
    for position in positions:
        for channel, idx in frame_files(scene, position):
            cloud = load_file(scene, channel, idx, frame)
            xyz.append(cloud["xyz"])
            intensity.append(cloud["intensity"])
            times.append(cloud["timestamp"])
            files.append((channel, scene["lidar"][channel]["filename"][idx], cloud["keep"]))

    times = np.concatenate(times)
    ages = (times.max() - times) / 1e6 if times.size else np.zeros(0)
    features = [np.concatenate(xyz), np.concatenate(intensity)]
    if USE_TIME:
        features.insert(0, ages)
    columns = np.column_stack(features)
    return torch.from_numpy(columns).float(), files


def scene_annotations(scene: dict, sample: int, ego_from_world: np.ndarray) -> dict[str, torch.Tensor]:
    """The boxes of key frame ``sample`` in the frame ``ego_from_world`` (4x4).

    Sizes become ``[length, width, height]`` and quaternions stay ``[w, x, y, z]``, as the rest of the
    codebase expects. Boxes of categories outside the detection classes are dropped.
    """
    lo, hi = scene["ann_offsets"][sample], scene["ann_offsets"][sample + 1]
    class_ids = scene["ann_class"][lo:hi]
    keep = class_ids > 0
    centers = scene["ann_center"][lo:hi][keep]
    sizes = scene["ann_size"][lo:hi][keep]
    rotations = scene["ann_rotation"][lo:hi][keep]

    ego_from_world_rotation = Quaternion(matrix=ego_from_world[:3, :3])
    centers = centers @ ego_from_world[:3, :3].T + ego_from_world[:3, 3]
    quats = np.array([(ego_from_world_rotation * Quaternion(r)).elements for r in rotations]).reshape(-1, 4)

    return {
        "class": torch.from_numpy(class_ids[keep].astype(np.int32)),
        "center": torch.from_numpy(centers).float(),
        "size": torch.from_numpy(sizes[:, [1, 0, 2]]).float(),
        "quats": torch.from_numpy(quats).float(),
    }


def read_pcd(path: str) -> dict[str, np.ndarray]:
    """Fields of a ``binary`` or ``binary_compressed`` (LZF, column-major) .pcd file as 1D arrays."""
    with open(path, "rb") as f:
        raw = f.read()

    header, pos = {}, 0
    while "DATA" not in header:
        end = raw.index(b"\n", pos)
        line = raw[pos:end].decode("ascii").strip()
        pos = end + 1
        if line and not line.startswith("#"):
            key, *values = line.split()
            header[key] = values

    if any(count != "1" for count in header["COUNT"]):
        raise NotImplementedError(f"Multi-count .pcd fields are not supported: {path}")
    names = header["FIELDS"]
    dtypes = [np.dtype(f"<{kind.lower()}{size}") for kind, size in zip(header["TYPE"], header["SIZE"])]
    num_points = int(header["POINTS"][0])

    kind = header["DATA"][0]
    if kind == "binary_compressed":
        compressed_size, size = struct.unpack_from("<II", raw, pos)
        body = _lzf_decompress(raw[pos + 8 : pos + 8 + compressed_size], size)
        columns, offset = {}, 0
        for name, dtype in zip(names, dtypes):
            columns[name] = np.frombuffer(body, dtype=dtype, count=num_points, offset=offset)
            offset += dtype.itemsize * num_points
        return columns
    if kind == "binary":
        records = np.frombuffer(raw, dtype=np.dtype(list(zip(names, dtypes))), count=num_points, offset=pos)
        return {name: records[name] for name in names}
    raise NotImplementedError(f"Unsupported .pcd DATA {kind!r}: {path}")


def _lzf_decompress_kernel(src, dst) -> int:
    """LZF (liblzf) decompression of ``src`` into the preallocated ``dst``; returns the bytes written."""
    ip = 0
    op = 0
    n = len(src)
    while ip < n:
        ctrl = int(src[ip])
        ip += 1
        if ctrl < 32:  # literal run
            ctrl += 1
            dst[op : op + ctrl] = src[ip : ip + ctrl]
            ip += ctrl
            op += ctrl
        else:  # back reference
            length = ctrl >> 5
            ref = op - ((ctrl & 0x1F) << 8) - 1
            if length == 7:
                length += int(src[ip])
                ip += 1
            ref -= int(src[ip])
            ip += 1
            length += 2
            if ref + length <= op:
                dst[op : op + length] = dst[ref : ref + length]
            else:  # overlaps its own output, so it has to go byte by byte
                for k in range(length):
                    dst[op + k] = dst[ref + k]
            op += length
    return op


_lzf_decompress_jit = numba.njit(_lzf_decompress_kernel) if numba is not None else None


def _lzf_decompress(compressed: bytes, size: int):
    if _lzf_decompress_jit is not None:
        out = np.empty(size, dtype=np.uint8)
        written = _lzf_decompress_jit(np.frombuffer(compressed, dtype=np.uint8), out)
    else:
        out = bytearray(size)
        written = _lzf_decompress_kernel(compressed, out)
    if written != size:
        raise ValueError(f"LZF decompressed to {written} bytes, expected {size}")
    return out
