"""NuScenes dataset."""

import os
import random
import numpy as np
import torch
from pyquaternion import Quaternion
from torch.utils.data import Dataset
from data.nuscenes_devkit_lite import (
    NuScenesLite,
    load_lidar_sweeps,
    load_lidarseg_labels,
    load_panoptic_labels,
    splits,
    transform_matrix,
)
from data.nuscenes_lidarseg import (
    IGNORE_ID,
    SEG_CLASS_NAME_TO_ID,
    THING_IDS,
    raw_lidarseg_to_seg_ids,
    split_panoptic_labels,
)
from data.base_dataset import BaseLidarDataset, centered_sweep_chunk, sample_sweep_chunk
from data.zod_dataset import (
    NoPointsException,
    _zero_pad_point_cloud,
    _zero_pad_point_cloud_with_labels,
    _reshape_annotations_dict,
)
from constants import (
    MAX_NUM_OBJECTS,
    MAX_NUM_POINTS,
    PC_RANGE,
    RANDOM_CROP_CENTER_RANGES,
    MIN_NUM_POINTS,
    USE_TIME,
    X_INDEX,
    Y_INDEX,
    Z_INDEX,
)

NUSCENES_DATA_ROOT = os.environ.get("NUSCENES_DATASET_ROOT", "/datasets/nuscenes/v1.0/")
RANDOM_CROP_CENTER_RANGE = RANDOM_CROP_CENTER_RANGES["NuScenes"]
MAX_TRY = 10
# A sample's sweeps are its key frame +-5: key frames are 2 Hz and sweeps 20 Hz, so these tile a scene.
SWEEPS_AROUND_KEYFRAME = 5

CLASSES = (
    "pedestrian",
    "car",
    "motorcycle",
    "bicycle",
    "bus",
    "truck",
    "construction_vehicle",
    "trailer",
    "barrier",
    "traffic_cone",
)
CLASS_NAMES = [cls for cls in CLASSES]
# nuscenes object class mapping
NUSCENES_TO_CLASSES = {
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    # "human.pedestrian.wheelchair": "ignore",
    # "human.pedestrian.stroller": "ignore",
    # "human.pedestrian.personal_mobility": "ignore",
    "human.pedestrian.police_officer": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    # "animal": "ignore",
    "vehicle.car": "car",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.truck": "truck",
    "vehicle.construction": "construction_vehicle",
    # "vehicle.emergency.ambulance": "ignore",
    # "vehicle.emergency.police": "ignore",
    "vehicle.trailer": "trailer",
    "movable_object.barrier": "barrier",
    "movable_object.trafficcone": "traffic_cone",
    # "movable_object.pushable_pullable": "ignore",
    # "movable_object.debris": "ignore",
    # "static_object.bicycle_rack": "ignore",
}
CLASS_NAME_TO_ID = {name: idx + 1 for idx, name in enumerate(CLASSES)}

NUSCENES_CLASSES = {
    "human.pedestrian.adult",
    "human.pedestrian.child",
    "human.pedestrian.wheelchair",
    "human.pedestrian.stroller",
    "human.pedestrian.personal_mobility",
    "human.pedestrian.police_officer",
    "human.pedestrian.construction_worker",
    "animal",
    "vehicle.car",
    "vehicle.motorcycle",
    "vehicle.bicycle",
    "vehicle.bus.bendy",
    "vehicle.bus.rigid",
    "vehicle.truck",
    "vehicle.construction",
    "vehicle.emergency.ambulance",
    "vehicle.emergency.police",
    "vehicle.trailer",
    "movable_object.barrier",
    "movable_object.trafficcone",
    "movable_object.pushable_pullable",
    "movable_object.debris",
    "static_object.bicycle_rack",
}


class NuScenesDataset(Dataset, BaseLidarDataset):
    """NuScenes Dataset for handling LiDAR and object detection annotations."""

    @classmethod
    def get_class_name_to_id(cls) -> dict[str, int]:
        return CLASS_NAME_TO_ID

    @classmethod
    def get_seg_class_name_to_id(cls) -> dict[str, int]:
        """The lidarseg benchmark classes, which are not the detection classes."""
        return SEG_CLASS_NAME_TO_ID

    @classmethod
    def get_seg_thing_ids(cls) -> tuple[int, ...]:
        return THING_IDS

    @classmethod
    def get_seg_ignore_id(cls) -> int | None:
        """Lidarseg id 0 is noise, which the benchmark neither trains nor scores."""
        return IGNORE_ID
    def __init__(
        self,
        train_flag: bool = True,
        load_annotations: bool = False,
        num_samples: int = None,
        version: str = "full",
        min_num_sweeps_per_sample: int = 1,
        max_num_sweeps_per_sample: int = 1,
        random_crop: bool = False,
        load_frame_id: bool = False,
        load_segmentation: bool = False,
        load_num_sweeps: bool = False,
    ):
        """Initialize the NuScenesDataset.
        Args:
            dataset_root: Path to dataset root.
            version: Dataset version ('v1.0-mini' or 'v1.0-trainval').
            train_flag: Load training set if True, validation set otherwise.
            load_annotations: Load object detection annotations if True.
            load_segmentation: Also return per-point semantic and instance ids.
                Both come from nuScenes-panoptic when that add-on is staged;
                otherwise the semantic ids come from nuScenes-lidarseg and the
                instance ids are all zero.
            min_num_sweeps_per_sample, max_num_sweeps_per_sample: Range the number of aggregated
                sweeps is drawn from per sample, out of the key frame +-SWEEPS_AROUND_KEYFRAME sweeps.
                Without labels (pre-training) a random chunk is aggregated into the sensor frame of a
                random sweep in it, even for a single sweep. With labels the chunk is centred on the key
                frame as reference and the other sweeps' points get the ignore label. Usually we should eval on 1 sweep though.
        """
        version = "v1.0-mini" if version == "mini" else "v1.0-trainval"
        available_vers = ["v1.0-trainval", "v1.0-mini"]
        assert version in available_vers
        if version == "v1.0-trainval":
            train_scenes = splits.train
            val_scenes = splits.val
        else:
            train_scenes = splits.mini_train
            val_scenes = splits.mini_val
            num_samples = 10 #if mini we want it to be fewer samples as this is for debug.

        try:
            self.nusc = NuScenesLite(version=version, dataroot=NUSCENES_DATA_ROOT, verbose=True)
        except FileNotFoundError:
            if version != "v1.0-mini":
                raise
            # Some releases ship only the trainval and test metadata. The mini
            # scenes are a subset of trainval and their records there are
            # identical -- same scene, sample, sample_data and annotation tokens
            # -- so the split can still be served from the trainval tables. The
            # scene filter below restricts them to the mini scenes. This costs a
            # full trainval table load (slow, and several GB), so it is a
            # fallback rather than the preferred path.
            print("v1.0-mini tables not found; falling back to v1.0-trainval restricted to the mini scenes")
            self.nusc = NuScenesLite(version="v1.0-trainval", dataroot=NUSCENES_DATA_ROOT, verbose=True)

        scenes_names = train_scenes if train_flag else val_scenes
        # Look the split scenes up by name. Indexing the scene table by a name's
        # position within the split list instead returns the first N table
        # entries, which is unrelated to the split and puts the same scenes in
        # both train and val.
        name_to_token = {scene["name"]: scene["token"] for scene in self.nusc.scene}
        scenes = {name_to_token[name] for name in scenes_names if name in name_to_token}
        print(
            f"number of {'training' if train_flag else 'validation'} "
            f"scenes: {len(scenes)} (of {len(scenes_names)} in the split list)"
        )
        self.samples = []
        for sample in self.nusc.sample:
            if sample["scene_token"] not in scenes:
                continue
            self.samples.append(sample)
        random.seed(42)  # set seed so that we get same samples
        if num_samples is None or num_samples > len(self.samples):
            # TODO: make sure no multiple samples from same scene as far as possible.
            num_samples = len(self.samples)
        self.samples = random.sample(self.samples, num_samples)  # Randomly sample from
        random.seed(None)
        print(f"Number of actual samples {len(self.samples)}")
        self._load_annotations = load_annotations
        self._load_frame_id = load_frame_id
        self._load_num_sweeps = load_num_sweeps
        self._min_nsweeps = min_num_sweeps_per_sample
        self._max_nsweeps = max_num_sweeps_per_sample
        # Labels only exist for key frames, so only label-free samples may use any sweep as reference.
        self._random_reference = not (load_annotations or load_segmentation)
        assert 1 <= min_num_sweeps_per_sample <= max_num_sweeps_per_sample
        self._random_crop = random_crop
        self._load_segmentation = load_segmentation

        if self._load_segmentation and not (self.nusc.has_lidarseg or self.nusc.has_panoptic):
            raise FileNotFoundError(
                "Segmentation labels were requested but this NuScenes release has "
                "no lidarseg.json. Stage the nuScenes-lidarseg add-on alongside "
                "the base data (several paths can be joined with ':' in "
                "NUSCENES_DATASET_ROOT)."
            )
        if self._load_segmentation and not self.nusc.has_panoptic:
            print(
                "No panoptic.json in this NuScenes release: instance ids will all be "
                "zero. Stage the nuScenes-panoptic add-on to get real ones."
            )

    def __len__(self):
        """Get length."""
        return len(self.samples)

    def __getitem__(self, index):  # type: ignore
        """Get item.
        Returns:
            - point_cloud: padded tensor of [x, y, z, intensity], with time prepended when USE_TIME is enabled.
            - annotations: dictionary of object detection annotations with the following structure:
                {
                    "class": Tensor of shape (MAX_OBJECTS)
                    "task_ids": Tensor of shape (MAX_OBJECTS)
                    "center": Tensor of shape (MAX_OBJECTS, 3),
                    representing [x, y, z]
                    "size": Tensor of shape (MAX_OBJECTS, 3),
                    representing [dx, dy, dz]
                    "quats": Tensor of shape (MAX_OBJECTS, 4),
                    representing quaternions [w, x, y, z]
                    "rot_mats": Tensor of shape (MAX_OBJECTS, 3, 3),
                    representing rotation matrix
                }
            - camera_matrices: dictionary of camera matrices with the following structure:
                {
                    "camera_extrinsics": Tensor of shape (4, 4)
                    "lidar_extrinsics": Tensor of shape (4, 4)
                    "camera_distortion": Tensor of shape (4)
                    "camera_field_of_view": Tensor of shape (2) in degrees
                }
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
                    if len(self.samples) <= 1:
                        raise
                    current_index = torch.randint(0, len(self.samples), size=(1,)).item()
                    no_points_tries = 0
            except Exception:
                raise

    def _getitem(self, index):  # type: ignore
        """Get item."""
        sample = self.samples[index]
        # get lidar data
        lidar_token = sample["data"]["LIDAR_TOP"]
        keyframe = self.nusc.get("sample_data", lidar_token)

        nsweeps = torch.randint(self._min_nsweeps, self._max_nsweeps + 1, size=(1,)).item()
        sweeps, reference = self._sweep_chunk(keyframe, nsweeps)
        lidar_points_tensor, seg_labels = self._get_sweeped_pointcloud(sample, sweeps, reference)  # (N,5), (N,2) or None

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
            raise NoPointsException(f"Too few points left after filtering for index {index}, only {mask.sum()} points remaining.")
        if seg_labels is None:
            lidar_points_tensor = _zero_pad_point_cloud(lidar_points_tensor[mask])
        else:
            lidar_points_tensor, seg_labels = _zero_pad_point_cloud_with_labels(
                lidar_points_tensor[mask], seg_labels[mask]
            )

        outs = (lidar_points_tensor,)
        if self._load_frame_id:
            frame_id = sample["token"]
            outs += (frame_id,)
        if self._load_num_sweeps:
            outs += (len(sweeps),)
            
        if self._load_annotations:
            annotations = self._get_annotations(sample, lidar_token)
            # Filter annotations based on pc range as well
            centers = annotations["center"]
            if centers.shape[0] > 0:
                xs, ys, zs = centers[:, 0], centers[:, 1], centers[:, 2]
                mask = (xs >= x_min) & (xs <= x_max) & (ys >= y_min) & (ys <= y_max) & (zs >= z_min) & (zs <= z_max)
                for key in annotations:
                    annotations[key] = annotations[key][mask]

            annotations = _reshape_annotations_dict(annotations)  # pad to fixed size

            # Since we are only using fov for limiting pointcloud to annotated region for now
            # and nuscenes has 360 annotations, we set this to a full FOV tensor to keep all points.
            field_of_view = torch.tensor([360.0, 360.0], dtype=torch.float32)
            outs += (annotations, field_of_view)

        if seg_labels is not None:
            outs += ({"semantic": seg_labels[:, 0], "instance": seg_labels[:, 1]},)

        return outs[0] if len(outs) == 1 else outs
    
    def _sweep_chunk(self, keyframe: dict, nsweeps: int) -> tuple[list[dict], int]:
        """``nsweeps`` consecutive sweeps around the key frame in time order and the index of the reference among them.

        The run and reference are random without labels, else the key frame is the reference in the middle.
        """
        before, after = [], []
        record = keyframe
        while len(before) < SWEEPS_AROUND_KEYFRAME and record["prev"]:
            record = self.nusc.get("sample_data", record["prev"])
            before.insert(0, record)
        record = keyframe
        while len(after) < SWEEPS_AROUND_KEYFRAME and record["next"]:
            record = self.nusc.get("sample_data", record["next"])
            after.append(record)
        pool = before + [keyframe] + after
        if self._random_reference:
            start, end, reference = sample_sweep_chunk(len(pool), nsweeps)
        else:
            start, end, reference = centered_sweep_chunk(len(pool), nsweeps, len(before))
        return pool[start:end], reference - start

    def _get_sweeped_pointcloud(self, sample_rec, sweeps, reference):
        """Load ``sweeps`` aggregated in the frame of ``sweeps[reference]`` and, when requested, the key frame's labels.

        Returns:
            ``(points, seg_labels)`` where ``points`` is (N, POINT_FEATURE_DIM)
            and ``seg_labels`` is (N,2) of [semantic, instance] ids, or None when
            labels were not requested. Labels are the key frame's, which must be the reference.
        """
        points, sweep_times, not_close_masks = load_lidar_sweeps(self.nusc, sweeps, sweeps[reference])
        if points.shape[1] == 0:
            raise NoPointsException("Empty lidar sweeps.")
        points = points.T  # shape (N,4): x,y,z,intensity
        points = torch.from_numpy(points).float()
        if USE_TIME:
            timestamps = torch.from_numpy(sweep_times.T).float()
            timestamps -= timestamps.min()
            points = torch.cat((timestamps, points), dim=1)

        if not self._load_segmentation:
            return points, None

        offset = sum(int(mask.sum()) for mask in not_close_masks[:reference])
        seg_labels = self._get_seg_labels(sample_rec, points.shape[0], not_close_masks[reference], offset)
        return points, seg_labels

    def _get_seg_labels(self, sample_rec, num_points, keyframe_mask, offset=0):
        """Per-point [semantic, instance] ids for the aggregated cloud, whose key frame points start at ``offset``."""
        lidar_token = sample_rec["data"]["LIDAR_TOP"]
        if self.nusc.has_panoptic:
            semantic, instance = split_panoptic_labels(
                load_panoptic_labels(self.nusc, lidar_token)
            )
        else:
            semantic = raw_lidarseg_to_seg_ids(load_lidarseg_labels(self.nusc, lidar_token))
            instance = np.zeros_like(semantic)

        # Labels are stored against the key frame's unfiltered points, so apply
        # the same close-range filter that the aggregation applied to them.
        semantic = semantic[keyframe_mask]
        instance = instance[keyframe_mask]

        labels = torch.zeros((num_points, 2), dtype=torch.long)
        num_labelled = semantic.shape[0]
        labels[offset : offset + num_labelled, 0] = torch.from_numpy(semantic)
        labels[offset : offset + num_labelled, 1] = torch.from_numpy(instance)
        return labels
    
    def _get_annotations(self, sample: dict, lidar_token: str) -> dict[str, torch.Tensor]:
        """Process and convert annotations to tensors."""
        class_ids = []
        center = []
        size = [] 
        quaternions = []
        velocities = []
        num_points = []
        # get lidar calibration
        lidar_data = self.nusc.get("sample_data", lidar_token)
        cs_record = self.nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
        ego_pose = self.nusc.get("ego_pose", lidar_data["ego_pose_token"])
        # transform from global to ego vehicle frame
        global_to_ego = transform_matrix(
            ego_pose["translation"], Quaternion(ego_pose["rotation"]), inverse=True
        )
        # transform from ego vehicle to sensor frame
        ego_to_sensor = transform_matrix(
            cs_record["translation"], Quaternion(cs_record["rotation"]), inverse=True
        )
        for ann_token in sample["anns"]:
            ann = self.nusc.get("sample_annotation", ann_token)
            
            if ann["category_name"] not in NUSCENES_CLASSES:
                continue
            dynamic_class = NUSCENES_TO_CLASSES.get(ann["category_name"], None)
            if dynamic_class is None:
                continue
            class_id = CLASS_NAME_TO_ID.get(dynamic_class, 0)
            if class_id == 0:
                continue
            box_center = np.array(ann["translation"])
            box_size = np.array(ann["size"])  # width, length, height?
            # reorder to length, width, height?
            box_size = np.array([box_size[1], box_size[0], box_size[2]])
            box_rotation = Quaternion(ann["rotation"])
            box_center_homo = np.array([box_center[0], box_center[1], box_center[2], 1.0])
            box_center_ego = global_to_ego @ box_center_homo
            box_center_sensor = ego_to_sensor @ box_center_ego
            box_center_sensor = box_center_sensor[:3]
            rotation_global_to_ego = Quaternion(ego_pose["rotation"]).inverse
            rotation_ego_to_sensor = Quaternion(cs_record["rotation"]).inverse
            box_rotation_sensor = rotation_ego_to_sensor * rotation_global_to_ego * box_rotation
            class_ids.append(class_id)
            center.append(box_center_sensor)
            size.append(box_size)
            velocities.append((rotation_ego_to_sensor * rotation_global_to_ego).rotate(self._box_velocity(ann))[:2])
            # The nuScenes detection eval drops boxes that no lidar or radar point hit.
            num_points.append(ann["num_lidar_pts"] + ann["num_radar_pts"])
            quaternions.append(
                [
                    box_rotation_sensor.w,
                    box_rotation_sensor.x,
                    box_rotation_sensor.y,
                    box_rotation_sensor.z,
                ]
            )
        tensors = (
            torch.tensor(np.array(class_ids), dtype=torch.int32),
            torch.tensor(np.array(center), dtype=torch.float32),
            torch.tensor(np.array(size), dtype=torch.float32),
            torch.tensor(np.array(quaternions), dtype=torch.float32),
            torch.tensor(np.array(velocities), dtype=torch.float32),
        )
        return _pad_and_format_annotations(tensors, torch.tensor(num_points, dtype=torch.int32))

    def _box_velocity(self, ann: dict, max_time_diff: float = 1.5) -> np.ndarray:
        """The devkit's `NuScenes.box_velocity`: global-frame velocity from the neighbouring annotations, NaN if unknown."""
        has_prev, has_next = ann["prev"] != "", ann["next"] != ""
        if not has_prev and not has_next:
            return np.full(3, np.nan)
        first = self.nusc.get("sample_annotation", ann["prev"]) if has_prev else ann
        last = self.nusc.get("sample_annotation", ann["next"]) if has_next else ann
        time_diff = 1e-6 * (
            self.nusc.get("sample", last["sample_token"])["timestamp"]
            - self.nusc.get("sample", first["sample_token"])["timestamp"]
        )
        if time_diff > (2 * max_time_diff if has_prev and has_next else max_time_diff):
            return np.full(3, np.nan)
        return (np.array(last["translation"]) - np.array(first["translation"])) / time_diff
   

def _pad_and_format_annotations(
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    num_points: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Pad or randomly trim annotation tensors to a fixed size."""
    class_ids, centers, sizes, quaternions, velocities = tensors
    num_objects = centers.shape[0]
    device = centers.device if centers.numel() > 0 else "cpu"
    # trim randomly if more than allowed
    if num_objects > MAX_NUM_OBJECTS:
        keep_indices = torch.randperm(num_objects)[:MAX_NUM_OBJECTS]
        class_ids = class_ids[keep_indices]
        centers = centers[keep_indices]
        sizes = sizes[keep_indices]
        quaternions = quaternions[keep_indices]
        velocities = velocities[keep_indices]
        num_points = num_points[keep_indices]
        num_objects = MAX_NUM_OBJECTS
    # padding
    padded = {
        "class": torch.zeros((MAX_NUM_OBJECTS), dtype=torch.int32, device=device),
        "center": torch.zeros((MAX_NUM_OBJECTS, 3), dtype=torch.float32, device=device),
        "size": torch.zeros((MAX_NUM_OBJECTS, 3), dtype=torch.float32, device=device),
        "quats": torch.zeros((MAX_NUM_OBJECTS, 4), dtype=torch.float32, device=device),
        "velocity": torch.zeros((MAX_NUM_OBJECTS, 2), dtype=torch.float32, device=device),
        "num_points": torch.zeros((MAX_NUM_OBJECTS), dtype=torch.int32, device=device),
    }
    if num_objects > 0:
        padded["class"][:num_objects] = class_ids
        padded["center"][:num_objects] = centers
        padded["size"][:num_objects] = sizes
        padded["quats"][:num_objects] = quaternions
        padded["velocity"][:num_objects] = velocities
        padded["num_points"][:num_objects] = num_points
    return padded
