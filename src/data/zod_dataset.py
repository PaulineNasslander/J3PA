"""Pytorch dataset for loading ZOD camera image and LiDAR cloud."""

import os
import pickle
from typing import Literal

import numpy as np
import torch
from pyquaternion import Quaternion
from zod import ZodFrames
from zod.constants import AnnotationProject, Camera, Lidar
from zod.data_classes.frame import _adjust_lidar_core_time
from zod.data_classes.sensor import LidarData
from zod.utils.compensation import motion_compensate_scanwise
from zod.zod_frames import ZodFrame

from constants import ZOD_LABELS_DIR, MAX_NUM_OBJECTS, MAX_NUM_POINTS, MIN_NUM_POINTS, PC_RANGE, RANDOM_CROP_CENTER_RANGES, USE_TIME, X_INDEX, Y_INDEX, Z_INDEX
from data.base_dataset import BaseLidarDataset, centered_sweep_chunk, sample_sweep_chunk

from zod.anno.object import OBJECT_SUBCLASSES

# class_names = ["Vehicle"]
# CLASS_NAME_TO_ID = {name: idx + 1 for idx, name in enumerate(class_names)}

# Try to evaluate all classes, including subclasses.
all_classes = set(OBJECT_SUBCLASSES)
# Remove classes that
all_classes -= {
    "Unclear",
    "Vehicle_Other",
    "VulnerableVehicle_Other",
    "VulnerableVehicle_Wheelchair",
}
ZOD_CLASS_NAMES = sorted(all_classes)
CLASS_NAMES = [cls for cls in ZOD_CLASS_NAMES]
CLASS_NAME_TO_ID = {name: idx + 1 for idx, name in enumerate(ZOD_CLASS_NAMES)}

DATASET_ROOT = os.environ.get("ZOD_DATASET_ROOT", "/datasets/zod/zodv1")
RANDOM_CROP_CENTER_RANGE = RANDOM_CROP_CENTER_RANGES["ZOD"]
MAX_TRY = 10

# def _encode_class(name, with_rider: bool, traffic_content_visible: bool, object_type: str):
#     """Encode classes."""
#     return CLASS_NAME_TO_ID.get(name, 0)  # Returns 0 if class is not found

def _encode_class(
    name: str, with_rider: bool, traffic_content_visible: bool, object_type: str
):
    """Encode classes."""
    full_name = name
    if name.startswith("Vehicle"):
        sub_class = object_type if object_type else "Other"
        full_name = name + "_" + sub_class
    elif "VulnerableVehicle" in name:
        sub_class = object_type if with_rider else "NoRider"
        full_name = name + "_" + sub_class
    elif "Sign" in name:  # sign and signal will be true here
        sub_class = "Front" if traffic_content_visible else "Back"
        full_name = name + "_" + sub_class
    # else: keep name as is.
    number = CLASS_NAME_TO_ID.get(full_name, 0)  # Returns 0 if class is not found
    if number == 0:
        return CLASS_NAME_TO_ID.get(name, 0)  # Returns 0 if class is not found
    return number

class ZODdataset(ZodFrames, BaseLidarDataset):
    """Dataset for LiDAR point clouds and images with object annotations."""

    # class_name_to_number = {name: idx + 1 for idx, name in enumerate(class_names)}

    @classmethod
    def get_class_name_to_id(cls) -> dict[str, int]:
        return CLASS_NAME_TO_ID

    def __init__(
        self,
        train_flag,
        version: Literal["mini", "full"] = "full",
        load_annotations=True,
        random_crop=False,
        mp=True,
        load_frame_id = False,
        load_segmentation: bool = False,
        load_num_sweeps: bool = False,
        min_num_sweeps_per_sample: int = 1,
        max_num_sweeps_per_sample: int = 1,
    ):
        """Initialize.
        
        Args:
            train_flag: bool, whether to load training or validation set
            version: str, "mini" or "full", which version of the dataset to load
            load_annotations: bool, whether to load OD annotations
            random_crop: bool, whether to apply random cropping to the point cloud
            mp: bool, whether to use multiprocessing in dataloading
            load_frame_id: bool, whether to load frame id
            load_num_sweeps: bool, whether to return the number of aggregated sweeps.
            load_segmentation: Also return per-point semantic and instance labels.
                As ZOD only contains 3D object annotations these are provided by annotating points 
                inside a bounding box by the class of the object. Other points are all labeled with 
                the "background class"
            min_num_sweeps_per_sample, max_num_sweeps_per_sample: Range the number of aggregated
                sweeps is drawn from per sample, out of the frame's ~21 sweeps (+-1 s around the key one).
                Without labels (pre-training) a random run is aggregated into the lidar frame of a random
                sweep in it, even for a single sweep. With labels the run is centred on the key sweep as
                reference. Segmentation (in_cone) only covers the key sweep, so it needs max == 1.
            
            The panoptic and semantic labels are precomputed and saved directly in the ZOD_LABELS_DIR 
            as frame_id.npz, each containing {"in_cone", "semantic", "instance"} keys. 

            NOTE: If load_segmentation is true then the points loaded will be limited to the 
            "in_cone" points, i.e. the points that are inside the camera frustum. This is because the 
            OD annotations from which these labels are computed only exist for the points inside the camera frustum. 
            Because I am not a consistent person this is not the case when only load_annotations is true.
        
        """
        super().__init__(DATASET_ROOT, version, mp)  # mp: multiprocessing in dataloading
        self._load_annotations = load_annotations
        self._load_frame_id = load_frame_id
        self._load_num_sweeps = load_num_sweeps
        _ids = self._train_ids if train_flag else self._val_ids
        _frames = {k: v for _, (k, v) in enumerate(self._infos.items()) if k in _ids}
        self._frames = TorchList(_frames.values())
        self._random_crop = random_crop
        self._load_segmentation = load_segmentation
        assert 1 <= min_num_sweeps_per_sample <= max_num_sweeps_per_sample
        self._min_nsweeps = min_num_sweeps_per_sample
        self._max_nsweeps = max_num_sweeps_per_sample
        # Labels only exist for the key sweep, so only label-free samples may use any sweep as reference.
        self._random_reference = not (load_annotations or load_segmentation)
        if max_num_sweeps_per_sample > 1 and load_segmentation:
            raise ValueError("ZOD segmentation labels only cover the key sweep, so they need max_num_sweeps_per_sample == 1.")
    
    def __getitem__(self, index):
        current_index = index
        no_points_tries = 0

        while True:
            try:
                return self._getitem(current_index)
            except NoPointsException:
                no_points_tries += 1

                if no_points_tries > MAX_TRY:
                    # If repeated random crops fail, fall back to another frame.
                    if len(self._frames) <= 1:
                        raise
                    current_index = torch.randint(0, len(self._frames), size=(1,)).item()
                    no_points_tries = 0
            except Exception:
                raise
            
        
    def _getitem(self, index):
        """Get image and lidar points from ZOD frame."""
        zod_frame = ZodFrame(info=self._frames[index])

        if self._random_reference or self._max_nsweeps > 1:
            nsweeps = torch.randint(self._min_nsweeps, self._max_nsweeps + 1, size=(1,)).item()
            lidar_data, nsweeps = _load_sweep_chunk(zod_frame, nsweeps, random_reference=self._random_reference, return_num_sweeps=True)
        else:
            lidar_data = zod_frame.get_lidar()[0]
            nsweeps = 1
        if lidar_data.timestamps.size == 0:
            raise NoPointsException(f"Empty lidar scan for index {index}.")
        features = [
            torch.from_numpy(lidar_data.points).float(),
            torch.from_numpy(lidar_data.intensity).float().unsqueeze(1),
        ]
        if USE_TIME:
            timestamps = lidar_data.timestamps.max() - lidar_data.timestamps
            features.insert(0, torch.from_numpy(timestamps).float().unsqueeze(1))
        lidar_points_tensor = torch.cat(features, dim=1)

        #Load precomputed segmentation and panoptic labels if requested and if so limit points to frustum to keep N same.
        if self._load_segmentation:
            frame_id = zod_frame.info.id
            label_path = os.path.join(ZOD_LABELS_DIR, f"{frame_id}.npz")
            if not os.path.exists(label_path):
                raise FileNotFoundError(f"Precomputed labels not found for frame {frame_id} at {label_path}. Please run precompute_zod_labels.py first.")
            labels = np.load(label_path)
            in_cone = torch.from_numpy(labels["in_cone"]).bool()
            semantic = torch.from_numpy(labels["semantic"]).long()
            instance = torch.from_numpy(labels["instance"]).long()

            lidar_points_tensor = lidar_points_tensor[in_cone]
            semantic = semantic[in_cone]
            instance = instance[in_cone]

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
        if self._load_segmentation:
            lidar_points_tensor, seg_labels = _zero_pad_point_cloud_with_labels(
                lidar_points_tensor[mask], torch.stack((semantic, instance), dim=1)[mask]
            )
        else:
            lidar_points_tensor = _zero_pad_point_cloud(lidar_points_tensor[mask])

        outs = (lidar_points_tensor,)
        if self._load_frame_id:
            frame_id = zod_frame.info.id
            outs += (frame_id,)
        if self._load_num_sweeps:
            outs += (nsweeps,)

        if self._load_annotations:
            annotation_dict = self.get_annotations(zod_frame.get_annotation(AnnotationProject.OBJECT_DETECTION))
            # Filter annotations based on pc range as well
            centers = annotation_dict["center"]
            if centers.shape[0] > 0:
                xs, ys, zs = centers[:, 0], centers[:, 1], centers[:, 2]
                mask = (xs >= x_min) & (xs <= x_max) & (ys >= y_min) & (ys <= y_max) & (zs >= z_min) & (zs <= z_max)
                for key in annotation_dict:
                    annotation_dict[key] = annotation_dict[key][mask]

            annotation_dict = _reshape_annotations_dict(annotation_dict)  # pad to fixed size

            # get horizontal and vertical fov for annotation cone
            field_of_view = (
                torch.from_numpy(zod_frame.calibration.cameras[Camera.FRONT].field_of_view).to(torch.float32).squeeze(0)
            )  # horizontal, vertical (degrees)

            outs += (annotation_dict, field_of_view)
        if self._load_segmentation:
            outs += ({"semantic": seg_labels[:, 0], "instance": seg_labels[:, 1]},)

        return outs[0] if len(outs) == 1 else outs

    def __len__(self):
        """Get length."""
        return len(self._frames)

    def get_annotations(self, annotations) -> dict:
        """Get annotations.

        returns: dictionary with class_name, center, size, quats tensors
        """
        class_name, center, size, quaternion_components = (
            [],
            [],
            [],
            [],
        )

        for annotation in annotations:
            class_label = _encode_class(
                annotation.name,
                annotation.with_rider,
                annotation.traffic_content_visible,
                annotation.object_type,
            )

            if class_label == 0 or annotation.box3d is None:
                continue

            class_name.append(class_label)
            center.append(annotation.box3d.center)
            size.append(annotation.box3d.size)
            quat = Quaternion(matrix=annotation.box3d.orientation.rotation_matrix)
            quaternion_components.append(np.array([quat.w, quat.x, quat.y, quat.z], dtype=np.float32))

        class_name_tensor = torch.tensor(np.array(class_name), dtype=torch.int32)
        center_tensor = torch.tensor(np.array(center), dtype=torch.float32)
        size_tensor = torch.tensor(np.array(size), dtype=torch.float32)
        quats_tensor = torch.tensor(np.array(quaternion_components), dtype=torch.float32)

        return {
            "class": class_name_tensor,
            "center": center_tensor,
            "size": size_tensor,
            "quats": quats_tensor,
        }


def _load_sweep_chunk(
    zod_frame: ZodFrame, nsweeps: int, random_reference: bool, return_num_sweeps: bool = False,
) -> LidarData | tuple[LidarData, int]:
    """``nsweeps`` consecutive sweeps of the frame in the lidar frame of the reference: random, or the key sweep in the middle."""
    lidar_frames = zod_frame.info.get_lidar_frames(Lidar.VELODYNE)
    if nsweeps > 1:
        # The last sweep's core time is ~30 ms past the frame's ego-motion poses, so it can't be compensated.
        lidar_frames = lidar_frames[:-1]
    if random_reference:
        start, end, reference = sample_sweep_chunk(len(lidar_frames), nsweeps)
    else:
        keyframe = lidar_frames.index(zod_frame.info.get_key_lidar_frame())
        start, end, reference = centered_sweep_chunk(len(lidar_frames), nsweeps, keyframe)
    scans = [lidar_frame.read() for lidar_frame in lidar_frames[start:end]]
    if any(scan.timestamps.size == 0 for scan in scans):
        # zod's _adjust_lidar_core_time takes timestamps.max(), which fails on an empty scan.
        raise NoPointsException(f"Empty lidar scan in frame {zod_frame.info.id}.")
    for scan in scans:
        # Frames sweeps are not pointwise compensated; ZodFrame.get_aggregated_lidar applies the same fix.
        _adjust_lidar_core_time(scan)
    calibration = zod_frame.calibration.lidars[Lidar.VELODYNE]
    target_timestamp = scans[reference - start].core_timestamp
    scans = [
        scan if i == reference - start
        else motion_compensate_scanwise(scan, zod_frame.ego_motion, calibration, target_timestamp)
        for i, scan in enumerate(scans)
    ]
    aggregated = scans[0]
    aggregated.extend(*scans[1:])
    return (aggregated, end - start) if return_num_sweeps else aggregated


def _zero_pad_point_cloud(point_cloud):
    """Zero-pad pointclouds to fixed length.

    Parameters:
    Point cloud, tensor of shape (N, C)
    Max_points, int of maximum points in a oint cloud

    Returns:
    Padded point cloud, tensor of shape (M, C) where M is maximum number of points
    Number of points in the input cloud, int
    """
    point_dim = point_cloud.shape[-1]
    num_original_points = point_cloud.shape[0]
    if point_cloud.shape[0] > MAX_NUM_POINTS:
        point_excess = num_original_points - MAX_NUM_POINTS  # How many points to remove
        indices = torch.randperm(num_original_points)[:point_excess]  # cloud idxs to random order -> select first excess (all unique)
        mask = torch.ones(num_original_points, dtype=torch.bool)
        mask[indices] = False  # set all indexes to remove to False
        padded_point_cloud = point_cloud[mask]

    elif point_cloud.shape[0] < MAX_NUM_POINTS:
        padded_point_cloud = point_cloud.new_zeros((MAX_NUM_POINTS, point_dim))
        padded_point_cloud[:num_original_points] = point_cloud

    else:
        padded_point_cloud = point_cloud  # already the perfect length:)

    return padded_point_cloud


def _zero_pad_point_cloud_with_labels(
    point_cloud: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad/trim a cloud to MAX_NUM_POINTS, keeping per-point labels aligned.

    ``_zero_pad_point_cloud`` draws its own random subset when a cloud is too
    large, so labels cannot be padded independently without losing the
    correspondence. Both are therefore indexed with a single shared selection.

    Args:
        point_cloud: (N, C) cloud.
        labels: (N, 2) per-point [semantic, instance] ids.

    Returns:
        ``(padded_cloud, padded_labels)`` of shapes (MAX_NUM_POINTS, C) and
        (MAX_NUM_POINTS, 2). Padding rows are zero, i.e. the ignore class.
    """
    num_points = point_cloud.shape[0]
    if num_points > MAX_NUM_POINTS:
        keep = torch.randperm(num_points)[:MAX_NUM_POINTS]
        keep, _ = torch.sort(keep)  # keep sweep order
        return point_cloud[keep], labels[keep]

    padded_cloud = torch.zeros((MAX_NUM_POINTS, point_cloud.shape[1]), dtype=point_cloud.dtype)
    padded_labels = torch.zeros((MAX_NUM_POINTS, labels.shape[1]), dtype=labels.dtype)
    padded_cloud[:num_points] = point_cloud
    padded_labels[:num_points] = labels
    return padded_cloud, padded_labels


def _reshape_annotations_dict(annotations: dict) -> dict:
    """Zero-pad annotations to fixed length.

    Parameters:
        annotations, dict of tensors of shape (N,...)

    Returns:
        annotations, dict of tensors of shape (M,...) where M is maximum number of objects
    """
    num_original_obj = annotations["center"].shape[0]
    if num_original_obj > MAX_NUM_OBJECTS:
        annotation_excess = num_original_obj - MAX_NUM_OBJECTS  # How many points to remove
        indices = torch.randperm(num_original_obj)[:annotation_excess]  # cloud idxs to random order -> select first excess (all unique)
        mask = torch.ones(num_original_obj, dtype=torch.bool)
        mask[indices] = False  # set all indexes to remove to False
        for key in annotations:
            annotations[key] = annotations[key][mask]
    else:
        for key in annotations:
            if key == "class":
                padded_tensor = torch.zeros(size=(MAX_NUM_OBJECTS,), dtype=torch.int32)
            elif key == "num_points":
                padded_tensor = torch.zeros(size=(MAX_NUM_OBJECTS,), dtype=torch.int32)
            elif key in ["center", "size"]:
                padded_tensor = torch.zeros(size=(MAX_NUM_OBJECTS, 3))
            elif key == "quats":
                padded_tensor = torch.zeros(size=(MAX_NUM_OBJECTS, 4))
            elif key == "velocity":
                padded_tensor = torch.zeros(size=(MAX_NUM_OBJECTS, 2))
            else:
                raise ValueError(f"Unknown annotation key: {key}")
            if num_original_obj > 0:
                padded_tensor[:num_original_obj] = annotations[key]
            annotations[key] = padded_tensor
    return annotations


class TorchList:
    """Python list stored as serialized objects in a torch tensor.

    Using this in data loaders will avoid memory copies between the dataloader workers.
    """

    def __init__(self, python_list: list):
        """Initialize."""
        serialized_list = [pickle.dumps(d, protocol=-1) for d in python_list]
        addresses = [len(s) for s in serialized_list]

        # Use np.frombuffer instead of torch.frombuffer directly to avoid warnings about nonwritable
        # buffers or arrays.
        serialized_list = [np.frombuffer(s, dtype=np.uint8) for s in serialized_list]
        serialized_list = np.concatenate(serialized_list)
        self._serialized_list = torch.from_numpy(serialized_list)

        self._addresses = torch.tensor(addresses, dtype=torch.int64)
        self._addresses = torch.cumsum(self._addresses, dim=-1)

    def __len__(self):
        """Length."""
        return len(self._addresses)

    def __getitem__(self, index):
        """Get item."""
        start_address = 0 if index == 0 else int(self._addresses[index - 1])
        end_address = int(self._addresses[index])
        list_entry_bytes = memoryview(self._serialized_list[start_address:end_address].numpy())
        return pickle.loads(list_entry_bytes)

class NoPointsException(Exception):
    """Exception for when no points are left after filtering."""
    