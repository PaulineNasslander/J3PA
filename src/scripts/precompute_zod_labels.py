"""Precompute per-point segmentation labels for ZOD.

ZOD annotations include 3D object boxes but no per-point labels, and the boxes only 
cover the front camera's field of view. 

This script precomputes, once per frame and caches to `ZOD_LABELS_DIR`, the following:

    in_cone   (N,) bool   points inside the annotated cone; everything outside
                          has no ground truth and must be ignored by supervision 
                          signal and evaluation.
    semantic  (N,) uint8  0 = background, 1..C = `zod_dataset.CLASS_NAME_TO_ID`.
                            Every point outside the cone is labelled background.
    instance  (N,) int16  0 = background, otherwise a per-frame box ordinal.
                            Every point outside the cone is labelled background.

All three are in the *raw* lidar point order, before any filter 
that `ZODdataset._getitem` applies. 
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from zod.constants import AnnotationProject, Camera
from zod.zod_frames import ZodFrame

sys.path.insert(0, str(Path(__file__).parent.parent))

from constants import USE_TIME, XYZ_SLICE, ZOD_LABELS_DIR
from data.zod_dataset import CLASS_NAME_TO_ID, ZODdataset
from evaluation.probe_utils import limit_pc_to_fov
from utils.utils import get_pointwise_labels

# The saved dtypes have to hold every label we can produce.
MAX_SEMANTIC_ID = np.iinfo(np.uint8).max
MAX_INSTANCE_ID = np.iinfo(np.int16).max


def _raw_point_cloud(zod_frame: ZodFrame) -> torch.Tensor:
    """Build the configured point features exactly as the loader does, uncropped."""
    lidar_data = zod_frame.get_lidar()
    features = [
        torch.from_numpy(lidar_data[0].points).float(),
        torch.from_numpy(lidar_data[0].intensity).float().unsqueeze(1),
    ]
    if USE_TIME:
        timestamps = lidar_data[0].timestamps
        timestamps = timestamps.max() - timestamps
        features.insert(0, torch.from_numpy(timestamps).float().unsqueeze(1))
    return torch.cat(features, dim=1)


def _in_annotated_cone(point_cloud: torch.Tensor, field_of_view: torch.Tensor) -> torch.Tensor:
    """Boolean mask of the points ZOD annotates, i.e. the front camera cone."""
    limited = limit_pc_to_fov(point_cloud.clone().unsqueeze(0), field_of_view.unsqueeze(0))
    return torch.any(limited[0] != 0, dim=-1)


def _pointwise_labels(positions: torch.Tensor, annotations: dict) -> tuple[np.ndarray, np.ndarray]:
    """Class and instance id of the box each point falls inside.

    `get_pointwise_labels` already implements the oriented
    box containment test. Calling it a second time with the class ids swapped for
    box ordinals reuses that identical test -- and its identical last-box-wins
    tie-break for overlapping boxes -- so the two label maps can never disagree
    about which box a point belongs to.

    Args:
        positions: (K, 3) point positions.
        annotations: unbatched dict of 'center' (M,3), 'size' (M,3), 'quats'
            (M,4) and 'class' (M,).

    Returns:
        ``(semantic, instance)``, both (K,) numpy arrays with 0 for background.
    """
    batch_indices = torch.zeros(positions.shape[0], dtype=torch.long, device=positions.device)
    batched = {key: value.to(positions.device).unsqueeze(0) for key, value in annotations.items()}
    semantic = get_pointwise_labels(positions, batch_indices, batched)

    num_boxes = batched["center"].shape[1]
    #change class to instance id , 1..num_boxes. Background is still 0.
    batched["class"] = torch.arange(1, num_boxes + 1, device=positions.device).unsqueeze(0)
    instance = get_pointwise_labels(positions, batch_indices, batched)

    return semantic.cpu().numpy(), instance.cpu().numpy()


def labels_for_frame(zod_frame: ZodFrame, dataset: ZODdataset, device: torch.device) -> dict:
    """Compute the three cached arrays for one frame."""
    point_cloud = _raw_point_cloud(zod_frame)
    field_of_view = (
        torch.from_numpy(zod_frame.calibration.cameras[Camera.FRONT].field_of_view)
        .to(torch.float32)
        .squeeze(0)
    )  # horizontal, vertical (degrees)
    in_cone = _in_annotated_cone(point_cloud, field_of_view)

    # Every box in the frame. Unlike `_getitem` this keeps boxes whose centre
    # falls outside PC_RANGE and skips the MAX_NUM_OBJECTS subsample, so a point
    # is never labelled background just because its box was dropped.
    annotations = dataset.get_annotations(
        zod_frame.get_annotation(AnnotationProject.OBJECT_DETECTION)
    )

    num_points = point_cloud.shape[0]
    semantic = np.zeros(num_points, dtype=np.uint8)
    instance = np.zeros(num_points, dtype=np.int16)

    num_boxes = annotations["center"].shape[0]
    assert num_boxes <= MAX_INSTANCE_ID, f"{num_boxes} boxes overflow the instance dtype"
    if num_boxes > 0:
        # Only the cone can contain annotated points, and restricting the test to
        # it keeps the (num_boxes, num_points, 3) intermediate a few times smaller.
        cone = in_cone.numpy()
        cone_semantic, cone_instance = _pointwise_labels(
            point_cloud[in_cone, XYZ_SLICE].to(device), annotations
        )
        semantic[cone] = cone_semantic
        instance[cone] = cone_instance

    return {"in_cone": in_cone.numpy(), "semantic": semantic, "instance": instance}


def main(args):
    assert len(CLASS_NAME_TO_ID) <= MAX_SEMANTIC_ID, "Class ids overflow the semantic dtype"
    device = torch.device(args.device)
    os.makedirs(args.output_directory, exist_ok=True)

    for split in args.splits:
        dataset = ZODdataset(
            train_flag=split == "train", version=args.version, load_annotations=True, mp=False
        )
        # Sharding by stride rather than by block so every shard sees a mix of
        # frame sizes and the jobs finish at roughly the same time.
        indices = range(args.shard_index, len(dataset), args.num_shards)
        print(f"[{split}] {len(dataset)} frames, this shard handles {len(indices)}")

        written, skipped = 0, 0
        for progress, index in enumerate(indices):
            zod_frame = ZodFrame(info=dataset._frames[index])
            output_path = os.path.join(args.output_directory, f"{zod_frame.info.id}.npz")
            if os.path.exists(output_path) and not args.overwrite:
                skipped += 1
                continue

            labels = labels_for_frame(zod_frame, dataset, device)
            # Background dominates, so compression takes these from ~800 kB to a
            # few tens of kB per frame.
            np.savez_compressed(output_path, **labels)
            written += 1

            if progress % args.log_every == 0:
                annotated = labels["semantic"].astype(bool).sum()
                print(
                    f"  [{split}] {progress + 1}/{len(indices)} {zod_frame.info.id} "
                    f"{labels['in_cone'].sum()} in cone, {annotated} on objects"
                )

        print(f"[{split}] wrote {written}, skipped {skipped} already present")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--version", choices=["mini", "full"], default="full")
    parser.add_argument("--splits", nargs="+", choices=["train", "val"], default=["train", "val"])
    parser.add_argument("--output_directory", type=str, default=ZOD_LABELS_DIR)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", help="Recompute frames already cached.")
    parser.add_argument("--log_every", type=int, default=100)

    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard_index must be in [0, --num_shards)")
    return args


if __name__ == "__main__":
    main(parse_args())
