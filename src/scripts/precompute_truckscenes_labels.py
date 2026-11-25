"""Precompute per-point segmentation labels for TruckScenes.

TruckScenes annotates 3D boxes at the 2 Hz key frames but has no per-point labels. Like for ZOD and
Argoverse 2, every point inside a box gets the box's class and a per-frame instance id, everything else
is background. Annotations cover the full 360 degrees.

Labels are stored per key-frame LiDAR *file* and concatenated in the order `TruckScenesDataset` merges
the six LiDARs. One `.npz` per file under `TRUCKSCENES_LABELS_DIR`, at the .pcd's relative path
(`samples/<channel>/<channel>_<timestamp>.npz`), with:

    in_cone   (N,) bool   always all-True (360 degree annotations), for schema parity with ZOD.
    semantic  (N,) uint8  0 = background, 1..C = `truckscenes_dataset.CLASS_NAME_TO_ID`.
    instance  (N,) int16  0 = background, otherwise the 1-based ordinal of the box within its key frame,
                          so the same box has the same id in every LiDAR file of that key frame.

All in the raw .pcd point order. Points are placed like `TruckScenesDataset` does (ego pose at each point's
timestamp) and tested against the boxes in the ego frame of their file. With that, the per-box point counts
match the release's `num_lidar_pts` far better than with one pose per file (median ratio 1.02 vs 0.85 on a
25 m/s highway scene); they are not exact, as the release's own counting is undocumented.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from constants import TRUCKSCENES_LABELS_DIR  # noqa: E402
from data.truckscenes_dataset import (  # noqa: E402
    CLASS_NAME_TO_ID,
    LIDAR_CHANNELS,
    label_path,
    load_file,
    load_scenes,
    pose_matrix,
    scene_annotations,
)

MAX_SEMANTIC_ID = np.iinfo(np.uint8).max
MAX_INSTANCE_ID = np.iinfo(np.int16).max


def _pointwise_labels(positions: torch.Tensor, annotations: dict) -> tuple[np.ndarray, np.ndarray]:
    """Class and 1-based box ordinal of the box each point falls inside, 0 outside all boxes.

    Same oriented-box test and last-box-wins tie-break for overlapping boxes as the ZOD/Argoverse 2
    label scripts.
    """
    device = positions.device
    centers = annotations["center"].to(device)
    half_sizes = annotations["size"].to(device) / 2  # [length, width, height] along the box's x, y, z
    w, x, y, z = annotations["quats"].to(device).unbind(-1)
    rotations = torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], dim=-1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], dim=-1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], dim=-1),
    ], dim=1)  # (M, 3, 3) box-to-ego
    local = torch.einsum("mji,mnj->mni", rotations, positions.unsqueeze(0) - centers.unsqueeze(1))
    inside = (local.abs() <= half_sizes.unsqueeze(1)).all(dim=-1)  # (M, N)

    semantic = torch.zeros(positions.shape[0], dtype=torch.long, device=device)
    instance = torch.zeros_like(semantic)
    for box, class_id in enumerate(annotations["class"].tolist()):
        semantic[inside[box]] = class_id
        instance[inside[box]] = box + 1
    return semantic.cpu().numpy(), instance.cpu().numpy()


def labels_for_file(scene: dict, channel: str, idx: int, device: torch.device) -> dict:
    """The three cached arrays for one key-frame LiDAR file."""
    record = scene["lidar"][channel]
    frame = (int(record["timestamp"][idx]), pose_matrix(record["pose"][idx]))
    ego_from_world = np.linalg.inv(frame[1])
    points = load_file(scene, channel, idx, frame, min_distance=0.0)["xyz"]
    num_points = points.shape[0]

    # Every box of the key frame: unlike `_getitem` no PC_RANGE filter and no MAX_NUM_OBJECTS subsample,
    # so a point is never labelled background just because its box was dropped.
    annotations = scene_annotations(scene, int(record["sample"][idx]), ego_from_world)
    num_boxes = annotations["center"].shape[0]
    assert num_boxes <= MAX_INSTANCE_ID, f"{num_boxes} boxes overflow the instance dtype"

    semantic = np.zeros(num_points, dtype=np.uint8)
    instance = np.zeros(num_points, dtype=np.int16)
    if num_boxes > 0:
        semantic[:], instance[:] = _pointwise_labels(torch.from_numpy(points).float().to(device), annotations)
    return {"in_cone": np.ones(num_points, dtype=bool), "semantic": semantic, "instance": instance}


def main(args):
    assert len(CLASS_NAME_TO_ID) <= MAX_SEMANTIC_ID, "Class ids overflow the semantic dtype"
    device = torch.device(args.device)

    for split in args.splits:
        scenes = load_scenes(split, args.version)
        if args.scenes:
            scenes = [scene for scene in scenes if scene["name"] in args.scenes]
        # Stride rather than block sharding so every shard gets a similar mix of scenes.
        scenes = scenes[args.shard_index :: args.num_shards]
        print(f"[{split}] this shard handles {len(scenes)} scenes, writing to {args.output_directory}", flush=True)

        written, skipped = 0, 0
        for progress, scene in enumerate(scenes):
            annotated_points, scene_written = 0, 0
            for channel in LIDAR_CHANNELS:
                record = scene["lidar"][channel]
                for idx in np.flatnonzero(record["is_key_frame"]):
                    out_path = label_path(record["filename"][idx], args.output_directory)
                    if os.path.exists(out_path) and not args.overwrite:
                        skipped += 1
                        continue
                    labels = labels_for_file(scene, channel, int(idx), device)
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    np.savez_compressed(out_path, **labels)
                    annotated_points += int(labels["semantic"].astype(bool).sum())
                    scene_written += 1
            written += scene_written

            if progress % args.log_every == 0:
                print(
                    f"  [{split}] {progress + 1}/{len(scenes)} {scene['name']} {scene_written} files, "
                    f"{annotated_points / max(scene_written, 1):.0f} points on objects per file",
                    flush=True,
                )
        print(f"[{split}] wrote {written}, skipped {skipped} already present", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", nargs="+", choices=["train", "val"], default=["train", "val"], help="Only these are annotated.")
    parser.add_argument("--version", choices=["full", "mini"], default="full")
    parser.add_argument("--output_directory", type=str, default=TRUCKSCENES_LABELS_DIR, help="Where TruckScenesDataset reads the labels from (TRUCKSCENES_LABELS_DIR).")
    parser.add_argument("--scenes", nargs="+", default=None, help="Only label these scene names (for spot checks).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", help="Recompute files already cached.")
    parser.add_argument("--log_every", type=int, default=10, help="Scenes between progress lines.")

    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard_index must be in [0, --num_shards)")
    return args


if __name__ == "__main__":
    main(parse_args())
