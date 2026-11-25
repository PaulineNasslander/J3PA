"""Precompute per-point segmentation labels for Argoverse 2.

Argoverse 2 annotations include 3D object cuboids but no per-point labels. Unlike
ZOD the cuboids cover the full 360 degrees around the ego vehicle, so -- apart
from the unannotated `lidar/*` and `test` logs, which this script skips -- every
point of an annotated sweep has ground truth.

This script precomputes, once per sweep and caches to `ARGOVERSE2_LABELS_DIR`
as `<split>/<log_id>/<timestamp_ns>.npz`, the following:

    in_cone   (N,) bool   points that carry ground truth. Always all-True here,
                          since Argoverse 2 annotates the full sweep. Kept so
                          this cache has the same schema as the ZOD one.
    semantic  (N,) uint8  0 = background, 1..C = `argoverse2_dataset.CLASS_NAME_TO_ID`.
    instance  (N,) int16  0 = background, otherwise a per-frame box ordinal.

All three are in the *raw* sweep order, before the PC_RANGE filter that
`Argoverse2Dataset._getitem` applies.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from constants import ARGOVERSE2_LABELS_DIR, XYZ_SLICE
from data.argoverse2_dataset import (
    CLASS_NAME_TO_ID,
    DATASET_ROOT,
    _annotation_path,
    _cuboids_to_annotations,
    _list_sequences,
    _load_sweep,
)
from utils.utils import get_pointwise_labels

# The saved dtypes have to hold every label we can produce.
MAX_SEMANTIC_ID = np.iinfo(np.uint8).max
MAX_INSTANCE_ID = np.iinfo(np.int16).max

EMPTY_ANNOTATIONS = {
    "class": torch.zeros(0, dtype=torch.int32),
    "center": torch.zeros(0, 3),
    "size": torch.zeros(0, 3),
    "quats": torch.zeros(0, 4),
}


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
    # change class to instance id, 1..num_boxes. Background is still 0.
    batched["class"] = torch.arange(1, num_boxes + 1, device=positions.device).unsqueeze(0)
    instance = get_pointwise_labels(positions, batch_indices, batched)

    return semantic.cpu().numpy(), instance.cpu().numpy()


def labels_for_frame(sweep_path: str, annotations: dict, device: torch.device) -> dict:
    """Compute the three cached arrays for one sweep."""
    point_cloud = _load_sweep(sweep_path)
    num_points = point_cloud.shape[0]

    semantic = np.zeros(num_points, dtype=np.uint8)
    instance = np.zeros(num_points, dtype=np.int16)

    # Every cuboid of the sweep. Unlike `_getitem` this keeps boxes whose centre
    # falls outside PC_RANGE and skips the MAX_NUM_OBJECTS subsample, so a point
    # is never labelled background just because its box was dropped.
    num_boxes = annotations["center"].shape[0]
    assert num_boxes <= MAX_INSTANCE_ID, f"{num_boxes} boxes overflow the instance dtype"
    if num_boxes > 0:
        semantic[:], instance[:] = _pointwise_labels(point_cloud[:, XYZ_SLICE].to(device), annotations)

    return {
        # Argoverse 2 annotates the full 360 degrees, so nothing is outside the
        # annotated region. The key stays for schema parity with the ZOD cache.
        "in_cone": np.ones(num_points, dtype=bool),
        "semantic": semantic,
        "instance": instance,
    }


def _annotations_by_timestamp(sequence: str) -> dict[int, dict]:
    """Read a log's cuboid table once and split it per sweep.

    The table holds every sweep of the log, so reading it per sweep -- the way
    `Argoverse2Dataset` does -- would re-parse it ~157 times per log.
    """
    cuboids = pd.read_feather(_annotation_path(sequence), use_threads=False)
    return {
        int(timestamp): _cuboids_to_annotations(group)
        for timestamp, group in cuboids.groupby("timestamp_ns")
    }


def main(args):
    assert len(CLASS_NAME_TO_ID) <= MAX_SEMANTIC_ID, "Class ids overflow the semantic dtype"
    device = torch.device(args.device)

    for split in args.splits:
        sequences = _list_sequences([split], None)
        # Sharding by stride rather than by block so every shard sees a mix of
        # log lengths and the jobs finish at roughly the same time.
        sequences = sequences[args.shard_index :: args.num_shards]
        print(f"[{split}] this shard handles {len(sequences)} logs", flush=True)

        written, skipped = 0, 0
        for progress, sequence in enumerate(sequences):
            output_directory = os.path.join(args.output_directory, sequence)
            os.makedirs(output_directory, exist_ok=True)

            lidar_directory = os.path.join(DATASET_ROOT, sequence, "sensors", "lidar")
            sweep_files = sorted(os.listdir(lidar_directory))
            todo = [
                sweep_file
                for sweep_file in sweep_files
                if args.overwrite
                or not os.path.exists(
                    os.path.join(output_directory, sweep_file.replace(".feather", ".npz"))
                )
            ]
            skipped += len(sweep_files) - len(todo)
            if not todo:
                continue

            annotations_by_timestamp = _annotations_by_timestamp(sequence)
            annotated_points = 0
            for sweep_file in todo:
                timestamp_ns = int(sweep_file.removesuffix(".feather"))
                labels = labels_for_frame(
                    os.path.join(lidar_directory, sweep_file),
                    # Every sweep is annotated in practice, but an empty table
                    # would silently mean "all background" rather than a crash.
                    annotations_by_timestamp.get(timestamp_ns, EMPTY_ANNOTATIONS),
                    device,
                )
                # Background dominates, so compression takes these from ~600 kB to
                # a few tens of kB per sweep.
                np.savez_compressed(
                    os.path.join(output_directory, f"{timestamp_ns}.npz"), **labels
                )
                annotated_points += int(labels["semantic"].astype(bool).sum())
                written += 1

            if progress % args.log_every == 0:
                print(
                    f"  [{split}] {progress + 1}/{len(sequences)} {sequence} "
                    f"{len(todo)} sweeps, {annotated_points / len(todo):.0f} points "
                    f"on objects per sweep",
                    flush=True,
                )

        print(f"[{split}] wrote {written}, skipped {skipped} already present", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val"],
        default=["train", "val"],
        help="Only these two splits are annotated.",
    )
    parser.add_argument("--output_directory", type=str, default=ARGOVERSE2_LABELS_DIR)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", help="Recompute sweeps already cached.")
    parser.add_argument("--log_every", type=int, default=10, help="Logs between progress lines.")

    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard_index must be in [0, --num_shards)")
    return args


if __name__ == "__main__":
    main(parse_args())
