"""Precompute per-point segmentation labels for Cirrus.

Cirrus annotates 3D boxes on its gaussian scans but has no per-point labels. Like for ZOD, Argoverse 2, TruckScenes
and KITTI, every point inside a box gets the box's class and a per-frame instance id, everything else is background.

One `.npz` per annotated gaussian scan (train and val) under `CIRRUS_LABELS_DIR`, as `dataset<k>/<timestamp_ns>.npz`,
with:

    semantic  (N,) uint8  0 = background, 1..C = `cirrus_dataset.CLASS_NAME_TO_ID`.
    instance  (N,) int16  0 = background, otherwise the 1-based ordinal of the box within its frame.

Both in the raw point order of the `.xyz` file, which `CirrusDataset` loads.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from constants import CIRRUS_LABELS_DIR  # noqa: E402
from data.cirrus_dataset import CLASS_NAME_TO_ID, _split_frames, label_path, read_boxes, read_scan  # noqa: E402
from scripts.precompute_kitti_labels import _pointwise_labels  # noqa: E402

MAX_SEMANTIC_ID = np.iinfo(np.uint8).max
MAX_INSTANCE_ID = np.iinfo(np.int16).max


def labels_for_frame(sequence: int, stamp: str, device: torch.device) -> dict:
    """The two cached arrays for one annotated gaussian scan."""
    points = read_scan(sequence, "gaussian", stamp)[:, :3]
    # Every box of the frame: unlike `_getitem` no PC_RANGE filter and no MAX_NUM_OBJECTS subsample, so a point is
    # never labelled background just because its box was dropped.
    annotations = read_boxes(sequence, stamp)
    num_boxes = annotations["center"].shape[0]
    assert num_boxes <= MAX_INSTANCE_ID, f"{num_boxes} boxes overflow the instance dtype"

    semantic = np.zeros(points.shape[0], dtype=np.uint8)
    instance = np.zeros(points.shape[0], dtype=np.int16)
    if num_boxes > 0:
        semantic[:], instance[:] = _pointwise_labels(torch.from_numpy(points).to(device), annotations)
    return {"semantic": semantic, "instance": instance}


def main(args):
    assert len(CLASS_NAME_TO_ID) <= MAX_SEMANTIC_ID, "Class ids overflow the semantic dtype"
    device = torch.device(args.device)

    frames = [frame for split in args.splits for frame in _split_frames(split)]
    # Stride rather than block sharding so every shard gets a similar mix of frames.
    frames = frames[args.shard_index :: args.num_shards]
    print(f"{args.splits}: this shard handles {len(frames)} frames, writing to {args.output_directory}", flush=True)

    written, skipped = 0, 0
    for progress, frame in enumerate(frames):
        out_path = label_path(frame["sequence"], frame["stamp"], args.output_directory)
        if os.path.exists(out_path) and not args.overwrite:
            skipped += 1
            continue
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        labels = labels_for_frame(frame["sequence"], frame["stamp"], device)
        np.savez_compressed(out_path, **labels)
        written += 1
        if progress % args.log_every == 0:
            print(
                f"  {progress + 1}/{len(frames)} dataset{frame['sequence']}/{frame['stamp']}: {labels['semantic'].size} points, "
                f"{int(labels['semantic'].astype(bool).sum())} on objects",
                flush=True,
            )
    print(f"wrote {written}, skipped {skipped} already present", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", nargs="+", choices=["train", "val"], default=["train", "val"], help="Splits of the annotated frames (see cirrus_dataset).")
    parser.add_argument("--output_directory", type=str, default=CIRRUS_LABELS_DIR, help="Where CirrusDataset reads the labels from (CIRRUS_LABELS_DIR).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", help="Recompute frames already cached.")
    parser.add_argument("--log_every", type=int, default=500, help="Frames between progress lines.")

    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard_index must be in [0, --num_shards)")
    return args


if __name__ == "__main__":
    main(parse_args())
