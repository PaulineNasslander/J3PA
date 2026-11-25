"""Probe semantic segmentation of a frozen j3pa encoder at several input voxel sizes.

For every voxel size a fresh linear (or with --nl_probe Linear-ReLU-Linear) head is trained on the frozen encoder run at that voxel size and
evaluated at the same voxel size. Results go to TensorBoard (``vs_sweep/<metric>`` with the voxel size in
mm as the step, so pointing TensorBoard at the parent directory overlays checkpoints) and to
``vs_sweep_results.json``, which plot_voxel_size_sweep.py reads to compare checkpoints.

Like run_eval, nothing in src/constants.py has to be changed for a run: its saved constants.py (or legacy constants_values.txt) is loaded
into the constants module before anything else is imported, except PROBE_* and the QK-LayerNorm backend so every run gets the same
probe and the current execution backend.

Checkpoints from code without voxel size inputs need that code's encoder: pass its src
directory as --code_dir, e.g. from ``git archive <revision> src | tar -x -C /path/to/legacy_code``.
Their encoder is run at another voxel size by overriding its fixed voxel size, which sets the grid and
the position normalisation like the voxel size argument of the current encoder does.

Example:
    PYTHONPATH=src python src/evaluation/voxel_size_sweep.py \
        --checkpoint /path/to/run/checkpoints/epoch_100.pt \
        --output_directory experiments/vs_sweep
"""

import argparse
import json
import os
import sys
from pathlib import Path


def _run_dir(checkpoint: str) -> Path:
    checkpoint = Path(checkpoint)
    return checkpoint.parent.parent if checkpoint.parent.name == "checkpoints" else checkpoint.parent


def _prepare_imports():
    """Import the project from --code_dir and load the run's constants, before anything below imports them."""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--code_dir", default=str(Path(__file__).resolve().parent.parent))
    pre_parser.add_argument("--checkpoint")
    pre_parser.add_argument("--constants")
    known, _ = pre_parser.parse_known_args()
    sys.path.insert(0, known.code_dir)
    if known.checkpoint is None:
        return

    import constants
    from evaluation.constants_loader import load_constants_file

    run_constants = load_constants_file(known.constants or str(_run_dir(known.checkpoint)))
    for name, value in run_constants.items():
        if hasattr(constants, name) and not name.startswith("PROBE_") and name != "ENC_QK_LAYERNORM_IMPL":
            setattr(constants, name, value)
    # The dump stores slices as an unparsable repr.
    if "XYZ_START" in run_constants:
        constants.XYZ_SLICE = slice(constants.XYZ_START, constants.XYZ_START + 3)


_prepare_imports()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

from constants import (  # noqa: E402
    PROBE_BATCH_SIZE,
    PROBE_HIDDEN_DIM,
    PROBE_LEARNING_RATE,
    PROBE_NUM_WORKERS,
    PROBE_SCHEDULER_MAX_STEPS,
    PROBE_SEED,
    XYZ_SLICE,
)
from evaluation.backbones import FROZEN, J3PABackbone  # noqa: E402
from evaluation.constants_loader import load_constants_file  # noqa: E402
from evaluation.run_eval import (  # noqa: E402
    SEMANTIC,
    ProbeModel,
    _build_head,
    _cycle,
    evaluate_segmentation,
    probe_feature_dim,
    train_head,
)
from evaluation.probe_utils import _cleanup_dataloader  # noqa: E402
from utils.multi_gpu_utils import get_gpu_device  # noqa: E402
from utils.utils import get_dataloader  # noqa: E402

RESULTS_FILE = "vs_sweep_results.json"


class VoxelSizeBackbone(nn.Module):
    """A frozen `J3PABackbone` whose encoder runs at ``self.voxel_size``."""

    def __init__(self, backbone: J3PABackbone):
        super().__init__()
        self.backbone = backbone
        self.voxel_size = backbone.voxel_size

    def forward(self, point_cloud, batch_indices):
        encoder = self.backbone.encoder
        if hasattr(encoder, "stage_voxel_sizes"):
            B = int(batch_indices.max()) + 1
            voxel_size = torch.full((B,), self.voxel_size, device=point_cloud.device, dtype=torch.float32)
            outputs = encoder(point_cloud, batch_indices, voxel_size=voxel_size)
        else:
            # Encoders without a voxel size argument use this for the grid and the position normalisation.
            encoder._voxel_size = self.voxel_size
            outputs = encoder(point_cloud, batch_indices)
        stage_features, stage_positions, stage_batch_indices, up_indices, _ = outputs
        return stage_features, stage_positions, stage_batch_indices, up_indices, point_cloud[..., XYZ_SLICE]


def _to_json(metrics: dict) -> dict:
    return {
        name: {k: float(v) for k, v in value.items()} if isinstance(value, dict) else float(value)
        for name, value in metrics.items()
    }


def main(args):
    device = torch.device(get_gpu_device())
    run_constants = load_constants_file(args.constants)
    output_dir = os.path.join(args.output_directory, args.label)
    os.makedirs(output_dir, exist_ok=True)

    # Read by train_head/evaluate_segmentation from run_eval.
    args.load_segmentation = True
    args.accumulation_steps = 1

    loader_kwargs = dict(
        version=args.version,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        load_annotations=True,  # rotate_cloud needs them
        dataset_type=args.dataset,
        load_segmentation=True,
    )
    train_loader, _ = get_dataloader(train_flag=True, shuffle=True, drop_last=True, seed=PROBE_SEED, **loader_kwargs)
    val_loader, _ = get_dataloader(train_flag=False, shuffle=False, drop_last=False, seed=PROBE_SEED, **loader_kwargs)
    args.class_name_to_id = train_loader.dataset.get_class_name_to_id()
    args.seg_class_name_to_id = train_loader.dataset.get_seg_class_name_to_id()
    args.seg_thing_ids = train_loader.dataset.get_seg_thing_ids()
    args.seg_ignore_id = train_loader.dataset.get_seg_ignore_id()
    args.num_seg_classes = len(args.seg_class_name_to_id) + 1  # + background/ignore at id 0

    backbone = VoxelSizeBackbone(
        J3PABackbone(checkpoint_path=args.checkpoint, constants_path=args.constants, use_teacher=not args.use_student)
    ).to(device=device)
    backbone.backbone.set_trainable(FROZEN)
    args.feature_dim, _ = probe_feature_dim(backbone, val_loader, args, device)
    if not any(abs(vs - backbone.voxel_size) < 1e-9 for vs in args.voxel_sizes):
        args.voxel_sizes = sorted(args.voxel_sizes + [backbone.voxel_size])

    summary = {
        "label": args.label,
        "checkpoint": args.checkpoint,
        "constants": args.constants,
        "pretrain_epoch": backbone.backbone.pretrain_epoch,
        "train_voxel_size": run_constants.get("VOXEL_SIZE"),
        "train_voxel_size_range": run_constants.get("VOXEL_SIZE_RANGE"),
        "num_steps": args.num_steps,
        "probe": "nl_probe" if args.nl_probe else "linear_probe",
        "dataset": args.dataset,
        "results": [],
    }
    writer = SummaryWriter(log_dir=output_dir)
    writer.add_text("config", "\n\n".join(f"{k}: {v}" for k, v in summary.items() if k != "results"))

    for voxel_size in args.voxel_sizes:
        voxel_size_mm = round(voxel_size * 1000)
        tag = f"vs_{voxel_size * 1000:.1f}mm"
        print(f"=== voxel size {voxel_size:.4f} m")
        backbone.voxel_size = voxel_size

        torch.manual_seed(PROBE_SEED)
        head = _build_head(SEMANTIC, args.feature_dim, args, hidden_dim=PROBE_HIDDEN_DIM if args.nl_probe else None).to(device=device)
        model = ProbeModel(backbone=backbone, head=head, task=SEMANTIC, ignore_logit=args.seg_ignore_id)
        optimizer = torch.optim.AdamW(head.parameters(), lr=PROBE_LEARNING_RATE)
        # Same schedule as the online probe, so step budgets are comparable.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, PROBE_SCHEDULER_MAX_STEPS, eta_min=PROBE_LEARNING_RATE * 1e-3)

        train_head(model, _cycle(train_loader), optimizer, scheduler, args, SEMANTIC, device, writer, tag, 0, args.num_steps, None)
        metrics = evaluate_segmentation(model, val_loader, args, SEMANTIC, device, writer, tag, args.num_steps)

        for name, value in metrics.items():
            if isinstance(value, dict):
                for class_name, class_value in value.items():
                    writer.add_scalar(f"vs_sweep_{name}/{class_name}", class_value, voxel_size_mm)
            else:
                writer.add_scalar(f"vs_sweep/{name}", value, voxel_size_mm)
        writer.flush()
        print("  " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items() if not isinstance(v, dict)))

        summary["results"].append({"voxel_size": voxel_size, **_to_json(metrics)})
        # Rewritten after every voxel size so a crashed sweep keeps what it finished.
        with open(os.path.join(output_dir, RESULTS_FILE), "w") as f:
            json.dump(summary, f, indent=2)

        del model, head, optimizer, scheduler
        torch.cuda.empty_cache()

    writer.close()
    _cleanup_dataloader(train_loader)
    _cleanup_dataloader(val_loader)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--constants", type=str, default=None, help="constants.py (or legacy constants_values.txt) of the run or its directory. Defaults to the run directory above checkpoints/.")
    parser.add_argument("--output_directory", type=str, required=True, help="Each sweep writes to <output_directory>/<label>.")
    parser.add_argument("--label", type=str, default=None, help="Name in the plot legend. Defaults to <run dir>_<checkpoint name>.")
    parser.add_argument("--code_dir", type=str, default=None, help="src directory to import the model code from, e.g. a master checkout for master checkpoints. Defaults to this src.")
    parser.add_argument("--voxel_sizes", type=float, nargs="+", default=np.geomspace(0.02, 0.1, 10).tolist(), help="Voxel sizes in meters. The run's own VOXEL_SIZE is always added.")
    parser.add_argument("--num_steps", type=int, default=1000, help="Probe training steps per voxel size.")
    parser.add_argument("--nl_probe", action="store_true", help="Linear-ReLU-Linear head instead of a linear one. A linear head mostly predicts background on ZOD.")
    parser.add_argument("--use_student", action="store_true", help="Use the student weights instead of the teacher/EMA weights.")
    parser.add_argument("--dataset", choices=["ZOD", "NuScenes", "Argoverse2", "TruckScenes"], default="ZOD")
    parser.add_argument("--version", choices=["mini", "full"], default="full")
    parser.add_argument("--batch_size", type=int, default=PROBE_BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=PROBE_NUM_WORKERS)
    parser.add_argument("--prefetch_factor", type=int, default=1)
    parser.add_argument("--limit_eval_batches", type=int, default=None, help="Evaluate only this many validation batches. Default is the whole split.")
    parser.add_argument("--num_vis_samples", type=int, default=0, help="Validation frames rendered to TensorBoard per voxel size.")
    args = parser.parse_args()

    run_dir = _run_dir(args.checkpoint)
    if args.constants is None:
        args.constants = str(run_dir)
    if args.label is None:
        args.label = f"{run_dir.name}_{Path(args.checkpoint).stem}"
    return args


if __name__ == "__main__":
    main(parse_args())
