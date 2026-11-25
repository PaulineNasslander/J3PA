"""Plot probe metrics over voxel size for every voxel_size_sweep.py run found under the given paths.

Example:
    python src/evaluation/plot_voxel_size_sweep.py experiments/vs_sweep --metrics mIoU allAcc --output vs_sweep.png
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_FILE = "vs_sweep_results.json"


def find_results(paths: list[str]) -> list[dict]:
    """Load every results file given directly or found recursively under a directory."""
    files = []
    for path in map(Path, paths):
        files.extend(sorted(path.rglob(RESULTS_FILE)) if path.is_dir() else [path])
    runs = []
    for file in files:
        with open(file) as f:
            runs.append(json.load(f))
    return runs


def main(args):
    runs = find_results(args.paths)
    if not runs:
        raise FileNotFoundError(f"No {RESULTS_FILE} under {args.paths}")

    fig, axes = plt.subplots(1, len(args.metrics), figsize=(6 * len(args.metrics), 4.5), squeeze=False)
    for ax, metric in zip(axes[0], args.metrics):
        for run in runs:
            results = sorted(run["results"], key=lambda r: r["voxel_size"])
            voxel_sizes_cm = [r["voxel_size"] * 100 for r in results]
            line, = ax.plot(voxel_sizes_cm, [r[metric] for r in results], marker="o", label=run["label"])
            if run.get("train_voxel_size") is not None:
                # The voxel size the run validated at (and its teacher used).
                ax.axvline(run["train_voxel_size"] * 100, color=line.get_color(), linestyle=":", alpha=0.6)
        ax.set_xscale("log")
        ax.set_xlabel("voxel size [cm]")
        ax.set_ylabel(metric)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_title(f"{metric} vs voxel size")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize="small", loc="lower center", ncol=1, frameon=False)
    fig.tight_layout(rect=(0, 0.06 * len(runs), 1, 1))
    fig.savefig(args.output, dpi=150)
    print(f"Wrote {args.output} with {len(runs)} run(s)")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help=f"Directories to search for {RESULTS_FILE}, or the files themselves.")
    parser.add_argument("--metrics", nargs="+", default=["mIoU"], help="Any scalar in the results, e.g. mIoU mAcc allAcc.")
    parser.add_argument("--output", type=str, default="vs_sweep.png")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
