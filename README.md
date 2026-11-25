# J3PA

Official project repository of the paper "J3PA: A Predictive Approach to Self-Supervised 3D Representation Learning on Point Clouds". It provides research code for self-supervised learning on LiDAR point clouds via JEPA. The repo includes teacher-to-student (self) distillation, semantic and panoptic probe evaluation, and scripts for reproducing feature
visualizations.

The code does not include datasets. Obtain each dataset under its own terms of use. 

Model checkpoints are found and downloaded [here](https://huggingface.co/collections/Zenseact/j3pa).

## Installation

The image defined in [Dockerfile](./Dockerfile) contains the complete environment needed for pretraining J3PA.

Build the image from the repository root:

```bash
docker build -f Dockerfile -t j3pa:latest .
```

Or install into a local `.venv` with [uv](https://docs.astral.sh/uv/). 
```bash
uv sync
source .venv/bin/activate
```


## Data locations

Dataset paths can be set with environment variables. Otherwise paths default to `/datasets`:

| Dataset | Environment variable |
| --- | --- |
| Argoverse 2 | `ARGOVERSE2_DATASET_ROOT` |
| Cirrus | `CIRRUS_DATASET_ROOT` |
| KITTI | `KITTI_DATASET_ROOT` |
| nuScenes | `NUSCENES_DATASET_ROOT` |
| OctoSense | `OCTOSENSE_DATASET_ROOT` |
| PandaSet | `PANDASET_DATASET_ROOT` |
| SemanticKITTI | `SEMANTICKITTI_DATASET_ROOT` |
| TruckScenes | `TRUCKSCENES_DATASET_ROOT` |
| Waymo | `WAYMO_DATASET_ROOT` |
| ZOD | `ZOD_DATASET_ROOT` |

For evaluation, datasets that do not include per-point labels but do have object detection annotations need generated per-point label caches. Their locations are configured with
`ZOD_LABELS_DIR`, `ARGOVERSE2_LABELS_DIR`, `TRUCKSCENES_LABELS_DIR`, and `CIRRUS_LABELS_DIR`.
Generate the label caches with `src/scripts/precompute_*_labels.py`.

## Training
Change settings before you launch the training by modifying the [constants.py](./src/constants.py) file.

Example run command for self-distillation training:
```bash
src/train_jepa.py -o experiments/j3pa_zod --dataset ZOD --torch_ddp
```

## Evaluation

`run_eval.py` runs the simple single-view protocol. `run_eval_bellsandwhistels.py` runs the augmented evaluation
protocol. 

```bash
src/evaluation/run_eval.py \
  --checkpoint /models/teacher/checkpoints/epoch_100.pt \
  --constants /models/teacher/constants.py \
  --output_directory /runs/eval_simple \
  --dataset ZOD \
  --semantic_segmentation --linear_probe

src/evaluation/run_eval_bellsandwhistels.py \
  --checkpoint /models/teacher/checkpoints/epoch_100.pt \
  --constants /models/teacher/constants.py \
  --output_directory /runs/eval_augmented \
  --dataset ZOD \
  --semantic_segmentation --linear_probe
```

For multi-GPU evaluation, launch with `torchrun` and add `--torch_ddp`. 

Logs of metrics and qualitative figures are saved to TensorBoard under the output directory.

## License

The J3PA pretraining code in this repository is released under the [Apache License 2.0](./LICENSE). Third-party code included in this repository and its licenses are listed in [NOTICE](./NOTICE). 
