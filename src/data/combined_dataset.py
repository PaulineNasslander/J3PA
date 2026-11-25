"""Dataset for combining multiple datasets into a single unified dataset.

To be used for pre-training where multiple datasets need to be accessed as a single dataset as no labels or other dataset specific information is required.

The dataset should load NUM_SAMPLES_PER_EPOCH samples in one epoch. This is the one “dispatcher dataset” (main dataset) that for each get item samples a dataset to draw 
from based on the proposed fractions. Then each dataset has its own PyTorch dataset where it keeps track of the scenes sampled from it and loops over all scenes such that 
each dataset picks samples with maximal diversity. When a new epoch starts, the sampling process continues from where it left off in a circular manner.

Proposed Fractions:
ZOD: 0.30 
Argoverse2: 0.25
NuScenes: 0.13
TruckScenes: 0.12
OctoSense: 0.11
PandaSet: 0.05
KITTI: 0.02
Cirrus: 0.02
"""

import numpy as np
from torch.utils.data import Dataset, Sampler

NUM_SAMPLES_PER_EPOCH = 50_000
NUM_SAMPLES_PER_EPOCH_MINI = 100
# Datasets with fraction 0 are not built at all.
DATASET_FRACTIONS = {"ZOD": 0.30, "Argoverse2": 0.25, "NuScenes": 0.13, "TruckScenes": 0.12, "OctoSense": 0.11, "PandaSet": 0.05, "KITTI": 0.02, "Cirrus": 0.02}
SEED = 0


class CombinedDataset(Dataset):
    """Label-free pre-training samples of several datasets, indexed by the ``(dataset, item)`` pairs
    that ``CombinedBatchSampler`` draws.
    """

    def __init__(
        self,
        num_sweeps_ranges: dict[str, tuple[int, int]],
        version: str = "full",
        random_crop: bool = False,
        load_frame_id: bool = False,
        load_num_sweeps: bool = False,
    ):
        """num_sweeps_ranges: per dataset (min, max) number of sweeps aggregated per sample."""
        self.names = [name for name, fraction in DATASET_FRACTIONS.items() if fraction > 0]
        self.datasets = [
            _build_pretrain_dataset(
                name,
                version=version,
                random_crop=random_crop,
                load_frame_id=load_frame_id,
                load_num_sweeps=load_num_sweeps,
                min_num_sweeps_per_sample=num_sweeps_ranges[name][0],
                max_num_sweeps_per_sample=num_sweeps_ranges[name][1],
            )
            for name in self.names
        ]
        self._load_frame_id = load_frame_id
        self.num_samples_per_epoch = NUM_SAMPLES_PER_EPOCH if version == "full" else NUM_SAMPLES_PER_EPOCH_MINI

    def __getitem__(self, key: tuple[int, int]):
        d, item = key
        sample = self.datasets[d][item]
        if not self._load_frame_id:
            return sample
        points, frame_id, *metadata = sample
        return points, f"{self.names[d]}/{frame_id}", *metadata


class CombinedBatchSampler(Sampler[list[tuple[int, int]]]):
    """Builds every batch of a ``CombinedDataset``: its mix of datasets, and which item of each.

    Each dataset is walked through shuffled orders of its items, one after another, so every item is drawn
    once before any is drawn again. The walk lives in this sampler in the main process, so it continues
    across epochs. All ranks draw the same batch mix and share the orders (same seeds), each taking every
    world_size-th entry, so together they consume each order exactly once, like a DistributedSampler.
    """

    def __init__(self, dataset: CombinedDataset, batch_size: int, rank: int = 0, world_size: int = 1):
        self._sizes = [len(d) for d in dataset.datasets]
        self._batch_size = batch_size
        self._rank = rank
        self._world_size = world_size
        self._num_batches = dataset.num_samples_per_epoch // batch_size // world_size
        fractions = np.array([DATASET_FRACTIONS[name] for name in dataset.names])
        self._fractions = fractions / fractions.sum()
        self._rng = np.random.default_rng(SEED)
        # Per dataset: draws made by this rank, and the shuffled order (and its pass number) being walked.
        self._draws = [0] * len(self._sizes)
        self._passes = [-1] * len(self._sizes)
        self._orders = [None] * len(self._sizes)

    def __len__(self):
        return self._num_batches

    def __iter__(self):
        for _ in range(self._num_batches):
            yield [(d, self._next_item(d)) for d in self._batch_datasets()]

    def _batch_datasets(self) -> list[int]:
        """Dataset of each sample of a batch: batch_size * fraction, the leftover slots drawn by the remainders."""
        exact = self._batch_size * self._fractions
        counts = np.floor(exact).astype(int)
        leftover = self._batch_size - counts.sum()
        if leftover:
            # Systematic sampling: each dataset gets a leftover slot with probability exactly its remainder (< 1).
            bounds = np.cumsum(exact - counts)
            bounds[-1] = leftover
            counts[np.searchsorted(bounds, self._rng.random() + np.arange(leftover), side="right")] += 1
        return np.repeat(np.arange(len(counts)), counts).tolist()

    def _next_item(self, d: int) -> int:
        position = self._rank + self._draws[d] * self._world_size
        self._draws[d] += 1
        pass_index, offset = divmod(position, self._sizes[d])
        if pass_index != self._passes[d]:
            self._orders[d] = np.random.default_rng((SEED, d, pass_index)).permutation(self._sizes[d])
            self._passes[d] = pass_index
        return int(self._orders[d][offset])


def _build_pretrain_dataset(name, **kwargs):
    """The label-free pre-training dataset of one source."""
    common = dict(load_annotations=False, load_segmentation=False, **kwargs)
    if name == "ZOD":
        from data.zod_dataset import ZODdataset

        return ZODdataset(train_flag=True, mp=True, **common)
    if name == "NuScenes":
        from data.nuscenes_dataset import NuScenesDataset

        return NuScenesDataset(train_flag=True, **common)
    if name == "Argoverse2":
        from data.argoverse2_dataset import build_argoverse2_dataset

        return build_argoverse2_dataset(train_flag=True, **common)
    if name == "TruckScenes":
        from data.truckscenes_dataset import build_truckscenes_dataset

        # One item per scene (a random frame of it), so the sampler walks every scene once per pass.
        return build_truckscenes_dataset(train_flag=True, **common)
    if name == "OctoSense":
        from data.octosense_dataset import build_octosense_dataset

        # One item per ~20 s segment of a drive (a random frame of it).
        return build_octosense_dataset(train_flag=True, **common)
    if name == "PandaSet":
        from data.pandaset_dataset import build_pandaset_dataset

        # One item per 8 s sequence (a random frame of it).
        return build_pandaset_dataset(train_flag=True, **common)
    if name == "KITTI":
        from data.kitti_dataset import build_kitti_dataset

        # One item per ~10 s segment of a raw drive (a random frame of it).
        return build_kitti_dataset(train_flag=True, **common)
    if name == "Cirrus":
        from data.cirrus_dataset import build_cirrus_dataset

        # One item per 1 Hz frame (its gaussian or uniform scan at random), always a single sweep.
        return build_cirrus_dataset(train_flag=True, **common)
    raise ValueError(f"Unknown dataset {name!r}")
