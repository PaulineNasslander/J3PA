"""Shared interface every LiDAR dataset in `data` implements.

Consumers ask the dataset for its class spaces (``loader.dataset.get_...()``)
instead of switching on a dataset name, so adding a dataset only means
implementing this interface.
"""

from abc import ABC, abstractmethod

import torch


def normalize_intensity(values, max_value: float):
    """Map intensity from the sensor's native [0, max_value] to [0, 255]."""
    return values * (255.0 / max_value)


def sample_sweep_chunk(available_num_sweeps: int, nsweeps: int) -> tuple[int, int, int]:
    """Randomly select consecutive sweeps out of the available sweeps.

    Args:
        available_num_sweeps: The total number of sweeps available.
        nsweeps: The number of consecutive sweeps to sample.

    Returns:
        A tuple of (start, end, reference) where start and end define the range of consecutive sweeps
        and reference is the index of the reference sweep within that range into which other sweeps should be motion compensated.
    """
    nsweeps = min(nsweeps, available_num_sweeps)
    start = torch.randint(0, available_num_sweeps - nsweeps + 1, size=(1,)).item()
    reference = torch.randint(start, start + nsweeps, size=(1,)).item()
    return start, start + nsweeps, reference


def centered_sweep_chunk(available_num_sweeps: int, nsweeps: int, keyframe: int) -> tuple[int, int, int]:
    """Like sample_sweep_chunk but deterministic (for evals): the keyframe is the reference, in the middle of the run."""
    nsweeps = min(nsweeps, available_num_sweeps)
    start = min(max(0, keyframe - nsweeps // 2), available_num_sweeps - nsweeps)
    return start, start + nsweeps, keyframe


class BaseLidarDataset(ABC):
    """The class spaces a dataset must be able to describe."""

    @classmethod
    @abstractmethod
    def get_class_name_to_id(cls) -> dict[str, int]:
        """Object-detection classes, 1-indexed, with 0 meaning background."""

    @classmethod
    def get_seg_class_name_to_id(cls) -> dict[str, int]:
        """Classes of the per-point labels.

        Defaults to the object classes, which is what the box-derived labels of
        ZOD and Argoverse2 are built from.
        """
        return cls.get_class_name_to_id()

    @classmethod
    def get_seg_thing_ids(cls) -> tuple[int, ...]:
        """Segmentation classes that carry instance ids; all of them by default."""
        return tuple(cls.get_seg_class_name_to_id().values())

    @classmethod
    def get_seg_ignore_id(cls) -> int | None:
        """Class id kept out of the loss and the metrics, ``None`` if there is none.

        Box-derived labels have none: their id 0 is ordinary background.
        """
        return None
