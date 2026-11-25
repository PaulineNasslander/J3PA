import os
import tempfile

import numpy as np
import numpy.typing as npt
import torch
from tqdm import tqdm


def dump_covariance(
    features: torch.Tensor,
    tmp_work_dir: str,
    data_idx: int,
    stage_idx: int = None,
) -> str:
    """Dump feature covariance matrices, feature means, and number of vectors for all data.
    Args:
        features: tensor of shape (N, C)
        tmp_work_dir: Directory to store temporary files in.
        data_idx: Index of the data point, used for naming the files.
    Returns:
        Three paths;
            Path to a stored covariance matrix.
            Path to a stored mean vector.
            Path to a stored number of samples.
    """
    # Compute the empirical covariance matrix using all features
    # We will work in fp64 to avoid numerical errors. This is OK since the algorithm is fast anyway.
    features = features.to(torch.float64)  # row added by pauline
    m = torch.mean(features, dim=0)
    features -= m
    cov = (1 / (features.shape[0] - 1)) * features.T @ features
    # Somehow converting to double again here makes results slightly more precise...
    # Maybe a statistical fluke in tests, or some GPU stuff?
    cov = cov.cpu().double().detach().numpy()
    m = m.cpu().double().detach().numpy()
    n = np.array(float(features.shape[0]))
    # Save the covariance matrix to disk.
    cov_path = os.path.join(tmp_work_dir, f"covariance_{stage_idx}_{data_idx}.npy")
    np.save(cov_path, cov)
    # Save the mean to disk.
    m_path = os.path.join(tmp_work_dir, f"mean_{stage_idx}_{data_idx}.npy")
    np.save(m_path, m)
    # Save the number of samples to disk
    n_path = os.path.join(tmp_work_dir, f"nsample_{stage_idx}_{data_idx}.npy")
    np.save(n_path, n)
    return cov_path, m_path, n_path


def _merge(  # noqa: PLR0913
    covariance_path_lhs: str,
    mean_path_lhs: str,
    n_sample_path_lhs: str,
    covariance_path_rhs: str,
    mean_path_rhs: str,
    n_sample_path_rhs: str,
    tmp_work_dir: str,
) -> tuple[str, str, str]:
    """Merge covariance, mean, and number of samples for two sets of videos.
    Args:
        covariance_path_lhs, mean_path_lhs, n_sample_path_lhs: Path to numpy binary containing
            local covariance estimate, feature mean and number of samples for one of the sets to
            merge
        covariance_path_rhs, mean_path_rhs, n_sample_path_rhs: Same as above for the second set.
        tmp_work_dir: Directory to store temporary files in.
    Returns:
        Three paths:
            1. Path to stored merged covariance matrix.
            2. Path to stored merged feature mean.
            3. Path to stored total number of aggregated feature vectors.
    """
    # Load everything in to memory.
    cov_lhs: npt.NDArray = np.load(covariance_path_lhs)
    mean_lhs: npt.NDArray = np.load(mean_path_lhs)
    size_lhs: npt.NDArray = np.load(n_sample_path_lhs)
    cov_rhs: npt.NDArray = np.load(covariance_path_rhs)
    mean_rhs: npt.NDArray = np.load(mean_path_rhs)
    size_rhs: npt.NDArray = np.load(n_sample_path_rhs)
    # Aggregate covariance matrices taking into account degrees of freedom and the shifting feature
    # means.
    # Someone might want to check my math on this since deriving this gets a bit messy, but it
    # seems to work out based on comparisons to results obtained with Scikit on a small data set.
    cov_term_1 = (size_lhs - 1) * cov_lhs + (size_rhs - 1) * cov_rhs
    # Mean offset must be (C, C) matrix to correct all elements appropriately.
    cov_term_2 = (
        size_lhs * size_rhs / (size_lhs + size_rhs) * (np.expand_dims((mean_lhs - mean_rhs), axis=1) @ np.expand_dims((mean_lhs - mean_rhs), axis=0))
    )
    cov = (cov_term_1 + cov_term_2) / (size_lhs + size_rhs - 1)
    # Average means.
    mean = (mean_lhs * size_lhs + mean_rhs * size_rhs) / (size_lhs + size_rhs)
    # Sum number of samples.
    n_sample = size_lhs + size_rhs
    # Delete the temporary files of everything that is now merged.
    os.remove(covariance_path_lhs)
    os.remove(mean_path_lhs)
    os.remove(n_sample_path_lhs)
    os.remove(covariance_path_rhs)
    os.remove(mean_path_rhs)
    os.remove(n_sample_path_rhs)
    # Create a secure temporary filename with a .npy suffix for storing new means, covariances,
    # and numbers of samples.
    cov_fd, cov_path = tempfile.mkstemp(dir=tmp_work_dir, suffix=".npy")
    mean_fd, mean_path = tempfile.mkstemp(dir=tmp_work_dir, suffix=".npy")
    n_sample_fd, n_sample_path = tempfile.mkstemp(dir=tmp_work_dir, suffix=".npy")
    # Close the open file descriptor because we just needed the name.
    os.close(cov_fd)
    os.close(mean_fd)
    os.close(n_sample_fd)
    # Save the new values.
    np.save(cov_path, cov)
    np.save(mean_path, mean)
    np.save(n_sample_path, n_sample)
    return cov_path, mean_path, n_sample_path


def _hierarchical_covariance(
    covariance_paths: list[str],
    mean_paths: list[str],
    nsample_paths: list[str],
    tmp_work_dir: str,
    pbar: tqdm,
) -> tuple[str, str, str]:
    r"""Hierarchically calculate the global feature covariance matrix.
    The goal of the algorithm is given a set of means, covariances, and the number of samples these
    correspond to, to compute the global covariance and feature mean.
    This is the main divide and conquer algorithm. It effectively makes a 2-tree where leafs get
    merged in a depth-first fashion starting from the bottom.
           *
         /   \
        /     \
       *       *
      / \     / \
     *   *   *   *
    / \ / \ / \ / \
    * * * * * * * *
    We start by traversing the tree to the bottom, where 2 leafs get merged. Then we go back
    upwards, and back down to merge the next leaves until we can traverse upwards.
    Args:
        covariance_paths: List of paths with feature covariance matrices.
        mean_paths: List of paths with feature means.
        nsample_paths: List of paths with number of samples means and covariances correspond to.
        tmp_work_dir: Directory to store temporary files in.
        pbar: tqdm progress bar to update.
    Returns:
        1. Global feature covariance.
        2. Global feature mean.
        3. Total number of feature vectors.
    """
    l = len(covariance_paths)  # noqa: E741
    if l == 0:
        return None, None, None
    if l == 1:
        return covariance_paths[0], mean_paths[0], nsample_paths[0]
    if l == 2:
        result = _merge(
            covariance_paths[0],
            mean_paths[0],
            nsample_paths[0],
            covariance_paths[1],
            mean_paths[1],
            nsample_paths[1],
            tmp_work_dir=tmp_work_dir,
        )
        pbar.update(1)
        return result
    # Note that first chunk can have 1 less element than the other if the number of entries
    # is uneven.
    div = l // 2
    cov_path_1, mean_path_1, n_sample_path_1 = _hierarchical_covariance(
        covariance_paths[:div],
        mean_paths[:div],
        nsample_paths[:div],
        tmp_work_dir=tmp_work_dir,
        pbar=pbar,
    )
    cov_path_2, mean_path_2, n_sample_path_2 = _hierarchical_covariance(
        covariance_paths[div:],
        mean_paths[div:],
        nsample_paths[div:],
        tmp_work_dir=tmp_work_dir,
        pbar=pbar,
    )
    result = _merge(
        cov_path_1,
        mean_path_1,
        n_sample_path_1,
        cov_path_2,
        mean_path_2,
        n_sample_path_2,
        tmp_work_dir=tmp_work_dir,
    )
    pbar.update(1)
    return result


def hirarcical_merging_covariances(
    cov_paths: list[str], mean_paths: list[str], nsample_paths: list[str], tmp_work_dir: str
) -> tuple[torch.Tensor, torch.Tensor]:
    print(f"Hierarchically merging {len(cov_paths)} covariances")
    total_merges = 0 if len(cov_paths) <= 1 else len(cov_paths) - 1
    with tqdm(
        total=total_merges,
        smoothing=0.05,
        desc="Merging covariances",
        unit="merge",
        mininterval=10,
        maxinterval=10,
        miniters=None,
        dynamic_ncols=False,
        ncols=80,
        bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    ) as pbar:
        avg_cov_path, avg_mean_path, nsample_path = _hierarchical_covariance(cov_paths, mean_paths, nsample_paths, tmp_work_dir, pbar=pbar)

    return avg_cov_path, avg_mean_path, nsample_path


def get_entropy_from_covariances(cov_paths: list[str], mean_paths: list[str], nsample_paths: list[str], tmp_work_dir: str):
    avg_cov_path, _, _ = hirarcical_merging_covariances(cov_paths, mean_paths, nsample_paths, tmp_work_dir)
    avg_cov = torch.from_numpy(np.load(avg_cov_path))
    os.remove(avg_cov_path)

    eigenvalues = torch.linalg.eigvalsh(avg_cov).cpu().numpy()
    eigenvalues = np.clip(eigenvalues, 0, None)  # clip since can get tiny neg values
    lambdas = eigenvalues / eigenvalues.sum()  # normalize
    entropy = -np.sum(lambdas * np.log(lambdas + 1e-8))
    return entropy / np.log(avg_cov.shape[0])  # normalize by ln(D) to get value in [0, 1]
