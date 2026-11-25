import os

import tempfile
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
import shutil
from torch.utils.tensorboard import SummaryWriter
from constants import BATCH_SIZE, NUM_EVAL_SWEEPS, NUM_SWEEPS_RANGES, NUM_WORKERS, PREFETCH_FACTOR
from data.zod_dataset import ZODdataset
from utils.multi_gpu_utils import torch_ddp_active
from utils.merge_cov import (
    dump_covariance,
    get_entropy_from_covariances,
    hirarcical_merging_covariances,
)

def filter_and_flatten(point_clouds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Filter padded zeros and flatten batched point clouds.
    Args:
        point_clouds (torch.Tensor): (B, N, C) tensor of point clouds with zero padding

    Returns:
        points: torch.Tensor: (M, C) tensor of filtered and flattened point clouds
        batch_indices: torch.Tensor: (M,) tensor indicating batch index for each point
    """
    B, N, _ = point_clouds.shape
    # Create mask for non-zero points
    mask = torch.any(point_clouds != 0, dim=-1)  # (B, N)

    # Filter and flatten point clouds
    batch_indices = (torch.arange(B).repeat_interleave(N)).to(point_clouds.device)
    points = point_clouds[mask]  # (M, C), where M is total number of non-zero points
    batch_indices = batch_indices[mask.flatten(0, 1)]  # (M,)

    # Safety guarantee that everything happening later in the code is safe, where empty tensor scatters
    # assume every index gets hit.
    present = torch.zeros(B, dtype=torch.bool, device=point_clouds.device)
    present[batch_indices] = True
    torch._assert_async(present.all())

    return points, batch_indices


def get_dataloader(
    train_flag,
    version,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    prefetch_factor=PREFETCH_FACTOR,
    shuffle=True,
    drop_last=False,
    load_annotations=False,
    load_frame_id=False,
    random_crop=False,
    dataset_type="ZOD",
    load_segmentation=False,
    seed=None,
    load_num_sweeps=False,
    aggregate_sweeps=False,
):
    """aggregate_sweeps: pre-training, each dataset aggregates a number of sweeps drawn from its NUM_SWEEPS_RANGES entry.
    Otherwise (probes, evals) every sample is NUM_EVAL_SWEEPS sweeps.
    """
    if aggregate_sweeps:
        num_sweeps_ranges = NUM_SWEEPS_RANGES
    else:
        num_sweeps_ranges = {name: (NUM_EVAL_SWEEPS, NUM_EVAL_SWEEPS) for name in NUM_SWEEPS_RANGES}
    if dataset_type not in ("Combined", "SemanticKITTI", "Waymo"):
        min_sweeps, max_sweeps = num_sweeps_ranges[dataset_type]
    if dataset_type in ("SemanticKITTI", "Waymo"):
        from data.semantickitti_dataset import SemanticKITTIDataset
        from data.waymo_dataset import WaymoDataset
        if aggregate_sweeps or random_crop or load_frame_id or load_num_sweeps or not (load_annotations and load_segmentation):
            raise ValueError(f"{dataset_type} is evaluation-only: labelled single scans with annotations and segmentation")
        dataset_cls = SemanticKITTIDataset if dataset_type == "SemanticKITTI" else WaymoDataset
        dataset = dataset_cls(train_flag=train_flag, version=version)
    elif dataset_type == "ZOD":
        dataset = ZODdataset(
            train_flag=train_flag,
            load_annotations=load_annotations,
            version=version,
            random_crop=random_crop,
            mp=True,
            load_frame_id=load_frame_id,
            load_segmentation=load_segmentation,
            load_num_sweeps=load_num_sweeps,
            min_num_sweeps_per_sample=min_sweeps,
            max_num_sweeps_per_sample=max_sweeps,
        )
    elif dataset_type == "NuScenes":
        from data.nuscenes_dataset import NuScenesDataset # Import here to avoid circular import when using ZOD
        dataset = NuScenesDataset(
            train_flag=train_flag,
            load_annotations=load_annotations,
            version=version,
            min_num_sweeps_per_sample=min_sweeps,
            max_num_sweeps_per_sample=max_sweeps,
            random_crop=random_crop,
            load_frame_id=load_frame_id,
            load_segmentation=load_segmentation,
            load_num_sweeps=load_num_sweeps,
        )
    elif dataset_type == "Argoverse2":
        from data.argoverse2_dataset import build_argoverse2_dataset # Import here to avoid circular import when using ZOD
        dataset = build_argoverse2_dataset(
            train_flag=train_flag,
            version=version,
            load_annotations=load_annotations,
            load_frame_id=load_frame_id,
            random_crop=random_crop,
            load_segmentation=load_segmentation,
            load_num_sweeps=load_num_sweeps,
            min_num_sweeps_per_sample=min_sweeps,
            max_num_sweeps_per_sample=max_sweeps,
        )
    elif dataset_type == "TruckScenes":
        from data.truckscenes_dataset import build_truckscenes_dataset
        dataset = build_truckscenes_dataset(
            train_flag=train_flag,
            version=version,
            load_annotations=load_annotations,
            load_frame_id=load_frame_id,
            random_crop=random_crop,
            load_segmentation=load_segmentation,
            load_num_sweeps=load_num_sweeps,
            min_num_sweeps_per_sample=min_sweeps,
            max_num_sweeps_per_sample=max_sweeps,
        )
    elif dataset_type == "OctoSense":
        from data.octosense_dataset import build_octosense_dataset
        dataset = build_octosense_dataset(
            train_flag=train_flag,
            version=version,
            load_annotations=load_annotations,
            load_frame_id=load_frame_id,
            random_crop=random_crop,
            load_segmentation=load_segmentation,
            load_num_sweeps=load_num_sweeps,
            min_num_sweeps_per_sample=min_sweeps,
            max_num_sweeps_per_sample=max_sweeps,
        )
    elif dataset_type == "PandaSet":
        from data.pandaset_dataset import build_pandaset_dataset
        dataset = build_pandaset_dataset(
            train_flag=train_flag,
            version=version,
            load_annotations=load_annotations,
            load_frame_id=load_frame_id,
            random_crop=random_crop,
            load_segmentation=load_segmentation,
            load_num_sweeps=load_num_sweeps,
            min_num_sweeps_per_sample=min_sweeps,
            max_num_sweeps_per_sample=max_sweeps,
        )
    elif dataset_type == "KITTI":
        from data.kitti_dataset import build_kitti_dataset
        dataset = build_kitti_dataset(
            train_flag=train_flag,
            version=version,
            load_annotations=load_annotations,
            load_frame_id=load_frame_id,
            random_crop=random_crop,
            load_segmentation=load_segmentation,
            load_num_sweeps=load_num_sweeps,
            min_num_sweeps_per_sample=min_sweeps,
            max_num_sweeps_per_sample=max_sweeps,
        )
    elif dataset_type == "Cirrus":
        from data.cirrus_dataset import build_cirrus_dataset
        dataset = build_cirrus_dataset(
            train_flag=train_flag,
            version=version,
            load_annotations=load_annotations,
            load_frame_id=load_frame_id,
            random_crop=random_crop,
            load_segmentation=load_segmentation,
            load_num_sweeps=load_num_sweeps,
            min_num_sweeps_per_sample=min_sweeps,
            max_num_sweeps_per_sample=max_sweeps,
        )
    elif dataset_type == "Combined":
        from data.combined_dataset import CombinedBatchSampler, CombinedDataset
        assert train_flag and not (load_annotations or load_segmentation), "Combined is a label-free pre-training dataset"
        dataset = CombinedDataset(
            version=version,
            random_crop=random_crop,
            load_frame_id=load_frame_id,
            load_num_sweeps=load_num_sweeps,
            num_sweeps_ranges=num_sweeps_ranges,
        )
        batch_sampler = CombinedBatchSampler(
            dataset,
            batch_size=batch_size,
            rank=dist.get_rank() if torch_ddp_active() else 0,
            world_size=dist.get_world_size() if torch_ddp_active() else 1,
        )
        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=True,
        )
        return dataloader, None
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    generator = torch.Generator().manual_seed(seed) if seed is not None else None

    # For DDP, use DistributedSampler
    if torch_ddp_active():
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(
            dataset,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed if seed is not None else 0,
        )
        shuffle = False  # Sampler handles shuffling
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,  # False when using sampler
        sampler=sampler,  # None for single GPU
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
        drop_last=drop_last,
        generator=generator,
    )
    return dataloader, sampler


class EntropyMetrics:
    """Tracks Gaussian entropy of register tokens and voxel features per encoder stage.

    Uses the on-disk covariance accumulation + hierarchical merging pipeline.
    Registers and voxel features use a shared tmp directory but different stage_idx
    offsets (_VOX_STAGE_OFFSET) so file names never collide.

    Usage:
        e = _EntropyMetrics(num_stages, tmp_work_dir)
        for batch in val_loader:
            e.accumulate(teach_feats, teach_batch_inds, teacher_regs, eval_idx, actual_batch_size, batch_size)
        e.merge_local()
        e.gather_ddp()
        if is_main_process():
            e.log(writer, epoch, output_dir)
        dist.barrier() if torch_ddp_active() else None
        e.cleanup()
    """

    _VOX_STAGE_OFFSET = 100  # keeps voxel filenames distinct from register filenames

    def __init__(self, num_stages: int, tmp_work_dir: str):
        self.num_stages   = num_stages
        self.tmp_work_dir = tmp_work_dir

        self.reg_paths = {"cov": {i: [] for i in range(num_stages)},
                          "mean": {i: [] for i in range(num_stages)},
                          "nsample": {i: [] for i in range(num_stages)}}
        self.vox_paths = {"cov": {i: [] for i in range(num_stages)},
                          "mean": {i: [] for i in range(num_stages)},
                          "nsample": {i: [] for i in range(num_stages)}}
        self._avg_reg = None   # set by merge_local()
        self._avg_vox = None
        self._gathered = None  # set by gather_ddp()

    def accumulate(self, teach_feats, teach_batch_inds, teacher_regs, eval_idx, actual_batch_size, batch_size_const):
        """Dump per-sample covariances for register tokens and subsampled voxel features."""
        for i in range(self.num_stages):
            for sample_idx in range(actual_batch_size):
                # data_idx only needs to be unique within this rank's tmp_work_dir,
                # so no cross-rank offset is required (each rank has its own directory).
                data_idx = eval_idx * batch_size_const + sample_idx
                # Register tokens: (R, C)
                if teacher_regs is not None:
                    cp, mp, nsp = dump_covariance(
                        teacher_regs[i][sample_idx],
                        tmp_work_dir=self.tmp_work_dir, data_idx=data_idx, stage_idx=i,
                    )
                    self.reg_paths["cov"][i].append(cp)
                    self.reg_paths["mean"][i].append(mp)
                    self.reg_paths["nsample"][i].append(nsp)

                # Voxel features:
                sample_mask = teach_batch_inds[i] == sample_idx
                vox = teach_feats[i][sample_mask].float()
                cp, mp, nsp = dump_covariance(
                    vox, tmp_work_dir=self.tmp_work_dir,
                    data_idx=data_idx, stage_idx=i + self._VOX_STAGE_OFFSET,
                )
                self.vox_paths["cov"][i].append(cp)
                self.vox_paths["mean"][i].append(mp)
                self.vox_paths["nsample"][i].append(nsp)

    def merge_local(self):
        """Merge per-sample covariances within this rank into one file per stage."""
        def _merge(paths):
            avg = {"cov": {}, "mean": {}, "nsample": {}}
            for i in range(self.num_stages):
                acp, amp, nsp = hirarcical_merging_covariances(
                    paths["cov"][i], paths["mean"][i], paths["nsample"][i], self.tmp_work_dir,
                )
                avg["cov"][i]     = os.path.relpath(acp, self.tmp_work_dir) if acp else acp
                avg["mean"][i]    = os.path.relpath(amp, self.tmp_work_dir) if amp else amp
                avg["nsample"][i] = os.path.relpath(nsp, self.tmp_work_dir) if nsp else nsp
            return avg
        self._avg_reg = _merge(self.reg_paths)
        self._avg_vox = _merge(self.vox_paths)

    def gather_ddp(self):
        """Gather merged path dicts and tmp dirs from all ranks (no-op on single GPU)."""
        if not torch_ddp_active():
            return
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, {
            "avg_reg": self._avg_reg,
            "avg_vox": self._avg_vox,
            "tmp_dir": self.tmp_work_dir,
        })
        self._gathered = gathered

    def log(self, writer: SummaryWriter, epoch: int, output_dir: str) -> None:
        """Copy files cross-rank, do final merge, compute Gaussian entropy, log. Main process only."""
        if torch_ddp_active():
            local_tmp = tempfile.mkdtemp(dir=output_dir)
            for rank_idx, g in enumerate(self._gathered):
                for f in os.listdir(g["tmp_dir"]):
                    shutil.copy2(
                        os.path.join(g["tmp_dir"], f),
                        os.path.join(local_tmp, f"{rank_idx}_{f}"),
                    )
            def _lists(key):
                return {
                    k: {
                        i: [
                            f"{rank_idx}_{g[key][k][i]}" if g[key][k][i] is not None else None
                            for rank_idx, g in enumerate(self._gathered)
                        ]
                        for i in range(self.num_stages)
                    }
                    for k in ("cov", "mean", "nsample")
                }
            reg_lists = _lists("avg_reg")
            vox_lists = _lists("avg_vox")
        else:
            local_tmp = self.tmp_work_dir
            def _wrap(avg):
                return {k: {i: [avg[k][i]] for i in range(self.num_stages)} for k in ("cov", "mean", "nsample")}
            reg_lists = _wrap(self._avg_reg)
            vox_lists = _wrap(self._avg_vox)

        for i in range(self.num_stages):
            for label, lists in (("register", reg_lists), ("voxel", vox_lists)):
                acps = [os.path.join(local_tmp, p) for p in lists["cov"][i] if p]
                amps = [os.path.join(local_tmp, p) for p in lists["mean"][i] if p]
                nsps = [os.path.join(local_tmp, p) for p in lists["nsample"][i] if p]
                if acps:
                    entropy = get_entropy_from_covariances(acps, amps, nsps, local_tmp)
                    writer.add_scalar(f"Entropy_val/{label}_stage_{i}", entropy, epoch)

        if torch_ddp_active():
            shutil.rmtree(local_tmp)

    def cleanup(self):
        """Remove this rank's tmp dir. Call after dist.barrier() in DDP."""
        if os.path.exists(self.tmp_work_dir):
            shutil.rmtree(self.tmp_work_dir)


def get_pointwise_labels(pos, batch_indices, annotations):
    """Assign per-point integer class labels from oriented bounding-box annotations.

    Args:
        pos: (N, 3) point positions.
        batch_indices: (N,) batch index per point.
        annotations: dict with keys 'center' (B,M,3), 'size' (B,M,3),
                     'quats' (B,M,4), 'class' (B,M).

    Returns:
        obj_t: (N,) long tensor; 0 = background, 1..C = foreground class.
    """
    centers = annotations["center"]   # (B, M, 3)
    sizes   = torch.relu(annotations["size"]) + 1e-7  # (B, M, 3)
    quats   = annotations["quats"]    # (B, M, 4)
    classes = annotations["class"]    # (B, M)

    obj_t = torch.zeros(pos.shape[0], device=pos.device, dtype=torch.long)

    for b_idx in range(centers.shape[0]):
        unpad_mask = (centers[b_idx] != 0).any(dim=-1)
        centers_b = centers[b_idx][unpad_mask]
        sizes_b   = sizes[b_idx][unpad_mask]
        quats_b   = quats[b_idx][unpad_mask]
        classes_b = classes[b_idx][unpad_mask]

        batch_mask = batch_indices == b_idx
        pos_b = pos[batch_mask]

        if centers_b.shape[0] == 0:
            continue

        w, x, y, z = quats_b[:, 0], quats_b[:, 1], quats_b[:, 2], quats_b[:, 3]
        R = torch.stack([
            torch.stack([1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)    ], dim=-1),
            torch.stack([2*(x*y + w*z),      1 - 2*(x*x + z*z), 2*(y*z - w*x)    ], dim=-1),
            torch.stack([2*(x*z - w*y),      2*(y*z + w*x),     1 - 2*(x*x + y*y)], dim=-1),
        ], dim=1)  # (M', 3, 3)

        rel_pos   = pos_b.unsqueeze(0) - centers_b.unsqueeze(1)           # (M', N_b, 3)
        local_pos = torch.einsum('mji,mnj->mni', R, rel_pos)              # (M', N_b, 3)
        half_sizes = sizes_b.unsqueeze(1) / 2.0                           # (M', 1, 3)
        inside = (local_pos.abs() <= half_sizes).all(dim=-1)              # (M', N_b)

        obj_t_b = torch.zeros(pos_b.shape[0], device=pos.device, dtype=torch.long)
        for m_idx in range(len(classes_b)):
            obj_t_b[inside[m_idx]] = classes_b[m_idx].long()
        obj_t[batch_mask] = obj_t_b

    return obj_t