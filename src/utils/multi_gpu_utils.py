import os
from datetime import timedelta

import torch


def torch_ddp_active() -> bool:
    """Return whether torch distributed is active."""
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_gpu_device() -> str:
    """Return name of gpu, used for torch.Tensor.to ."""
    if torch_ddp_active():
        return f"cuda:{get_gpu_local_rank()}"
    return "cuda"


def get_gpu_local_rank() -> int:
    """Return local gpu rank (i.e., device index)."""
    return int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", os.environ.get("LOCAL_RANK", 0)))


def init_multigpu_via_torch_ddp(timeout_s: int = 6 * 3600):
    """Initialize multi-gpu training.

    This needs to be done at the start of training, before anything else. After this has been
    initialized, the torch.distributed library can be used. There are utilities in
    dl/torch/multi_gpu/utils.py .
    """
    world_rank = os.environ.get(
        "OMPI_COMM_WORLD_RANK",
        os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)),
    )
    world_size = os.environ.get(
        "OMPI_COMM_WORLD_SIZE",
        os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)),
    )
    local_device = torch.device(f"cuda:{get_gpu_local_rank()}")

    # Set device in accordance with:
    # https://pytorch.org/tutorials/beginner/ddp_series_multigpu.html#constructing-the-process-group
    torch.cuda.set_device(local_device)

    # Long: evals sync only once after a pass that can run for hours, so ranks drift far apart before it.
    torch.distributed.init_process_group(
        backend="nccl",
        rank=int(world_rank),
        world_size=int(world_size),
        timeout=timedelta(seconds=timeout_s),
        device_id=local_device,
    )


def torch_ddp_run_on_master_only(func):
    """Create function wrapper to run given function only if on master process.

    When running torch DDP, a few tasks -- e.g., checkpoint writing -- should only be done
    by the master process unless these tasks have been explicitly distributed somehow.
    """

    def wrapper(*args, **kwargs):
        if is_main_process():
            result = func(*args, **kwargs)
            return result
        return None

    return wrapper


def is_main_process() -> bool:
    """Return whether process is main process."""
    return get_gpu_world_rank() == 0


def get_gpu_world_rank() -> int:
    """Return world gpu rank."""
    if torch_ddp_active():
        return torch.distributed.get_rank()
    return 0
