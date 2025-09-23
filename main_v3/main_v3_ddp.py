import os
from typing import Optional

import torch
import torch.distributed as dist
from datetime import timedelta


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def init_distributed(backend: str = "nccl", timeout_sec: int = 1800) -> bool:
    """
    Initialize torch.distributed if launched by torchrun.
    Returns True if distributed was initialized in this call, False otherwise.
    Safe to call multiple times.
    """
    if is_dist_avail_and_initialized():
        return False

    # torchrun sets these envs
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            # set the current gpu device for this process
            torch.cuda.set_device(local_rank)
        # init process group
        dist.init_process_group(
            backend=backend,
            timeout=timedelta(seconds=timeout_sec),
        )
        return True

    return False


def get_world_size() -> int:
    if is_dist_avail_and_initialized():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    if is_dist_avail_and_initialized():
        return dist.get_rank()
    return 0


def get_local_rank() -> int:
    # torchrun sets LOCAL_RANK
    try:
        return int(os.environ.get("LOCAL_RANK", "0"))
    except Exception:
        return 0


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_dist_avail_and_initialized():
        dist.barrier()


def cleanup() -> None:
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


def reduce_tensor(t: torch.Tensor, op: str = "mean") -> torch.Tensor:
    """
    All-reduce a tensor across processes. Returns a new tensor on the same device.
    op: "sum" or "mean"
    """
    if not is_dist_avail_and_initialized():
        return t

    rt = t.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if op == "mean":
        rt /= get_world_size()
    return rt


__all__ = [
    "init_distributed",
    "is_dist_avail_and_initialized",
    "get_world_size",
    "get_rank",
    "get_local_rank",
    "is_main_process",
    "barrier",
    "cleanup",
    "reduce_tensor",
]
