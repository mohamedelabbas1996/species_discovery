"""Checkpointing example."""

from __future__ import annotations


import os
import shutil
import signal
import uuid
import warnings
from logging import getLogger as get_logger
from pathlib import Path
from types import FrameType
from typing import Any, TypedDict
import pandas as pd

import random
import numpy as np
import torch
from torch import Tensor, nn
from tqdm import tqdm

import wandb


def set_random_seeds(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_num_workers() -> int:
    """Gets the optimal number of DatLoader workers to use in the current job."""
    if "SLURM_CPUS_PER_TASK" in os.environ:
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return torch.multiprocessing.cpu_count()


SCRATCH = Path(os.environ["SCRATCH"])
SLURM_JOBID = os.environ["SLURM_JOB_ID"]

CHECKPOINT_FILE_NAME = "checkpoint.pth"

logger = get_logger(__name__)


class RunState(TypedDict):
    """Typed dictionary containing the state of the training run which is saved at each epoch.

    Using type hints helps prevent bugs and makes your code easier to read for both humans and
    machines (e.g. Copilot). This leads to less time spent debugging and better code suggestions.
    """

    wandb_run_id: int
    epoch: int
    total_steps: int  # for lr schedular
    best_acc: float
    model_state: dict[str, Tensor]
    optimizer_state: dict[str, Tensor]
    scheduler_state: dict[str, Tensor]
    random_state: tuple[Any, ...]
    numpy_random_state: dict[str, Any]
    torch_random_state: Tensor
    torch_cuda_random_state: list[Tensor]

    # for OSR
    embedding: Tensor
    post_proj: Tensor
    labels: Tensor
    df: pd.DataFrame

    # WOODS
    lam: Tensor
    lam2: Tensor
    in_constraint_weight: float
    ce_constraint_weight: float


def load_checkpoint(checkpoint_dir: Path, **torch_load_kwargs) -> RunState | None:
    """Loads the latest checkpoint if possible, otherwise returns `None`."""
    checkpoint_file = checkpoint_dir / CHECKPOINT_FILE_NAME
    restart_count = int(os.environ.get("SLURM_RESTART_COUNT", 0))
    if restart_count:
        logger.info(f"NOTE: This job has been restarted {restart_count} times by SLURM.")

    if not checkpoint_file.exists():
        logger.debug(f"No checkpoint found in checkpoints dir ({checkpoint_dir}).")
        if restart_count:
            logger.warning(
                RuntimeWarning(
                    f"This job has been restarted {restart_count} times by SLURM, but no "
                    "checkpoint was found! This either means that your checkpointing code is "
                    "broken, or that the job did not reach the checkpointing portion of your "
                    "training loop."
                )
            )
        return None

    checkpoint_state: dict = torch.load(checkpoint_file, **torch_load_kwargs)

    missing_keys = set(checkpoint_state.keys()) - RunState.__required_keys__
    if missing_keys:
        warnings.warn(
            RuntimeWarning(
                f"Checkpoint at {checkpoint_file} is missing the following keys: {missing_keys}. "
                f"Ignoring this checkpoint."
            )
        )
        return None

    logger.debug(f"Resuming from the checkpoint file at {checkpoint_file}")
    state: RunState = checkpoint_state  # type: ignore
    return state


def save_checkpoint(checkpoint_dir: Path, is_best: bool, state: RunState):
    """Saves a checkpoint with the current state of the run in the checkpoint dir.

    The best checkpoint is also updated if `is_best` is `True`.

    Parameters
    ----------
    checkpoint_dir: The checkpoint directory.
    is_best: Whether this is the best checkpoint so far.
    state: The dictionary containing all the things to save.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_dir / CHECKPOINT_FILE_NAME

    # Use a unique ID to avoid any potential collisions.
    unique_id = uuid.uuid1()
    temp_checkpoint_file = checkpoint_file.with_suffix(f".temp{unique_id}")

    torch.save(state, temp_checkpoint_file)
    os.replace(temp_checkpoint_file, checkpoint_file)

    if is_best:
        best_checkpoint_file = checkpoint_file.with_name("model_best.pth")
        temp_best_checkpoint_file = best_checkpoint_file.with_suffix(f".temp{unique_id}")
        shutil.copyfile(checkpoint_file, temp_best_checkpoint_file)
        os.replace(temp_best_checkpoint_file, best_checkpoint_file)


def signal_handler(signum: int, frame: FrameType | None):
    """Called before the job gets pre-empted or reaches the time-limit.

    This should run quickly. Performing a full checkpoint here mid-epoch is not recommended.
    """
    signal_enum = signal.Signals(signum)
    print(f"Job received a {signal_enum.name} signal!", flush=True)
    # logger.error(f"Job received a {signal_enum.name} signal!")
    # Perform quick actions that will help the job resume later.
    # If you use Weights & Biases: https://docs.wandb.ai/guides/runs/resuming#preemptible-sweeps
    if wandb.run:
        wandb.mark_preempting()


def print_metrics(metrics):
    return " - ".join([f"{k}: {v}" for k, v in metrics.items()])
