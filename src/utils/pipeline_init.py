import torch
import wandb
import signal
from types import FrameType
from timm.scheduler import CosineLRScheduler
import numpy as np


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
