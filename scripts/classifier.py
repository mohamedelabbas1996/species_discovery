import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

import openood.utils.comm as comm
from openood.utils import Config

import os
import signal
import wandb

from src.utils import signal_handler, RunState, load_checkpoint, save_checkpoint, get_num_workers

SCRATCH = os.environ["SCRATCH"]
try:
    SLURM_TMPDIR = Path(os.environ["SLURM_TMPDIR"])
except:
    SLURM_TMPDIR = None
SLURM_JOBID = os.environ["SLURM_JOB_ID"]


def train_epoch(epoch, model, train_loader, optimizer, scheduler, batch_size, device):
    model.train()

    total_loss, correct, total = 0, 0, 0
    total_steps = (int(len(train_loader) / batch_size) + 1) * epoch

    for batch in train_loader:
        data = batch["data"].to(device)
        labels = batch["label"].to(device)

        # forward
        outputs = model(data)
        loss = F.cross_entropy(outputs, labels)

        # backward
        total_steps += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step_update(num_updates=total_steps)

        # compute loss and accuracy
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total

    return avg_loss, accuracy


def eval_epoch(model, val_loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, total_epochs, device):
    model.to(device)
    for epoch in range(total_epochs):
        train_loss, train_acc = train_epoch(epoch, model, train_loader, criterion, optimizer, scheduler, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)

        # update checkpoint

        # remember best accuracy and save the current state.
        val_accuracy = val_acc.avg
        is_best = val_accuracy > best_acc
        best_acc = max(val_accuracy, best_acc)

        model_state_dict = model.state_dict()
        optimizer_state = optimizer.state_dict()
        scheduler_state = scheduler.state_dict()

        save_checkpoint(
            checkpoint_dir,
            is_best,
            RunState(
                wandb_run_id=wandb.run.id,
                epoch=epoch + 1,
                model_state=model_state_dict,
                optimizer_state=optimizer_state,
                scheduler_state=scheduler_state,
                best_acc=best_acc,
            ),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", type=str, default=None)
    args = parser.parse_args()

    # load training configuration
    config = training_helper.load_config(args.config_path)
    training_helper.set_random_seeds(config["random_seed"])
    # load datasets
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = get_num_workers()

    train_loader, val_loader = get_dataloader(config)

    # set up checkpoint
    run_dir = Path(os.path.join(SCRATCH, config["model_save_path"])) if config["model_save_path"] is not None else None
    checkpoint_dir = run_dir / SLURM_JOBID / "checkpoints" if run_dir is not None else None

    if checkpoint_dir:
        pass

    # set up wandb
    if config["wandb_project"]:
        pass

    print(f"device: {device}", flush=True)
    print(f"num_workers: {num_workers}", flush=True)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    signal.signal(signal.SIGTERM, signal_handler)  # Before getting pre-empted and requeued.
    signal.signal(signal.SIGUSR1, signal_handler)  # Before reaching the end of the time limit.

    train()


if __name__ == "__main__":
    main()
