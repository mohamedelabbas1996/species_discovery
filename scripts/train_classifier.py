import torch
import wandb
import os
from pathlib import Path
import signal
from types import FrameType


from src.networks import get_network

# from src.trainers import get_trainer
# from src.evaluators import get_evaluator

from src.utils.checkpoint import RunState, load_checkpoint, save_checkpoint
from src.utils.config import setup_config
from src.utils.pipeline_init import signal_handler, print_metrics


# init
config = setup_config()
device = "cuda" if torch.cuda.is_available() else "cpu"

# getting environment variables
SCRATCH = os.environ["SCRATCH"]
try:
    SLURM_TMPDIR = Path(os.environ["SLURM_TMPDIR"])
except:
    SLURM_TMPDIR = None
SLURM_JOBID = os.environ["SLURM_JOB_ID"]

# loader_dict = get_dataloader(config)
# train_loader = loader_dict["train"]
# val_loader = loader_dict["val"]

# initialize
start_epoch, best_acc = 1, 0.0
net = get_network(config.network)

print(net)

exit()
wandb_run_id = None
run_dir = Path(os.path.join(SCRATCH, config.run_dir))
run_dir.mkdir(parents=True, exist_ok=True)
checkpoint_dir = run_dir / SLURM_JOBID / "checkpoints"
checkpoint: RunState | None = load_checkpoint(checkpoint_dir, map_location=device)
trainer = get_trainer(net, train_loader, val_loader, config)
evaluator = get_evaluator(config)

if checkpoint:
    wandb_run_id = checkpoint["wandb_run_id"]
    start_epoch = checkpoint["epoch"] + 1  # +1 to start at the next epoch.
    best_acc = checkpoint["best_acc"]
    trainer.net.load_state_dict(checkpoint["model_state"])
    trainer.optimizer.load_state_dict(checkpoint["optimizer_state"])

    print(
        f"Checkpoints found in {checkpoint_dir}.\nResuming training at epoch {start_epoch} (best_acc={best_acc:.2%}).",
        flush=True,
    )
else:
    print(
        f"No checkpoints found in {checkpoint_dir}. Training from scratch.",
        flush=True,
    )


if wandb_run_id == None:
    wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=config.wandb.name + "_" + SLURM_JOBID,
        resume="allow",
        config=config,
    )
else:
    wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=config.wandb.name + "_" + SLURM_JOBID,
        id=wandb_run_id,
        resume="must",
        config=config,
    )


signal.signal(signal.SIGTERM, signal_handler)  # Before getting pre-empted and requeued.
signal.signal(signal.SIGUSR1, signal_handler)  # Before reaching the end of the time limit.

# # add checkpointing
for epoch_idx in range(start_epoch, config.optimizer.num_epochs + 1):
    epoch_metrics, epoch_metrics_formated = {}, {}
    net, train_metrics = trainer.train_epoch(epoch_idx)[:2]
    epoch_metrics["train/loss"] = train_metrics["loss"]
    epoch_metrics_formated["train_loss"] = f"{train_metrics['loss']:.4f}"
    test_metrics = evaluator.eval_acc(net, val_loader, epoch_idx=epoch_idx)
    epoch_metrics["val/loss"] = test_metrics["loss"]
    epoch_metrics["val/acc"] = test_metrics["acc"]
    epoch_metrics_formated["val_loss"] = f"{ test_metrics['loss']:.4f}"
    epoch_metrics_formated["val_acc"] = f"{ test_metrics['acc']:.4f}"
    val_accuracy = test_metrics["acc"]
    is_best = val_accuracy > best_acc
    best_acc = max(val_accuracy, best_acc)

    if config.network.name == "arpl_net":
        model_state_dict = net["netF"].state_dict()
    else:
        model_state_dict = net.state_dict()

    optimizer_state = trainer.optimizer.state_dict()
    scheduler_state = trainer.scheduler.state_dict()
    save_checkpoint(
        checkpoint_dir,
        is_best,
        RunState(
            wandb_run_id=wandb.run.id,
            epoch=epoch_idx + 1,
            model_state=model_state_dict,
            optimizer_state=optimizer_state,
            scheduler_state=scheduler_state,
            best_acc=best_acc,
        ),
    )

    print(f" {print_metrics(epoch_metrics_formated)}", flush=True)
    wandb.log({**epoch_metrics})


print("Finished Training", flush=True)
