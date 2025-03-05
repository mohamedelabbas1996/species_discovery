import torch
import wandb
import os
from pathlib import Path

import pathlib
import numpy as np
import pandas as pd


from src.networks import get_network
from src.utils.checkpoint import RunState, load_checkpoint, save_checkpoint
from src.utils.config import setup_config
from src.utils.pipeline_init import signal_handler, print_metrics
from src.utils.dataset import get_dataloader
from src.evaluators import get_evaluator


# init
config = setup_config()
dataset_name = config.dataset.name

device = "cuda" if torch.cuda.is_available() else "cpu"


network_name = config.network.name if config.network.name != "dinov2" else config.network.vit_name
wandb.init(
    project=config.wandb.project,
    entity=config.wandb.entity,
    name=f"{config.dataset.name}_{network_name}",
    resume="allow",
    config=config,
)


dataloader_dict = get_dataloader(config)
net = get_network(config.network)

evaluator = get_evaluator(config)

metrics = evaluator.eval_clustering(net, dataloader_dict)

wandb.log({**metrics})
