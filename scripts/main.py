import torch
import wandb
import os
from pathlib import Path

# from src.clustering.utils import extract_features, clustering, cluster_acc
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


id_loader_dict = get_dataloader(config)
net = get_network(config.network)

# if config.network.checkpoint:
#     checkpoint = torch.load(
#         config.network.checkpoint,
#         weights_only=False,
#     )
#     weights = checkpoint["model_state"]

#     net.load_state_dict(weights)
net.eval()
net.to(device)

# save_dir = ("/").join(config.network.checkpoint.split("/")[:-1])
# save_dir += f"/{config.dataset.name}/{config.trainer.name}"
# print(save_dir)
# pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

labeled_dataloader = id_loader_dict["test_labeled"]
unlabeled_dataloader = id_loader_dict["test_unlabeled"]

evaluator = get_evaluator(config)

metrics = evaluator.eval_clustering(
    net,
    labeled_dataloader=labeled_dataloader,
    unlabeled_dataloader=unlabeled_dataloader,
)

print(f"ACC: {metrics['ACC']} NMI: {metrics['NMI']} ARI: {metrics['ARI']}")
