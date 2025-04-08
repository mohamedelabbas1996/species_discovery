import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.utils import Config
from timm.scheduler import CosineLRScheduler


class BaseTrainer:
    def __init__(
        self, net: nn.Module, train_loader: DataLoader, config: Config
    ) -> None:

        self.net = net
        self.train_loader = train_loader
        self.config = config

        self.optimizer = torch.optim.AdamW(
            net.parameters(),
            lr=config.optimizer["lr"],
            weight_decay=config.optimizer["weight_decay"],
        )

        steps_per_epoch = (
            int(config.dataset.train.len / config.dataset.train.batch_size) + 1
        )
        total_steps = int(config.optimizer.num_epochs * steps_per_epoch)
        warmup_steps = int(config.optimizer.warmup_epochs * steps_per_epoch)

        self.scheduler = CosineLRScheduler(
            self.optimizer,
            t_initial=(total_steps - warmup_steps),
            warmup_t=warmup_steps,
            warmup_prefix=True,
            cycle_limit=1,
            t_in_epochs=False,
        )

    def train_epoch(self, epoch_idx):
        self.net.train()

        loss_avg = 0.0
        train_dataiter = iter(self.train_loader)
        total_steps = (
            int(self.config.dataset.train.len / self.config.dataset.train.batch_size)
            + 1
        ) * epoch_idx

        for batch in tqdm(
            self.train_loader,
            position=0,
            leave=True,
        ):
            data = batch["data"].cuda()
            target = batch["label"].cuda()

            # forward
            logits_classifier = self.net(data)
            loss = F.cross_entropy(logits_classifier, target)

            # backward
            total_steps += 1
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step_update(num_updates=total_steps)

            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        metrics = {}
        metrics["epoch_idx"] = epoch_idx
        metrics["loss"] = loss_avg

        return self.net, metrics
