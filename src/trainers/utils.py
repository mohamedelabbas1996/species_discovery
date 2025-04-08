from torch.utils.data import DataLoader

from src.utils import Config


from .base_trainer import BaseTrainer
from .gcd_trainer import GCDTrainer


def get_trainer(net, train_loader: DataLoader, val_loader: DataLoader, config: Config):
    if type(train_loader) is DataLoader:
        trainers = {"base": BaseTrainer, "gcd": GCDTrainer}

        # TODO: update net

        return trainers[config.trainer.name](net, train_loader, config)
