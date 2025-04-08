import os
import torch
from torch.utils.data import DataLoader
from .preprocessor import get_preprocessor
from PIL import Image

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, config, preprocessor) -> None:
        super(BaseDataset, self).__init__()

        with open(config.imglist_pth) as imgfile:
            self.imglist = imgfile.readlines()
        self.orig_ids = list(range(len(self.imglist)))
        self.data_dir = config.data_dir
        self.transform_image = preprocessor

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):

        line = self.imglist[index].strip("\n")
        tokens = line.split(" ", 1)
        image_name, extra_str = tokens[0], tokens[1]
        path = os.path.join(self.data_dir, image_name)
        sample = dict()
        sample["image_name"] = image_name
        #  kwargs = {"name": self.name, "path": path, "tokens": tokens}
        image = Image.open(path).convert("RGB")
        sample["data"] = self.transform_image(image)
        sample["label"] = int(extra_str)

        return sample


def get_num_workers() -> int:
    """Gets the optimal number of DatLoader workers to use in the current job."""
    if "SLURM_CPUS_PER_TASK" in os.environ:
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return torch.multiprocessing.cpu_count()


def get_dataloader(config):
    dataset_config = config.dataset
    dataloader_dict = {}
    for split in dataset_config.split_names:
        split_config = dataset_config[split]
        preprocessor = get_preprocessor(config, split)
        dataset = BaseDataset(split_config, preprocessor)
        dataloader = DataLoader(
            dataset,
            shuffle=split == "train",
            num_workers=get_num_workers(),
            batch_size=split_config.batch_size,
        )
        dataloader_dict[split] = dataloader

    return dataloader_dict
