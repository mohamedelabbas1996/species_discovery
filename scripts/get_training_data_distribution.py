import numpy as np
from collections import Counter

from src.utils.config import setup_config


def get_class_prior():
    num_classes = config.dataset.num_classes
    cls_idx = []
    label_filename = config.dataset.train.imglist_pth
    with open(label_filename, "r") as f:
        for line in f.readlines():
            segs = line.strip().split(" ")
            cls_idx.append(int(segs[-1]))
    cls_idx = np.array(cls_idx, dtype="int")
    label_stat = Counter(cls_idx)
    cls_num = [-1 for _ in range(num_classes)]
    for i in range(num_classes):
        cat_num = int(label_stat[i])
        cls_num[i] = cat_num
    targets = cls_num / np.sum(cls_num)

    return targets


config = setup_config()
targets = get_class_prior()
print(targets)
