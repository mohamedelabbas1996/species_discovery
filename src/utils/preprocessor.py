import torch
import numpy as np
from torchvision import transforms

import torchvision.transforms.functional as F
from functools import partial


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, "constant")


def random_resize(image, full_size=300):
    random_num = np.random.uniform()
    if random_num <= 0.25:
        transform = transforms.Resize((int(0.5 * full_size), int(0.5 * full_size)))
        image = transform(image)
    elif random_num <= 0.5:
        transform = transforms.Resize((int(0.25 * full_size), int(0.25 * full_size)))
        image = transform(image)

    return image


def get_preprocessor(dataset_config, split):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    input_size = dataset_config.image_size
    split_config = dataset_config[split]
    ops = []

    # if config.rescale_size:
    #     ops += [transforms.Resize((config.rescale_size, config.rescale_size))]

    if split == "train":
        if split_config.use_mixres:
            f_random_resize = partial(random_resize, full_size=input_size)
            ops += [transforms.Lambda(f_random_resize)]

        if split_config.dataaug == "randaug":
            ops += [
                transforms.RandomResizedCrop(input_size, scale=(0.3, 1)),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(num_ops=2, magnitude=9),
            ]

        elif split_config.dataaug == "simple":
            ops += [
                transforms.Resize((input_size, input_size)),
                transforms.RandomHorizontalFlip(),
            ]
    else:
        ops += [transforms.Resize((input_size, input_size))]

    ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(ops)
