import torch
import numpy as np
from torchvision import transforms

import torchvision.transforms.functional as F
from functools import partial

import open_clip


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]


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


def base_preprocessor(dataset_config, split):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    image_size = dataset_config.image_size
    split_config = dataset_config[split]
    # padding
    ops = [SquarePad()]

    if split == "train":
        # print(split_config.use_mixres)
        # print(split_config.dataaug)
        # mix resolution
        if split_config.use_mixres:
            f_random_resize = partial(random_resize, full_size=image_size)
            ops += [transforms.Lambda(f_random_resize)]

        if split_config.dataaug == "randaug":
            ops += [
                transforms.RandomResizedCrop(image_size, scale=(0.3, 1)),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(num_ops=2, magnitude=9),
            ]

        elif split_config.dataaug == "simple":
            ops += [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
            ]
    else:
        ops += [transforms.Resize((image_size, image_size))]

    ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(ops)


def bioclip_preprocessor(dataset_config, split):
    _, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        "hf-hub:imageomics/bioclip"
    )

    if split == "train":
        return preprocess_train
    else:
        return preprocess_val


def contrastive_preprocessor(dataset_config, split):
    image_size = dataset_config.image_size
    interpolation = 3
    crop_pct = 0.875
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if split == "train":

        f_random_resize = partial(random_resize, full_size=image_size)  # mix resolution

        train_transform = transforms.Compose(
            [
                SquarePad(),
                transforms.Lambda(f_random_resize),
                transforms.Resize(int(image_size / crop_pct), interpolation),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(),
                transforms.ToTensor(),
                transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
            ]
        )

        return ContrastiveLearningViewGenerator(
            base_transform=train_transform, n_views=2
        )

    else:
        test_transform = transforms.Compose(
            [
                SquarePad(),
                transforms.Resize(int(image_size / crop_pct), interpolation),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
            ]
        )

        return test_transform


def get_preprocessor(config, split):
    dataset_config = config.dataset
    preprocessor_dict = {
        "base": base_preprocessor,
        "bioclip": bioclip_preprocessor,
        "contrastive": contrastive_preprocessor,
    }
    return preprocessor_dict[config.preprocessor.name](dataset_config, split)
