# import mmcv
from copy import deepcopy
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from .dino_v2 import DINOv2Wrapper


def get_network(network_config):
    num_classes = network_config.num_classes

    if network_config.name == "dinov2":
        backbone = torch.hub.load("facebookresearch/dinov2", network_config.vit_name)
        net = DINOv2Wrapper(backbone, num_classes)

    elif network_config.name == "bioclip":
        pass

    if network_config.pretrained:
        net.load_state_dict(torch.load(network_config.checkpoint), strict=False)

    print("Model Loading {} Completed!".format(network_config.name))

    return net
