# import mmcv
from copy import deepcopy
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from .dino_v2 import DINOv2Wrapper
from .bioclip import BioCLIPWrapper
from .resnet50 import ResNet50
from .mobilenet import MobileNet
from .extended_net import ExtendedNet


def get_network(network_config):

    num_classes = network_config.num_classes

    if network_config.name == "dinov2":
        backbone = torch.hub.load("facebookresearch/dinov2", network_config.vit_name)
        net = DINOv2Wrapper(backbone, num_classes)

    elif network_config.name == "bioclip":
        net = BioCLIPWrapper(num_classes)

    elif network_config.name == "resnet50":
        net = ResNet50(num_classes=num_classes)

    elif network_config.name == "mobilenet":
        net = MobileNet(num_classes=num_classes)

    elif network_config.name == "extended_net":
        backbone = get_network(network_config.backbone)
        backbone.fc = nn.Identity()
        net = ExtendedNet(backbone, network_config.num_closed_set, num_classes)

    if network_config.pretrained:
        if type(net) is dict:
            if isinstance(network_config.checkpoint, list):
                for subnet, checkpoint in zip(net.values(), network_config.checkpoint):
                    if checkpoint is not None:
                        if checkpoint != "none":
                            subnet.load_state_dict(torch.load(checkpoint), strict=False)
            elif isinstance(network_config.checkpoint, str):
                ckpt = torch.load(network_config.checkpoint)
                subnet_ckpts = {k: {} for k in net.keys()}
                for k, v in ckpt.items():
                    for subnet_name in net.keys():
                        if k.startwith(subnet_name):
                            subnet_ckpts[subnet_name][k.replace(subnet_name + ".", "")] = v
                            break

                for subnet_name, subnet in net.items():
                    subnet.load_state_dict(subnet_ckpts[subnet_name])

        elif network_config.name == "bit" and not network_config.normal_load:
            net.load_from(np.load(network_config.checkpoint))
        elif network_config.name == "vit":
            pass
        else:
            checkpoint = torch.load(network_config.checkpoint)
            if "model_state" in checkpoint.keys():
                checkpoint = checkpoint["model_state"]
            try:
                missing_keys, unexpected_keys = net.load_state_dict(checkpoint, strict=False)
                if len(missing_keys) > 0:
                    print("missing_keys: ", missing_keys)
                if len(unexpected_keys) > 0:
                    print("unexpected: ", unexpected_keys)
            except RuntimeError:
                # sometimes fc should not be loaded
                loaded_pth = torch.load(network_config.checkpoint)
                loaded_pth.pop("fc.weight")
                loaded_pth.pop("fc.bias")
                net.load_state_dict(loaded_pth, strict=False)
        print("Model Loading {} Completed!".format(network_config.name))

    # if network_config.num_gpus > 1:
    #     if type(net) is dict:
    #         for key, subnet in zip(net.keys(), net.values()):
    #             net[key] = torch.nn.parallel.DistributedDataParallel(
    #                 subnet.cuda(), device_ids=[comm.get_local_rank()], broadcast_buffers=True
    #             )
    #     else:
    #         net = torch.nn.parallel.DistributedDataParallel(
    #             net.cuda(), device_ids=[comm.get_local_rank()], broadcast_buffers=True
    #         )

    if network_config.num_gpus > 0:
        if type(net) is dict:
            for subnet in net.values():
                subnet.cuda()
        else:
            net.cuda()

    cudnn.benchmark = True
    return net


# # import mmcv
# from copy import deepcopy
# import numpy as np
# import torch
# import torch.backends.cudnn as cudnn
# import torch.nn as nn

# from .dino_v2 import DINOv2Wrapper
# from .bioclip import BioCLIPWrapper
# from .resnet50 import ResNet50
# from .mobilenet import MobileNet
# from .extended_net import ExtendedNet

# device = "cuda" if torch.cuda.is_available() else "cpu"


# def get_network(network_config):
#     num_classes = network_config.num_classes

#     if network_config.name == "dinov2":
#         backbone = torch.hub.load("facebookresearch/dinov2", network_config.vit_name)
#         net = DINOv2Wrapper(backbone, num_classes)

#     elif network_config.name == "bioclip":
#         net = BioCLIPWrapper(num_classes)

#     elif network_config.name == "resnet50":
#         net = ResNet50(num_classes=num_classes)

#     elif network_config.name == "mobilenet":
#         net = MobileNet(num_classes=num_classes)

#     elif network_config.name == "extended_net":
#         backbone = get_network(network_config.backbone)
#         backbone.fc = nn.Identity()
#         net = ExtendedNet(backbone, network_config.num_closed_set, num_classes)

#     if network_config.pretrained:
#         if type(net) is dict:
#             if isinstance(network_config.checkpoint, list):
#                 for subnet, checkpoint in zip(net.values(), network_config.checkpoint):
#                     if checkpoint is not None:
#                         if checkpoint != "none":
#                             subnet.load_state_dict(torch.load(checkpoint), strict=False)
#             elif isinstance(network_config.checkpoint, str):
#                 ckpt = torch.load(network_config.checkpoint)
#                 subnet_ckpts = {k: {} for k in net.keys()}
#                 for k, v in ckpt.items():
#                     for subnet_name in net.keys():
#                         if k.startwith(subnet_name):
#                             subnet_ckpts[subnet_name][k.replace(subnet_name + ".", "")] = v
#                             break

#                 for subnet_name, subnet in net.items():
#                     subnet.load_state_dict(subnet_ckpts[subnet_name])

#         elif network_config.name == "bit" and not network_config.normal_load:
#             net.load_from(np.load(network_config.checkpoint))
#         elif network_config.name == "vit":
#             pass
#         else:
#             try:
#                 missing_keys, unexpected_keys = net.load_state_dict(torch.load(network_config.checkpoint), strict=False)
#                 print(missing_keys)
#                 print(unexpected_keys)
#             except RuntimeError:
#                 # sometimes fc should not be loaded
#                 loaded_pth = torch.load(network_config.checkpoint)
#                 loaded_pth.pop("fc.weight")
#                 loaded_pth.pop("fc.bias")
#                 net.load_state_dict(loaded_pth, strict=False)
#         print("Model Loading {} Completed!".format(network_config.name))

#     return net
