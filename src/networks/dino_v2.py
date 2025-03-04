import torch
import torch.nn as nn


class DINOv2Wrapper(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        feature_size = self.backbone.embed_dim
        self.fc = nn.Linear(feature_size, num_classes)

    def forward(self, x, return_feature=False):
        feature = self.backbone(x)
        pred = self.fc(feature)
        if return_feature:
            return pred, feature
        else:
            return pred
