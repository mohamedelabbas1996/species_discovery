import torch
import torch.nn as nn
from .projection_head import ProjectionHead


class DINOv2Wrapper(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        feature_size = self.backbone.embed_dim
        self.proj_head = ProjectionHead(in_dim=feature_size, out_dim=66536)

    def forward(
        self, x, return_feature=False, return_both_features=False, return_both=False
    ):
        feature = self.backbone(x)
        if return_feature:
            return feature

        proj_feat, pred = self.proj_head(feature, return_both=True)
        if return_both_features:
            return feature, proj_feat
        if return_both:
            return pred, feature
        else:
            return pred
