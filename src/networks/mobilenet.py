import timm
import torch.nn as nn
import torch
from torchvision import models


class MobileNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        base_model = timm.create_model(
            "mobilenetv3_large_100", pretrained=True, num_classes=0
        )

        self.forward_features = nn.Sequential(*list(base_model.children())[:-2])
        self.forward_head = base_model.forward_head
        self.classifier = base_model.classifier

    def forward(self, x, return_feature=False, return_both=False):

        feature = self.forward_features(x)
        feature = self.forward_head(feature, pre_logits=True)

        if return_feature:
            return feature

        pred = self.classifier(feature)

        if return_both:
            return pred, feature
        else:
            return pred
