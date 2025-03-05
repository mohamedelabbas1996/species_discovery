import timm
import torch.nn as nn


class MobileNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = timm.create_model("mobilenetv3_large_100", pretrained=True, num_classes=num_classes)

    def forward(self, x, return_feature=False):
        feature = self.model.features(x)
        feature = feature.mean([2, 3])
        pred = self.model.classifier(feature)
        if return_feature:
            return pred, feature
        else:
            return pred
