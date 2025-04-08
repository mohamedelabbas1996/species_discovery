import torch.nn as nn


class ExtendedNet(nn.Module):
    def __init__(self, backbone, num_classes, num_all_classes):
        super(ExtendedNet, self).__init__()

        self.backbone = backbone
        if hasattr(self.backbone, "fc"):
            self.backbone.fc = nn.Identity()

        try:
            feature_size = backbone.feature_size
        except AttributeError:
            feature_size = backbone.module.feature_size

        self.fc = nn.Linear(feature_size, num_classes)
        self.fc_extended = nn.Linear(feature_size, num_all_classes)

    def forward(
        self, x, return_pred_extended=False, return_feature=False, return_both=False
    ):
        feature = self.backbone(x, return_feature=True)
        if return_feature:
            return feature
        pred = self.fc(feature)
        if return_pred_extended:
            pred_extended = self.fc_extended(feature)
            return pred, pred_extended
        if return_both:
            return pred, feature
        else:
            return pred
