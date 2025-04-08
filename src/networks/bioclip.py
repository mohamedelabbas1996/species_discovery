from bioclip.predict import BaseClassifier
import torch
import torch.nn as nn
import open_clip


class BioCLIPWrapper(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip")
        self.backbone = model
        feature_size = self.backbone.visual.output_dim
        self.fc = nn.Linear(feature_size, num_classes)

    def forward(self, x, return_feature=False, return_both=False):
        feature = self.backbone.encode_image(x)
        if return_feature:
            return feature
        pred = self.fc(feature)
        if return_both:
            return pred, feature
        else:
            return pred
