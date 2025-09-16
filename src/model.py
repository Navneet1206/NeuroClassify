import torch
import torch.nn as nn
from torchvision import models


class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes: int = 4, freeze_until_layer: int = 0):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
        # Freeze layers optionally
        if freeze_until_layer > 0:
            ct = 0
            for child in self.backbone.children():
                ct += 1
                if ct <= freeze_until_layer:
                    for p in child.parameters():
                        p.requires_grad = False

    def forward(self, x):
        return self.backbone(x)
