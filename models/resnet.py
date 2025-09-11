# ──────────────────────────────────────────────────────────────────────────────
# file: resnet18_model.py
# ──────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torchvision import models
from base_model import BaseModel

class ResNet18Model(BaseModel):
    def __init__(self, num_classes=4, dropout=0.3, freeze_until_layer='layer4', pretrained=True):
        super().__init__(num_classes, dropout)
        self.freeze_until_layer = freeze_until_layer
        self.pretrained = pretrained
        self.model = self.build_model()

    def build_model(self):
        model = models.resnet18(pretrained=self.pretrained)

        # Freeze layers
        for name, param in model.named_parameters():
            if self.freeze_until_layer not in name:
                param.requires_grad = False

        # Replace fc
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(256, self.num_classes)
        )
        return model

    def forward(self, x):
        return self.model(x)
