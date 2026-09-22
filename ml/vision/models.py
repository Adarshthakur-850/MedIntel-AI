import torch
import torch.nn as nn
from torchvision import models

class MedicalResNet18(nn.Module):
    def __init__(self, pretrained=True, num_classes=1):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        
        # Target layer for Grad-CAM access
        self.target_layer = self.backbone.layer4[1].conv2
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def extract_features(self, x):
        # Extract 512-dim embedding before head
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x):
        return self.backbone(x)

def build_vision_model(arch="resnet18", pretrained=True):
    if arch == "resnet18":
        return MedicalResNet18(pretrained=pretrained, num_classes=1)
    else:
        raise ValueError(f"Unsupported vision architecture: {arch}")
