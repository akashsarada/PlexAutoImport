import torch
import torch.nn as nn
import torchvision

class MobileNetV3Small(nn.Module):
    def __init__(self, num_classes: int = 3, pretrained: bool = True) -> None:
        super().__init__()
        weights = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = torchvision.models.mobilenet_v3_small(weights=weights)
        in_features = self.model.classifier[0].in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
