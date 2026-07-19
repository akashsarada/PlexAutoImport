import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class CategorySorter(nn.Module):
    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.entry = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            DepthwiseSeparableConv(32, 64, stride=2),
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 192, stride=2),
            DepthwiseSeparableConv(192, 256, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.blocks(self.entry(x))))


def param_count(model: nn.Module = None) -> int:
    if model is None:
        model = CategorySorter()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    return total


if __name__ == "__main__":
    _model = CategorySorter()
    _total = param_count(_model)
    assert _total < 250_000, f"Model has {_total:,} parameters, exceeds 250k limit"
    x = torch.randn(2, 3, 128, 128)
    out = _model(x)
    assert out.shape == (2, 3), f"Unexpected output shape: {out.shape}"
