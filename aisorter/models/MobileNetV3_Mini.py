import torch
import torch.nn as nn

class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        reduced_channels = max(1, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced_channels, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, 1, bias=True),
            nn.Hardsigmoid()
        )
    def forward(self, x):
        return x * self.se(x)


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, exp_channels, out_channels, kernel_size, stride, use_se, act_layer):
        super().__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels
        
        layers = []
        # Expand 1x1
        if exp_channels != in_channels:
            layers.append(nn.Conv2d(in_channels, exp_channels, 1, stride=1, padding=0, bias=False))
            layers.append(nn.BatchNorm2d(exp_channels))
            layers.append(act_layer())
        
        # Depthwise
        padding = (kernel_size - 1) // 2
        layers.append(nn.Conv2d(exp_channels, exp_channels, kernel_size, stride, padding, groups=exp_channels, bias=False))
        layers.append(nn.BatchNorm2d(exp_channels))
        layers.append(act_layer())
        
        # Squeeze-and-Excitation
        if use_se:
            layers.append(SqueezeExcitation(exp_channels))
            
        # Project 1x1
        layers.append(nn.Conv2d(exp_channels, out_channels, 1, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3Mini(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # Stem layer
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish()
        )
        
        # Core blocks (MobileNetV3 style: kernel, expansion, output, SE, activation, stride)
        self.bneck = nn.Sequential(
            InvertedResidual(16, 16, 16, kernel_size=3, stride=1, use_se=True, act_layer=nn.ReLU),
            InvertedResidual(16, 48, 24, kernel_size=3, stride=2, use_se=False, act_layer=nn.ReLU),
            InvertedResidual(24, 72, 24, kernel_size=3, stride=1, use_se=False, act_layer=nn.ReLU),
            InvertedResidual(24, 72, 40, kernel_size=5, stride=2, use_se=True, act_layer=nn.Hardswish),
            InvertedResidual(40, 120, 40, kernel_size=5, stride=1, use_se=True, act_layer=nn.Hardswish),
            InvertedResidual(40, 120, 80, kernel_size=3, stride=2, use_se=False, act_layer=nn.Hardswish),
            InvertedResidual(80, 240, 80, kernel_size=3, stride=1, use_se=False, act_layer=nn.Hardswish),
        )
        
        # Last conv stage
        self.conv_last = nn.Sequential(
            nn.Conv2d(80, 240, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(240),
            nn.Hardswish()
        )
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(240, 120),
            nn.Hardswish(),
            nn.Dropout(0.3),
            nn.Linear(120, num_classes)
        )
        
    def forward(self, x):
        x = self.stem(x)
        x = self.bneck(x)
        x = self.conv_last(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
