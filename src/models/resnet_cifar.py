"""CIFAR ResNet32 with an expandable linear classifier."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class CifarResNet(nn.Module):
    def __init__(self, blocks_per_stage: int = 5, feature_dim: int = 64):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, blocks_per_stage, stride=1)
        self.layer2 = self._make_layer(32, blocks_per_stage, stride=2)
        self.layer3 = self._make_layer(feature_dim, blocks_per_stage, stride=2)
        self.feature_dim = feature_dim
        self.classifier: nn.Linear | None = None

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for block_stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, block_stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        return torch.flatten(out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.classifier is None:
            raise RuntimeError("Classifier has not been initialized")
        return self.classifier(self.extract_features(x))

    def expand_classifier(self, num_classes: int) -> Tuple[int, int]:
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")

        old_classifier = self.classifier
        old_classes = 0 if old_classifier is None else old_classifier.out_features
        if num_classes < old_classes:
            raise ValueError("Cannot shrink the classifier")
        if num_classes == old_classes:
            return old_classes, num_classes

        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        new_classifier = nn.Linear(self.feature_dim, num_classes).to(device=device, dtype=dtype)
        nn.init.kaiming_normal_(new_classifier.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(new_classifier.bias)

        if old_classifier is not None:
            with torch.no_grad():
                new_classifier.weight[:old_classes].copy_(old_classifier.weight)
                new_classifier.bias[:old_classes].copy_(old_classifier.bias)

        self.classifier = new_classifier
        return old_classes, num_classes


def resnet32() -> CifarResNet:
    return CifarResNet(blocks_per_stage=5, feature_dim=64)
