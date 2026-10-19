"""Randomly initialized CIFAR-style ResNet-18 used in the experiments."""

import torch
from torchvision.models import resnet18


def make_model(num_classes: int, seed: int = 1234):
    torch.manual_seed(seed)
    model = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1,
                                  padding=1, bias=False)
    model.maxpool = torch.nn.Identity()
    return model
