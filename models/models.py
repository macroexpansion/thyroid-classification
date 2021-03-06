import torch
import torch.nn as nn

from torchvision import models


def ResNet50(pretrained=False):
    net = models.resnet50(pretrained=pretrained)
    net.fc = nn.Linear(in_features=2048, out_features=3)

    return net