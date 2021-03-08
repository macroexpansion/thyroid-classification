import torch
import torch.nn as nn

from torchvision import models


def ResNet50(pretrained=False, mode="eval", load_weight=None):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if pretrained and load_weight:
        raise Exception("load_weight must be None when pretrained is True")

    net = models.resnet50(pretrained=pretrained)
    net.fc = nn.Linear(in_features=2048, out_features=3)

    if load_weight == "best":
        net.load_state_dict(torch.load("weights/resnet50/best-resnet50.pt", map_location=torch.device(device)))
    elif load_weight == "last":
        net.load_state_dict(torch.load("weights/resnet50/last-resnet50.pt", map_location=torch.device(device)))

    if mode == "eval":
        net.eval()
        for param in net.parameters():
            param.grad = None
    return net