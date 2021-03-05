from utils import set_seed
from train import train
from torchvision import models

import torch
import torch.nn as nn
import torch.optim as optim


def ResNet50(pretrained=False):
    net = models.resnet50(pretrained=pretrained)
    net.fc = nn.Linear(in_features=2048, out_features=3)

    return net


if __name__ == "__main__":
    set_seed(0)
    net = ResNet50(pretrained=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=5e-4)
    train(model=net, loss=criterion, optimizer=optimizer, num_epochs=1000, batch_size=64, seed=3, model_name="resnet50")
