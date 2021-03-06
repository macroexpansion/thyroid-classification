from utils import set_seed
from train import train
from models import ResNet50

import torch
import torch.nn as nn
import torch.optim as optim


def train_resnet():
    set_seed(0)
    net = ResNet50(pretrained=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=5e-4)
    train(model=net, loss=criterion, optimizer=optimizer, num_epochs=1000, batch_size=10, seed=3, model_name="resnet50")


if __name__ == "__main__":
    train_resnet()
