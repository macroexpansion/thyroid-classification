from utils import set_seed
from dataloader import dataloader
from torchvision import transforms
from train import train


if __name__ == "__main__":
    set_seed(0)
    train()
