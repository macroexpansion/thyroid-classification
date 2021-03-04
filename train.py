from utils import set_seed
from dataloader import dataloader
from torchvision import transforms

set_seed(0)
preprocess = {
    "train": transforms.Compose([transforms.ToTensor()]),
    "valid": transforms.Compose([transforms.ToTensor()]),
    "test": transforms.Compose([transforms.ToTensor()]),
}


def train():
    for phase in ["train", "valid"]:
        data_loader = dataloader(data=phase, batch_size=1, transform=preprocess[phase])
        for data in data_loader:
            print(len(data_loader))
            break

    for phase in ["test"]:
        data_loader = dataloader(data=phase, batch_size=1, transform=preprocess[phase])
        for data in data_loader:
            print(len(data_loader))
            break


__all__ = ["train"]
