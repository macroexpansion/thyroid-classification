import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import random

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from torchvision.datasets import ImageFolder
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import set_seed


def datacounter():
    data = {}
    for folder in os.listdir("data"):
        data[folder] = len(os.listdir(os.path.join("data", folder)))
    return data


def split_data(dataset, data="train", test_split_size=0.2, valid_split_size=0.2, seed=1):
    set_seed(seed)
    X_indices = np.array([idx for idx, data in enumerate(dataset.imgs)])
    y = np.array([data[1] for data in dataset.imgs])
    train_indices, test_indices, y_train, y_test = train_test_split(X_indices, y, test_size=test_split_size, random_state=seed)
    if data == "test":
        test_sampler = SubsetRandomSampler(test_indices)
        return {"test": test_sampler}
    train_indices, valid_indices, y_train, y_valid = train_test_split(
        train_indices, y_train, test_size=valid_split_size, random_state=seed
    )

    # print(f"train: {np.bincount(y_train)}, valid: {np.bincount(y_valid)}, test: {np.bincount(y_test)}")

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)

    return {"train": train_sampler, "valid": valid_sampler}


def dataloader(data="train", batch_size=16, transform=transforms.ToTensor(), seed=1):
    path = "data"

    all_data = ImageFolder(root=path, transform=transform)
    # calc_norm(all_data)
    samplers = split_data(all_data, data=data, seed=seed)
    dataloader = DataLoader(all_data, batch_size=batch_size, sampler=samplers[data], pin_memory=True, num_workers=2)

    return dataloader
