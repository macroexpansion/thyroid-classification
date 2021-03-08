import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import random

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import set_seed
from .dataset import IndexedImageFolder


def datacounter():
    data = {}
    for folder in os.listdir("data"):
        data[folder] = len(os.listdir(os.path.join("data", folder)))
    return data


def split_data(dataset, data="train", test_split_size=0.2, valid_split_size=0.2, seed=1, random_sample=True):
    set_seed(seed)
    X_indices = np.array([idx for idx, data in enumerate(dataset.imgs)])
    y = np.array([data[1] for data in dataset.imgs])
    train_indices, test_indices, y_train, y_test = train_test_split(X_indices, y, test_size=test_split_size, random_state=seed)
    if data == "test":
        test_sampler = SubsetRandomSampler(test_indices) if random_sample else SequentialSampler(test_indices)
        return {"test": test_sampler, "test_size": len(test_indices)}
    train_indices, valid_indices, y_train, y_valid = train_test_split(
        train_indices, y_train, test_size=valid_split_size, random_state=seed
    )

    print(f"train: {np.bincount(y_train)}, valid: {np.bincount(y_valid)}, test: {np.bincount(y_test)}")

    train_sampler = SubsetRandomSampler(train_indices) if random_sample else SequentialSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices) if random_sample else SequentialSampler(valid_indices)

    return {"train": train_sampler, "train_size": len(train_indices), "valid": valid_sampler, "valid_size": len(valid_indices)}


def dataloader(data="train", batch_size=16, transform=transforms.ToTensor(), seed=1, random_sample=True):
    path = "data"

    all_data = IndexedImageFolder(root=path, transform=transform)
    samplers = split_data(all_data, data=data, seed=seed, random_sample=random_sample)
    if data == "test":
        testloader = DataLoader(all_data, batch_size=batch_size, sampler=samplers[data], pin_memory=True, num_workers=2)
        return all_data, {"test": testloader}, {"test": samplers["test_size"]}
    else:
        trainloader = DataLoader(all_data, batch_size=batch_size, sampler=samplers["train"], pin_memory=True, num_workers=2)
        validloader = DataLoader(all_data, batch_size=batch_size, sampler=samplers["valid"], pin_memory=True, num_workers=2)
        return (
            all_data,
            {"train": trainloader, "valid": validloader},
            {
                "train": samplers["train_size"],
                "valid": samplers["valid_size"],
            },
        )
