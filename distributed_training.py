import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch
import os
import time
import torch.multiprocessing as mp
import torch.distributed as dist

from torchvision import transforms
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from utils import set_seed, EarlyStopping
from dataloader import dataloader, FixedSizePadding, distributed_dataloader
from logs import log_writer
from models import ResNet50


preprocess = {
    "train": transforms.Compose([transforms.ToTensor(), FixedSizePadding()]),
    "valid": transforms.Compose([transforms.ToTensor(), FixedSizePadding()]),
}


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def distributed_train(world_size: int = 2):
    mp.spawn(
        _distributed_train,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )


def _distributed_train(rank, world_size, num_epochs: int = 5, batch_size: int = 128, seed: int = 3, model_name: str = "resnet50"):
    setup(rank, world_size)

    model = ResNet50(pretrained=False, load_weight=None).to(rank)
    loss_fn = nn.CrossEntropyLoss().to(rank)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    scaler = GradScaler()

    model = DDP(model, device_ids=[rank])

    dataset, data_loader, data_size = distributed_dataloader(
        rank, world_size, data="train", batch_size=batch_size, transform=preprocess["train"], seed=seed
    )
    for epoch in range(1, num_epochs):
        for phase in ["train", "valid"]:
            if phase == "train":
                # print(f"Training on rank {rank}")
                model.train()
            else:
                # print(f"Validating on rank {rank}")
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0

            for images, labels, _ in tqdm(data_loader[phase]):
                labels = labels.to(rank)

                with torch.set_grad_enabled(phase == "train"):
                    with autocast():
                        outputs = model(images)
                        _, preds = torch.max(outputs, 1)
                        loss = loss_fn(outputs, labels)

                        if phase == "train":
                            optimizer.zero_grad()
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()

                running_loss += loss * images.size(0)
                running_corrects += torch.sum(preds == labels.data)

            dist.reduce(running_loss, 0, op=dist.ReduceOp.SUM)
            dist.reduce(running_corrects, 0, op=dist.ReduceOp.SUM)
            if rank == 0:
                print(f"loss: {running_loss.item() / (2 * data_size[phase])}")
                print(f"accuracy: {running_corrects.item() / (2 * data_size[phase])}")

    cleanup()


if __name__ == "__main__":
    distributed_train()
