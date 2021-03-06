import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import os
import time

from utils import set_seed, EarlyStopping
from dataloader import dataloader, FixedSizePadding
from torchvision import transforms
from tqdm import tqdm


preprocess = {
    "train": transforms.Compose([transforms.ToTensor(), FixedSizePadding()]),
    "valid": transforms.Compose([transforms.ToTensor(), FixedSizePadding()]),
}


def train(model, loss, optimizer, num_epochs=500, batch_size=16, seed=3, model_name="resnet50"):
    use_gpu = torch.cuda.is_available()
    device = "cuda:0" if use_gpu else "cpu"
    if use_gpu:
        print("Using CUDA")
        model.cuda()

    es = EarlyStopping(mode="max", patience=3)
    since = time.time()

    best_acc = 0.0

    for epoch in range(1, num_epochs):
        start = time.time()
        print("-" * 10)
        print("Epoch {}/{}".format(epoch, num_epochs))

        # training and validateing
        data_loader = dataloader(batch_size=batch_size, transform=preprocess["train"], seed=seed)
        for phase in ["train", "valid"]:
            print(phase)
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0

            for images, labels in tqdm(data_loader[phase]):
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    loss_ = loss(outputs, labels)

                    if phase == "train":
                        loss_.backward()
                        optimizer.step()

                running_loss += loss_.item() * images.size(0)
                running_corrects += torch.sum(preds == labels.data)

                del images, labels, outputs, preds
                torch.cuda.empty_cache()

            data_size = batch_size  # len(data_loader)
            epoch_loss = running_loss / data_size
            epoch_acc = running_corrects / data_size

            print(f"\t--> Loss: {epoch_loss}")
            print(f"\t--> Accuracy: {epoch_acc}")

            # save last weight
            if not os.path.exists(f"weights/{model_name}"):
                os.makedirs(f"weights/{model_name}")
            torch.save(model.state_dict(), f"weights/{model_name}/last-{model_name}.pt")

            # logging
            if phase == "valid":
                if es.step(epoch_acc.cpu()):
                    time_elapsed = time.time() - since
                    print("Early Stopping")
                    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
                    print("Best val Acc: {:4f}".format(best_acc))
                    return

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    print("Update best acc: {:4f}".format(best_acc))
                    torch.save(model.state_dict(), f"weights/{model_name}/best-{model_name}.pt")

    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best val Acc: {:4f}".format(best_acc))


__all__ = ["train"]
