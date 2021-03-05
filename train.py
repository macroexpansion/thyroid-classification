import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import os
import time

from utils import set_seed, EarlyStopping
from dataloader import dataloader
from torchvision import transforms
from tqdm import tqdm

set_seed(0)


class FixedSizePadding:
    def __init__(self, max_width=366, max_height=258):
        self.MAX_WIDTH = max_width
        self.MAX_HEIGHT = max_height

    def __call__(self, image):
        _, h, w = image.size()
        left_pad, right_pad, top_pad, bot_pad = 0, 0, 0, 0
        if h < self.MAX_HEIGHT:
            diff = self.MAX_HEIGHT - h
            top_pad = diff // 2
            bot_pad = diff // 2 if diff % 2 == 0 else diff // 2 + 1
        if w < self.MAX_WIDTH:
            diff = self.MAX_WIDTH - w
            left_pad = diff // 2
            right_pad = diff // 2 if diff % 2 == 0 else diff // 2 + 1

        image = F.pad(image, (left_pad, right_pad, top_pad, bot_pad), "constant", 0)

        return image


preprocess = {
    "train": transforms.Compose([transforms.ToTensor(), FixedSizePadding()]),
    "valid": transforms.Compose([transforms.ToTensor(), FixedSizePadding()]),
    "test": transforms.Compose([transforms.ToTensor(), FixedSizePadding()]),
}


def show_image(image: torch.Tensor):
    plt.imshow(image.squeeze().permute(1, 2, 0))
    plt.show()


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
        for phase in ["train", "valid"]:
            if phase == "train":
                print(phase)
                model.train()
            else:
                print(phase)
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0

            data_loader = dataloader(data=phase, batch_size=batch_size, transform=preprocess[phase], seed=seed)
            for images, labels in tqdm(data_loader):
                images = images.to(device)
                labels = labels.to(device)
                # print(images, labels)
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

            data_size = len(data_loader)
            epoch_loss = running_loss / data_size
            epoch_acc = running_corrects / data_size

            if phase == "train":
                print(f"\t--> Loss: {epoch_loss}")
                print(f"\t--> Accuracy: {epoch_acc}")
            else:
                print(f"\t--> Loss: {epoch_loss}")
                print(f"\t--> Accuracy: {epoch_acc}")

            if not os.path.exists(f"weights/{model_name}"):
                os.makedirs(f"weights/{model_name}")
            torch.save(model.state_dict(), f"weights/{model_name}/last-{model_name}.pt")

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


def evaluate(model):
    for param in model.parameters():
        param.grad = None

    for phase in ["test"]:
        with torch.no_grad():
            data_loader = dataloader(data=phase, batch_size=16, transform=preprocess[phase], seed=seed)
            for images, labels in data_loader:
                print(image.size())


__all__ = ["train", "evaluate"]
