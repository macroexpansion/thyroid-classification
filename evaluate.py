import torch
import time

from models import ResNet50
from utils import show_image
from dataloader import dataloader, FixedSizePadding
from tqdm import tqdm
from torchvision import transforms


preprocess = {
    "train": transforms.Compose([transforms.ToTensor(), FixedSizePadding()]),
    "valid": transforms.Compose([transforms.ToTensor(), FixedSizePadding()]),
}


def evaluate(model, batch_size=16, seed=3, model_name="resnet50"):

    for param in model.parameters():
        param.grad = None
    model.eval()

    use_gpu = torch.cuda.is_available()
    device = "cuda:0" if use_gpu else "cpu"
    if use_gpu:
        print("Using CUDA")
        model.cuda()

    with torch.no_grad():
        since = time.time()
        # running_loss = 0.0
        running_corrects = 0.0

        all_predictions = torch.tensor([]).type(torch.int16)
        all_labels = torch.tensor([]).type(torch.int16)

        data_loader = dataloader(batch_size=batch_size, transform=preprocess["train"], seed=seed)
        for images, labels in tqdm(data_loader["train"]):
            images = images.to(device)
            labels = labels.to(device)

        #     outputs = model(images)
        #     _, preds = torch.max(outputs, 1)
        #     all_predictions = torch.cat((all_predictions, preds), 0).type(torch.int16)
        #     all_labels = torch.cat((all_labels, labels), 0).type(torch.int16)

        #     running_corrects += torch.sum(preds == labels.data)

        #     del images, labels, outputs, preds
        #     torch.cuda.empty_cache()

        # data_size = len(data_loader)
        # epoch_acc = running_corrects / data_size

        # print(f"\t--> Loss: {epoch_loss}")
        # print(f"\t--> Accuracy: {epoch_acc}")

        # time_elapsed = time.time() - since
        # print("Evaluating complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))


if __name__ == "__main__":
    net = ResNet50()
    evaluate(net)