import torch.nn.functional as F
import torch
import torch.nn as nn

from skimage import io
from torchvision import models
from torchvision import transforms


def ResNet50(pretrained=False, mode="eval"):
    net = models.resnet50(pretrained=pretrained)
    net.fc = nn.Linear(in_features=2048, out_features=3)

    net.load_state_dict(torch.load("trained/best-resnet50.pt", map_location=torch.device("cpu")))

    if mode == "eval":
        net.eval()
        for param in net.parameters():
            param.grad = None
    return net


class FixedSizePadding:
    def __init__(self, max_width: int = 366, max_height: int = 258):
        self.MAX_WIDTH = max_width
        self.MAX_HEIGHT = max_height

    def __call__(self, image: torch.Tensor):
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


def inference(model, image):
    use_gpu = torch.cuda.is_available()
    device = "cuda:0" if use_gpu else "cpu"
    if use_gpu:
        print("Using CUDA")
        model.cuda()

    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        _, pred = torch.max(output, 1)
        return pred.item()


if __name__ == "__main__":
    transforms = transforms.Compose([transforms.ToTensor(), FixedSizePadding(), lambda x: x.unsqueeze(0)])

    image = io.imread("data/2/chuongthichin1960_20190116_Small-_Part_0001_2_0.jpg")
    image = transforms(image)

    net = ResNet50()

    prediction = inference(net, image)
    print(prediction)