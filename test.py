import torch

a = torch.tensor([]).type(torch.int16)
b = torch.tensor([0, 0, 0, 0, 0]).type(torch.int16)
a = torch.cat((a, b), 0).type(torch.int16)
print(a)