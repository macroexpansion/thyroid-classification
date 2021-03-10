import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import argparse

from torch.nn.parallel import DistributedDataParallel as DDP

parser.add_argument("--local_rank", type=int)


def example(rank, world_size):

    # create default process group
    torch.distributed.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    # create local model
    model = nn.Linear(10, 10).to(rank)
    # construct DDP model
    ddp_model = DDP(model, device_ids=[rank])
    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # forward pass
    outputs = ddp_model(torch.randn(20, 10).to(rank))
    labels = torch.randn(20, 10).to(rank)
    # backward pass
    loss_fn(outputs, labels).backward()
    # update parameters
    optimizer.step()


def main():
    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()
    # torch.cuda.set_device(args.local_rank)

    world_size = 2
    mp.spawn(example, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    # torch.backends.cudnn.benchmark = True
    main()
