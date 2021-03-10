#!/bin/bash
python -m torch.distributed.launch --nproc_per_node=1 --use_env distributed_training.py
