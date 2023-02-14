#!/bin/bash
CUDA_VISIBLE_DEVICES=6 python schnet_qm9.py --target='G' --seed=50 --pe='lappe'
CUDA_VISIBLE_DEVICES=6 python schnet_qm9.py --target='G' --seed=100 --pe='lappe'