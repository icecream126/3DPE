#!/bin/bash
CUDA_VISIBLE_DEVICES=3 python schnetpack_qm9.py --target='mu' --seed=42
# CUDA_VISIBLE_DEVICES=0 python schnet_qm9.py --target='mu' --seed=50
# CUDA_VISIBLE_DEVICES=0 python schnet_qm9.py --target='mu' --seed=100