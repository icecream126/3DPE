#!/bin/bash
CUDA_VISIBLE_DEVICES=4 python schnetpack_qm9.py --target='homo' --seed=42
# CUDA_VISIBLE_DEVICES=1 python schnet_qm9.py --target='homo' --seed=50
# CUDA_VISIBLE_DEVICES=1 python schnet_qm9.py --target='homo' --seed=100