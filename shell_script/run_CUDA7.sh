#!/bin/bash
CUDA_VISIBLE_DEVICES=7 python dig_schnet_qm9.py --target='homo' --seed=42
# CUDA_VISIBLE_DEVICES=1 python schnet_qm9.py --target='homo' --seed=50
# CUDA_VISIBLE_DEVICES=1 python schnet_qm9.py --target='homo' --seed=100