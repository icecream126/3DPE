#!/bin/bash
CUDA_VISIBLE_DEVICES=2 python dig_schnet_qm9.py --target='zpve' --seed=42
CUDA_VISIBLE_DEVICES=2 python dig_schnet_qm9.py --target='U0' --seed=42
CUDA_VISIBLE_DEVICES=2 python dig_schnet_qm9.py --target='U' --seed=42
CUDA_VISIBLE_DEVICES=2 python dig_schnet_qm9.py --target='zpve' --seed=50
CUDA_VISIBLE_DEVICES=2 python dig_schnet_qm9.py --target='U0' --seed=50
CUDA_VISIBLE_DEVICES=2 python dig_schnet_qm9.py --target='U' --seed=50
CUDA_VISIBLE_DEVICES=2 python dig_schnet_qm9.py --target='zpve' --seed=100
CUDA_VISIBLE_DEVICES=2 python dig_schnet_qm9.py --target='U0' --seed=100
CUDA_VISIBLE_DEVICES=2 python dig_schnet_qm9.py --target='U' --seed=100