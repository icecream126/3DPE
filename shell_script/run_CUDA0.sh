#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python dig_schnet_qm9.py --target='mu' --seed=42
CUDA_VISIBLE_DEVICES=0 python dig_schnet_qm9.py --target='alpha' --seed=42
CUDA_VISIBLE_DEVICES=0 python dig_schnet_qm9.py --target='homo' --seed=42
CUDA_VISIBLE_DEVICES=0 python dig_schnet_qm9.py --target='mu' --seed=50
CUDA_VISIBLE_DEVICES=0 python dig_schnet_qm9.py --target='alpha' --seed=50
CUDA_VISIBLE_DEVICES=0 python dig_schnet_qm9.py --target='homo' --seed=50
CUDA_VISIBLE_DEVICES=0 python dig_schnet_qm9.py --target='mu' --seed=100
CUDA_VISIBLE_DEVICES=0 python dig_schnet_qm9.py --target='alpha' --seed=100
CUDA_VISIBLE_DEVICES=0 python dig_schnet_qm9.py --target='homo' --seed=100