#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python dig_schnet_qm9.py --target='lumo' --seed=42
CUDA_VISIBLE_DEVICES=1 python dig_schnet_qm9.py --target='gap' --seed=42
CUDA_VISIBLE_DEVICES=1 python dig_schnet_qm9.py --target='r2' --seed=42
CUDA_VISIBLE_DEVICES=1 python dig_schnet_qm9.py --target='lumo' --seed=50
CUDA_VISIBLE_DEVICES=1 python dig_schnet_qm9.py --target='gap' --seed=50
CUDA_VISIBLE_DEVICES=1 python dig_schnet_qm9.py --target='r2' --seed=50
CUDA_VISIBLE_DEVICES=1 python dig_schnet_qm9.py --target='lumo' --seed=100
CUDA_VISIBLE_DEVICES=1 python dig_schnet_qm9.py --target='gap' --seed=100
CUDA_VISIBLE_DEVICES=1 python dig_schnet_qm9.py --target='r2' --seed=100