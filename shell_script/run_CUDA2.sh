#!/bin/bash
CUDA_VISIBLE_DEVICES=2 python dig_schnet_qm9.py --target='homo' --seed=42 --pe='signinv'
CUDA_VISIBLE_DEVICES=2 python dig_schnet_qm9.py --target='r2' --seed=42 --pe='signinv'
CUDA_VISIBLE_DEVICES=2 python dig_schnet_qm9.py --target='homo' --seed=50 --pe='signinv'
CUDA_VISIBLE_DEVICES=2 python dig_schnet_qm9.py --target='r2' --seed=50 --pe='signinv'
CUDA_VISIBLE_DEVICES=2 python dig_schnet_qm9.py --target='homo' --seed=100 --pe='signinv'
CUDA_VISIBLE_DEVICES=2 python dig_schnet_qm9.py --target='r2' --seed=100 --pe='signinv'