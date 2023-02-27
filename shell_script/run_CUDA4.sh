#!/bin/bash
CUDA_VISIBLE_DEVICES=4 python dig_schnet_qm9.py --target='mu' --seed=42 --pe='signinv'
CUDA_VISIBLE_DEVICES=4 python dig_schnet_qm9.py --target='alpha' --seed=42 --pe='signinv'
CUDA_VISIBLE_DEVICES=4 python dig_schnet_qm9.py --target='mu' --seed=50 --pe='signinv'
CUDA_VISIBLE_DEVICES=4 python dig_schnet_qm9.py --target='alpha' --seed=50 --pe='signinv'
CUDA_VISIBLE_DEVICES=4 python dig_schnet_qm9.py --target='mu' --seed=100 --pe='signinv'
CUDA_VISIBLE_DEVICES=4 python dig_schnet_qm9.py --target='alpha' --seed=100 --pe='signinv'