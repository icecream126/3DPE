#!/bin/bash
CUDA_VISIBLE_DEVICES=6 python dig_schnet_qm9.py --target='zpve' --seed=42 --pe='signinv'
CUDA_VISIBLE_DEVICES=6 python dig_schnet_qm9.py --target='U0' --seed=42 --pe='signinv'
CUDA_VISIBLE_DEVICES=6 python dig_schnet_qm9.py --target='zpve' --seed=50 --pe='signinv'
CUDA_VISIBLE_DEVICES=6 python dig_schnet_qm9.py --target='U0' --seed=50 --pe='signinv'
CUDA_VISIBLE_DEVICES=6 python dig_schnet_qm9.py --target='zpve' --seed=100 --pe='signinv'
CUDA_VISIBLE_DEVICES=6 python dig_schnet_qm9.py --target='U0' --seed=100 --pe='signinv'