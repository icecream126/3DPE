#!/bin/bash
CUDA_VISIBLE_DEVICES=5 python dig_schnet_qm9.py --target='lumo' --seed=42 --pe='signinv'
CUDA_VISIBLE_DEVICES=5 python dig_schnet_qm9.py --target='gap' --seed=42 --pe='signinv'
CUDA_VISIBLE_DEVICES=5 python dig_schnet_qm9.py --target='lumo' --seed=50 --pe='signinv'
CUDA_VISIBLE_DEVICES=5 python dig_schnet_qm9.py --target='gap' --seed=50 --pe='signinv'
CUDA_VISIBLE_DEVICES=5 python dig_schnet_qm9.py --target='lumo' --seed=100 --pe='signinv'
CUDA_VISIBLE_DEVICES=5 python dig_schnet_qm9.py --target='gap' --seed=100 --pe='signinv'