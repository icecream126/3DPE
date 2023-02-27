#!/bin/bash
CUDA_VISIBLE_DEVICES=3 python dig_schnet_qm9.py --target='U' --seed=42 --pe='signinv'
CUDA_VISIBLE_DEVICES=3 python dig_schnet_qm9.py --target='Cv' --seed=42 --pe='signinv'
CUDA_VISIBLE_DEVICES=3 python dig_schnet_qm9.py --target='U' --seed=50 --pe='signinv'
CUDA_VISIBLE_DEVICES=3 python dig_schnet_qm9.py --target='Cv' --seed=50 --pe='signinv'
CUDA_VISIBLE_DEVICES=3 python dig_schnet_qm9.py --target='U' --seed=100 --pe='signinv'
CUDA_VISIBLE_DEVICES=3 python dig_schnet_qm9.py --target='Cv' --seed=100 --pe='signinv'