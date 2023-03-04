#!/bin/bash
CUDA_VISIBLE_DEVICES=6 python dig_schnet_qm9.py --target='mu' --seed=42 --k=8 --pe='simpPC' 
CUDA_VISIBLE_DEVICES=6 python dig_schnet_qm9.py --target='U0' --seed=42 --k=8 --pe='simpPC'