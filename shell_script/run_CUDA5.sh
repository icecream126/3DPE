#!/bin/bash
CUDA_VISIBLE_DEVICES=5 python dig_schnet_qm9.py --target='mu' --seed=42 --k=6 --pe='simpPC' 
CUDA_VISIBLE_DEVICES=5 python dig_schnet_qm9.py --target='U0' --seed=42 --k=6 --pe='simpPC'