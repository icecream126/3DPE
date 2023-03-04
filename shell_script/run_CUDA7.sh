#!/bin/bash
CUDA_VISIBLE_DEVICES=7 python dig_schnet_qm9.py --target='mu' --seed=42 --k=9 --pe='simpPC' 
CUDA_VISIBLE_DEVICES=7 python dig_schnet_qm9.py --target='U0' --seed=42 --k=9 --pe='simpPC'