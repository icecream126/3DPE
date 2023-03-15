#!/bin/bash
CUDA_VISIBLE_DEVICES=6 python dig_schnet_qm9.py --target='lumo' --seed=42 --k=2 --pe='simpPC' --sigma_idx=8
CUDA_VISIBLE_DEVICES=6 python dig_schnet_qm9.py --target='lumo' --seed=42 --k=2 --pe='simpPC' --sigma_idx=9 
CUDA_VISIBLE_DEVICES=6 python dig_schnet_qm9.py --target='mu' --seed=42 --k=2 --pe='simpPC' --sigma_idx=0 