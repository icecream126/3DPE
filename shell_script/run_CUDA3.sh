#!/bin/bash
CUDA_VISIBLE_DEVICES=3 python dig_schnet_qm9.py --target='H' --seed=42
CUDA_VISIBLE_DEVICES=3 python dig_schnet_qm9.py --target='G' --seed=42
CUDA_VISIBLE_DEVICES=3 python dig_schnet_qm9.py --target='Cv' --seed=42
CUDA_VISIBLE_DEVICES=3 python dig_schnet_qm9.py --target='H' --seed=50
CUDA_VISIBLE_DEVICES=3 python dig_schnet_qm9.py --target='G' --seed=50
CUDA_VISIBLE_DEVICES=3 python dig_schnet_qm9.py --target='Cv' --seed=50
CUDA_VISIBLE_DEVICES=3 python dig_schnet_qm9.py --target='H' --seed=100
CUDA_VISIBLE_DEVICES=3 python dig_schnet_qm9.py --target='G' --seed=100
CUDA_VISIBLE_DEVICES=3 python dig_schnet_qm9.py --target='Cv' --seed=100