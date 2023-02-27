#!/bin/bash
CUDA_VISIBLE_DEVICES=7 python dig_schnet_qm9.py --target='H' --seed=42 --pe='signinv'
CUDA_VISIBLE_DEVICES=7 python dig_schnet_qm9.py --target='G' --seed=42 --pe='signinv'
CUDA_VISIBLE_DEVICES=7 python dig_schnet_qm9.py --target='H' --seed=50 --pe='signinv'
CUDA_VISIBLE_DEVICES=7 python dig_schnet_qm9.py --target='G' --seed=50 --pe='signinv'
CUDA_VISIBLE_DEVICES=7 python dig_schnet_qm9.py --target='H' --seed=100 --pe='signinv'
CUDA_VISIBLE_DEVICES=7 python dig_schnet_qm9.py --target='G' --seed=100 --pe='signinv'