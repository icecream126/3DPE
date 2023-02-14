#!/bin/bash
CUDA_VISIBLE_DEVICES=7 python schnet_qm9.py --target='Cv' --seed=50 --pe='lappe'
CUDA_VISIBLE_DEVICES=7 python schnet_qm9.py --target='Cv' --seed=100 --pe='lappe'