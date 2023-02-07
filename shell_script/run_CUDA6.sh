#!/bin/bash
CUDA_VISIBLE_DEVICES=6 python spherenet_qm9.py --target='U' --seed=100  --pe='hkpe'
CUDA_VISIBLE_DEVICES=6 python spherenet_qm9.py --target='H' --seed=100 --pe='hkpe'