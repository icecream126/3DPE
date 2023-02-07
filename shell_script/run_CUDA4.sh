#!/bin/bash
CUDA_VISIBLE_DEVICES=4 python spherenet_qm9.py --target='gap' --seed=100 --pe='hkpe'
CUDA_VISIBLE_DEVICES=4 python spherenet_qm9.py --target='r2' --seed=100 --pe='hkpe'
