#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python spherenet_qm9.py --target='mu' --seed=100 --pe='hkpe'
CUDA_VISIBLE_DEVICES=0 python spherenet_qm9.py --target='alpha' --seed=100 --pe='hkpe'