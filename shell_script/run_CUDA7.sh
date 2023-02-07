#!/bin/bash
CUDA_VISIBLE_DEVICES=7 python spherenet_qm9.py --target='G' --seed=100 --pe='hkpe'
CUDA_VISIBLE_DEVICES=7 python spherenet_qm9.py --target='Cv' --seed=100 --pe='hkpe'