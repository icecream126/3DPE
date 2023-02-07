#!/bin/bash
CUDA_VISIBLE_DEVICES=5 python spherenet_qm9.py --target='zpve' --seed=100 --pe='hkpe'
CUDA_VISIBLE_DEVICES=5 python spherenet_qm9.py --target='U0' --seed=100 --pe='hkpe'