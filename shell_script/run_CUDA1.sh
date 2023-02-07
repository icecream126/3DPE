#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python spherenet_qm9.py --target='homo' --seed=100 --pe='hkpe'
CUDA_VISIBLE_DEVICES=1 python spherenet_qm9.py --target='lumo' --seed=100  --pe='hkpe'