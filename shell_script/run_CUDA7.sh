#!/bin/bash
CUDA_VISIBLE_DEVICES=7 python spherenet_qm9.py --target='mu' --seed=50 --pe='lappe'
CUDA_VISIBLE_DEVICES=7 python spherenet_qm9.py --target='alpha' --seed=50 --pe='lappe'
CUDA_VISIBLE_DEVICES=7 python spherenet_qm9.py --target='homo' --seed=50  --pe='lappe'
CUDA_VISIBLE_DEVICES=7 python spherenet_qm9.py --target='lumo' --seed=50 --pe='lappe'
CUDA_VISIBLE_DEVICES=7 python spherenet_qm9.py --target='gap' --seed=50  --pe='lappe'
CUDA_VISIBLE_DEVICES=7 python spherenet_qm9.py --target='r2' --seed=50 --pe='lappe'