#!/bin/bash
CUDA_VISIBLE_DEVICES=2 python spherenet_qm9.py --target='alpha' --seed=42 --pe='lappe'
CUDA_VISIBLE_DEVICES=2 python spherenet_qm9.py --target='homo' --seed=42  --pe='lappe'
CUDA_VISIBLE_DEVICES=2 python spherenet_qm9.py --target='lumo' --seed=42 --pe='lappe'
CUDA_VISIBLE_DEVICES=2 python spherenet_qm9.py --target='gap' --seed=42  --pe='lappe'
CUDA_VISIBLE_DEVICES=2 python spherenet_qm9.py --target='r2' --seed=42 --pe='lappe'