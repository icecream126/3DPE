#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python spherenet_qm9.py --target='mu' --seed=42 --pe='hkpe'
CUDA_VISIBLE_DEVICES=0 python spherenet_qm9.py --target='alpha' --seed=42 --pe='hkpe'
CUDA_VISIBLE_DEVICES=0 python spherenet_qm9.py --target='homo' --seed=42  --pe='hkpe'
CUDA_VISIBLE_DEVICES=0 python spherenet_qm9.py --target='lumo' --seed=42 --pe='hkpe'
CUDA_VISIBLE_DEVICES=0 python spherenet_qm9.py --target='gap' --seed=42  --pe='hkpe'
CUDA_VISIBLE_DEVICES=0 python spherenet_qm9.py --target='r2' --seed=42 --pe='hkpe]'