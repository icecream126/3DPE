#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python spherenet_qm9.py --target='zpve' --seed=42 --pe='hkpe'
CUDA_VISIBLE_DEVICES=1 python spherenet_qm9.py --target='U0' --seed=42  --pe='hkpe'
CUDA_VISIBLE_DEVICES=1 python spherenet_qm9.py --target='U' --seed=42  --pe='hkpe'
CUDA_VISIBLE_DEVICES=1 python spherenet_qm9.py --target='H' --seed=42 --pe='hkpe'
CUDA_VISIBLE_DEVICES=1 python spherenet_qm9.py --target='G' --seed=42  --pe='hkpe'
CUDA_VISIBLE_DEVICES=1 python spherenet_qm9.py --target='Cv' --seed=42 --pe='hkpe'