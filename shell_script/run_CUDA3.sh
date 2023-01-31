#!/bin/bash
CUDA_VISIBLE_DEVICES=3 python spherenet_qm9.py --target='zpve' --seed=42 --pe='lappe'
CUDA_VISIBLE_DEVICES=3 python spherenet_qm9.py --target='U' --seed=42  --pe='lappe'
CUDA_VISIBLE_DEVICES=3 python spherenet_qm9.py --target='H' --seed=42 --pe='lappe'
CUDA_VISIBLE_DEVICES=3 python spherenet_qm9.py --target='G' --seed=42  --pe='lappe'
CUDA_VISIBLE_DEVICES=3 python spherenet_qm9.py --target='Cv' --seed=42 --pe='lappe'