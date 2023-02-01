#!/bin/bash
CUDA_VISIBLE_DEVICES=3 python spherenet_qm9.py --target='zpve' --seed=50 --pe='hkpe'
CUDA_VISIBLE_DEVICES=3 python spherenet_qm9.py --target='U0' --seed=50  --pe='hkpe'
CUDA_VISIBLE_DEVICES=3 python spherenet_qm9.py --target='U' --seed=50  --pe='hkpe'
CUDA_VISIBLE_DEVICES=3 python spherenet_qm9.py --target='H' --seed=50 --pe='hkpe'
CUDA_VISIBLE_DEVICES=3 python spherenet_qm9.py --target='G' --seed=50  --pe='hkpe'
CUDA_VISIBLE_DEVICES=3 python spherenet_qm9.py --target='Cv' --seed=50 --pe='hkpe'