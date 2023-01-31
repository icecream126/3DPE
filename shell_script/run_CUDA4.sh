#!/bin/bash
CUDA_VISIBLE_DEVICES=4 python spherenet_qm9.py --target='mu' --seed=50
CUDA_VISIBLE_DEVICES=4 python spherenet_qm9.py --target='alpha' --seed=50
CUDA_VISIBLE_DEVICES=4 python spherenet_qm9.py --target='homo' --seed=50
CUDA_VISIBLE_DEVICES=4 python spherenet_qm9.py --target='lumo' --seed=50
CUDA_VISIBLE_DEVICES=4 python spherenet_qm9.py --target='gap' --seed=50
CUDA_VISIBLE_DEVICES=4 python spherenet_qm9.py --target='r2' --seed=50
CUDA_VISIBLE_DEVICES=4 python spherenet_qm9.py --target='zpve' --seed=50
CUDA_VISIBLE_DEVICES=4 python spherenet_qm9.py --target='U0' --seed=50
CUDA_VISIBLE_DEVICES=4 python spherenet_qm9.py --target='U' --seed=50
CUDA_VISIBLE_DEVICES=4 python spherenet_qm9.py --target='H' --seed=50
CUDA_VISIBLE_DEVICES=4 python spherenet_qm9.py --target='G' --seed=50
CUDA_VISIBLE_DEVICES=4 python spherenet_qm9.py --target='Cv' --seed=50