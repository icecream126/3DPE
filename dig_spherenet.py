from dig.threedgraph.dataset import QM93D
from dig.threedgraph.method import SphereNet
from dig.threedgraph.evaluation import ThreeDEvaluator
from dig.threedgraph.method import run
from dig.threedgraph.utils import positional_encoding

import torch

import argparse
import sys

parser = argparse.ArgumentParser(description='Argparse')

parser.add_argument('--target', type=str, default='mu')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--pe', type=str, default=None)
parser.add_argument('--k', type=int, default=2)


args = parser.parse_args()

target = args.target # targets = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']
seed = args.seed
pe = args.pe
k = args.k

cutoff=5.0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the dataset and split
dataset = QM93D(root='dataset/')

if pe is not None:
        dataset = positional_encoding(dataset, pe, k, cutoff)
dataset.data.y = dataset.data[target]
split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=110000, valid_size=10000, seed=seed)
train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]

print('len(train_dataset) : ',len(train_dataset))
print('len(valid_dataset) : ',len(valid_dataset))
print('len(test_dataset) : ',len(test_dataset))



# Define model, loss, and evaluation
model = SphereNet(energy_and_force=False, cutoff=5.0, num_layers=4,
                hidden_channels=128, out_channels=1, int_emb_size=64,
                basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8, out_emb_channels=256,
                num_spherical=3, num_radial=6, envelope_exponent=5,
                num_before_skip=1, num_after_skip=2, num_output_layers=3, positional_encoding=pe, k=k)                 
loss_func = torch.nn.L1Loss()
evaluation = ThreeDEvaluator()

# Train and evaluate
run3d = run()
run3d.run(target, device, train_dataset, valid_dataset, test_dataset, model, loss_func, evaluation, seed, pe,k,
        epochs=300, batch_size=32, vt_batch_size=32, lr=0.0005, lr_decay_factor=0.5, lr_decay_step_size=50)
