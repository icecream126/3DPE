from dig.threedgraph.dataset import QM93D, QM9LapPE, QM9SimplePCLapPE, QM9SignInvLapPE
from dig.threedgraph.method import SchNet
from dig.threedgraph.evaluation import ThreeDEvaluator
from dig.threedgraph.method import run

import torch

import argparse
import sys

parser = argparse.ArgumentParser(description='Argparse')

parser.add_argument('--target', type=str, default='mu')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--pe', type=str, default=None)
parser.add_argument('--k', type=int, default=2)
parser.add_argument('--epoch',type=int, default=300)
parser.add_argument('--sigma',type=int, default=1)

args = parser.parse_args()

target = args.target # targets = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']
seed = args.seed
pe = args.pe
k = args.k
epoch = args.epoch
cutoff=10.0
sigma = args.sigma


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device : ',device)

if pe=='lappe' :
        dataset = QM9LapPE(k=k, cutoff=cutoff)
elif pe=='signinv':
        dataset = QM9SignInvLapPE(k=k, cutoff=cutoff)
elif pe=='simpPC':
        dataset = QM9SimplePCLapPE(k=k, cutoff=cutoff, sigma=sigma)
else:
        dataset = QM93D(root='dataset/')

print('dataset : ',dataset)
print('dataset.data : ',dataset.data)
dataset.data.y = dataset.data[target]
split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=110000, valid_size=10000, seed=seed)
train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]

print('len(train_dataset) : ',len(train_dataset))
print('len(valid_dataset) : ',len(valid_dataset))
print('len(test_dataset) : ',len(test_dataset))

# Define model, loss, and evaluation
model = SchNet(energy_and_force=False, cutoff=10.0, num_layers=6, hidden_channels=128, out_channels=1, num_filters=128, num_gaussians=50, positional_encoding=pe, k=k)   
loss_func = torch.nn.L1Loss()
evaluation = ThreeDEvaluator()

# Train and evaluate
run3d = run()
run3d.run(target, device, train_dataset, valid_dataset, test_dataset, model, loss_func, evaluation,seed, pe,k,
        epochs=epoch, batch_size=32, vt_batch_size=32, lr=0.0005, lr_decay_factor=0.5, lr_decay_step_size=50)
