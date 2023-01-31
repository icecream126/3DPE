from dig.threedgraph.dataset import QM93D
from dig.threedgraph.method import SchNet
from dig.threedgraph.evaluation import ThreeDEvaluator
from dig.threedgraph.method import run

import torch

targets = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']

for target in targets:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the dataset and split
    dataset = QM93D(root='dataset/')
    dataset.data.y = dataset.data[target]
    split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=110000, valid_size=10000, seed=42)
    train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]

    # Define model, loss, and evaluation
    model = SchNet(energy_and_force=False, cutoff=10.0, num_layers=6, hidden_channels=128, out_channels=1, num_filters=128, num_gaussians=50)   
    loss_func = torch.nn.L1Loss()
    evaluation = ThreeDEvaluator()

    # Train and evaluate
    run3d = run()
    run3d.run(target, device, train_dataset, valid_dataset, test_dataset, model, loss_func, evaluation,
            epochs=20, batch_size=32, vt_batch_size=32, lr=0.0005, lr_decay_factor=0.5, lr_decay_step_size=15)
