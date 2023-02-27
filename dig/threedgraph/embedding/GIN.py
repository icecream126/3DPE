import torch
import torch.nn as nn
# import dgl
# from dgl.nn.pytorch import GraphConv, GATConv, GINConv
from torch_geometric.nn.conv import GINConv

# from embedding.mlp import MLP

import torch.nn.functional as F

from typing import Callable, Optional, Union

from torch import Tensor

from torch_geometric.utils.convert import to_networkx
# import dgl
# from dgl import from_networkx

device='cuda'

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, use_bn=False, use_ln=False, dropout=0.5, activation='relu', residual=False):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        if use_bn: self.bns = nn.ModuleList()
        if use_ln: self.lns = nn.ModuleList()
        
        if num_layers == 1:
            # linear mapping
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            if use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels))
            if use_ln: self.lns.append(nn.LayerNorm(hidden_channels))
            for layer in range(num_layers-2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                if use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels))
                if use_ln: self.lns.append(nn.LayerNorm(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError('Invalid activation')
        self.use_bn = use_bn
        self.use_ln = use_ln
        self.dropout = dropout
        self.residual = residual
    
    @torch.no_grad()        
    def forward(self, x):
        x_prev = x
        for i, lin in enumerate(self.lins[:-1]):
            x = x
            x = lin(x)
            x = self.activation(x)
            if self.use_bn:
                if x.ndim == 2:
                    x = self.bns[i](x)
                elif x.ndim == 3:
                    x = self.bns[i](x.transpose(2,1)).transpose(2,1).to('cuda')
                else:
                    raise ValueError('invalid dimension of x')
            if self.use_ln: x = self.lns[i](x)
            if self.residual and x_prev.shape == x.shape: x = x + x_prev
            x = F.dropout(x, p=self.dropout, training=self.training)
            x_prev = x
        x = self.lins[-1](x)
        if self.residual and x_prev.shape == x.shape:
            x = x + x_prev
        return x

class GIN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers, use_bn=True, dropout=0.5, activation='relu'):
        super(GIN, self).__init__()
        self.layers = nn.ModuleList()
        if use_bn: self.bns = nn.ModuleList()
        self.use_bn = use_bn
        self.activation = activation# activations[activation]
        # input layer
        update_net = MLP(in_channels, hidden_channels, hidden_channels, 2, use_bn=use_bn, dropout=dropout, activation=activation)
        # self.layers.append(GINConv(update_net, 'sum'))
        self.layers.append(GINConv(update_net)) # problem here,,,, I should use DGL graph anyway because torch geometric cannot handle eigenvector input.

        # hidden layers
        for i in range(n_layers - 2):
            update_net = MLP(hidden_channels, hidden_channels, hidden_channels, 2, use_bn=use_bn, dropout=dropout, activation=activation)
            # self.layers.append(GINConv(update_net, 'sum'))
            self.layers.append(GINConv(update_net))
            if use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels))
        # output layer
        update_net = MLP(hidden_channels, hidden_channels, out_channels, 2, use_bn=use_bn, dropout=dropout, activation=activation)
        # self.layers.append(GINConv(update_net, 'sum'))
        self.layers.append(GINConv(update_net))

        if use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.dropout = nn.Dropout(p=dropout)

    @torch.no_grad()
    def forward(self, g, x):
        for i, layer in enumerate(self.layers):
            if i != 0:
                x = self.dropout(x)
                if self.use_bn:
                    if x.ndim == 2:
                        x = self.bns[i-1](x)
                    elif x.ndim == 3:
                        x = self.bns[i-1](x.transpose(2,1)).transpose(2,1)
                    else:
                        raise ValueError('invalid x dim')

            x = layer(x=x, edge_index = g.edge_index)
        return x
    

class GINDeepSigns(nn.Module):
    """ Sign invariant neural network
        f(v1, ..., vk) = rho(enc(v1) + enc(-v1), ..., enc(vk) + enc(-vk))
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, k, use_bn=False, use_ln=False, dropout=0.5, activation='relu'):
        super(GINDeepSigns, self).__init__()
        self.enc = GIN(in_channels, hidden_channels, out_channels, num_layers, use_bn=use_bn, dropout=dropout, activation=activation) # Problem Here
        # rho_dim = out_channels * k
        rho_dim = 2*k
        self.rho = MLP(rho_dim, hidden_channels, k, num_layers, use_bn=use_bn, dropout=dropout, activation=activation)
        self.k = k

    @torch.no_grad()
    def forward(self, g, x):
        
        x = self.enc(g, x) + self.enc(g, -x)
        orig_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        x = self.rho(x)
        x = x.reshape(orig_shape[0], self.k, 1)
        return x