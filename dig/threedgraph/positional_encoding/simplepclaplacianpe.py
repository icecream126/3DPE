import numpy as np
import torch 

from torch_geometric.data import Data

from torch_geometric.utils import (
    get_laplacian,
    to_scipy_sparse_matrix,
)
from torch_sparse import SparseTensor
from typing import Optional, Tuple, Union

from torch import Tensor
import torch.nn.functional as F


from torch_geometric.typing import OptTensor
from torch_geometric.transforms import Distance
from torch_geometric.utils import to_dense_adj

from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.linalg import eigs
from scipy import sparse
from scipy.linalg import expm

class SimplePCLaplacianEigenvectorPE():
    def __init__(
        self,
        k: int,
        is_undirected: bool = False,
        **kwargs,
    ):
        self.k = k
        self.is_undirected = is_undirected
        self.kwargs = kwargs

    def __call__(self,pos, sigma, edge_index) -> Data:
        # Code Basline from here :  https://github.com/vlivashkin/pygkernels/blob/master/pygkernels/measure/kernel.py
        # First Make weighted Adjacency matrix
        A = to_dense_adj(edge_index).squeeze()

        pos = pos.cpu()
        Dist = squareform(pdist(pos))
        pos = pos.cuda()
        W = torch.tensor(expm(-Dist/(2*sigma**2))) # w_ij = e^{-||x_i-x_j||^2 / 2\sigma^2} # matrix exponential
        W = torch.mul(W, A)
        D = torch.diag(torch.sum(W,1))
        
        
        # Finally construct laplacian matrix from weighted adjacency matrix
        L = D-W
        # L[L!=L]= 0
        # L[L==float("inf")]=10000
        # print('L is nan : ',L.isnan().any())
        # print('L is inf : ',L.isinf().any()) 

        n = pos.shape[0] # number of nodes

        eigval, eigvec = np.linalg.eig(L)
        idx = eigval.argsort()
        eigval, eigvec = eigval[idx], np.real(eigvec[:,idx])
        print('eigval')
        print(eigval)
        print('eigvec')
        print(eigvec)
        pe = torch.from_numpy(eigvec[:,1:self.k+1]).float()

        if n <= self.k:
            pe = F.pad(pe, (0, self.k - n + 1), value=float('0'))

        # https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/train/train_molecules_graph_regression.py
        pe = pe.cuda()
        sign_flip = torch.rand(pe.size(1)).cuda()
        sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
        pe *= sign_flip
        
        return pe


if __name__=='__main__':
    k = 2
    pos = torch.rand(7,3)
    edge_index = torch.tensor([[0,1,1,1,2,2,3,3,3,3,4,5,5,6],
                  [1,0,2,3,1,3,1,2,5,6,5,3,4,3]])
    sigma_list = [ 0.1, 1, 10, 100]

    for sigma in sigma_list:
        print('pos')
        print(pos)
        simpPC = SimplePCLaplacianEigenvectorPE(k=k)
        print('sigma :',sigma)
        pe = simpPC(pos=pos, sigma=sigma,edge_index = edge_index)
        print()

    lappe = LaplacianEigenvectorPE(k=k)
    lappe(pos.shape[0], edge_index)

