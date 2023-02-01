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

from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.linalg import eigs
from scipy import sparse

class HeatKernelEigenvectorPE():
    def __init__(
        self,
        k: int,
        is_undirected: bool = False,
        **kwargs,
    ):
        self.k = k
        self.is_undirected = is_undirected
        self.kwargs = kwargs

    def __call__(self,pos) -> Data:
        # Code Basline from here :  https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/data/molecules.py
        pos = pos.cpu()
        D = squareform(pdist(pos))
        pos = pos.cuda()

        n = pos.shape[0] # number of nodes

        eigval, eigvec = np.linalg.eig(D)
        idx = eigval.argsort()
        eigval, eigvec = eigval[idx], np.real(eigvec[:,idx])
        pe = torch.from_numpy(eigvec[:,1:self.k+1]).float()

        if n <= self.k:
            pe = F.pad(pe, (0, self.k - n + 1), value=float('0'))

        # https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/train/train_molecules_graph_regression.py
        pe = pe.cuda()
        sign_flip = torch.rand(pe.size(1)).cuda()
        sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
        pe *= sign_flip
        
        return pe