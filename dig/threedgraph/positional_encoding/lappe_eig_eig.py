import numpy as np
import torch 

from torch_geometric.data import Data

from torch_geometric.utils import (
    get_laplacian,
    to_scipy_sparse_matrix,
)
from torch_sparse import SparseTensor
from typing import Optional, Tuple, Union

import torch.nn.functional as F
from torch import Tensor

from torch_geometric.typing import OptTensor
from torch_geometric.transforms import Distance

from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.linalg import eigs
from scipy import sparse
import math

# def SinEncoding(e):
#     constant=100

#     e = torch.tensor(e)
#     ee = torch.tensor(e * constant) # \epsilon * eigval
        
#     return ee



class LapEigEig():
    r"""Adds the Laplacian eigenvector positional encoding from the
    `"Benchmarking Graph Neural Networks" <https://arxiv.org/abs/2003.00982>`_
    paper to the given graph
    (functional name: :obj:`add_laplacian_eigenvector_pe`).

    Args:
        k (int): The number of non-trivial eigenvectors to consider.
        attr_name (str, optional): The attribute name of the data object to add
            positional encodings to. If set to :obj:`None`, will be
            concatenated to :obj:`data.x`.
            (default: :obj:`"laplacian_eigenvector_pe"`)
        is_undirected (bool, optional): If set to :obj:`True`, this transform
            expects undirected graphs as input, and can hence speed up the
            computation of eigenvectors. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :meth:`scipy.sparse.linalg.eigs` (when :attr:`is_undirected` is
            :obj:`False`) or :meth:`scipy.sparse.linalg.eigsh` (when
            :attr:`is_undirected` is :obj:`True`).
    """
    def __init__(
        self,
        k: int,
        is_undirected: bool = False,
        **kwargs,
    ):
        self.k = k
        self.is_undirected = is_undirected
        self.kwargs = kwargs

    def __call__(self, num_nodes, edge_index) -> Data:
        from scipy.sparse.linalg import eigs, eigsh
        eig_fn = eigs if not self.is_undirected else eigsh

        # num_nodes = data.num_nodes
        edge_index, edge_weight = get_laplacian(
            edge_index,
            normalization='sym',
            num_nodes=num_nodes,
        )

        L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes).toarray()

        eigval, eigvec = np.linalg.eig(L)
        idx = eigval.argsort()
        eigval, eigvec = eigval[idx], np.real(eigvec[:,idx])

        # enc_eigval = SinEncoding(eigval)

        # print('eigval : ',eigval.shape)
        # print(eigval)
        # print('eigvec : ',eigvec.shape)
        # print(eigvec)
        pe = torch.from_numpy(eigvec[:,1:self.k+1]).float()
        # print('eigvecpe : ',pe.shape)
        # print(pe)

        if num_nodes <= self.k:
            pe = F.pad(pe, (0, self.k - num_nodes + 1), value=float('0'))

        # https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/train/train_molecules_graph_regression.py
        pe = pe.cuda()
        sign_flip = torch.rand(pe.size(1)).cuda()
        sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
        pe *= sign_flip

        eigval=torch.Tensor(eigval).cuda()
        # https://stackoverflow.com/questions/67267440/torch-concat-1d-to-a-2d-tensor
        pe=torch.cat((pe,eigval.reshape(-1,1)), dim=1) # concat sign flip augmented eigenvector and encoded eigenvalue

        
        return pe


if __name__=='__main__':
    k = 2
    pos = torch.rand(7,3)
    edge_index = torch.tensor([[0,1,1,1,2,2,3,3,3,3,4,5,5,6],
                  [1,0,2,3,1,3,1,2,5,6,5,3,4,3]])
    # sigma_list = [ 0.1, 1, 10, 100]

    lappe = LapEigEig(k=2)
    pe=lappe(num_nodes=7, edge_index=edge_index)

    print('pe : ',pe.shape)
    print(pe)