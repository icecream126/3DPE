import numpy as np
import torch 

from torch_geometric.data import Data, InMemoryDataset

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
import sys


# sys.path.append('/home/hsjang/hmkim/3DPE/dig/threedgraph/embedding')
sys.path.append('/home/guest_khm/hyomin/3DPE/dig/threedgraph/embedding')


from GIN import GINDeepSigns


class CleanLaplacianEigenvectorPE(InMemoryDataset):
    # Returning eigenvector with no sign-flip augmentation
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

    def __call__(self, g, num_nodes, edge_index) -> Data:
        # https://github.com/cptq/SignNet-BasisNet/blob/main/GraphPrediction/configs/gin/GIN_ZINC_LapPE_signinv_GIN.json
        # sign_inv_net = GINDeepSigns(1, net_params['hidden_dim'], net_params['phi_out_dim'], net_params['sign_inv_layers'], net_params['pos_enc_dim'], use_bn=True, dropout=net_params['dropout'], activation=net_params['sign_inv_activation'])
        
        sign_inv_net = GINDeepSigns(2, 95, 4, 8, 2, use_bn=True, dropout=0.0, activation="relu") # Problem Here
        sign_inv_net = sign_inv_net.to('cuda')
        
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
        pe = torch.from_numpy(eigvec[:,1:self.k+1]).float()

        if num_nodes <= self.k:
            pe = F.pad(pe, (0, self.k - num_nodes + 1), value=float('0'))

        # https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/train/train_molecules_graph_regression.py
        # uncomment this to use gpu
        pe = pe.cuda()
        g = g.to('cuda')
        pe = sign_inv_net(g, pe)

        return pe
