from ..positional_encoding.laplacianpe import LaplacianEigenvectorPE
from ..positional_encoding.randomwalkpe import RandomWalkPE
from ..positional_encoding.heatkernelpe import HeatKernelEigenvectorPE
from torch_geometric.data import InMemoryDataset

import torch
from sklearn.utils import shuffle

from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import radius_graph

class QM93DPE(InMemoryDataset):
    def __init__(self, data_list):
        super(QM93DPE, self).__init__('../dataset/')
        self.data, self.slices = self.collate(data_list)
    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train':train_idx, 'valid':val_idx, 'test':test_idx}
        return split_dict



def positional_encoding(dataset, pe, k, cutoff):
    # dataset_size = len(dataset.data.y)
    data_list = []

    if pe == 'lappe':
        print('lappe')
        lappe = LaplacianEigenvectorPE(k)
        for data in dataset:
            edge_index = radius_graph(data.pos, r=cutoff)
            data.pe = lappe(data.pos.shape[0], edge_index)
            data_list.append(data)
        
    elif pe == 'hkpe':
        print('hkpe')
        hkpe = HeatKernelEigenvectorPE(k)
        for data in dataset:
            data.pe = hkpe(data.pos)
            data_list.append(data)


    elif pe == 'rwpe':
        print('rwpe')
        rwpe = RandomWalkPE(dataset, k)
        for data in dataset:
            data.pe = rwpe(data.pos.shape[0], edge_index)
    

    dataset = QM93DPE(data_list)
    return dataset