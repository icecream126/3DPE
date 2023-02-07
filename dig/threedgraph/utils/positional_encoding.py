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


class QM9LapPE(InMemoryDataset):
    def __init__(self, data_list, dataset, k, cutoff):
        super(QM9LapPE, self).__init__('../dataset/qm9/lappe/k_2')
        # self.data, self.slices = self.collate(data_list)
        self.data, self.slices = torch.load(self.processd_paths[0])
        self.orig_qm9 = dataset
        self.k = k
        self.cutoff = cutoff

    @property
    def raw_file_names(self):
        return 'qm9_lappe_k_'+k+'_'+cutoff+'.npz'

    @property
    def processed_file_names(self):
        return 'qm9_lappe_k_'+k+'_'+cutoff+'.pt'
        

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train':train_idx, 'valid':val_idx, 'test':test_idx}
        return split_dict

    def process(self):
        data_list = []
        lappe = LaplacianEigenvectorPE(k)
        for data in self.orig_qm9:
            edge_index = radius_graph(data.pos, r=cutoff)
            data.pe = lappe(data.pos.shape[0], edge_index)
            data_list.append(data)




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
