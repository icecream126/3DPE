# from ..positional_encoding.import LaplacianEigenvectorPE
# from ..positional_encoding.randomwalkpe import RandomWalkPE
# from ..positional_encoding.heatkernelpe import HeatKernelEigenvectorPE

import sys
# sys.path.append('/home/hsjang/hmkim/3DPE/dig/threedgraph/positional_encoding')
# sys.path.append('/home/hsjang/hmkim/3DPE/dig/threedgraph/dataset')

sys.path.append('/home/guest_khm/hyomin/3DPE/dig/threedgraph/positional_encoding')
sys.path.append('/home/guest_khm/hyomin/3DPE/dig/threedgraph/dataset')


from laplacianpe import LaplacianEigenvectorPE
from randomwalkpe import RandomWalkPE
from heatkernelpe import HeatKernelEigenvectorPE
from PygQM93D import QM93D


import torch
from sklearn.utils import shuffle

from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import radius_graph


class QM9LapPE(InMemoryDataset):
    
    # def __init__(self,k, cutoff): 
    # def __init__(self, data_list, k, cutoff) : # Before first processing
    
    # After first processing
    def __init__(self,k, cutoff): 
        self.k = k
        # self.data_list = data_list # Before first processing
        self.orig_qm9 = QM93D(root='dataset/')
        self.cutoff = cutoff
        super(QM9LapPE, self).__init__('./dataset/qm9/lappe/')
        self.data, self.slices = torch.load(self.processed_paths[0]) # After first processing
        # self.data, self.slices = self.collate(data_list) # Before first processing
        
        
    def __str__(self):
        return f"k is {self.k}, cutoff is {self.cutoff}"

    @property
    def raw_file_names(self):
        return 'qm9_lappe_k_'+str(self.k)+'_cutoff_'+str(self.cutoff)+'.npz'

    @property
    def processed_file_names(self):
        return 'qm9_lappe_k_'+str(self.k)+'_cutoff_'+str(self.cutoff)+'.pt'
        return 'qm9_lappe_k_'+str(self.k)+'_cutoff_'+str(self.cutoff)+'.pt'
        

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train':train_idx, 'valid':val_idx, 'test':test_idx}
        return split_dict

    
    def process(self):
        data, slices = self.collate(data_list)

        print('Saving lappe with k = '+str(self.k)+' and cutoff = '+str(self.cutoff)+'...')
        torch.save((data, slices), self.processed_paths[0])
    


if __name__=="__main__":
    cutoff=10.0
    k=9
    origdataset = QM93D()
    data_list = []
    lappe = LaplacianEigenvectorPE(k=k)
    for data in origdataset:
        edge_index = radius_graph(data.pos, r=cutoff)
        data.pe = lappe(data.pos.shape[0], edge_index)
        data_list.append(data)




    dataset = QM9LapPE(data_list = data_list, k=k, cutoff=cutoff) # Change these parameters as you want
    print(dataset.data)

# print(dataset)
# dataset.process()
# print(dataset)
# print(dataset.data.z.shape)
# print(dataset.data.pos.shape)
# print(dataset.data.pe.shape)
# target = 'mu'
# dataset.data.y = dataset.data[target]
# print(dataset.data.y.shape)
# print(dataset.data.y)
# print(dataset.data.mu)
# split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=110000, valid_size=10000, seed=42)
# print(split_idx) 
# print(dataset[split_idx['train']])
# train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# data = next(iter(train_loader))
# print(data)