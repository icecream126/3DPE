# from ..positional_encoding.import LaplacianEigenvectorPE
# from ..positional_encoding.randomwalkpe import RandomWalkPE
# from ..positional_encoding.heatkernelpe import HeatKernelEigenvectorPE

import sys
# sys.path.append('/home/hsjang/hmkim/3DPE/dig/threedgraph/positional_encoding')
# sys.path.append('/home/hsjang/hmkim/3DPE/dig/threedgraph/dataset')

sys.path.append('/home/guest_khm/hyomin/3DPE/dig/threedgraph/positional_encoding')
sys.path.append('/home/guest_khm/hyomin/3DPE/dig/threedgraph/dataset')


from simplepclaplacianpe import SimplePCLaplacianEigenvectorPE
from PygQM93D import QM93D


import torch
from sklearn.utils import shuffle

from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import radius_graph


class QM9SimplePCLapPE(InMemoryDataset):
    
    # Before first processing
    # def __init__(self, data_list, k, cutoff, sigma) :
    
    # After first processing
    def __init__(self,k, cutoff, sigma): 
        self.k = k
        # self.data_list = data_list # Before first processing
        self.orig_qm9 = QM93D(root='dataset/')
        self.cutoff = cutoff
        self.sigma = sigma
        super(QM9SimplePCLapPE, self).__init__('./dataset/qm9/lappe/')
        self.data, self.slices = torch.load(self.processed_paths[0]) # After first processing
        # self.data, self.slices = self.collate(data_list) # Before first processing
        
        
    def __str__(self):
        return f"k is {self.k}, cutoff is {self.cutoff}"

    @property
    def raw_file_names(self):
        return 'qm9_simp_PC_lappe_k_'+str(self.k)+'_cutoff_'+str(self.cutoff)+'_sigma_'+str(self.sigma)+'.npz'

    @property
    def processed_file_names(self):
        return 'qm9_simp_PC_lappe_k_'+str(self.k)+'_cutoff_'+str(self.cutoff)+'_sigma_'+str(self.sigma)+'.pt'
        

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
    sigma=0.01 # hyperparameter.. # Try 0.01, 1, 10
    k=4
    origdataset = QM93D()
    data_list = []
    lappe = SimplePCLaplacianEigenvectorPE(k=k)
    cnt=1
    for data in origdataset:
        edge_index = radius_graph(data.pos, r=cutoff)
        data.pe = lappe(data.pos, sigma=sigma)
        data_list.append(data)
        print('Processed # of data : ',cnt,' / ',len(origdataset))
        cnt+=1


    dataset = QM9SimplePCLapPE(data_list = data_list, k=k, cutoff=cutoff, sigma=sigma) # Change these parameters as you want
    print(dataset.data)