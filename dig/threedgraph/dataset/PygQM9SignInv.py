# from ..positional_encoding.import LaplacianEigenvectorPE
# from ..positional_encoding.randomwalkpe import RandomWalkPE
# from ..positional_encoding.heatkernelpe import HeatKernelEigenvectorPE

import sys
# sys.path.append('/home/hsjang/hmkim/3DPE/dig/threedgraph/positional_encoding')
# sys.path.append('/home/hsjang/hmkim/3DPE/dig/threedgraph/dataset')

sys.path.append('/home/guest_khm/hyomin/3DPE/dig/threedgraph/positional_encoding')
sys.path.append('/home/guest_khm/hyomin/3DPE/dig/threedgraph/dataset')



from cleanlaplacianpe import CleanLaplacianEigenvectorPE
from laplacianpe import LaplacianEigenvectorPE
from randomwalkpe import RandomWalkPE
from heatkernelpe import HeatKernelEigenvectorPE
from PygQM93D import QM93D
import gc
import pickle
import numpy as np


import torch
from sklearn.utils import shuffle

from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import radius_graph

import multiprocessing as mp 
from joblib import Parallel, delayed


class QM9SignInvLapPE(InMemoryDataset):
    
    # Before first processing
    # def __init__(self, data_list, k, cutoff) : 
    
    # After first processing
    def __init__(self,k, cutoff): 
        self.k = k
        # self.data_list = data_list # Before first processing
        self.orig_qm9 = QM93D(root='dataset/')
        self.cutoff = cutoff
        super(QM9SignInvLapPE, self).__init__('./dataset/qm9/signinv/k_'+str(k))
        # print('self.processed_paths[0] : ',self.processed_paths[0]) #/home/hsjang/hmkim/3DPE/dataset/qm9/signinv/k_2/processed/cutoff_10.0.pt
        self.data, self.slices = torch.load(self.processed_paths[0]) # After first processing
        # self.data, self.slices = self.collate(self.data_list) # Before first processing
        
        
    def __str__(self):
        return f"k is {self.k}, cutoff is {self.cutoff}"

    @property
    def raw_file_names(self):
        return 'cutoff_'+str(self.cutoff)+'.npz'

    @property
    def processed_file_names(self):
        return 'cutoff_'+str(self.cutoff)+'.pt'
        

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        print('data size : ',data_size)
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train':train_idx, 'valid':val_idx, 'test':test_idx}
        return split_dict

    
    def process(self):
        print('SignInv processing..')
        data, slices = self.collate(self.data_list)

        print('Saving signinv lappe with k = '+str(self.k)+' and cutoff = '+str(self.cutoff)+'...')
        torch.save((data, slices), self.processed_paths[0])
    
def same_storage(x, y):
    return x.storage().data_ptr() == y.storage().data_ptr()

def make_pe_list(data):
    cutoff=10.0
    data.edge_index = radius_graph(data.pos, r=cutoff)
    data.pe = lappe(data, data.pos.shape[0], data.edge_index).squeeze()
    gc.collect()
    torch.cuda.empty_cache()
    return data

if __name__=="__main__":
    
    num_core = mp.cpu_count()
    cutoff=10.0
    origdataset = QM93D()
    data_list = []
    # data_list = data_list.to('cpu')
    
    lappe = CleanLaplacianEigenvectorPE(2)
    
    count=1
    '''
    with Parallel(n_jobs=num_core) as parallel:
        data_list = parallel(delayed(make_pe_list)(data) for data in origdataset)
        
    '''
    for data in origdataset:
        data.edge_index = radius_graph(data.pos, r=cutoff)
        data.pe = lappe(data, data.pos.shape[0], data.edge_index).squeeze() # maybe squeeze?
        data_list.append(data.pe.cpu())
        print('# of processed data : ',count,'/',len(origdataset))
        count+=1
        # if count==10:
        #     break
        
    
    # pe_np = np.array(data_list)
    signinv_data = []
    count=0
    print('=== Now Generating PE Dataset ===')
    for data in origdataset:
        data.pe = data_list[count]
        signinv_data.append(data)
        print("Appending data pe : ",count, ' / ',len(origdataset))
        count+=1
        # if count==9:
        #     break

    # np.save('signinvpe.npy',pe_np)



    dataset = QM9SignInvLapPE(data_list = signinv_data, k=2, cutoff=cutoff) # Change these parameters as you want
    print(dataset.data)