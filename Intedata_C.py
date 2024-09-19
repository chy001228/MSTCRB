import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
import Cod_C


class Intedata_C(Dataset):
    def __init__(self,protein):
        super(Intedata_C, self).__init__()
        Kmer, NCP, DPCP, knf, Y = Cod_C.all_data(protein)
        pair = np.load('./Datasets/circRNA-RBP/'+protein+'/pair.npy', allow_pickle=True)
        self.Kmer = Kmer
        self.NCP = NCP
        self.DPCP = DPCP
        self.knf = knf
        self.pair = pair
        self.Y = Y
        dict_data = {'Kmer': Kmer, 'NCP':NCP, 'DPCP':DPCP, 'knf': knf, 'pair': pair, 'Y': Y}
        self.dict_data = dict_data


    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx]) for key, value in self.dict_data.items()}
        return item

    def __len__(self):
        return len(self.Y)





