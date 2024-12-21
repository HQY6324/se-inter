import torch
from torch.utils.data import Dataset
import os
import pickle

class HemoDataset(Dataset):
    def __init__(self, file_list):
        """
        Args:
            file_list (list): 包含所有pkl文件路径的列表。
        """
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        
        with open(file_path, 'rb') as file:
            data_dict = pickle.load(file)
        
        pdb_id = data_dict['pdb_id']
        data = data_dict['data']
        label = data_dict['label']
        
        return pdb_id, data, label
