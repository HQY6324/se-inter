import torch
from torch.utils.data import Dataset
import numpy as np
import os
import fnmatch
import pickle




class HemoDataset(Dataset):
    
    def __init__(self, folder_list,esm_msa_dir, monomer_msa_dir,apc_dir,di_dir,esm2pair_dir,label_dir):


        self.folder_list = folder_list
        self.esm_msa_dir = esm_msa_dir
        self.monomer_msa_dir = monomer_msa_dir
#        self.hmm_dir = hmm_dir
        self.apc_dir = apc_dir
        self.di_dir = di_dir
#        self.alnstats_dir = alnstats_dir
        self.esm2pair_dir=esm2pair_dir
        self.label_dir = label_dir
 

    def __len__(self):
        return len(self.folder_list)

    def __getitem__(self, idx):
        folder_path = self.folder_list[idx]
        esm2_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pt')])
        folder_name = os.path.basename(folder_path)
        print(folder_name)
        # 确保每个文件夹中有两个文件
        assert len(esm2_files) == 2, "Each folder must contain exactly two feature files."
        
        # 加载受体和配体特征
        rec_esm2 = torch.load(esm2_files[0])
        lig_esm2 = torch.load(esm2_files[1])


#        esm2_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pkl')])
#        folder_name = os.path.basename(folder_path)
#        print(folder_name)
#
#        # 确保每个文件夹中有两个文件
#        assert len(esm2_files) == 2, "Each folder must contain exactly two feature files."
#
#        # 加载受体和配体特征
#        with open(esm2_files[0], 'rb') as f:
#             rec_esm2 = pickle.load(f)
#
#        with open(esm2_files[1], 'rb') as f:
#             lig_esm2 = pickle.load(f)


        rec_esm2 = rec_esm2["esm_feats"]
        lig_esm2 = lig_esm2["esm_feats"]

        L1 = rec_esm2.shape[0]
        L2 = lig_esm2.shape[0]
        L=L1+L2
        
        
        esm_msa_files = fnmatch.filter(os.listdir(self.esm_msa_dir), f'{folder_name}*.pkl')
        if esm_msa_files:
            esm_msa_filename = esm_msa_files[0]  # 取匹配到的第一个文件
            esm_msa_path = os.path.join(self.esm_msa_dir, esm_msa_filename)
            with open(esm_msa_path, 'rb') as file:  # 打开文件进行二进制读取
                 esm_msa_feat = pickle.load(file)
            esm_msa_feature_1d = esm_msa_feat["esm_msa_1d"]
            esm_msa_row_attentions = esm_msa_feat["row_attentions"]
        else:
            raise FileNotFoundError(f"No ESM-MSA file found for prefix {folder_name}")




        esm2pair_files = fnmatch.filter(os.listdir(self.esm2pair_dir), f'{folder_name}*.pkl')
        if esm2pair_files:
            esm2pair_filename = esm2pair_files[0]
            esm2pair_path = os.path.join(self.esm2pair_dir, esm2pair_filename)
            with open(esm2pair_path, 'rb') as file:
                esm2pair_feat = pickle.load(file)
            esm2pair_row_attentions = esm2pair_feat["esm_feats"][:, :, 1:(L1 + 1), 1 + L1:(L + 1)]
            dims = esm2pair_row_attentions.shape
            merged_dim = dims[0] * dims[1]
            esm2pair_row_attentions_reshaped = esm2pair_row_attentions.reshape(merged_dim, dims[2], dims[3])
        else:
            raise FileNotFoundError(f"No ESM2 pair file found for prefix {folder_name}")



        monomer_msa_folder = os.path.join(self.monomer_msa_dir, folder_name)
        rec_monomer_msa_file = fnmatch.filter(os.listdir(monomer_msa_folder), os.path.basename(esm2_files[0])[:6] + '*.pkl')[0]
        lig_monomer_msa_file = fnmatch.filter(os.listdir(monomer_msa_folder), os.path.basename(esm2_files[1])[:6] + '*.pkl')[0]

        rec_monomer_msa = pickle.load(open(os.path.join(monomer_msa_folder, rec_monomer_msa_file), 'rb'))
        lig_monomer_msa = pickle.load(open(os.path.join(monomer_msa_folder, lig_monomer_msa_file), 'rb'))

        rec_monomer_msa_1d = rec_monomer_msa["esm_msa_1d"]
        lig_monomer_msa_1d = lig_monomer_msa["esm_msa_1d"]


        rec1d = np.concatenate( [rec_esm2,rec_monomer_msa_1d], axis=-1 )
        lig1d = np.concatenate( [lig_esm2,lig_monomer_msa_1d], axis=-1 )
        com2d = np.concatenate( [esm_msa_row_attentions.reshape(144, L, L)[:,:L1,L1:L1+L2],esm2pair_row_attentions_reshaped], axis=0)



        # 生成标签文件名
        pdb_id = folder_name
        label_filename = os.path.basename(folder_path) + ".txt"
        label_path = os.path.join(self.label_dir, label_filename)
        label = np.loadtxt(label_path)

        data = {
            "rec1d": rec1d,
            "lig1d": lig1d,
            "com2d": com2d,
        }

        return pdb_id, data, torch.tensor(label, dtype=torch.float)
