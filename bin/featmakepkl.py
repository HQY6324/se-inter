import torch
import numpy as np
import os
import fnmatch
import pickle


def save_features_to_pkl(output_dir, esm2_dir, esm_msa_dir, monomer_msa_dir,esm2pair_dir, label_dir):
    os.makedirs(output_dir, exist_ok=True)
    # 获取主目录下的所有子文件夹
    folder_list = [os.path.join(esm2_dir, d) for d in os.listdir(esm2_dir) if os.path.isdir(os.path.join(esm2_dir, d))]

    # 顺序处理文件夹
    for folder_path in folder_list:
        process_folder(folder_path, output_dir, esm_msa_dir, monomer_msa_dir,esm2pair_dir, label_dir)

def process_folder(folder_path, output_dir, esm_msa_dir, monomer_msa_dir,esm2pair_dir, label_dir):
    # 加载特征数据
    pdb_id, data, label = extract_features(folder_path, esm_msa_dir, monomer_msa_dir,esm2pair_dir, label_dir)

    # 创建字典
    feature_dict = {
        "pdb_id": pdb_id,
        "data": data,
        "label": label
    }

    # 保存到pkl文件，文件名为小文件夹的文件名
    folder_name = os.path.basename(folder_path)
    output_file = os.path.join(output_dir, f"{folder_name}.pkl")
    print(f"序列化文件: {output_file}")
    with open(output_file, 'wb') as f:
        pickle.dump(feature_dict, f)

def extract_features(folder_path, esm_msa_dir, monomer_msa_dir,apc_dir, di_dir,esm2pair_dir, label_dir):
    esm2_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pt')])
    folder_name = os.path.basename(folder_path)
    # 确保每个文件夹中有两个文件
    assert len(esm2_files) == 2, "Each folder must contain exactly two feature files."

    # 加载受体和配体特征
    rec_esm2 = torch.load(esm2_files[0])
    lig_esm2 = torch.load(esm2_files[1])

    rec_esm2 = rec_esm2["esm_feats"]
    lig_esm2 = lig_esm2["esm_feats"]
    
    L1 = rec_esm2.shape[0]
    L2 = lig_esm2.shape[0]
    L = L1 + L2

    esm_msa_files = fnmatch.filter(os.listdir(esm_msa_dir), f'{folder_name}*.pkl')
    if esm_msa_files:
        esm_msa_filename = esm_msa_files[0]
        esm_msa_path = os.path.join(esm_msa_dir, esm_msa_filename)
        with open(esm_msa_path, 'rb') as file:
            esm_msa_feat = pickle.load(file)
        esm_msa_feature_1d = esm_msa_feat["esm_msa_1d"]
        esm_msa_row_attentions = esm_msa_feat["row_attentions"]
    else:
        raise FileNotFoundError(f"No ESM-MSA file found for prefix {folder_name}")

    esm2pair_files = fnmatch.filter(os.listdir(esm2pair_dir), f'{folder_name}*.pkl')
    if esm2pair_files:
        esm2pair_filename = esm2pair_files[0]
        esm2pair_path = os.path.join(esm2pair_dir, esm2pair_filename)
        with open(esm2pair_path, 'rb') as file:
            esm2pair_feat = pickle.load(file)
        esm2pair_row_attentions = esm2pair_feat["esm_feats"][:, :, 1:(L1 + 1), 1 + L1:(L + 1)]
        dims = esm2pair_row_attentions.shape
        merged_dim = dims[0] * dims[1]
        esm2pair_row_attentions_reshaped = esm2pair_row_attentions.reshape(merged_dim, dims[2], dims[3])
    else:
        raise FileNotFoundError(f"No ESM2 pair file found for prefix {folder_name}")

    monomer_msa_folder = os.path.join(monomer_msa_dir, folder_name)
    rec_monomer_msa_file = fnmatch.filter(os.listdir(monomer_msa_folder), os.path.basename(esm2_files[0])[:6] + '*.pkl')[0]
    lig_monomer_msa_file = fnmatch.filter(os.listdir(monomer_msa_folder), os.path.basename(esm2_files[1])[:6] + '*.pkl')[0]

    rec_monomer_msa = pickle.load(open(os.path.join(monomer_msa_folder, rec_monomer_msa_file), 'rb'))
    lig_monomer_msa = pickle.load(open(os.path.join(monomer_msa_folder, lig_monomer_msa_file), 'rb'))

    rec_monomer_msa_1d = rec_monomer_msa["esm_msa_1d"]
    lig_monomer_msa_1d = lig_monomer_msa["esm_msa_1d"]



    rec1d = np.concatenate([rec_esm2,rec_monomer_msa_1d], axis=-1)
    lig1d = np.concatenate([lig_esm2,lig_monomer_msa_1d], axis=-1)
    com2d = np.concatenate([esm_msa_row_attentions.reshape(144, L, L)[:,:L1,L1:L1+L2],esm2pair_row_attentions_reshaped], axis=0)

    pdb_id = folder_name
    label_filename = os.path.basename(folder_path) + ".txt"
    label_path = os.path.join(label_dir, label_filename)
    label = torch.tensor(np.loadtxt(label_path), dtype=torch.float)

    data = {
        "rec1d": rec1d,
        "lig1d": lig1d,
        "com2d": com2d,
    }

    return pdb_id, data, label

if __name__ == "__main__":
    output_dir = '/home/huangqiyuan/datav2/bigtrainset/5pkl'
    esm2_dir = '/home/huangqiyuan/datav2/bigtrainset/ext/esm2650single'
    esm_msa_dir = '/home/huangqiyuan/datav2/bigtrainset/uniref100feat/home/huangqiyuan/datav2/bigtrainset/uniref100feat/pairesm/train'
    monomer_msa_dir = '/home/huangqiyuan/datav2/bigtrainset/uniref100feat/home/huangqiyuan/datav2/bigtrainset/uniref100feat/singleesm/train'
    esm2pair_dir = '/home/huangqiyuan/datav2/bigtrainset/esm2650pair'
    label_dir = '/home/huangqiyuan/datav2/bigtrainset/contactmap'

    save_features_to_pkl(output_dir, esm2_dir, esm_msa_dir, monomer_msa_dir,esm2pair_dir, label_dir)
