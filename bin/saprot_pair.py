import os
import torch
import esm
import numpy as np
import pickle as pkl
from Bio import SeqIO
from typing import Tuple
import string
# Assuming load_esm_saprot is correctly defined in utils.esm_loader
from utils.esm_loader import load_esm_saprot

torch.set_num_threads(8)

def read_sequence(filename: str) -> Tuple[str, str]:
    """ Reads the first sequence from a FASTA file. """
    record = next(SeqIO.parse(filename, "fasta"))
    return record.description, str(record.seq)

# load model
def load_esm(path):
    model, alphabet = load_esm_saprot(path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    return model, batch_converter, device

def get_esm_feats(model, batch_converter, device, sequence):
    """ Extract features using the ESM model. """
    seq_labels, seq_strs, seq_tokens = batch_converter(sequence)
    len_seq = len(seq_strs[0].strip())
        
    with torch.no_grad():
        seq_tokens = seq_tokens.to(device=device, non_blocking=True)
        results = model(seq_tokens, repr_layers=[33],need_head_weights=True)
        token_attn = results['attentions'].squeeze().cpu().numpy()
    return token_attn

def generate_data_from_folder(model_path, data_folder_path, output_folder_path):
    """ Generate features for FASTA files in the specified directory and save them to a .pkl file. """
    model, batch_converter, device = load_esm(model_path)
    os.makedirs(output_folder_path, exist_ok=True)  # Ensure output folder exists
    
    fasta_files = [os.path.join(data_folder_path, f) for f in os.listdir(data_folder_path) if f.endswith('.fasta')]
    for fasta_file in fasta_files:
        print(f"正在序列化文件: {fasta_file}")
        sequence = [read_sequence(fasta_file)]
        esm_feats = get_esm_feats(model, batch_converter, device, sequence)
        target = os.path.splitext(os.path.basename(fasta_file))[0]
        data = {'esm_feats': esm_feats}
        output_file_path = os.path.join(output_folder_path, target + "_esm_feats.pkl")
        print(f"正在序列化文件: {output_file_path}")
        with open(output_file_path, 'wb') as f:
            pkl.dump(data, f, protocol=4)

if __name__ == "__main__":
    model_path = '/home/huangqiyuan/esm/SaProt_650M_AF2.pt'
    data_folder_path = '/home/huangqiyuan/datav2/db55/sapairfastas'
    output_folder_path = '/home/huangqiyuan/datav2/db55/saesm2pair'
    generate_data_from_folder(model_path, data_folder_path, output_folder_path)