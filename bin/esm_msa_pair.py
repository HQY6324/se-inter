import os, sys
import esm
import torch
import argparse
import string
import itertools
import numpy as np
import pickle as pkl
from Bio import SeqIO
from typing import List, Tuple

torch.set_num_threads(8)

# read the Multiple Sequence Alignment (MSA)
def remove_insertions(sequence: str) -> str:
    return sequence.translate(translation)

def read_msa(filename: str, nseq: int) -> List[Tuple[str, str]]:
    return [(record.description, remove_insertions(str(record.seq)))
            for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)
            if len(record.seq) <= 1000]

# load model
def load_esm(path):
    model, alphabet = esm.pretrained.load_model_and_alphabet(path)
    model = model.eval()  # .cuda()
    batch_converter = alphabet.get_batch_converter()
    return model, batch_converter

# inference of ESM-MSA-1b
def get_esm_msa_feats(esm1b, esm1b_batch_converter, seq_list):
    esm1b_batch_labels, esm1b_batch_strs, esm1b_batch_tokens = esm1b_batch_converter(seq_list)
    with torch.no_grad():
        results = esm1b(esm1b_batch_tokens, repr_layers=[12], return_contacts=True)
    token_representations = results["representations"][12].mean(1)
    sequence_representations = []
    for i, seq in enumerate(seq_list):
        sequence_representations.append(np.array(token_representations[i, 1: len(seq[0][1]) + 1].cpu()))
    return sequence_representations[0], np.squeeze(np.array(results['row_attentions'].cpu()))[:, :, 1:, 1:]

def generate_data_from_file(model_path, data_folder_path, output_folder_path):
    os.makedirs(output_folder_path, exist_ok=True)
    esm1b, esm1b_batch_converter = load_esm(model_path)
    for filename in os.listdir(data_folder_path):
        print(f"Processing file: {filename}")  # Print the current file being processed
        target = filename.split(".")[0]
        msa_data = [read_msa(os.path.join(data_folder_path, filename), 512)]
        esm_msa_1d, row_attentions = get_esm_msa_feats(esm1b, esm1b_batch_converter, msa_data)
        data = {'esm_msa_1d': esm_msa_1d, 'row_attentions': row_attentions}
        with open(os.path.join(output_folder_path, target + "_esm_msa.pkl"), 'wb') as f:
            pkl.dump(data, f, protocol=3)

if __name__ == "__main__":
    deletekeys = dict.fromkeys(string.ascii_lowercase)
    deletekeys["."] = None
    deletekeys["*"] = None
    translation = str.maketrans(deletekeys)

    model_path = '/home/huangqiyuan/esm/esm_msa1_t12_100M_UR50S.pt'
    data_folder_path = '/home/huangqiyuan/datav2/db55/pairmsa'
    output_folder_path = '/home/huangqiyuan/datav2/db55/pairesm'
    generate_data_from_file(model_path, data_folder_path, output_folder_path)
