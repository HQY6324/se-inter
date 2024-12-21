from Bio import SeqIO
from Bio.PDB import PDBParser
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.PDB.Polypeptide import protein_letters_3to1
import os
import warnings
from Bio import PDB
from Bio.PDB.PDBExceptions import PDBConstructionWarning


def extract_combined_sequence(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("PDB_structure", pdb_path)
    combined_sequence = ""
    for model in structure:
        for chain in model:
            for res in chain.get_residues():
                if res.get_id()[0] == ' ':  # 只处理标准氨基酸残基
                    res_id = res.get_resname()
                    try:
                        single_letter = protein_letters_3to1[res_id]
                        combined_sequence += single_letter
                    except KeyError:
                        pass
    return SeqRecord(Seq(combined_sequence), id=os.path.basename(pdb_path), description="")


def process_directory_structure(pdb_directory_path, output_fasta_directory_path):
    warnings.simplefilter('ignore', PDBConstructionWarning)
    # 确保输出目录存在
    os.makedirs(output_fasta_directory_path, exist_ok=True)

    for subfolder in os.listdir(pdb_directory_path):
        subfolder_path = os.path.join(pdb_directory_path, subfolder)
        if os.path.isdir(subfolder_path):
            output_subfolder_path = os.path.join(output_fasta_directory_path, subfolder)
            os.makedirs(output_subfolder_path, exist_ok=True)  # 创建对应的子文件夹

            for pdb_file in os.listdir(subfolder_path):
                pdb_path = os.path.join(subfolder_path, pdb_file)
                seq_record = extract_combined_sequence(pdb_path)

                # 提取pdbid和后缀
                pdbid = pdb_file.split('.')[0]
                output_fasta_path = os.path.join(output_subfolder_path, f"{pdbid}.fasta")

                # 写入序列到单独的FASTA文件
                SeqIO.write([seq_record], output_fasta_path, "fasta")
                print(f"FASTA file written for {pdbid} at {output_fasta_path}")


if __name__ == '__main__':
    pdb_directory_path = '/home/huangqiyuan/casp/singlepdb2'
    output_fasta_directory_path = '/home/huangqiyuan/casp/singlefastas2'
    process_directory_structure(pdb_directory_path, output_fasta_directory_path)
