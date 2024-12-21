import os
from Bio.PDB import PDBParser
from foldseek_util import get_struc_seq

def get_chain_id(pdb_path):
    """提取 PDB 文件中的链名"""
    parser = PDBParser()
    structure = parser.get_structure("structure", pdb_path)
    # 假设每个 PDB 文件只有一条链，获取第一个链的 ID
    for model in structure:
        for chain in model:
            return chain.id  # 返回链的 ID

def write_combined_seq_to_fasta(combined_seq, pdb_path, output_dir):
    """将 combined_seq 写入指定目录下的 FASTA 文件"""
    # 获取文件名（去除路径和扩展名）
    file_name = os.path.splitext(os.path.basename(pdb_path))[0]
    # 生成目标目录下的 FASTA 文件名
    fasta_file = os.path.join(output_dir, f"{file_name}.fasta")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 写入 FASTA 文件
    with open(fasta_file, 'w') as f:
        f.write(f">{file_name}\n")
        f.write(f"{combined_seq}\n")

def process_pdb_files_in_folder(foldseek_path, input_dir, output_dir):
    """处理大文件夹中的所有小文件夹中的 PDB 文件"""
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".pdb"):
                pdb_path = os.path.join(root, file)

                print(f"Processing PDB file: {pdb_path}")
                
                # 获取相对路径，并创建对应的输出目录结构
                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                
                # 动态获取 PDB 文件中的链名
                chain_id = get_chain_id(pdb_path)

                # 使用提取的链名调用 get_struc_seq
                parsed_seqs = get_struc_seq(foldseek_path, pdb_path, [chain_id], plddt_mask=False)[chain_id]
                seq, foldseek_seq, combined_seq = parsed_seqs

                # 将 combined_seq 写入对应的输出目录中的 FASTA 文件
                write_combined_seq_to_fasta(combined_seq, pdb_path, output_subdir)

if __name__ == "__main__":
    # Example usage:
    foldseek_path = "/home/huangqiyuan/software/foldseek/bin/foldseek"
    input_dir = "/home/huangqiyuan/datav2/db55/singlepdb"  # 大文件夹路径，包含很多小文件夹
    output_dir = "/home/huangqiyuan/datav2/db55/safastas"  # 指定输出的大文件夹路径

    process_pdb_files_in_folder(foldseek_path, input_dir, output_dir)
