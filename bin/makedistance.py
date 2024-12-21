import numpy as np
import os

def read_pdb(pdb_file):
    """
    读取 PDB 文件，返回一个包含每个残基原子坐标的列表和残基标识的列表，保持残基顺序，排除未知残基。
    """
    aa_mapping = {
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY',
        'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'MSE', 'PHE', 'PRO',
        'SER', 'THR', 'TRP', 'TYR', 'VAL'
    }
    residues = []
    residue_ids = []
    current_res_id = None
    current_coords = []
    current_res_name = None
    with open(pdb_file, 'r') as file:
        for line in file:
            if line.startswith("ATOM"):
                res_name = line[17:20].strip()   # 残基名称
                res_seq = line[22:26].strip()    # 残基序列编号
                icode = line[26].strip()         # 插入代码
                res_num = res_seq + icode        # 组合残基编号和插入代码

                element = line[76:78].strip()    # 元素符号

                # 检查是否为标准氨基酸残基
                if res_name not in aa_mapping:
                    continue  # 跳过未知残基

                # 检查是否为重原子（非氢原子）
                if element == 'H' or element == 'D':
                    continue

                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])

                if res_num != current_res_id:
                    if current_res_id is not None:
                        # 保存前一个残基的信息
                        residues.append(np.array(current_coords))
                        residue_ids.append(current_res_id)
                    # 开始新的残基
                    current_res_id = res_num
                    current_coords = []
                # 添加原子坐标到当前残基的列表
                current_coords.append((x, y, z))

        # 处理最后一个残基
        if current_res_id is not None and current_coords:
            residues.append(np.array(current_coords))
            residue_ids.append(current_res_id)

    return residues, residue_ids

def compute_contact_matrix(residues1, residues2, threshold=8.0):
    """
    计算两个残基集合之间的接触矩阵，元素为 0 或 1，1 表示残基间的最小原子距离小于阈值。
    """
    num_residues1 = len(residues1)
    num_residues2 = len(residues2)
    contact_matrix = np.zeros((num_residues1, num_residues2), dtype=int)

    for i in range(num_residues1):
        coords1 = residues1[i]
        for j in range(num_residues2):
            coords2 = residues2[j]

            # 计算两个坐标数组之间的距离矩阵
            dist_array = np.linalg.norm(coords1[:, np.newaxis, :] - coords2[np.newaxis, :, :], axis=2)

            # 检查是否存在距离小于阈值的原子对
            if np.any(dist_array < threshold):
                contact_matrix[i, j] = 1
    return contact_matrix

def process_subfolder(subfolder, output_folder):
    """处理包含两个 PDB 文件的子文件夹，并保存接触矩阵。"""
    pdb_files = [os.path.join(subfolder, f) for f in os.listdir(subfolder) if f.endswith('.pdb')]
    if len(pdb_files) != 2:
        print(f"Skipping {subfolder}, it does not contain exactly two PDB files.")
        return

    # 提取残基和接触矩阵
    residues1, residue_ids1 = read_pdb(pdb_files[0])
    residues2, residue_ids2 = read_pdb(pdb_files[1])
    contact_matrix = compute_contact_matrix(residues1, residues2, threshold=8.0)

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    output_filename = os.path.join(output_folder, f"{os.path.basename(subfolder)}.txt")
    np.savetxt(output_filename, contact_matrix, fmt='%d')

    print(f"Processed {subfolder} and saved contact matrix to {output_filename}")

def process_folder(input_folder, output_folder):
    """遍历包含子文件夹的主文件夹。"""
    for subfolder_name in os.listdir(input_folder):
        subfolder_path = os.path.join(input_folder, subfolder_name)
        if os.path.isdir(subfolder_path):
            process_subfolder(subfolder_path, output_folder)

if __name__ == '__main__':
    input_folder_path = '../../nlp/testdataset/db55/singlepdb'   # 调整为您的主文件夹路径
    output_folder_path = '../../nlp/testdataset/db55/contactmap'  # 调整为您想要保存接触矩阵的文件夹路径
    process_folder(input_folder_path, output_folder_path)

