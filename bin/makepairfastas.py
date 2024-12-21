import os

def concatenate_fasta_files(source_dir, target_dir):
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)
    
    # 遍历源目录中的所有子目录
    for subdir in os.listdir(source_dir):
        subdir_path = os.path.join(source_dir, subdir)
        if os.path.isdir(subdir_path):
            fasta_files = [f for f in os.listdir(subdir_path) if f.endswith('.fasta')]
            # 确保找到两个fasta文件
            if len(fasta_files) == 2:
                sequences = []
                # 读取fasta文件并提取序列
                for fasta_file in fasta_files:
                    with open(os.path.join(subdir_path, fasta_file), 'r') as file:
                        lines = file.readlines()
                        sequence = ''.join(line.strip() for line in lines if not line.startswith('>'))
                        sequences.append(sequence)

                # 拼接序列
                combined_sequence = ''.join(sequences)
                
                # 写入新的fasta文件
                target_file_path = os.path.join(target_dir, f"{subdir}.fasta")
                with open(target_file_path, 'w') as file:
                    file.write(f">{subdir}\n{combined_sequence}\n")
                print(f"Created combined FASTA file at '{target_file_path}'")

if __name__ == "__main__":
    source_directory = '/home/huangqiyuan/datav2/db55/safastas'  # 源目录路径
    target_directory = '/home/huangqiyuan/datav2/db55/sapairfastas'  # 目标目录路径
    concatenate_fasta_files(source_directory, target_directory)


