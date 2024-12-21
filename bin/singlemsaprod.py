import os
import subprocess


def process_fasta_files(input_folder_path, output_folder_path, database_basename_path):
    # 确保输出文件夹存在
    os.makedirs(output_folder_path, exist_ok=True)

    # 遍历输入文件夹中的所有子文件夹
    for subdir in os.listdir(input_folder_path):
        subdir_path = os.path.join(input_folder_path, subdir)
        if os.path.isdir(subdir_path):
            output_subdir_path = os.path.join(output_folder_path, subdir)
            os.makedirs(output_subdir_path, exist_ok=True)

            # 遍历每个子文件夹中的所有FASTA文件
            for filename in os.listdir(subdir_path):
                if filename.endswith('.fasta'):
                    input_file = os.path.join(subdir_path, filename)
                    output_file = os.path.join(output_subdir_path, filename.replace('.fasta', '.a3m'))

                    # 构建hhblits命令
                    cmd = [
                        'hhblits',
                        '-i', input_file,
                        '-o', os.devnull,  # 忽略标准输出
                        '-oa3m', output_file,
                        '-d', database_basename_path,
                        '-n', '3',
                        '--cpu','32'
                    ]

                    # 执行hhblits命令
                    subprocess.run(cmd)
                    print(f"Processing of {filename} in {subdir} is complete.")

    print("HHblits processing of all directories is complete.")


if __name__ == '__main__':
    input_folder_path = '/home/huangqiyuan/datav2/db55/singlefastas'
    output_folder_path = '/home/huangqiyuan/datav2/db55/singlemsa'
    database_basename_path = '/home/huangqiyuan/database/UniRef30_2021_03/UniRef30_2021_03'

    process_fasta_files(input_folder_path, output_folder_path, database_basename_path)
