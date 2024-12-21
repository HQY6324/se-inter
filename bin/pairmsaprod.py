import os, sys
import string
import numpy as np


def extract_taxid(file, gap_cutoff=0.8):
    lines = open(file, 'r').readlines()
    query = lines[1].strip().translate(translation)
    seq_len = len(query)

    msas = [query]
    sid = [0]
    for line in lines[2:]:

        if line[0] == ">":
            if "TaxID=" in line:
                content = line.split("TaxID=")[1]
                if len(content) > 0:
                    try:
                        sid.append(int(content.split()[0]))
                    except:
                        sid.append(0)
            elif "OX=" in line:
                content = line.split("OX=")[1]
                if len(content) > 0:
                    try:
                        sid.append(int(content.split()[0]))
                    except:
                        sid.append(0)
            else:
                sid.append(0)
            continue

        seq = line.strip().translate(translation)
        gap_fra = float(seq.count('-')) / seq_len
        if gap_fra <= gap_cutoff:
            msas.append(seq)
        else:
            sid.pop(-1)

    if len(msas) != len(sid):
        print("ERROR: len(msas) != len(sid)")
        print(len(msas), len(sid))
        exit()

    return msas, np.array(sid)


def cal_identity(query, sub_msas):
    """
    Args:
        query : str
        sub_msas : List[str]
    Return:
        identity : np.array
    """

    identity = np.zeros((len(sub_msas)))
    seq_len = len(query)
    ones = np.ones(seq_len)
    for idx, seq in enumerate(sub_msas):
        match = [query[i] == seq[i] for i in range(seq_len)]
        counts = np.sum(ones[match])
        identity[idx] = counts / seq_len

    return identity


def alignment(msas1, sid1, msas2, sid2, top=True):
    # obtain the same species and delete species=0
    smatch = np.intersect1d(sid1, sid2)
    smatch = smatch[np.argsort(smatch)]
    smatch = np.delete(smatch, 0)

    query1 = msas1[0]
    query2 = msas2[0]
    aligns = [query1 + query2]

    for id in smatch:

        index1 = np.where(sid1 == id)[0]
        sub_msas1 = [msas1[idx] for idx in index1]
        identity1 = cal_identity(query1, sub_msas1)
        sort_idx1 = np.argsort(-identity1)

        index2 = np.where(sid2 == id)[0]
        sub_msas2 = [msas2[idx] for idx in index2]
        identity2 = cal_identity(query2, sub_msas2)
        sort_idx2 = np.argsort(-identity2)

        if top == True:
            aligns.append(sub_msas1[sort_idx1[0]] + \
                          sub_msas2[sort_idx2[0]])
        else:
            num = min(len(sub_msas1), len(sub_msas2))
            for i in range(num):
                aligns.append(sub_msas1[sort_idx1[i]] + \
                              sub_msas2[sort_idx2[i]])

    return aligns


def write_a3m(pdb, aligns, out_path):
    n = len(aligns)
    with open(out_path, 'w') as f:
        f.write(">" + pdb + "\n")
        f.write(aligns[0] + "\n")

        for idx, aligned_seq in enumerate(aligns[1:]):
            f.write(">" + str(idx + 1) + "\n")
            f.write(aligned_seq + "\n")


if __name__ == "__main__":
    # 新的逻辑开始
    deletekeys = dict.fromkeys(string.ascii_lowercase)
    deletekeys["."] = None
    deletekeys["*"] = None
    translation = str.maketrans(deletekeys)
    
    input_folder_path = '/home/huangqiyuan/datav2/db55/singlemsa/db55/singlemsa'  # 指定大文件夹路径
    output_folder_path = '/home/huangqiyuan/datav2/db55/pairmsa'  # 指定输出文件夹路径

    # 确保输出文件夹存在
    os.makedirs(output_folder_path, exist_ok=True)

    # 遍历大文件夹中的每个小文件夹
    for pdb_folder_name in os.listdir(input_folder_path):
        pdb_folder_path = os.path.join(input_folder_path, pdb_folder_name)

        # 确保是文件夹
        if os.path.isdir(pdb_folder_path):
            a3m_files = [f for f in os.listdir(pdb_folder_path) if f.endswith('.a3m')]
            # 确保找到两个A3M文件
            if len(a3m_files) == 2:
                a3m1_path = os.path.join(pdb_folder_path, a3m_files[0])
                a3m2_path = os.path.join(pdb_folder_path, a3m_files[1])

                # 处理A3M文件，生成配对MSA
                msas1, sid1 = extract_taxid(a3m1_path)
                msas2, sid2 = extract_taxid(a3m2_path)
                aligns = alignment(msas1, sid1, msas2, sid2, top=True)

                # 构建输出文件名和路径
                out_path = os.path.join(output_folder_path, f"{pdb_folder_name}_paired.a3m")
                write_a3m(pdb_folder_name, aligns, out_path)

                print(f"Processed and saved: {out_path}")