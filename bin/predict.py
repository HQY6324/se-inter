import sys
import torch
import numpy as np
from bin.data_loader import HemoDataset
from torch.utils.data import DataLoader
import os
from bin.model import resnet18
from bin.PpiLoss import PpiLoss




def concat(A_f1d, B_f1d, p2d):
    
    def rep_new_axis(mat, rep_num, axis):
        return torch.repeat_interleave(torch.unsqueeze(mat,axis=axis),rep_num,axis=axis)
    
    len_channel,lenA = A_f1d.shape
    len_channel,lenB = B_f1d.shape        
    
    row_repeat = rep_new_axis(A_f1d, lenB, 2)
    col_repeat = rep_new_axis(B_f1d, lenA, 1)        

    return  torch.unsqueeze(torch.cat((row_repeat, col_repeat, p2d),axis=0),0)



def test_model(model, criterion, test_loader, device,predictions_folder):
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 在测试阶段不计算梯度
        for pdb_name,inputs, targets in test_loader:
            print(pdb_name)
            rec1d = inputs["rec1d"]
            lig1d = inputs["lig1d"]
            com2d = inputs["com2d"]
            rec1d = rec1d.transpose(0, 1)
            lig1d = lig1d.transpose(0, 1)
            rec1d = rec1d.to(device).squeeze().float()
            lig1d = lig1d.to(device).squeeze().float()
            com2d = com2d.to(device).squeeze().float()
            targets = targets.to(device).squeeze().float()

            outputs = model(rec1d, lig1d, com2d)
            loss = criterion(outputs, targets)
            print(f'Loss for {pdb_name}: {loss.item()}')

            # 将预测的距离矩阵转换为numpy数组
            predicted_matrix = outputs.squeeze().cpu().numpy()
            save_path = os.path.join(predictions_folder, f'{pdb_name}__prediction.txt')

            # 保存预测的距离矩阵到文件
            np.savetxt(save_path, predicted_matrix, fmt='%0.3f')
            print(f'Saved prediction to {save_path}')

if __name__ == '__main__':
    predictions_folder = '../predict'
    if not os.path.exists(predictions_folder):
        os.makedirs(predictions_folder)
    # 定义设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化模型
    model = resnet18().to(device)
   
    # 加载模型参数
    model.load_state_dict(torch.load('../model/model.pth', map_location=device))
    model.to(device)

    # 定义损失函数
    criterion_ppi = PpiLoss(alpha=0.83, reduction='sum')
#    criterion = FocalLoss(5.0,0.8)


    # 准备测试数据集
    esm_msa_dir = '../example/pairesm'
    esm2_dir = '../example/saesm2single'
    monomer_esm_dir='../example/singleesm'
    esm2pair_dir = '../example/saesm2pair'
    label_dir = '../example/contactmap'




    

    all_folders = [os.path.join(esm2_dir, d) for d in os.listdir(esm2_dir) if os.path.isdir(os.path.join(esm2_dir, d))]
    
    test_set = HemoDataset(all_folders,esm_msa_dir, monomer_esm_dir,esm2pair_dir,label_dir)
    test_loader = DataLoader(test_set, batch_size=None, shuffle=False)
    # 运行测试
    test_model(model, criterion_ppi, test_loader, device,predictions_folder)


