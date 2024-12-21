
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from bin.dataset import HemoDataset  # 导入新的Dataset类
import torch
import torch.optim as optim
import random
import os
import numpy as np
from torch.utils.data import DataLoader
from bin.model import resnet18
from bin.PpiLoss import PpiLoss

def concat(A_f1d, B_f1d, p2d):
    def rep_new_axis(mat, rep_num, axis):
        return torch.repeat_interleave(torch.unsqueeze(mat, axis=axis), rep_num, axis=axis)
    
    len_channel, lenA = A_f1d.shape
    len_channel, lenB = B_f1d.shape        
    
    row_repeat = rep_new_axis(A_f1d, lenB, 2)
    col_repeat = rep_new_axis(B_f1d, lenA, 1)        

    return torch.unsqueeze(torch.cat((row_repeat, col_repeat, p2d), axis=0), 0)

def top_statistics_ppi(pred_map, contact_map, Topk_list):
    count = 0
    single_statistics = np.ones((len(Topk_list)))

    L = min(contact_map.shape[0], contact_map.shape[1])

    Label = contact_map.flatten()
    pred_map = pred_map.flatten()

    for Topk in Topk_list:
        if isinstance(Topk, str):
            Topk = int(L / int(Topk[Topk.index('/') + 1:]))
            if Topk < 1:
                Topk = 1

        SortedP = torch.topk(pred_map, Topk, largest=True, sorted=True)[1]
        TTL = torch.sum(Label[SortedP]).item()

        single_statistics[count] = TTL / Topk
        count += 1

    return single_statistics

if __name__ == '__main__':
    ###################              load dataset               ###################
    feature_dir = '/home/huangqiyuan/datav2/bigtrainset/5pkl'  # 包含所有pkl文件的目录
    all_files = [os.path.join(feature_dir, f) for f in os.listdir(feature_dir) if f.endswith('.pkl')]

    for i in range(10):
        random.shuffle(all_files)

    train_files = all_files[:6300]  # 前6300个作为训练集
    valid_files = all_files[6300:]  # 余下的作为验证集

    trainset = HemoDataset(train_files)
    validset = HemoDataset(valid_files)

    train_loader = DataLoader(trainset, batch_size=None, shuffle=True, num_workers=6, prefetch_factor=3, persistent_workers=True)
    valid_loader = DataLoader(validset, batch_size=None, shuffle=True, num_workers=6, prefetch_factor=3, persistent_workers=True)
    max_aa = 400

    ###################               import net                ###################
    device = torch.device('cuda')
    model = resnet18().to(device)
#    model=ContactMapUnet().to(device)
    criterion_ppi = PpiLoss(alpha=0.83, reduction='sum')
    optimizer = optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', eps=1e-6, patience=3, factor=0.1, verbose=True)

    epoch_num = 40

    ###################             top statistics              ###################
    topk_ppi = ['L/5', 'L/10', 'L/20', 50, 20, 10, 5, 1]
    dict_statics = {'min_loss': np.inf, 'valid_loss': []}

    for key in topk_ppi:
        dict_statics[key] = {'highest': 0, 'save': '', 'train_acc': [], 'valid_acc': []}

    ###################               save model                ###################
    epoch_target = {}

    savepth = '/home/huangqiyuan/datav2/modelpath/model23/5pkl/model'

    for key in topk_ppi:
        dict_statics[key]['save'] = '{0}_{1}.pkl'.format(savepth, str(key).replace('/', '_'))
    loss_save = f'{savepth}_minloss.pth'

    ###################                training                 ###################
    for epoch in range(epoch_num):

        for phase in ['train', 'valid']:
            print('\n')
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = valid_loader

            acc_all = np.zeros((0, len(topk_ppi)))
            acc_batch = np.zeros((0, len(topk_ppi)))
            running_loss = 0.0
            optimizer.zero_grad()

            for i, (pdb_name, inputs, targets) in enumerate(dataloader):                     
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
                                
                la, lb = targets.shape
                starta = 0 if la <= max_aa else np.random.randint(0, la - max_aa + 1)
                startb = 0 if lb <= max_aa else np.random.randint(0, lb - max_aa + 1)
                    
                rec1d = rec1d[:, starta:(starta + max_aa)]
                lig1d = lig1d[:, startb:(startb + max_aa)]
                com2d = com2d[:, starta:(starta + max_aa), startb:(startb + max_aa)]
                targets = targets[starta:(starta + max_aa), startb:(startb + max_aa)]              

                                         
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(rec1d, lig1d, com2d)
                    # 检查 outputs 和 targets 的形状是否一致
                    if outputs.shape != targets.shape:
				     # 如果 outputs 和 targets 的形状不一致，尝试转置 outputs
                        if outputs.shape == targets.transpose(0, 1).shape:
                            outputs = outputs.transpose(0, 1)  # 转置输出矩阵
                        else:
                            print(f"输出和标签的形状不匹配：outputs {outputs.shape} vs targets {targets.shape}")
                            continue  # 跳过该批次，继续下一个数据
				 
                    loss = criterion_ppi(outputs, targets)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                running_loss += loss.item()

                ##################          statistics           ##################
                accuracy = top_statistics_ppi(outputs, targets, topk_ppi)
                acc_all = np.vstack([acc_all, accuracy])
                acc_batch = np.vstack([acc_batch, accuracy])

                if (i + 1) % 100 == 0:
                    mean_acc = np.mean(acc_all[-100:], 0) * 100
                    print(
                        f'[{epoch:3d}, {i + 1:4d}]  loss:{running_loss:11.2f} {"  ".join([f"{j.item():7.3f}" for j in mean_acc])}')
                    batch_loss = 0
                    acc_batch = np.zeros((0, len(topk_ppi)))
                if (i + 1) == len(dataloader):
                    mean_acc = np.mean(acc_all, 0) * 100
                    print(
                        f'[{epoch:3d}, {i + 1:4d}]  loss:{running_loss:11.2f} {"  ".join([f"{j.item():7.3f}" for j in mean_acc])}')

            if phase == 'valid':
                scheduler.step(running_loss)
                dict_statics['valid_loss'].append(running_loss)
                for index, key in enumerate(topk_ppi):
                    dict_statics[key]['valid_acc'].append(mean_acc[index])
            else:
                for index, key in enumerate(topk_ppi):
                    dict_statics[key]['train_acc'].append(mean_acc[index])

        ##################                 save                  ##################
        for key in topk_ppi:
            acc = dict_statics[key]['valid_acc'][-1]
            highest = dict_statics[key]['highest']
            if acc > highest:
                print(f'save_{str(key):5s}:{acc:6.3f}  highest: {highest:6.3f}  delta:{acc - highest:6.3f}')
                dict_statics[key]['highest'] = acc

                if os.path.exists(dict_statics[key]['save']):
                    os.remove(dict_statics[key]['save'])
                torch.save(model.state_dict(), dict_statics[key]['save'])

        if running_loss < dict_statics['min_loss']:
            print('save_minloss:%11.2f    %11.2f' % (running_loss, dict_statics['min_loss']))
            dict_statics['min_loss'] = running_loss
            torch.save(model.state_dict(), loss_save)

        torch.save(model.state_dict(), savepth + f'__{epoch}.pkl')
