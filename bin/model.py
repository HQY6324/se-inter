# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
from collections import OrderedDict
from torch import einsum
from einops import rearrange



""" conv1d """
def conv1d( in_channels: int, 
            out_channels: int, 
            kernel_size: int, 
            stride: int = 1, 
            padding: str = "same", 
            dilation: int = 1, 
            group: int = 1, 
            bias: bool = False) -> nn.Conv1d:

    if padding == "same":
        padding = int((kernel_size - 1)/2)

    return nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, group, bias)



class Block_ResNet1D(nn.Module):

    def __init__(self,
        in_channels  : int,
        out_channels : int,
        kernel_size  : int,
        stride       : int = 1,
        downsample = None,
        padding      : str = "same",
        dilation     : int = 1,
        group        : int = 1,
        bias         : bool = False,
        track_running_stats_ : bool = True,
        norm         : str = "IN",
        activation   : str = "Relu"):

        super(Block_ResNet1D, self).__init__()

        if norm == "BN":
            self.bn1 = nn.BatchNorm1d(in_channels, affine=True, track_running_stats=track_running_stats_)
            self.bn2 = nn.BatchNorm1d(out_channels, affine=True, track_running_stats=track_running_stats_)
        elif norm == "IN":
            self.bn1 = nn.InstanceNorm1d(in_channels, affine=True, track_running_stats=track_running_stats_)
            self.bn2 = nn.InstanceNorm1d(out_channels, affine=True, track_running_stats=track_running_stats_)

        if activation == "ELU":
            self.relu1 = nn.ELU()
            self.relu2 = nn.ELU()
        elif activation == "Relu":
            self.relu1 = nn.LeakyReLU(negative_slope=0.01,inplace=True)
            self.relu2 = nn.LeakyReLU(negative_slope=0.01,inplace=True)

        self.conv1 = conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, group, bias)
        self.conv2 = conv1d(out_channels, out_channels, kernel_size, stride, padding, dilation, group, bias)
        # SE layers
        self.fc1 = nn.Conv1d(out_channels, out_channels//8, kernel_size=1)
        self.fc2 = nn.Conv1d(out_channels//8, out_channels, kernel_size=1)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.01,inplace=False)

        self.downsample = downsample

    def forward(self, x):

        identity = x

        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2(x)

        global_avg_pool = nn.AdaptiveAvgPool1d(1)
        w = global_avg_pool(x)
        w = self.leakyrelu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        x = x * w

        if self.downsample != None :
            identity = self.downsample(identity)

        x += identity

        return x

   
class OuterProductMean(nn.Module):
    def __init__(self, dim, pair_dim) -> None:
        super().__init__()

        self.reduction = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 16)
        )
        self.to_out = nn.Linear(16 * 16, pair_dim)

    def forward(self, x,y):
        x = self.reduction(x)
        y = self.reduction(y)
        x_outer = einsum('bmi,bnj->bmnij', x, y)       
        out = self.to_out(rearrange(x_outer, '... i j -> ... (i j)'))

        return out
        

def concat(A_f1d, B_f1d, p2d):
    
    def rep_new_axis(mat, rep_num, axis):
        return torch.repeat_interleave(torch.unsqueeze(mat,axis=axis),rep_num,axis=axis)
    
    len_channel,lenA = A_f1d.shape
    len_channel,lenB = B_f1d.shape        
    
    row_repeat = rep_new_axis(A_f1d, lenB, 2)
    col_repeat = rep_new_axis(B_f1d, lenA, 1)        


    return  torch.unsqueeze(torch.cat((row_repeat, col_repeat, p2d),axis=0),0)


def make_conv_layer(in_channels,
                    out_channels,
                    kernel_size,
                    padding_size,
                    non_linearity=True,
                    instance_norm=False,
                    dilated_rate=1):
    layers = []

    layers.append(
        ('conv', nn.Conv2d(in_channels, out_channels, kernel_size,
                           padding=padding_size, dilation=dilated_rate, bias=False)))
    # layers.append(
    #     ('dropout', nn.Dropout2d(p=0.3, inplace=True)))
    if instance_norm:
        layers.append(('in', nn.InstanceNorm2d(num_features=out_channels,momentum=0.1,
                                        affine=True,track_running_stats=False)))
    if non_linearity:
        layers.append(('leaky', nn.LeakyReLU(negative_slope=0.01,inplace=False)))

    layers.append(
        ('conv2', nn.Conv2d(out_channels, out_channels, kernel_size,
                           padding=padding_size, dilation=dilated_rate, bias=False)))
    if instance_norm:
        layers.append(('in2', nn.InstanceNorm2d(num_features=out_channels,momentum=0.1,
                                        affine=True,track_running_stats=False)))

    return nn.Sequential(OrderedDict(layers))


def make_1x1_layer(in_channels,
                    out_channels,
                    kernel_size,
                    padding_size,
                    non_linearity=True,
                    instance_norm=False,
                    dilated_rate=1):
    layers = []

    layers.append(
        ('conv', nn.Conv2d(in_channels, out_channels, kernel_size=1,
                           padding=0, dilation=1, bias=False)))
    if instance_norm:
        layers.append(('in', nn.InstanceNorm2d(num_features=out_channels,momentum=0.1,
                                        affine=True,track_running_stats=False)))
    if non_linearity:
        layers.append(('leaky', nn.LeakyReLU(negative_slope=0.01,inplace=False)))

    return nn.Sequential(OrderedDict(layers))

class BasicBlock_2D(nn.Module):
    expansion = 1

    def __init__(self, in_channels,
                       out_channels,
                       dilated_rate):
        super(BasicBlock_2D, self).__init__()

        self.leakyrelu = nn.LeakyReLU(negative_slope=0.01,inplace=False)
        self.dilated_rate = dilated_rate
        self.concatenate = False
        self.threshold = [1,20,40]
        self.Bool_in = True
        self.Bool_nl = True
        # SE layers
        self.fc1 = nn.Conv2d(out_channels, out_channels//8, kernel_size=1)
        self.fc2 = nn.Conv2d(out_channels//8, out_channels, kernel_size=1)


        self.conv_3x3 = make_conv_layer(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=(3,3),
                                        padding_size=(1,1),
                                        non_linearity=self.Bool_nl,
                                        instance_norm=self.Bool_in,
                                        dilated_rate=(dilated_rate,dilated_rate))

        if dilated_rate in self.threshold:
            self.conv_1xn = make_conv_layer(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=(1,15),
                                            padding_size=(0,7*dilated_rate),
                                            non_linearity=self.Bool_nl,
                                            instance_norm=self.Bool_in,
                                            dilated_rate=(1,dilated_rate))

            self.conv_nx1 = make_conv_layer(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=(15,1),
                                            padding_size=(7*dilated_rate,0),
                                            non_linearity=self.Bool_nl,
                                            instance_norm=self.Bool_in,
                                            dilated_rate=(dilated_rate,1))

            if self.concatenate:
                self.conv_1x1 = make_1x1_layer(in_channels=in_channels*3,
                                               out_channels=out_channels,
                                               kernel_size=(1,1),
                                               padding_size=(0,0),
                                               non_linearity=self.Bool_nl,
                                               instance_norm=self.Bool_in,
                                               dilated_rate=(1,1))

    def forward(self, x):

        out = x

        identity1 = self.conv_3x3(x)

        if self.dilated_rate in self.threshold:
            identity2 = self.conv_1xn(x)
            identity3 = self.conv_nx1(x)

            if self.concatenate:
                identity = torch.cat((identity1,identity2,identity3),1)
                identity = self.conv_1x1(identity)
            else:
                identity = identity1+identity2+identity3

        else:
            identity = identity1

            
        global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        w = global_avg_pool(identity)
        w = self.leakyrelu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        identity = identity * w
        out = out+identity

        if self.Bool_nl:
            out = self.leakyrelu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, blocks_num):
        super(ResNet, self).__init__()
        self.in_channel = 128

        self.reduction2d = nn.Linear(660, 64)
        self.OuterProductMean = OuterProductMean(256, 64)

        def get_downsample(in_channels, out_channels, kernel_size=1, stride=1, padding='same'):
           if padding == "same":
               padding = int((kernel_size - 1) / 2)
           return nn.Sequential(
               conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
               nn.InstanceNorm1d(out_channels)  # 或者使用 BatchNorm，根据需要选择
       )

        
         # 第一层残差块，输入维度和输出维度不同
        self.residual_block_rec = nn.Sequential(
            Block_ResNet1D(1280, 256,kernel_size=1,downsample=get_downsample(1280, 256, kernel_size=1)),  # 第一层
            *[Block_ResNet1D(256, 256,kernel_size=1) for _ in range(5)]  # 后续层
        )
        
        self.residual_block_lig = nn.Sequential(
            Block_ResNet1D(1280, 256,kernel_size=1,downsample=get_downsample(1280, 256, kernel_size=1)),  # 第一层
            *[Block_ResNet1D(256, 256,kernel_size=1) for _ in range(5)]  # 后续层
        )

        
        self.hidden_layer = self._make_layer(in_channel=self.in_channel, out_channel=self.in_channel,
                                             block_num=blocks_num,dilated_rate=1)

        self.output_layer = make_1x1_layer(in_channels=self.in_channel,
                                          out_channels=1,
                                          kernel_size=(1,1),
                                          padding_size=(0,0),
                                          non_linearity=False,
                                          instance_norm=False,
                                          dilated_rate=(1,1))

        self.Sig = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.01, mode='fan_in',nonlinearity='leaky_relu')


    def _make_layer(self, in_channel, out_channel, block_num, dilated_rate):

        layers = []

        for index in range(block_num):
            layers.append(('block'+str(index),BasicBlock_2D(in_channel, out_channel, dilated_rate)))

        return nn.Sequential(OrderedDict(layers))

    def forward(self, rec1d, lig1d, com2d):

        rec1d = rec1d.unsqueeze(0)
        lig1d = lig1d.unsqueeze(0)
        com2d = com2d.unsqueeze(0).permute(0,2,3,1)
        com2d = self.reduction2d(com2d)
    
        rec1d = self.residual_block_rec(rec1d)
        lig1d = self.residual_block_lig(lig1d)
        
        rec1d = rec1d.permute(0, 2, 1)
        lig1d = lig1d.permute(0, 2, 1)

        pair = self.OuterProductMean(rec1d, lig1d)

        combined_pair = torch.cat((pair, com2d), dim=3)
        combined_pair = combined_pair.permute(0, 3, 1, 2)

        
        x = self.hidden_layer(combined_pair)

        x = self.output_layer(x)

        x = torch.squeeze(x)
        x = torch.clamp(x,min=-15,max=15)
        x = self.Sig(x)

        return x




def resnet18():
    return ResNet(9)