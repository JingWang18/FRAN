import torch
import torch.nn as nn
import torch.nn.functional as F

import dtcwt
from pytorch_wavelets import DWT1DForward

import pdb


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x, is_target=False): # x.shape -> [64, 64, 300]
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool1d(x, kernel_size=x.size(2), stride=x.size(2))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type=='max':
                max_pool = F.max_pool1d(x, kernel_size=x.size(2), stride=x.size(2))
                channel_att_raw = self.mlp(max_pool)
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        scale = F.sigmoid(channel_att_sum).unsqueeze(2).expand_as(x) # channel_att_sum.shape -> [64, 64]
        if is_target:
            scale = torch.ones_like(scale).cuda() - scale
        return x * scale

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm1d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1), torch.std(x,1).unsqueeze(1)), dim=1)
        # return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 3
        self.compress = ChannelPool()
        self.spatial = BasicConv(3, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)

    def sigmoid(self, x):
        return 1./(1.+torch.exp(-x))

    def forward(self, x, is_target=False):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = self.sigmoid(x_out) # broadcasting
        if is_target:
            scale = torch.ones_like(scale).cuda() - scale
        return x_compress * scale


class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
        self.conv_freq_1 = nn.Conv1d(1, 32, kernel_size=4, stride=1, padding=1)
        self.conv_freq_2 = nn.Conv1d(1, 64, kernel_size=5, stride=1, padding=2)
        self.conv_freq_3 = nn.Conv1d(1, 64, kernel_size=5, stride=1, padding=2)

        self.conv_time_1 = nn.Conv1d(1, 32, kernel_size=4, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv_time_2 = nn.Conv1d(32, 64, kernel_size=4, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv_time_3 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(64)

        self.maxpool = nn.MaxPool1d(stride=2, kernel_size=2)  # max is better than average
        self.avgpool = nn.AvgPool1d(stride=2, kernel_size=2)
        self.relu = nn.ReLU()

        self.channel_1 = ChannelGate(32, pool_types=['avg', 'max'])
        self.SpatialGate = SpatialGate()

        self.transform = DWT1DForward(wave='haar', J=3).cuda()
        self.channel_1 = ChannelGate(32, pool_types=['avg','max'])
        # self.channel_2 = ChannelGate(64, pool_types=['avg', 'max'])

    def forward(self, x, is_target=False):
        x_0 = x
        # db: zh[0] -> 64,1,605    haar: 600
        #     zh[1] -> 64,1,308          300
        #     zh[2] -> 64,1,159          150
        zl, zh = self.transform(x)
        z1 = self.conv_freq_1(zh[0]) # 64, 16, 600
        z2 = self.conv_freq_2(zh[1]) # 64, 32, 300
        # z3 = self.conv_freq_3(zh[2]) # 64, 64, 150

        x = self.maxpool(self.relu(self.bn1(self.conv_time_1(x_0))))+z1
        x = self.maxpool(self.relu(self.bn2(self.conv_time_2(x))))+z2
        # x = self.maxpool(self.relu(self.bn3(self.conv_time_3(x)))) + z3

        # x = self.SpatialGate(x)

        return x

class Predictor(nn.Module):
    def __init__(self, prob=0.5):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(64*300, 1000)
        self.bn1_fc = nn.BatchNorm1d(1000)
        self.fc3 = nn.Linear(1000, 3)
        self.bn_fc3 = nn.BatchNorm1d(3)
        self.relu = nn.ReLU()
        self.prob = prob

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        x = x.view(x.size(0), 64*300)
        x = F.dropout(x, training=self.training, p=self.prob)
        x = self.relu(self.bn1_fc(self.fc1(x)))
        x = self.fc3(x)
        return x
