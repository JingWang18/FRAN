import torch
import torch.nn as nn
import torch.nn.functional as F

import pywt
import pytorch_wavelets.dwt.lowlevel as lowlevel

from pytorch_wavelets import DWT1DForward, DTCWTForward
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
        # return torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1), torch.std(x,1).unsqueeze(1)), dim=1)
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 3
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)

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
        self.linear_1 = nn.Linear(360000, 600)
        self.linear_2 = nn.Linear(9000, 300)

        self.conv11 = nn.Conv1d(1, 12, kernel_size=5, stride=1, padding=2)
        self.conv12 = nn.Conv1d(12, 12, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(12)
        self.conv21 = nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2)
        self.conv22 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(12)
        self.relu = nn.Sigmoid()
        self.maxpool = nn.MaxPool1d(stride=2, kernel_size=2)  # average

        self.channel_1 = ChannelGate(32, pool_types=['avg', 'max'])
        self.SpatialGate = SpatialGate()

        # self.dwt_1 = DWT1DForward(wave='db6', J=1).cuda()
        # self.dwt_2 = DWT1DForward(wave='db6', J=1).cuda()

        self.dwt_1 = DTCWTForward(J=2, biort='near_sym_b', qshift='qshift_b').cuda()

        # self.channel_1 = ChannelGate(32, pool_types=['max'])
        # self.channel_2 = ChannelGate(64, pool_types=['avg', 'max'])

    def forward(self, x, is_target=False):
        # wave = pywt.Wavelet('db1')
        # filts = lowlevel.prep_filt_afb1d(wave.dec_lo, wave.dec_hi)

        # x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        # Wavelet transform with 3 levels
        x = x.unsqueeze(3).expand(x.shape[0], x.shape[1], x.shape[2], x.shape[2])
        x_0 = x
        _, z = self.dwt_1(x)
        # z[0] is the real part and z[1] is the imaginary part
        # z[0] -> 64, 1, 6, 600, 600, 2 where 6 is 6 orientations and 2 is the real and imaginary parts
        z_1, z_2 = z[0], z[1] # z_n n is the level index
        z_1 = z_1.view(64*12,360000)
        z_2 = z_2.view(64*12,9000)
        z_1 = self.linear_1(z_1).view(64,12,600)
        z_2 = self.linear_2(z_2).view(64,12,300)

        x = self.maxpool(self.bn1(self.conv11(x_0))) + z_1
        x = self.maxpool(self.bn2(self.conv12(x))) + z_2

        # x = self.SpatialGate(x)

        # x = self.SpatialGate(x, is_target)
        # x = self.SpatialGate(x)
        # x = self.channel_2(x)

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
