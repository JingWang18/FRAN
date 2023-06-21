import torch
import torch.nn as nn
import torch.nn.functional as F
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
    def forward(self, x): # x.shape -> [64, 64, 300]
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

        scale = F.sigmoid(100*channel_att_sum).unsqueeze(2).expand_as(x) # channel_att_sum.shape -> [64, 64]
        return x * scale


class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=4, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv21 = nn.Conv1d(32, 64, kernel_size=4, stride=1, padding=2)
        self.bn21 = nn.BatchNorm1d(64)
        self.relu = nn.Sigmoid()
        self.maxpool = nn.AvgPool1d(stride=2, kernel_size=2)

        self.channel_1 = ChannelGate(32, pool_types=['avg', 'max'])
        self.channel_2 = ChannelGate(64, pool_types=['avg', 'max'])

    def forward(self, x, is_deconv=False):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.channel_1(x)
        x = self.maxpool(self.relu(self.bn21(self.conv21(x))))
        x = self.channel_2(x)
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
