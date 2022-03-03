'''
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from decimal import *

from .utils import MSDecoder



class PointNet_MS(nn.Module):
    def __init__(self, args): 
        super(PointNet_MS, self).__init__()
        self.num_add = args.num_add

        self.conv1 = nn.Conv1d(args.in_channels, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dim)

        self.decoder = MSDecoder(in_dim=(args.emb_dim + args.num_add + 3), 
                                layers=[2048, 2048, 2048, 2048, 2048], 
                                out_dim=(int(Decimal(str(args.out_dim)) // Decimal(str(args.resolution)))), 
                                dropout=args.dropout)

    def forward(self, x, env, idx_base): 
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()

        # add the encoded adduct
        if self.num_add == 1:
            x = torch.cat((x, torch.unsqueeze(env, 1)), 1)
        elif self.num_add > 1:
            x = torch.cat((x, env), 1)

        # decoder
        x = self.decoder(x)
        return x