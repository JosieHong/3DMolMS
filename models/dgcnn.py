"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from decimal import *

from .utils import MSDecoder



def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    # print(x.size(), pairwise_distance.size())
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, device, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    # device = torch.device('cuda')
    device = torch.device("cuda:" + str(device)) if torch.cuda.is_available() else torch.device("cpu")

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    
    return feature


class DGCNN_MS(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN_MS, self).__init__()
        self.device = args.device
        self.num_add = args.num_add
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn6 = nn.BatchNorm1d(args.emb_dim)

        self.conv1 = nn.Sequential(nn.Conv2d(args.in_channels*2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(256*2, 512, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv6 = nn.Sequential(nn.Conv1d(1024, args.emb_dim, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.merge = nn.Sequential(nn.Linear(args.emb_dim*2, args.emb_dim), 
                                   nn.BatchNorm1d(args.emb_dim), 
                                   nn.LeakyReLU(negative_slope=0.2))

        self.decoder = MSDecoder(in_dim=(args.emb_dim + args.num_add + 3), 
                                layers=[2048, 2048, 2048, 2048, 2048], 
                                out_dim=(int(Decimal(str(args.out_dim)) // Decimal(str(args.resolution)))), 
                                dropout=args.dropout)

    def forward(self, x, env, idx_base): 
        batch_size = x.size(0)

        x = get_graph_feature(x, device=self.device, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, device=self.device, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, device=self.device, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, device=self.device, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x4, device=self.device, k=self.k)
        x = self.conv5(x)
        x5 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv6(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = self.merge(x)

        # add the encoded adduct
        if self.num_add == 1:
            x = torch.cat((x, torch.unsqueeze(env, 1)), 1)
        elif self.num_add > 1:
            x = torch.cat((x, env), 1)

        # decoder
        x = self.decoder(x)
        return x