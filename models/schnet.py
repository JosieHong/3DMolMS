import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from decimal import *

from .utils import ShiftedSoftPlus, MSDecoder

# refer: 
# https://lifesci.dgl.ai/_modules/dgllife/model/gnn/schnet.html



class RBFExpansion(nn.Module):
    
    def __init__(self, low=0., high=30., gap=0.1):
        super(RBFExpansion, self).__init__()

        num_centers = int(np.ceil((high - low) / gap))
        self.centers = np.linspace(low, high, num_centers)
        self.centers = nn.Parameter(torch.tensor(self.centers).float(), requires_grad=False)
        self.gamma = 1 / gap

    def reset_parameters(self): 
        device = self.centers.device
        self.centers = nn.Parameter(
            self.centers.clone().detach().float(), requires_grad=False).to(device)

    def forward(self, edge_dists): 
        radial = edge_dists - self.centers
        coef = - self.gamma
        return torch.exp(coef * (radial ** 2))



class CFConv(nn.Module):
    def __init__(self, hidden_dim): 
        super(CFConv, self).__init__()
        self.rbf = RBFExpansion(low=0., high=30., gap=0.1)
        self.fc1 = nn.Linear(3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act1 = ShiftedSoftPlus(beta=1, origin=0.5, threshold=20)
        self.act2 = ShiftedSoftPlus(beta=1, origin=0.5, threshold=20)

    def forward(self, x, r):
        r = self.rbf(r)

        x = x.permute(0, 2, 1)
        r = r.permute(0, 2, 1)
        
        r = self.act1(self.fc1(r))
        r = self.act2(self.fc2(r))
        
        x = torch.mul(x, r).permute(0, 2, 1)
        return x

class Interaction(nn.Module):
    def __init__(self, hidden_dim):
        super(Interaction, self).__init__()
        self.fc1 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.cfconv = CFConv(hidden_dim)
        self.fc2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc3 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.activation = ShiftedSoftPlus(beta=1, origin=0.5, threshold=20)

    def forward(self, h, r): 
        x = self.fc1(h)
        x = self.cfconv(x, r)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = h + x
        return x


class SchNet_MS(nn.Module):

    def __init__(self, args): 
        super(SchNet_MS, self).__init__()
        self.device = args.device
        self.num_add = args.num_add
        self.k = args.k

        self.embedding = nn.Conv1d(args.in_channels, 64, 1)
        self.interaction_layers = nn.ModuleList(
            [
                Interaction(64),
                Interaction(64),
                Interaction(64)
            ]
        )
        self.layers = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            ShiftedSoftPlus(beta=1, origin=0.5, threshold=20),
            nn.Conv1d(64, args.emb_dim, 1), 
        )
        self.decoder = MSDecoder(in_dim=(args.emb_dim + args.num_add + 3), 
                                layers=[2048, 2048, 2048, 2048, 2048], 
                                out_dim=(int(Decimal(str(args.out_dim)) // Decimal(str(args.resolution)))), 
                                dropout=args.dropout)

    def forward(self, x, env, idx_base): 
        r = x[:, :3, :]

        x = self.embedding(x)
        for interaction_ in self.interaction_layers:
            x = interaction_(x, r)
        x = self.layers(x)        
        x = torch.sum(x, dim=2)

        if self.num_add == 1:
            x = torch.cat((x, torch.unsqueeze(env, 1)), 1)
        elif self.num_add > 1:
            x = torch.cat((x, env), 1)

        # decoder
        x = self.decoder(x)
        return x