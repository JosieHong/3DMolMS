import torch
import torch.nn as nn
import torch.nn.functional as F
from decimal import *
from typing import Tuple



# ----------------------------------------
# >>>           encoder part           <<<
# ----------------------------------------
class MolConv(nn.Module):
	def __init__(self, in_dim, out_dim, k, remove_xyz=False):
		super(MolConv, self).__init__()
		self.k = k
		self.in_dim = in_dim
		self.out_dim = out_dim
		self.remove_xyz = remove_xyz

		self.dist_ff = nn.Sequential(nn.Conv2d(1, 1, kernel_size=1, bias=False),
								nn.BatchNorm2d(1),
								nn.Sigmoid())
		self.gm2m_ff = nn.Sequential(nn.Conv2d(k, 1, kernel_size=1, bias=False),
								nn.BatchNorm2d(1),
								nn.Sigmoid())

		if remove_xyz: 
			self.update_ff = nn.Sequential(nn.Conv2d(in_dim-3, out_dim, kernel_size=1, bias=False),
								nn.BatchNorm2d(out_dim),
								nn.LeakyReLU(negative_slope=0.02))
		else:
			self.update_ff = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
								nn.BatchNorm2d(out_dim),
								nn.LeakyReLU(negative_slope=0.02))

		self._reset_parameters()

	def _reset_parameters(self): 
		for m in self.modules(): 
			if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
				nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
			
			elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)): 
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, x: torch.Tensor, 
						idx_base: torch.Tensor) -> torch.Tensor: 
		dist, gm2, feat_c, feat_n = self._generate_feat(x, idx_base, k=self.k, remove_xyz=self.remove_xyz) 
		'''Returned features: 
		dist: torch.Size([batch_size, 1, point_num, k])
		gm2: torch.Size([batch_size, k, point_num, k]) 
		feat_c: torch.Size([batch_size, in_dim, point_num, k]) 
		feat_n: torch.Size([batch_size, in_dim, point_num, k])
		'''
		w1 = self.dist_ff(dist)
		w2 = self.gm2m_ff(gm2)
		
		feat = torch.mul(w1, w2) * feat_n + (1 - torch.mul(w1, w2)) * feat_c
		feat = self.update_ff(feat)
		feat = feat.mean(dim=-1, keepdim=False)
		return feat

	def _generate_feat(self, x: torch.Tensor, 
								idx_base: torch.Tensor, 
								k: int, 
								remove_xyz: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
		batch_size, num_dims, num_points = x.size()
		
		# local graph (knn)
		inner = -2*torch.matmul(x.transpose(2, 1), x)
		xx = torch.sum(x**2, dim=1, keepdim=True)
		pairwise_distance = -xx - inner - xx.transpose(2, 1)
		dist, idx = pairwise_distance.topk(k=k, dim=2) # (batch_size, num_points, k)
		dist = - dist

		idx = idx + idx_base
		idx = idx.view(-1)

		x = x.transpose(2, 1).contiguous() # (batch_size, num_points, num_dims) -> (batch_size*num_points, num_dims) 
		graph_feat = x.view(batch_size*num_points, -1)[idx, :]
		graph_feat = graph_feat.view(batch_size, num_points, k, num_dims)

		# gram matrix
		gm_matrix = torch.matmul(graph_feat, graph_feat.permute(0, 1, 3, 2))

		# double gram matrix
		sub_feat = gm_matrix[:, :, :, 0].unsqueeze(3)
		sub_gm_matrix = torch.matmul(sub_feat, sub_feat.permute(0, 1, 3, 2))
		sub_gm_matrix = F.normalize(sub_gm_matrix, dim=1) 
		
		x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

		if remove_xyz:
			return dist.unsqueeze(3).permute(0, 3, 1, 2).contiguous(), \
					sub_gm_matrix.permute(0, 3, 1, 2).contiguous(), \
					x[:, :, :, 3:].permute(0, 3, 1, 2).contiguous(), \
					graph_feat[:, :, :, 3:].permute(0, 3, 1, 2).contiguous()
		else:
			return dist.unsqueeze(3).permute(0, 3, 1, 2).contiguous(), \
					sub_gm_matrix.permute(0, 3, 1, 2).contiguous(), \
					x.permute(0, 3, 1, 2).contiguous(), \
					graph_feat.permute(0, 3, 1, 2).contiguous()
	
	def __repr__(self):
		return self.__class__.__name__ + ' k = ' + str(self.k) + ' (' + str(self.in_dim) + ' -> ' + str(self.out_dim) + ')'

class Encoder(nn.Module):
	def __init__(self, in_dim, layers, emb_dim, k): 
		super(Encoder, self).__init__()
		self.emb_dim = emb_dim
		self.hidden_layers = nn.ModuleList([MolConv(in_dim=in_dim, out_dim=layers[0], k=k, remove_xyz=True)])
		for i in range(1, len(layers)): 
			if i == 1:
				self.hidden_layers.append(MolConv(in_dim=layers[i-1], out_dim=layers[i], k=k, remove_xyz=False))
			else:
				self.hidden_layers.append(MolConv(in_dim=layers[i-1], out_dim=layers[i], k=k, remove_xyz=False))
		
		self.conv = nn.Sequential(nn.Conv1d(emb_dim, emb_dim, kernel_size=1, bias=False),
								   nn.BatchNorm1d(emb_dim), 
								   nn.LeakyReLU(negative_slope=0.2))

		self.merge = nn.Sequential(nn.Linear(emb_dim*2, emb_dim), 
								   nn.BatchNorm1d(emb_dim), 
								   nn.LeakyReLU(negative_slope=0.2))
		self._reset_parameters()

	def _reset_parameters(self): 
		for m in self.merge: 
			if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
				nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
			
			elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)): 
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, x: torch.Tensor,  
						idx_base: torch.Tensor) -> torch.Tensor: 
		xs = []
		for i, hidden_layer in enumerate(self.hidden_layers): 
			if i == 0: 
				tmp_x = hidden_layer(x, idx_base)
			else: 
				tmp_x = hidden_layer(xs[-1], idx_base)
			# apply the mask
			# tmp_mask = mask.repeat(1, tmp_x.size(1), 1)
			# tmp_x = torch.mul(tmp_x, tmp_mask) 
			xs.append(tmp_x)

		x = torch.cat(xs, dim=1)
		x = self.conv(x)
		p1 = F.adaptive_max_pool1d(x, 1).squeeze().view(-1, self.emb_dim)
		p2 = F.adaptive_avg_pool1d(x, 1).squeeze().view(-1, self.emb_dim)
		
		x = torch.cat((p1, p2), 1)
		x = self.merge(x)
		return x



# ----------------------------------------
# >>>           decoder part           <<<
# ----------------------------------------
class FCResBlock(nn.Module): 
    def __init__(self, in_dim: int, out_dim: int, dropout: float=0.) -> torch.Tensor: 
        super(FCResBlock, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.linear1 = nn.Linear(in_dim, out_dim, bias=False) 
        self.bn1 = nn.LayerNorm(out_dim)

        self.linear2 = nn.Linear(out_dim, out_dim, bias=False)
        self.bn2 = nn.LayerNorm(out_dim)

        self.linear3 = nn.Linear(out_dim, out_dim, bias=False)
        self.bn3 = nn.LayerNorm(out_dim)

        self.dp = nn.Dropout(dropout)
            
        self._reset_parameters()

    def _reset_parameters(self): 
        for m in self.modules(): 
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
            
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)): 
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        identity = x
        
        x = self.bn1(self.linear1(x))
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.bn2(self.linear2(x))
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.bn3(self.linear3(x))
        
        x = x + F.interpolate(identity.unsqueeze(1), size=x.size()[1]).squeeze()

        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.dp(x)
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_dim) + ' -> ' + str(self.out_dim) + ')'

class MSDecoder(nn.Module): 
    def __init__(self, in_dim, layers, out_dim, dropout): 
        super(MSDecoder, self).__init__()
        self.blocks = nn.ModuleList([FCResBlock(in_dim=in_dim, out_dim=layers[0])])
        for i in range(len(layers)-1): 
            if len(layers) - i > 3:
                self.blocks.append(FCResBlock(in_dim=layers[i], out_dim=layers[i+1]))
            else:
                self.blocks.append(FCResBlock(in_dim=layers[i], out_dim=layers[i+1], dropout=dropout))

        self.fc = nn.Linear(layers[-1], out_dim)
        
        self._reset_parameters()

    def _reset_parameters(self): 
        nn.init.kaiming_normal_(self.fc.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        # x = self.fc(x)
        # return F.glu(torch.cat((x, x), dim=1))
        return self.fc(x)



# -----------------------------------
# >>>           3DMolMS           <<<
# -----------------------------------
class MolNet_MS(nn.Module): 
	def __init__(self, config): 
		super(MolNet_MS, self).__init__()
		self.add_num = config['add_num']

		self.encoder = Encoder(in_dim=int(config['in_dim']), 
									layers=config['encode_layers'],
									emb_dim=int(config['emb_dim']), 
									k=int(config['k']))
		self.decoder = MSDecoder(in_dim=int(config['emb_dim'] + config['add_num']), 
								layers=config['decode_layers'], 
								out_dim=int(Decimal(str(config['max_mz'])) // Decimal(str(config['resolution']))), 
								dropout=config['dropout'])
		
		for m in self.modules(): 
			if isinstance(m, nn.Conv1d):
				nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
			elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				m.weight.data.normal_(mean=0.0, std=1.0)
				if m.bias is not None:
					m.bias.data.zero_()

	def forward(self, x: torch.Tensor, 
						env: torch.Tensor, 
						idx_base: torch.Tensor) -> torch.Tensor: 
		'''
		Input: 
			x:      point set, torch.Size([batch_size, 14, atom_num])
			env:    experimental condiction
			idx_base:   idx for local knn
		'''
		x = self.encoder(x, idx_base) # torch.Size([batch_size, emb_dim])

		# add the encoded adduct
		if self.add_num == 1:
			x = torch.cat((x, torch.unsqueeze(env, 1)), 1)
		elif self.add_num > 1:
			x = torch.cat((x, env), 1)

		# decoder
		x = self.decoder(x)
		return x



class MolNet_RT(nn.Module): 
	def __init__(self, config): 
		super(MolNet_RT, self).__init__()
		
		self.encoder = Encoder(in_dim=int(config['in_dim']), 
									layers=config['encode_layers'],
									emb_dim=int(config['emb_dim']), 
									k=int(config['k']))
		self.decoder = MSDecoder(in_dim=int(config['emb_dim']), 
								layers=config['decode_layers'], 
								out_dim=1, 
								dropout=config['dropout'])
		
		for m in self.modules(): 
			if isinstance(m, nn.Conv1d):
				nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
			elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				m.weight.data.normal_(mean=0.0, std=1.0)
				if m.bias is not None:
					m.bias.data.zero_()

	def forward(self, x: torch.Tensor, 
						idx_base: torch.Tensor) -> torch.Tensor: 
		'''
		Input: 
			x:      point set, torch.Size([batch_size, 14, atom_num])
			env:    experimental condiction
			idx_base:   idx for local knn
		'''
		x = self.encoder(x, idx_base) # torch.Size([batch_size, emb_dim])

		# decoder
		x = self.decoder(x)
		return x
