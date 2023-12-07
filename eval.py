import os
import argparse
import numpy as np
from tqdm import tqdm
import yaml
from pyteomics import mgf

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from molmspack.molnet import MolNet_MS
from molmspack.dataset import MolMS_Dataset
from molmspack.data_utils import filter_spec, mgf2pkl

global batch_size
batch_size = 1



def get_lr(optimizer):
	for param_group in optimizer.param_groups:
		return param_group['lr']

def reg_criterion(outputs, targets): 
	# cosine similarity
	t = nn.CosineSimilarity(dim=1)
	spec_cosi = torch.mean(1 - t(outputs, targets))
	return spec_cosi

def eval_step(model, device, loader, batch_size, num_points): 
	model.eval()
	accuracy = 0
	with tqdm(total=len(loader)) as bar:
		for step, batch in enumerate(loader):
			_, x, y, env = batch
			x = x.to(device=device, dtype=torch.float)
			x = x.permute(0, 2, 1)
			y = y.to(device=device, dtype=torch.float)
			env = env.to(device=device, dtype=torch.float)
			idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

			with torch.no_grad(): 
				pred = model(x, env, idx_base) 
				pred = pred / torch.max(pred) # normalize the output
				
			bar.set_description('Eval')
			bar.update(1)
	
			# recover sqrt spectra to original spectra
			y = torch.pow(y, 2)
			pred = torch.pow(pred, 2)
			# post process
			pred = pred.detach().cpu().apply_(lambda x: x if x > 0.01 else 0).to(device)

			accuracy += F.cosine_similarity(pred, y, dim=1).mean().item()
	return accuracy / (step + 1)

def init_random_seed(seed):
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	return



if __name__ == "__main__": 
	parser = argparse.ArgumentParser(description='Molecular Mass Spectra Prediction (Train)')
	parser.add_argument('--test_data', type=str, default='./data/qtof_etkdg_test.pkl',
						help='path to test data (pkl)')
	parser.add_argument('--save_pkl', action='store_true', 
						help='save converted pkl file')
	parser.add_argument('--precursor_type', type=str, default='All', choices=['All', '[M+H]+', '[M-H]-'], 
                        help='Precursor type')
	parser.add_argument('--model_config_path', type=str, default='./config/molnet.yml',
						help='path to model and training configuration')
	parser.add_argument('--data_config_path', type=str, default='./config/preprocess_etkdg.yml',
						help='path to configuration')
	parser.add_argument('--resume_path', type=str, required=True, 
						help='Path to pretrained model')
	
	parser.add_argument('--seed', type=int, default=42,
						help='Seed for random functions')
	parser.add_argument('--device', type=int, default=0,
						help='Which gpu to use if any')
	parser.add_argument('--no_cuda', action='store_true', 
						help='Enables CUDA training')
	args = parser.parse_args()

	init_random_seed(args.seed)
	with open(args.model_config_path, 'r') as f: 
		config = yaml.load(f, Loader=yaml.FullLoader)
	print('Load the model & training configuration from {}'.format(args.model_config_path))

	# 1. Data
	# convert precursor type to encoded precursor type for filtering
	with open(args.data_config_path, 'r') as f: 
		data_config = yaml.load(f, Loader=yaml.FullLoader)
		precursor_encoder = {}
		for k, v in data_config['encoding']['precursor_type'].items(): 
			precursor_encoder[k] = ','.join([str(int(i)) for i in v])
		precursor_encoder['All'] = False
	
	test_format = args.test_data.split('.')[-1]
	if test_format == 'mgf': # convert mgf file into pkl 
		origin_spectra = mgf.read(args.test_data)
		
		print('Filter spectra...')
		filter_spectra, _ = filter_spec(origin_spectra, 
										data_config['all'], 
										type2charge=data_config['encoding']['type2charge'])
		pkl_dict = mgf2pkl(filter_spectra, data_config['encoding'])

	elif test_format == 'pkl':
		with open(args.test_data, 'rb') as file: 
			pkl_dict = pickle.load(file)

	else:
		raise ValueError('Unsupported format: {}'.format(test_format))

	# same the pkl, so do not need to convert it again next time
	pkl_path = args.test_data.replace('.'+test_format, '.pkl')
	if args.save_pkl: 
		if not os.path.exists(pkl_path): 
			raise OSError('The pkl file exists. Do not need to save it again. ')

		with open(pkl_path, 'wb') as f: 
			pickle.dump(pkl_dict, f)
			print('Save converted pkl file to {}'.format(pkl_path))

	valid_set = MolMS_Dataset(pkl_dict, precursor_encoder[args.precursor_type], mode='data')
	valid_loader = DataLoader(
					valid_set,
					batch_size=batch_size, 
					shuffle=True, 
					num_workers=config['train']['num_workers'], 
					drop_last=True)

	# 2. Model
	device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() and not args.no_cuda else torch.device("cpu")
	print(f'Device: {device}')

	model = MolNet_MS(config['model']).to(device)
	num_params = sum(p.numel() for p in model.parameters())
	print(f'{str(model)} #Params: {num_params}')

	# 3. Evalution
	print("Load the checkpoints...")
	model.load_state_dict(torch.load(args.resume_path, map_location=device)['model_state_dict'])

	valid_acc = eval_step(model, device, valid_loader, 
							batch_size=batch_size, num_points=config['model']['max_atom_num'])
	print("Validation: Acc: {}".format(valid_acc))

	
