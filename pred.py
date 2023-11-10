import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml
import pickle
from pyteomics import mgf

import torch
from torch.utils.data import DataLoader

from rdkit import Chem
# ignore the warning
from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import Descriptors

from molnetpack.molnet import MolNet_MS
from molnetpack.dataset import Mol_Dataset
from molnetpack.data_utils import ce2nce, parse_collision_energy, conformation_array, precursor_calculator, generate_ms

global batch_size
batch_size = 1



def spec_convert(spec, resolution):
	x = []
	y = []
	for i, j in enumerate(spec):
		if j != 0: 
			x.append(str(i*resolution))
			y.append(str(j))
	return {'m/z': ','.join(x), 'intensity': ','.join(y)}

def csv2pkl_dict(csv_path, encoder, save_pkl): 
	pkl_path = csv_path.replace('.csv', '.pkl')
	df = pd.read_csv(csv_path)
	data = []
	for idx, row in df.iterrows(): 
		# mol array
		good_conf, xyz_arr, atom_type = conformation_array(smiles=row['SMILES'], 
															conf_type=encoder['conf_type']) 
		# There are some limitations of conformation generation methods. 
		# e.g. https://github.com/rdkit/rdkit/issues/5145
		# Let's skip the unsolvable molecules. 
		if not good_conf: # filter 1
			print('Can not generate correct conformation: {}'.format(row['SMILES']))
			continue
		if xyz_arr.shape[0] > encoder['max_atom_num']: # filter 2
			print('Atomic number ({}) exceed the limitation ({})'.format(encoder['max_atom_num'], xyz_arr.shape[0]))
			continue
		# filter 3
		rare_atom_flag = False
		rare_atom = ''
		for atom in list(set(atom_type)):
			if atom not in encoder['atom_type'].keys(): 
				rare_atom_flag = True
				rare_atom = atom
				break
		if rare_atom_flag:
			print('Unsupported atom type: {}'.format(rare_atom))
			continue

		atom_type_one_hot = np.array([encoder['atom_type'][atom] for atom in atom_type])
		assert xyz_arr.shape[0] == atom_type_one_hot.shape[0]

		mol_arr = np.concatenate([xyz_arr, atom_type_one_hot], axis=1)
		mol_arr = np.pad(mol_arr, ((0, encoder['max_atom_num']-xyz_arr.shape[0]), (0, 0)), constant_values=0)
		
		# env array
		precursor_mz = precursor_calculator(row['Precursor_Type'], mass=Descriptors.MolWt(Chem.MolFromSmiles(row['SMILES'])))
		nce = ce2nce(ce=row['Collision_Energy'], 
						precursor_mz=precursor_mz, 
						charge=row['Charge'])
		if row['Precursor_Type'] not in encoder['precursor_type'].keys(): # filter 4
			print('Unsupported precusor type: {}'.format(row['Precursor_Type']))
			continue
		precursor_type_one_hot = encoder['precursor_type'][row['Precursor_Type']]
		env_arr = np.array([nce] + precursor_type_one_hot)

		data.append({'title': row['ID'], 'mol': mol_arr, 'env': env_arr})

	if save_pkl: 
		with open(csv_path.replace('.csv', '_no_spec.pkl'), 'wb') as f: 
			pickle.dump(data, f)
			print('Save converted pkl file to {}'.format(csv_path.replace('.csv', '.mgf')))
	return data

def mgf2pkl_dict(mgf_path, encoder, save_pkl, with_spec=False): 
	pkl_path = mgf_path.replace('.mgf', '.pkl')
	supp = mgf.read(mgf_path)
	data = []
	for idx, spec in enumerate(tqdm(supp)): 
		# mol array
		good_conf, xyz_arr, atom_type = conformation_array(smiles=spec['params']['smiles'], 
															conf_type=encoder['conf_type']) 
		if not good_conf: # filter 1
			print('Can not generate correct conformation: {}'.format(spec['params']['smiles']))
			continue 
		if xyz_arr.shape[0] > encoder['max_atom_num']: # filter 2
			print('Atomic number ({}) exceed the limitation ({})'.format(encoder['max_atom_num'], xyz_arr.shape[0]))
			continue
		# filter 3
		rare_atom_flag = False
		rare_atom = ''
		for atom in list(set(atom_type)):
			if atom not in encoder['atom_type'].keys(): 
				rare_atom_flag = True
				rare_atom = atom
				break
		if rare_atom_flag:
			print('Unsupported atom type: {}'.format(rare_atom))
			continue
		
		atom_type_one_hot = np.array([encoder['atom_type'][atom] for atom in atom_type])
		assert xyz_arr.shape[0] == atom_type_one_hot.shape[0]

		mol_arr = np.concatenate([xyz_arr, atom_type_one_hot], axis=1)
		mol_arr = np.pad(mol_arr, ((0, encoder['max_atom_num']-xyz_arr.shape[0]), (0, 0)), constant_values=0)

		# env array
		if 'charge' not in spec['params'].keys(): 
			print('Empty charge. We will assume it as charge 1.')
			charge = 1
		elif isinstance(spec['params']['charge'], list): # convert pyteomics.auxiliary.structures.ChargeList to int
			charge = int(spec['params']['charge'][0])
		precursor_mz = precursor_calculator(spec['params']['precursor_type'], 
											mass=Descriptors.MolWt(Chem.MolFromSmiles(spec['params']['smiles'])))
		ce, nce = parse_collision_energy(ce_str=spec['params']['collision_energy'], 
									precursor_mz=precursor_mz, 
									charge=charge)
		if ce == None and nce == None:
			print('Unsupported collision energy: {}'.format(spec['params']['collision_energy']))
			continue
		if spec['params']['precursor_type'] not in encoder['precursor_type'].keys(): # filter 4
			print('Unsupported precusor type: {}'.format(spec['params']['precursor_type']))
			continue
		precursor_type_one_hot = encoder['precursor_type'][spec['params']['precursor_type']]
		env_arr = np.array([nce] + precursor_type_one_hot)

		if with_spec:
			spec_arr = generate_ms(x=spec['m/z array'], 
								y=spec['intensity array'], 
								precursor_mz=spec['params']['precursor_mz'], 
								resolution=encoder['resolution'], 
								max_mz=encoder['max_mz'], 
								charge=charge)
			data.append({'title': spec['params']['title'], 'mol': mol_arr, 'spec': spec_arr, 'env': env_arr})
		else:
			data.append({'title': spec['params']['title'], 'mol': mol_arr, 'env': env_arr})
	
	if save_pkl: 
		if with_spec:
			pkl_path = mgf_path.replace('.mgf', '.pkl')
		else:
			pkl_path = mgf_path.replace('.mgf', '_no_spec.pkl')
		with open(pkl_path, 'wb') as f: 
			pickle.dump(data, f)
			print('Save converted pkl file to {}'.format(pkl_path))
	return data

def pred_step(model, device, loader, batch_size, num_points): 
	model.eval()
	id_list = []
	pred_list = []
	with tqdm(total=len(loader)) as bar:
		for step, batch in enumerate(loader):
			ids, x, env = batch
			x = x.to(device=device, dtype=torch.float)
			x = x.permute(0, 2, 1)
			env = env.to(device=device, dtype=torch.float)
			idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

			with torch.no_grad(): 
				pred = model(x, env, idx_base) 
				pred = pred / torch.max(pred) # normalize the output
				
			bar.set_description('Eval')
			bar.update(1)
	
			# recover sqrt spectra to original spectra
			pred = torch.pow(pred, 2)
			# post process
			pred = pred.detach().cpu().apply_(lambda x: x if x > 0.01 else 0)

			id_list += ids
			pred_list.append(pred)

	pred_list = torch.cat(pred_list, dim = 0)
	return id_list, pred_list

def init_random_seed(seed):
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	return



if __name__ == "__main__": 
	parser = argparse.ArgumentParser(description='Mass Spectrum Prediction (Train)')
	parser.add_argument('--test_data', type=str, default='./data/agilent_qtof_etkdg_test.pkl',
						help='path to test data (pkl)')
	parser.add_argument('--save_pkl', action='store_true', 
						help='Save converted pkl file')
	parser.add_argument('--with_spec', action='store_true', 
						help='Save spectra in converted pkl file')
	parser.add_argument('--precursor_type', type=str, default='All', choices=['All', '[M+H]+', '[M-H]-'], 
                        help='Precursor type')
	parser.add_argument('--model_config_path', type=str, default='./config/molnet.yml',
						help='path to model and training configuration')
	parser.add_argument('--data_config_path', type=str, default='./config/preprocess_etkdg.yml',
						help='path to data configuration')
	parser.add_argument('--resume_path', type=str, default='', 
						help='Path to pretrained model')
	parser.add_argument('--result_path', type=str, default='', 
						help='Path to saving results')
	
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
	test_format = args.test_data.split('.')[-1]
	if test_format == 'csv': 
		# convert csv file into pkl 
		with open(args.data_config_path, 'r') as f: 
			data_config = yaml.load(f, Loader=yaml.FullLoader)
		print('Load the data configuration from {}'.format(args.data_config_path))
		pkl_dict = csv2pkl_dict(args.test_data, data_config['encoding'], args.save_pkl)
	elif test_format == 'pkl': 
		with open(args.test_data, 'rb') as file: 
			pkl_dict = pickle.load(file)
	elif test_format == 'mgf':
		# convert mgf file into pkl 
		with open(args.data_config_path, 'r') as f: 
			data_config = yaml.load(f, Loader=yaml.FullLoader)
		print('Load the data configuration from {}'.format(args.data_config_path))
		pkl_dict = mgf2pkl_dict(args.test_data, data_config['encoding'], args.save_pkl, args.with_spec) 
	else:
		raise ValueError('Unsupported format: {}'.format(test_format))

	# convert precursor type to encoded precursor type for filtering
	with open(args.data_config_path, 'r') as f: 
		tmp = yaml.load(f, Loader=yaml.FullLoader)
		precursor_encoder = {}
		for k, v in tmp['encoding']['precursor_type'].items(): 
			precursor_encoder[k] = ','.join([str(int(i)) for i in v])
		precursor_encoder['All'] = False
		del tmp

	valid_set = Mol_Dataset(pkl_dict, precursor_encoder[args.precursor_type])
	valid_loader = DataLoader(
					valid_set,
					batch_size=batch_size, 
					shuffle=False, 
					num_workers=config['train']['num_workers'], 
					drop_last=True)

	# 2. Model
	device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() and not args.no_cuda else torch.device("cpu")
	print(f'Device: {device}')

	model = MolNet_MS(config['model']).to(device)
	num_params = sum(p.numel() for p in model.parameters())
	print(f'{str(model)} #Params: {num_params}')

	# 3. Evaluation
	print("Load the checkpoints...")
	model.load_state_dict(torch.load(args.resume_path, map_location=device)['model_state_dict'])
	best_valid_acc = torch.load(args.resume_path, map_location=device)['best_val_acc']

	id_list, pred_list = pred_step(model, device, valid_loader, 
									batch_size=batch_size, num_points=config['model']['max_atom_num'])
	pred_list = [spec_convert(spec, config['model']['resolution']) for spec in pred_list.tolist()]
	pred_mz = [pred['m/z'] for pred in pred_list]
	pred_intensity = [pred['intensity'] for pred in pred_list]

	# 4. Output the results
	result_dir = "".join(args.result_path.split('/')[:-1])
	os.makedirs(result_dir, exist_ok = True)

	res_df = pd.DataFrame({'ID': id_list, 'Pred M/Z': pred_mz, 'Pred Intensity': pred_intensity})
	if args.result_path[-3:] == 'csv': # save results to csv file
		res_df.to_csv(args.result_path, sep='\t')
	elif args.result_path[-3:] == 'mgf': # save results to mgf file
		spectra = []
		prefix = 'pred'
		for idx, row in res_df.iterrows(): 
			spectrum = {
				'params': {
					'title': row['ID'], 
					'mslevel': '2', 
					'organism': '3DMolMS_v1.1', 
					'spectrumid': prefix+'_'+str(idx), 
				},
				'm/z array': np.array([float(i) for i in row['Pred M/Z'].split(',')]),
				'intensity array': np.array([float(i)*1000 for i in row['Pred Intensity'].split(',')])
			} 
			spectra.append(spectrum)
		mgf.write(spectra, args.result_path, file_mode="w", write_charges=False)
	else:
		raise Exception("Not implemented output format. Please choose `.csv` or `.mgf`.")

	print('Save the test results to {}'.format(args.result_path))