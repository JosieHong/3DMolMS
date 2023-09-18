import os
import argparse
import yaml
from tqdm import tqdm
import numpy as np
import pickle

from rdkit import Chem
# ignore the warning
from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*')

from data_utils import conformation_array



def filter_mol(suppl, config): 
	clean_suppl = []
	smiles_list = []
	for idx, mol in enumerate(tqdm(suppl)): 
		if mol == None: continue

		# Filter by atom number and atom type 
		if len(mol.GetAtoms()) > config['max_atom_num'] or len(mol.GetAtoms()) < config['min_atom_num']: continue
		is_compound_countain_rare_atom = False 
		for atom in mol.GetAtoms(): 
			if atom.GetSymbol() not in config['atom_type']:
				is_compound_countain_rare_atom = True
				break
		if is_compound_countain_rare_atom: continue

		clean_suppl.append(mol)
		smiles_list.append(Chem.MolToSmiles(mol))
	return clean_suppl, smiles_list

def random_split(suppl, smiles_list, test_ratio=0.1):
	test_smiles = np.random.choice(smiles_list, int(len(smiles_list)*test_ratio), replace=False)

	train_mol = []
	test_mol = []
	for mol in suppl: 
		smiles = Chem.MolToSmiles(mol)
		if smiles in test_smiles:
			test_mol.append(mol)
		else:
			train_mol.append(mol)
	return test_mol, train_mol

def sdf2arr(suppl, encoder):
	'''data format
	[
		{'title': <str>, 'mol': <numpy array>, 'rt': <numpy array>}, 
		{'title': <str>, 'mol': <numpy array>, 'rt': <numpy array>}, 
		....
	]
	'''
	data = []
	for idx, mol in enumerate(tqdm(suppl)): 
		# mol array
		good_conf, xyz_arr, atom_type = conformation_array(smiles=Chem.MolToSmiles(mol), 
															conf_type=encoder['conf_type']) 
		# There are some limitations of conformation generation methods. 
		# e.g. https://github.com/rdkit/rdkit/issues/5145
		# Let's skip the unsolvable molecules. 
		if not good_conf: 
			continue
		atom_type_one_hot = np.array([encoder['atom_type'][atom] for atom in atom_type])
		assert xyz_arr.shape[0] == atom_type_one_hot.shape[0]

		mol_arr = np.concatenate([xyz_arr, atom_type_one_hot], axis=1)
		mol_arr = np.pad(mol_arr, ((0, encoder['max_atom_num']-xyz_arr.shape[0]), (0, 0)), constant_values=0)
		
		data.append({'title': mol.GetProp('PUBCHEM_COMPOUND_CID'), 'mol': mol_arr, 'rt': np.array([mol.GetProp('RETENTION_TIME')]).astype(np.float64)})
	return data
	


if __name__ == "__main__": 
	parser = argparse.ArgumentParser(description='Preprocess the Data')
	parser.add_argument('--raw_dir', type=str, default='./data/origin/',
						help='path to raw data')
	parser.add_argument('--pkl_dir', type=str, default='./data/',
						help='path to pkl data')
	parser.add_argument('--dataset', type=str, nargs='+', required=True, choices=['metlin'],
						help='dataset name')
	parser.add_argument('--data_config_path', type=str, default='./config/preprocess_etkdg.yml',
						help='path to configuration')
	args = parser.parse_args()
	
	if 'metlin' in args.dataset: 
		assert os.path.exists(os.path.join(args.raw_dir, 'SMRT_dataset.sdf'))
	
	# load the configurations
	with open(args.data_config_path, 'r') as f: 
		config = yaml.load(f, Loader=yaml.FullLoader)
	
	if 'metlin' in args.dataset: 
		# 1. load data
		print('\n>>> Step 1: load the dataset;')
		suppl = Chem.SDMolSupplier(os.path.join(args.raw_dir, 'SMRT_dataset.sdf'))
		suppl = [m for m in suppl if m != None and m.HasProp('PUBCHEM_COMPOUND_CID') and m.HasProp('RETENTION_TIME')]
		print('Load {} data from METLIN Dataset...'.format(len(suppl)))

		# 2. randomly split spectra into training and test set according to [smiles]
		print('\n>>> Step 2: filter out molecules by certain rules; randomly split SMILES into training set and test set;')
		suppl, smiles_list = filter_mol(suppl, config['metlin_rt'])
		test_mol, train_mol = random_split(suppl, 
									 		list(set(smiles_list)), 
											test_ratio=0.1)
		print('Get {} test data and {} training data'.format(len(test_mol), len(train_mol)))

		# 3. encoding data into arrays
		print('\n>>> Step 3: encode all the data into pkl format;')
		test_data = sdf2arr(test_mol, config['encoding'])
		out_path = os.path.join(args.pkl_dir, 'metlin_{}_test.pkl'.format(config['encoding']['conf_type']))
		with open(out_path, 'wb') as f: 
			pickle.dump(test_data, f)
			print('Save {}'.format(out_path))
			
		train_data = sdf2arr(train_mol, config['encoding'])
		out_path = os.path.join(args.pkl_dir, 'metlin_{}_train.pkl'.format(config['encoding']['conf_type']))
		with open(out_path, 'wb') as f: 
			pickle.dump(train_data, f)
			print('Save {}'.format(out_path))

	# implement other retention time dataset here 
	# elif ... in args.dataset: 
	# 	...