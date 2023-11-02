import os
import argparse
import yaml
from tqdm import tqdm
import numpy as np
import pickle
from pyteomics import mgf

from rdkit import Chem
# ignore the warning
from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker

from data_utils import sdf2mgf, filter_spec, generate_ms, parse_collision_energy, conformation_array



def spec2arr(spectra, encoder): 
	'''data format
	[
		{'title': <str>, 'mol': <numpy array>, 'env': <numpy array>, 'spec': <numpy array>}, 
		{'title': <str>, 'mol': <numpy array>, 'env': <numpy array>, 'spec': <numpy array>}, 
		....
	]
	'''
	data = []
	for idx, spectrum in enumerate(tqdm(spectra)): 
		# mol array
		good_conf, xyz_arr, atom_type = conformation_array(smiles=spectrum['params']['smiles'], 
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
		
		# spec array
		good_spec, spec_arr = generate_ms(x=spectrum['m/z array'], 
								y=spectrum['intensity array'], 
								precursor_mz=float(spectrum['params']['precursor_mz']), 
								resolution=encoder['resolution'], 
								max_mz=encoder['max_mz'], 
								charge=int(encoder['type2charge'][spectrum['params']['precursor_type']]))
		if not good_spec: # after binning, some spectra do not have enough peaks' number
			continue

		# env array
		ce, nce = parse_collision_energy(ce_str=spectrum['params']['collision_energy'], 
								precursor_mz=float(spectrum['params']['precursor_mz']), 
								charge=int(encoder['type2charge'][spectrum['params']['precursor_type']]))
		if ce == None and nce == None:
			continue
		precursor_type_one_hot = encoder['precursor_type'][spectrum['params']['precursor_type']]
		env_arr = np.array([nce] + precursor_type_one_hot)

		data.append({'title': spectrum['params']['title'], 'mol': mol_arr, 'spec': spec_arr, 'env': env_arr})
	return data



if __name__ == "__main__": 
	parser = argparse.ArgumentParser(description='Preprocess the Data')
	parser.add_argument('--raw_dir', type=str, default='./data/origin/',
						help='path to raw data')
	parser.add_argument('--pkl_dir', type=str, default='./data/',
						help='path to pkl data')
	parser.add_argument('--dataset', type=str, nargs='+', required=True, choices=['agilent', 'nist', 'mona', 'waters'], 
						help='dataset name')
	parser.add_argument('--instrument_type', type=str, nargs='+', required=True, choices=['qtof', 'hcd'], 
						help='dataset name')
	parser.add_argument('--train_ratio', type=float, default=0.9,
						help='Ratio for train set')
	parser.add_argument('--maxmin_pick', action='store_true', 
						help='If using MaxMin algorithm to pick training molecules')
	parser.add_argument('--data_config_path', type=str, default='./config/preprocess_etkdgv3.yml',
						help='path to configuration')
	args = parser.parse_args()

	assert args.train_ratio < 1. 

	if 'agilent' in args.dataset:
		assert os.path.exists(os.path.join(args.raw_dir, 'Agilent_Combined.sdf'))
		assert os.path.exists(os.path.join(args.raw_dir, 'Agilent_Metlin.sdf'))
		assert os.path.exists(os.path.join(args.raw_dir, 'hr_msms_nist.SDF'))
	if 'nist' in args.dataset:
		assert os.path.exists(os.path.join(args.raw_dir, 'hr_msms_nist.SDF'))
	if 'mona' in args.dataset:
		# assert os.path.exists(os.path.join(args.raw_dir, 'MoNA-export-All_LC-MS-MS_QTOF.sdf'))
		assert os.path.exists(os.path.join(args.raw_dir, 'MoNA-export-All_LC-MS-MS_Agilent_QTOF.sdf'))
	if 'waters' in args.dataset:
		assert os.path.exists(os.path.join(args.raw_dir, 'waters_qtof.mgf'))
	
	# load the configurations
	with open(args.data_config_path, 'r') as f: 
		config = yaml.load(f, Loader=yaml.FullLoader)

	# 1. convert original format to mgf 
	print('\n>>> Step 1: convert original format to mgf;')
	origin_spectra = {}
	if 'agilent' in args.dataset: 
		spectra1 = sdf2mgf(path=os.path.join(args.raw_dir, 'Agilent_Combined.sdf'), prefix='agilent_combine')
		spectra2 = sdf2mgf(path=os.path.join(args.raw_dir, 'Agilent_Metlin.sdf'), prefix='agilent_metlin')
		origin_spectra['agilent'] = spectra1 + spectra2
	if 'nist' in args.dataset: 
		origin_spectra['nist'] = sdf2mgf(path=os.path.join(args.raw_dir, 'hr_msms_nist.SDF'), prefix='nist20')
	if 'mona' in args.dataset: 
		origin_spectra['mona'] = sdf2mgf(path=os.path.join(args.raw_dir, 'MoNA-export-All_LC-MS-MS_QTOF.sdf'), prefix='mona_qtof')
	if 'waters' in args.dataset:
		origin_spectra['waters'] = mgf.read(os.path.join(args.raw_dir, 'waters_qtof.mgf'))
		print('Load {} data from {}'.format(len(origin_spectra['waters']), os.path.join(args.raw_dir, 'waters_qtof.mgf')))

	# 2. filter the spectra
	# 3. split spectra into training and test set according to smiles
	# Note that there is not overlapped molecules between training set and tes set. 
	# 4. generate 3d conformattions & encoding data into arrays
	print('\n>>> Step 2: filter out spectra by certain rules; \n\
\tStep 3: split SMILES into training set and test set; \n\
\tStep 4: encode all the data into pkl format;')
	for ins in args.instrument_type: 
		spectra = []
		smiles_list = []
		for ds in args.dataset:
			config_name = ds + '_' + ins
			if config_name not in config.keys(): 
				continue
			print('({}) Filter {} spectra...'.format(ins, config_name))
			filter_spectra, filter_smiles_list = filter_spec(origin_spectra[ds], 
																config[config_name], 
																type2charge=config['encoding']['type2charge'])
			# mgf.write(filter_spectra, './data/mgf_debug/{}.mgf'.format(config_name), file_mode="w") # save mgf debug
			filter_smiles_list = list(set(filter_smiles_list))
			spectra += filter_spectra
			smiles_list += filter_smiles_list
			del filter_spectra, filter_smiles_list
		smiles_list = list(set(smiles_list))
		
		if args.maxmin_pick: 
			print('({}) Split training and test set by MaxMin algorithm...'.format(ins))
			# maxmin algorithm picking training smiles (Since the training set so far is not large enough, 
			# we'd like to make a full utilizing of our datasets. It worth noting that MaxMin picker is 
			# for application, and the experiments in our paper spliting the molecules randomly according 
			# to their smiles. )
			fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3)
			fp_list = [fpgen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles_list]
			picker = MaxMinPicker()
			train_indices = picker.LazyBitVectorPick(fp_list, len(fp_list), int(len(fp_list)*args.train_ratio), seed=42)
			train_indices = list(train_indices)
		else:
			print('({}) Split training and test set by randomly choose...'.format(ins))
			train_indices = np.random.choice(len(smiles_list), int(len(smiles_list)*args.train_ratio), replace=False)

		train_smiles_list = [smiles_list[i] for i in range(len(smiles_list)) if i in train_indices]
		test_smiles_list = [s for s in smiles_list if s not in train_smiles_list]
		print('({}) Get {} training compounds and {} test compounds'.format(ins, len(train_smiles_list), len(test_smiles_list)))

		train_spectra = []
		test_spectra = []
		for _, spectrum in enumerate(tqdm(spectra)): 
			smiles = spectrum['params']['smiles']
			if smiles in train_smiles_list: 
				train_spectra.append(spectrum)
			else: 
				test_spectra.append(spectrum)
		del spectra, smiles_list, test_smiles_list
		print('({}) Get {} training spectra and {} test spectra'.format(ins, len(train_spectra), len(test_spectra)))

		print('({}) Convert spectra and molecules data into arrays...'.format(ins))
		# test
		test_data = spec2arr(test_spectra, config['encoding'])
		out_path = os.path.join(args.pkl_dir, '{}_{}_test.pkl'.format(ins, config['encoding']['conf_type']))
		with open(out_path, 'wb') as f: 
			pickle.dump(test_data, f)
			print('Save {}'.format(out_path))
		# train
		train_data = spec2arr(train_spectra, config['encoding'])
		out_path = os.path.join(args.pkl_dir, '{}_{}_train.pkl'.format(ins, config['encoding']['conf_type']))
		with open(out_path, 'wb') as f: 
			pickle.dump(train_data, f)
			print('Save {}'.format(out_path))

	print('Done!')

