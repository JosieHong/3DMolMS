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

from data_utils import generate_ms, parse_collision_energy, conformation_array



def sdf2mgf(path, prefix): 
	'''mgf format
	[{
		'params': {
			'title': prefix_<index>, 
			'precursor_type': <precursor_type (e.g. [M+NH4]+ and [M+H]+)>, 
			'precursor_mz': <precursor m/z>,
			'molmass': <isotopic mass>, 
			'ms_level': <ms_level>, 
			'ionmode': <POSITIVE|NEGATIVE>, 
			'source_instrument': <source_instrument>,
			'instrument_type': <instrument_type>, 
			'collision_energy': <collision energe>, 
			'smiles': <smiles>, 
			'inchi_key': <inchi_key>, 
		},
		'm/z array': mz_array,
		'intensity array': intensity_array
	}, ...]
	'''
	supp = Chem.SDMolSupplier(path)
	print('Read {} data from {}'.format(len(supp), path))

	spectra = []
	for idx, mol in enumerate(tqdm(supp)): 
		if mol == None or not mol.HasProp('MASS SPECTRAL PEAKS'): 
			continue
		
		mz_array = []
		intensity_array = []
		raw_ms = mol.GetProp('MASS SPECTRAL PEAKS').split('\n')
		for line in raw_ms:
			mz_array.append(float(line.split()[0]))
			intensity_array.append(float(line.split()[1]))
		mz_array = np.array(mz_array)
		intensity_array = np.array(intensity_array)

		inchi_key = 'Unknown' if not mol.HasProp('INCHIKEY') else mol.GetProp('INCHIKEY')
		instrument = 'Unknown' if not mol.HasProp('INSTRUMENT') else mol.GetProp('INSTRUMENT')
		spectrum = {
			'params': {
				'title': prefix+'_'+str(idx), 
				'precursor_type': mol.GetProp('PRECURSOR TYPE'),
				'precursor_mz': mol.GetProp('PRECURSOR M/Z'),
				'molmass': mol.GetProp('EXACT MASS'),
				'ms_level': mol.GetProp('SPECTRUM TYPE'), 
				'ionmode': mol.GetProp('ION MODE'), 
				'source_instrument': instrument,
				'instrument_type': mol.GetProp('INSTRUMENT TYPE'), 
				'collision_energy': mol.GetProp('COLLISION ENERGY'), 
				'smiles': Chem.MolToSmiles(mol, isomericSmiles=True), 
				'inchi_key': inchi_key, 
			},
			'm/z array': mz_array,
			'intensity array': intensity_array
		} 
		spectra.append(spectrum)
	return spectra

def filter_spec(spectra, config, type2charge): 
	clean_spectra = []
	smiles_list = []
	for idx, spectrum in enumerate(tqdm(spectra)): 
		smiles = spectrum['params']['smiles'] 
		mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
		if mol == None: continue

		# Filter by collision energy
		if spectrum['params']['collision_energy'].startswith('ramp'): # we can not process this type collision energy now
			continue

		# Filter by instrument type
		instrument_type = spectrum['params']['instrument_type']
		if instrument_type != config['intrument_type']: continue

		# Filter by instrument
		instrument = spectrum['params']['source_instrument']
		if instrument not in config['instrument']: continue

		# Filter by mslevel
		mslevel = spectrum['params']['ms_level']
		if mslevel != config['ms_level']: continue

		# Filter by atom number and atom type 
		if len(mol.GetAtoms()) > config['max_atom_num'] or len(mol.GetAtoms()) < config['min_atom_num']: continue
		is_compound_countain_rare_atom = False 
		for atom in mol.GetAtoms(): 
			if atom.GetSymbol() not in config['atom_type']:
				is_compound_countain_rare_atom = True
				break
		if is_compound_countain_rare_atom: continue

		# Filter by precursor type
		precursor_type = spectrum['params']['precursor_type']
		if precursor_type not in config['precursor_type']: continue

		# Filt by peak number
		if len(spectrum['m/z array']) < config['min_peak_num']: continue

		# Filter by max m/z
		if np.max(spectrum['m/z array']) < config['min_mz'] or np.max(spectrum['m/z array']) > config['max_mz']: continue

		# add charge
		spectrum['params']['charge'] = type2charge[precursor_type]
		clean_spectra.append(spectrum)
		smiles_list.append(smiles)
	return clean_spectra, smiles_list

def random_split(spectra, smiles_list, test_ratio=0.1):
	test_smiles = np.random.choice(smiles_list, int(len(smiles_list)*test_ratio), replace=False)

	train_spectra = []
	test_spectra = []
	for spectrum in spectra:
		smiles = spectrum['params']['smiles'] 
		if smiles in test_smiles:
			test_spectra.append(spectrum)
		else:
			train_spectra.append(spectrum)
	return test_spectra, train_spectra

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
		spec_arr = generate_ms(x=spectrum['m/z array'], 
								y=spectrum['intensity array'], 
								precursor_mz=float(spectrum['params']['precursor_mz']), 
								resolution=encoder['resolution'], 
								max_mz=encoder['max_mz'], 
								charge=int(encoder['type2charge'][spectrum['params']['precursor_type']]))
		
		# env array
		ce, nce = parse_collision_energy(ce_str=spectrum['params']['collision_energy'], 
								precursor_mz=float(spectrum['params']['precursor_mz']), 
								charge=int(encoder['type2charge'][spectrum['params']['precursor_type']]))
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
	parser.add_argument('--dataset', type=str, nargs='+', required=True, choices=['qtof', 'hcd'],
						help='dataset name')
	parser.add_argument('--data_config_path', type=str, default='./config/preprocess_etkdg.yml',
						help='path to configuration')
	args = parser.parse_args()

	mgf_dir = './data/mgf/' # for debug only
	if 'qtof' in args.dataset:
		assert os.path.exists(os.path.join(args.raw_dir, 'Agilent_Combined.sdf'))
		assert os.path.exists(os.path.join(args.raw_dir, 'Agilent_Metlin.sdf'))
		assert os.path.exists(os.path.join(args.raw_dir, 'hr_msms_nist.SDF'))
	if 'hcd' in args.dataset:
		assert os.path.exists(os.path.join(args.raw_dir, 'hr_msms_nist.SDF'))
	
	# load the configurations
	with open(args.data_config_path, 'r') as f: 
		config = yaml.load(f, Loader=yaml.FullLoader)

	# 1. convert original format to mgf 
	print('\n>>> Step 1: convert original format to mgf;')
	if 'qtof' in args.dataset: 
		spectra1 = sdf2mgf(path=os.path.join(args.raw_dir, 'Agilent_Combined.sdf'), prefix='agilent_combine')
		spectra2 = sdf2mgf(path=os.path.join(args.raw_dir, 'Agilent_Metlin.sdf'), prefix='agilent_metlin')
		agilent_spectra = spectra1 + spectra2

		nist_spectra = sdf2mgf(path=os.path.join(args.raw_dir, 'hr_msms_nist.SDF'), prefix='nist20')
		
	elif 'hcd' in args.dataset: # if hcd and qtof are both in args.dataset, we do not need to load nist20 twice
		nist_spectra = sdf2mgf(path=os.path.join(args.raw_dir, 'hr_msms_nist.SDF'), prefix='nist20')
	
	# 2. filter the spectra
	# 3. randomly split spectra into training and test set according to [smiles]
	# Note that there is not overlapped molecules between training set and tes set. 
	print('\n>>> Step 2 & 3: filter out spectra by certain rules; randomly split SMILES into training set and test set;')
	if 'qtof' in args.dataset: 
		print('Filter Agilent QTOF spectra...')
		agilent_qtof_spectra, agilent_qtof_smiles_list = filter_spec(agilent_spectra, config['agilent_qtof'], type2charge=config['encoding']['type2charge'])
		
		print('Filter NIST20 QTOF spectra...')
		nist_qtof_spectra, nist_qtof_smiles_list = filter_spec(nist_spectra, config['nist_qtof'], type2charge=config['encoding']['type2charge'])

		qtof_test_spectra, qtof_train_spectra = random_split(agilent_qtof_spectra+nist_qtof_spectra, 
													list(set(agilent_qtof_smiles_list+nist_qtof_smiles_list)), 
													test_ratio=0.1)
		del agilent_qtof_spectra
		del nist_qtof_spectra
		del agilent_spectra
		print('Get {} training spectra and {} test spectra'.format(len(qtof_train_spectra), len(qtof_test_spectra)))

	if 'hcd' in args.dataset: 
		print('Filter NIST20 HCD spectra...')
		nist_hcd_spectra, nist_hcd_smiles_list = filter_spec(nist_spectra, config['nist_hcd'], type2charge=config['encoding']['type2charge'])

		hcd_test_spectra, hcd_train_spectra = random_split(nist_hcd_spectra, 
													list(set(nist_hcd_smiles_list)), 
													test_ratio=0.1)
		del nist_hcd_spectra
		del nist_spectra
		print('Get {} training spectra and {} test spectra'.format(len(hcd_train_spectra), len(hcd_test_spectra)))

	# 4. generate 3d conformattions & encoding data into arrays
	print('\n>>> Step 4: encode all the data into pkl format;')
	if 'qtof' in args.dataset: 
		print('Convert QTOF spectra and molecules data into arrays...')
		# test
		test_data = spec2arr(qtof_test_spectra, config['encoding'])
		out_path = os.path.join(args.pkl_dir, 'qtof_{}_re0.01_test.pkl'.format(config['encoding']['conf_type']))
		with open(out_path, 'wb') as f: 
			pickle.dump(test_data, f)
			print('Save {}'.format(out_path))
		# train
		train_data = spec2arr(qtof_train_spectra, config['encoding'])
		out_path = os.path.join(args.pkl_dir, 'qtof_{}_re0.01_train.pkl'.format(config['encoding']['conf_type']))
		with open(out_path, 'wb') as f: 
			pickle.dump(train_data, f)
			print('Save {}'.format(out_path))

	if 'hcd' in args.dataset: 
		print('Convert HCD spectra and molecules data into arrays...')
		# test
		test_data = spec2arr(hcd_test_spectra, config['encoding'])
		out_path = os.path.join(args.pkl_dir, 'hcd_{}_test.pkl'.format(config['encoding']['conf_type']))
		with open(out_path, 'wb') as f: 
			pickle.dump(test_data, f)
			print('Save {}'.format(out_path))
		# train
		train_data = spec2arr(hcd_train_spectra, config['encoding'])
		out_path = os.path.join(args.pkl_dir, 'hcd_{}_train.pkl'.format(config['encoding']['conf_type']))
		with open(out_path, 'wb') as f: 
			pickle.dump(train_data, f)
			print('Save {}'.format(out_path))
		
	print('Done!')

