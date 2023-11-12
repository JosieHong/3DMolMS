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
from rdkit.Chem import Descriptors

from molmspack.data_utils import filter_mol, sdf2pkl_with_cond



if __name__ == "__main__": 
	parser = argparse.ArgumentParser(description='Preprocess the Data')
	parser.add_argument('--raw_dir', type=str, default='./data/hmdb/',
						help='path to raw data')
	parser.add_argument('--pkl_dir', type=str, default='./data/hmdb/',
						help='path to pkl data')
	parser.add_argument('--data_config_path', type=str, default='./config/preprocess_etkdgv3.yml',
						help='path to configuration')
	args = parser.parse_args()

	assert os.path.exists(os.path.join(args.raw_dir, 'structures.sdf'))

	# load the configurations
	with open(args.data_config_path, 'r') as f: 
		config = yaml.load(f, Loader=yaml.FullLoader)

	# load the data in sdf
	supp = Chem.SDMolSupplier(os.path.join(args.raw_dir, 'structures.sdf'))
	print('Read {} data from {}'.format(len(supp), os.path.join(args.raw_dir, 'structures.sdf')))

	# filter the molecules 
	supp, _ = filter_mol(supp, config['hmdb'])
	print('Filter to get {} molecules'.format(len(supp)))

	# convert data to pkl (with multiple meta data)
	# ID,SMILES,Precursor_Type,Source_Instrument,Collision_Energy
	collision_energies = ['20 eV', '40 eV']
	precursor_types = ['[M+H]+', '[M-H]-']
	hmdb_pkl = sdf2pkl_with_cond(supp, 
								config['encoding'], 
								collision_energies, 
								precursor_types)

	# save
	out_path = os.path.join(args.pkl_dir, 'hmdb_{}.pkl'.format(config['encoding']['conf_type']))
	with open(out_path, 'wb') as f: 
		pickle.dump(hmdb_pkl, f)
		print('Save {}'.format(out_path))
	
	print('Done!')
