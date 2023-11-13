import numpy as np
from tqdm import tqdm

from rdkit import Chem
# ignore the warning
from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import Descriptors

from .utils import conformation_array, precursor_calculator, parse_collision_energy, generate_ms



'''pkl for training format
[
	{'title': <str>, 'mol': <numpy array>, 'env': <numpy array>, 'spec': <numpy array>}, 
	{'title': <str>, 'mol': <numpy array>, 'env': <numpy array>, 'spec': <numpy array>}, 
	....
]
where 'env' is formated as np.array([normalized collision energy, one hot encoding of precursor types]). 
'''

'''pkl for prediction format
[
	{'title': <str>, 'mol': <numpy array>, 'env': <numpy array>, 'smiles': <string>}, 
	{'title': <str>, 'mol': <numpy array>, 'env': <numpy array>, 'smiles': <string>}, 
	....
]
where 'env' is formated as np.array([normalized collision energy, one hot encoding of precursor types]). 
'''

# used in training ------------------------------------------------------------------------------
def mgf2pkl(spectra, encoder): 
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



# used in prediction ------------------------------------------------------------------------------
def csv2pkl_wfilter(csv_path, encoder): 
	'''
	This function is only used in prediction, so by default, the spectra are not contained. 
	'''
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

		data.append({'title': row['ID'], 'smiles': row['SMILES'], 'mol': mol_arr, 'env': env_arr})
	return data

def mgf2pkl_wfilter(mgf_path, encoder, with_spec=False): 
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

		data.append({'title': spec['params']['title'], 'smiles': spec['params']['smiles'], 
					'mol': mol_arr, 'env': env_arr})
	return data



# used in generating reference library ------------------------------------------------------------------------------
def sdf2pkl_with_cond(suppl, encoder, collision_energies, precursor_types): 
	data = []
	for idx, mol in enumerate(tqdm(suppl)): 
		# mol array
		good_conf, xyz_arr, atom_type = conformation_array(smiles=Chem.MolToSmiles(mol, isomericSmiles=True), 
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
		
		# env array
		for ce_str in collision_energies: 
			for add in precursor_types: 
				precursor_mz = precursor_calculator(add, Descriptors.ExactMolWt(mol))
				ce, nce = parse_collision_energy(ce_str=ce_str, 
										precursor_mz=precursor_mz, 
										charge=int(encoder['type2charge'][add]))
				if ce == None and nce == None: 
					continue
				precursor_type_one_hot = encoder['precursor_type'][add]
				env_arr = np.array([nce] + precursor_type_one_hot)

				data.append({'title': mol.GetProp('DATABASE_ID')+'_'+ce_str+'_'+str(add), 
							'smiles': Chem.MolToSmiles(mol, isomericSmiles=True), 'mol': mol_arr, 'env': env_arr})
	return data


