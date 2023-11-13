import numpy as np
from tqdm import tqdm

from rdkit import Chem
# ignore the warning
from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*')



def filter_spec(spectra, config, type2charge): 
	clean_spectra = []
	smiles_list = []
	for idx, spectrum in enumerate(tqdm(spectra)): 
		smiles = spectrum['params']['smiles'] 
		mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
		if mol == None: continue

		# Filter by collision energy
		if spectrum['params']['collision_energy'].startswith('ramp') or \
				spectrum['params']['collision_energy'].startswith('Ramp') or \
				spectrum['params']['collision_energy'].endswith('ramp'): # we can not process this type collision energy now
			continue

		# Filter by instrument type
		instrument_type = spectrum['params']['instrument_type']
		if instrument_type not in config['intrument_type']: continue

		# Filter by instrument (MoNA contains too many intrument names to filter out)
		if 'instrument' in config.keys(): 
			instrument = spectrum['params']['source_instrument']
			if instrument not in config['instrument']: continue

		# Filter by mslevel
		if 'ms_level' in config.keys(): 
			mslevel = spectrum['params']['ms_level']
			if mslevel != config['ms_level']: continue

		# Filter by atom number and atom type 
		if not check_atom(mol, config, in_type='molh'): 
			continue

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

def filter_mol(suppl, config): 
	clean_suppl = []
	smiles_list = []
	for idx, mol in enumerate(tqdm(suppl)): 
		if mol == None: continue
		mol = Chem.AddHs(mol)

		# Filter by atom number and atom type 
		if not check_atom(mol, config, in_type='molh'): 
			continue

		clean_suppl.append(mol)
		smiles_list.append(Chem.MolToSmiles(mol))
	return clean_suppl, smiles_list

def check_atom(x, config, in_type='smiles'):
	assert in_type in ['smiles', 'mol', 'molh']
	if in_type == 'smiles': 
		mol = Chem.AddHs(Chem.MolFromSmiles(x))
	elif in_type == 'mol': 
		mol = Chem.AddHs(x)
	else:
		mol = x

	if len(mol.GetAtoms()) > config['max_atom_num'] or len(mol.GetAtoms()) < config['min_atom_num']: 
		return False

	is_compound_countain_rare_atom = False 
	for atom in mol.GetAtoms(): 
		if atom.GetSymbol() not in config['atom_type']:
			is_compound_countain_rare_atom = True
			break
	if is_compound_countain_rare_atom: 
		return False
	return True