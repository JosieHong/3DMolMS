import numpy as np
import re
from decimal import *
import requests
from tqdm import tqdm

from rdkit import Chem
# ignore the warning
from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import AllChem



# -----------------------------------
# >>>    file-level functions     <<<
# -----------------------------------
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
		if mol == None or \
			not mol.HasProp('MASS SPECTRAL PEAKS') or \
			not mol.HasProp('PRECURSOR TYPE') or \
			not mol.HasProp('PRECURSOR M/Z') or \
			not mol.HasProp('SPECTRUM TYPE') or \
			not mol.HasProp('COLLISION ENERGY'): 
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



# -----------------------------------
# >>>     elemental functions     <<<
# -----------------------------------
def generate_ms(x, y, precursor_mz, resolution=1, max_mz=1500, charge=1): 
	'''
	Input:  x   [float list denotes the x-coordinate of peaks]
			y   [float list denotes the y-coordinate of peaks]
			precursor_mz	[float denotes the parention]
			resolution	[float denotes the resolution of spectra]
			max_mz		[integer denotes the maxium m/z value of spectra]
			charge		[float denotes the charge of spectra]
	Return: ms	[numpy array denotes the mass spectra]
	'''
	# generate isotropic peaks (refers to Kaiyuan's codes:
	# https://github.com/lkytal/PredFull/blob/master/train_model.py)
	isotropic_peaks = []
	for delta in (0, 1, 2):
		tmp = precursor_mz + delta / charge
		isotropic_peaks.append(int(tmp // resolution))

	# prepare parameters
	precursor_mz = Decimal(str(precursor_mz)) 
	resolution = Decimal(str(resolution))
	max_mz = Decimal(str(max_mz))
	right_bound = int(precursor_mz // resolution) # make precursor_mz as the right bound

	# init mass spectra vector: add "0" to y data
	ms = [0] * int(max_mz // resolution)

	# convert x, y to vector
	for idx, val in enumerate(x): 
		val = int(round(Decimal(str(val)) // resolution))
		if val >= right_bound: # remove precusor peak
			continue
		if val in isotropic_peaks: 
			continue
		ms[val] += y[idx]

	# normalize to 0-1
	if np.max(ms) - np.min(ms) == 0: 
		print('The maximum intensity and minimum intensity of this spectrum are the same!')
		print('right bound', right_bound)
		for i, j in zip(x, y):
			print(i, j)
		return False, np.array(ms)
	ms = (ms - np.min(ms)) / (np.max(ms) - np.min(ms))

	# smooth out large values
	ms = np.sqrt(np.array(ms)) 
	return True, ms

def ce2nce(ce, precursor_mz, charge):
	charge_factor = {1: 1, 2: 0.9, 3: 0.85, 4: 0.8, 5: 0.75, 6: 0.75, 7: 0.75, 8: 0.75}
	return ce * 500 * charge_factor[charge] / precursor_mz

def parse_collision_energy(ce_str, precursor_mz, charge=1): 
	# ratio constants for NCE
	charge_factor = {1: 1, 2: 0.9, 3: 0.85, 4: 0.8, 5: 0.75, 6: 0.75, 7: 0.75, 8: 0.75}

	ce = None
	nce = None
	
	# match collision energy (eV)
	matches_ev = {
		# NIST20
		r"^[\d]+[.]?[\d]*$": lambda x: float(x), 
		r"^[\d]+[.]?[\d]*[ ]?eV$": lambda x: float(x.rstrip(" eV")), 
		r"^[\d]+[.]?[\d]*[ ]?ev$": lambda x: float(x.rstrip(" ev")), 
		r"^[\d]+[.]?[\d]*[ ]?v$": lambda x: float(x.rstrip(" v")), 
		r"^NCE=[\d]+[.]?[\d]*% [\d]+[.]?[\d]*eV$": lambda x: float(x.split()[1].rstrip("eV")),
		r"^nce=[\d]+[.]?[\d]*% [\d]+[.]?[\d]*ev$": lambda x: float(x.split()[1].rstrip("ev")),
		# MassBank
		r"^[\d]+[.]?[\d]*[ ]?V$": lambda x: float(x.rstrip(" V")), 
		# r"^ramp [\d]+[.]?[\d]*-[\d]+[.]?[\d]* (ev|v)$":  lambda x: float((float(re.split(' |-', x)[1]) + float(re.split(' |-', x)[2])) /2), # j0siee: cannot process this ramp ce
		r"^[\d]+[.]?[\d]*-[\d]+[.]?[\d]*$": lambda x: float((float(x.split('-')[0]) + float(x.split('-')[1])) /2), 
		r"^hcd[\d]+[.]?[\d]*$": lambda x: float(x.lstrip('hcd')), 
	}
	for k, v in matches_ev.items(): 
		if re.match(k, ce_str): 
			ce = v(ce_str)
			break
	# match collision energy (NCE)
	matches_nce = {
		# MassBank
		r"^[\d]+[.]?[\d]*[ ]?[%]? \(nominal\)$": lambda x: float(x.rstrip('% (nominal)')), 
		r"^[\d]+[.]?[\d]*[ ]?nce$": lambda x: float(x.rstrip(' nce')), 
		r"^[\d]+[.]?[\d]*[ ]?\(nce\)$": lambda x: float(x.rstrip(' (nce)')), 
		r"^NCE=[\d]+\%$": lambda x: float(x.lstrip('NCE=').rstrip('%')), 
		# casmi
		r"^[\d]+[.]?[\d]*[ ]?\(nominal\)$": lambda x: float(x.rstrip("(nominal)").rstrip(' ')), 
	}
	for k, v in matches_nce.items(): 
		if re.match(k, ce_str): 
			nce = v(ce_str) * 0.01
			break
	
	# unknown collision energy
	if ce_str == 'Unknown': 
		ce = 40

	if nce == None and ce != None: 
		nce = ce * 500 * charge_factor[charge] / precursor_mz
	elif ce == None and nce != None:
		ce = nce * precursor_mz / (500 * charge_factor[charge])
	else:
		# raise Exception('Collision energy parse error: {}'.format(ce_str))
		return None, None
	return ce, nce

def conformation_array(smiles, conf_type): 
	# convert smiles to molecule
	if conf_type == 'etkdg': 
		mol = Chem.MolFromSmiles(smiles)
		mol_from_smiles = Chem.AddHs(mol)
		AllChem.EmbedMolecule(mol_from_smiles)

	elif conf_type == 'etkdgv3': 
		mol = Chem.MolFromSmiles(smiles)
		mol_from_smiles = Chem.AddHs(mol)
		AllChem.EmbedMolecule(mol_from_smiles, AllChem.ETKDGv3()) 

	elif conf_type == '2d':
		mol = Chem.MolFromSmiles(smiles)
		mol_from_smiles = Chem.AddHs(mol)
		rdDepictor.Compute2DCoords(mol_from_smiles)

	elif conf_type == 'omega': 
		raise ValueError('OMEGA conformation will be supported soon. ')
	else:
		raise ValueError('Unsupported conformation type. {}'.format(conf_type))

	# get the x,y,z-coordinates of atoms
	try: 
		conf = mol_from_smiles.GetConformer()
	except:
		return False, None, None
	xyz_arr = conf.GetPositions()
	# center the x,y,z-coordinates
	centroid = np.mean(xyz_arr, axis=0)
	xyz_arr -= centroid
	
	# concatenate with atom attributes
	xyz_arr = xyz_arr.tolist()
	for i, atom in enumerate(mol_from_smiles.GetAtoms()):
		xyz_arr[i] += [atom.GetDegree()]
		xyz_arr[i] += [atom.GetExplicitValence()]
		xyz_arr[i] += [atom.GetMass()/100]
		xyz_arr[i] += [atom.GetFormalCharge()]
		xyz_arr[i] += [atom.GetNumImplicitHs()]
		xyz_arr[i] += [int(atom.GetIsAromatic())]
		xyz_arr[i] += [int(atom.IsInRing())]
	xyz_arr = np.array(xyz_arr)
	
	# get the atom types of atoms
	atom_type = [atom.GetSymbol() for atom in mol_from_smiles.GetAtoms()]
	return True, xyz_arr, atom_type

def precursor_calculator(precursor_type, mass):
	if precursor_type == '[M+H]+':
		return mass + 1.007276 
	elif precursor_type == '[M+Na]+':
		return mass + 22.989218
	elif precursor_type == '[2M+H]+':
		return 2 * mass + 1.007276
	elif precursor_type == '[M-H]-':
		return mass - 1.007276
	else:
		raise ValueError('Unsupported precursor type: {}'.format(precursor_type))



# -----------------------------------
# >>>  post-processing functions  <<<
# -----------------------------------
global CACTUS 
CACTUS = "https://cactus.nci.nih.gov/chemical/structure/{0}/{1}"

def smiles_to_iupac(smiles):
	global CACTUS 
	rep = "iupac_name"
	url = CACTUS.format(smiles, rep)
	try: 
		response = requests.get(url)
		response.raise_for_status()
		return response.text
	except:
		print("Sorry, your structure identifier could not be resolved (the request returned a HTML 404 status message)")
		return ""

def smiles_to_inchi(smiles):
	global CACTUS 
	rep = "stdinchi"
	url = CACTUS.format(smiles, rep)
	try: 
		response = requests.get(url)
		response.raise_for_status()
		return response.text
	except:
		print("Sorry, your structure identifier could not be resolved (the request returned a HTML 404 status message)")
		return ""

def smiles_to_inchikey(smiles):
	global CACTUS 
	rep = "stdinchikey"
	url = CACTUS.format(smiles, rep)
	try: 
		response = requests.get(url)
		response.raise_for_status()
		return response.text
	except:
		print("Sorry, your structure identifier could not be resolved (the request returned a HTML 404 status message)")
		return ""