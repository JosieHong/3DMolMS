'''
Date: 2021-07-08 18:37:32
LastEditors: yuhhong
LastEditTime: 2022-12-08 01:04:38
'''
import torch
from torch.utils.data import Dataset

import re
import numpy as np
from tqdm import tqdm
from decimal import *

from rdkit import Chem



class BaseDataset(Dataset): 
	def __init__(self, supp, gzsupp, num_points, num_ms, resolution, data_augmentation): 
		self.ENCODE_ATOM = {'C': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
							'H': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
							'O': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
							'N': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 
							'F': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 
							'S': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 
							'Cl': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 
							'P': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 
							'B': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 
							'Br': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 
							'I': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}
		self.ENCODE_INST = {'HCD': 0, 'QTOF': 1}
		self.supp = supp
		self.gzsupp = gzsupp
		self.num_points = num_points
		self.num_ms = num_ms
		self.resolution = resolution
		
		self.ENCODE_ADD = {}
		self.ENCODE_CE = {}
		self.MAP_INST = {}

		# data for training
		self.mol_idx = {}
		self.mol_list = []
		self.point_sets = []
		self.ms_sets = []
		self.adduct_sets = []
		self.instru_sets = []
		self.collision_energy_sets = []
		# data for identification
		self.ids = []
		self.smiles = []
		# other parameters
		self.data_augmentation = data_augmentation
	
	def init(self): 
		# For mol file
		print('Loading molecules...')
		for i, mol in tqdm(enumerate(self.gzsupp)): 
			self.mol_list.append(mol)
			clean_id = mol.GetProp('clean_id')
			self.mol_idx[clean_id] = i

		# For ms file
		print('Loading spectra...')
		for _, spectrum in tqdm(enumerate(self.supp)):
			smiles = spectrum['params']['smiles']
			clean_id = spectrum['params']['clean_id']
			mol = self.mol_list[self.mol_idx[clean_id]] # concat ms with its 3D molecule by `clean_id`

			x = spectrum['m/z array'].tolist()
			y = spectrum['intensity array'].tolist()
			pepmass = spectrum['params']['pepmass'][0]
			ms = self.generate_ms(x, y, pepmass, resolution=self.resolution)
			ce = self.parse_collision_energy(spectrum['params']['collision_energy'].lower(), pepmass)
			if np.max(ms) == 0: # filt out all 0 spectra
				continue
			if spectrum['params']['precursor_type'] not in self.ENCODE_ADD.keys():
				continue

			# load data
			self.point_sets.append(mol)
			self.ms_sets.append(ms)
			self.adduct_sets.append(self.ENCODE_ADD[spectrum['params']['precursor_type']])
			self.collision_energy_sets.append(ce)
			self.instru_sets.append(self.ENCODE_INST[self.MAP_INST[spectrum['params']['instrument_type'].lower()]])
			# additional information
			self.ids.append(spectrum['params']['clean_id'])
			self.smiles.append(smiles)
	
	def parse_collision_energy(self, ce_str, pepmass, charge=2): 
		# ratio constants for NCE
		charge_factor = {1: 1, 2: 0.9, 3: 0.85, 4: 0.8, 5: 0.75, 6: 0.75, 7: 0.75, 8: 0.75}

		ce = None
		if ce_str == 'unknown': # give the unknown collision energy an average value
			ce = 20 * 500 * charge_factor[charge] / pepmass
		elif ce_str in self.ENCODE_CE.keys(): 
			ce = self.ENCODE_CE[ce_str]
		else: # parase ce for NIST20 & MassBank
			# match collision energy (eV)
			matches_ev = {
				# NIST20
				r"^[\d]+[.]?[\d]*$": lambda x: float(x), 
				r"^[\d]+[.]?[\d]*[ ]?eV$": lambda x: float(x.rstrip(" eV")), 
				r"^[\d]+[.]?[\d]*[ ]?ev$": lambda x: float(x.rstrip(" ev")), 
				r"^[\d]+[.]?[\d]*[ ]?v$": lambda x: float(x.rstrip(" v")), 
				r"^NCE=[\d]+[.]?[\d]*% [\d]+[.]?[\d]*eV$": lambda x: float(x.split()[1].rstrip("eV")),
				# MassBank
				r"^[\d]+[.]?[\d]*[ ]?v$": lambda x: float(x.rstrip(" v")), 
				# r"^ramp [\d]+[.]?[\d]*-[\d]+[.]?[\d]* (ev|v)$":  lambda x: float((float(re.split(' |-', x)[1]) + float(re.split(' |-', x)[2])) /2), # j0siee: cannot process this ramp ce
				r"^[\d]+[.]?[\d]*-[\d]+[.]?[\d]*$": lambda x: float((float(x.split('-')[0]) + float(x.split('-')[1])) /2), 
				r"^hcd[\d]+[.]?[\d]*$": lambda x: float(x.lstrip('hcd')), 
			}
			for k, v in matches_ev.items(): 
				if re.match(k, ce_str): 
					ce = v(ce_str)
					# convert ce to nce
					ce = ce * 500 * charge_factor[charge] / pepmass
					break
			# match collision energy (NCE)
			matches_nce = {
				# MassBank
				r"^[\d]+[.]?[\d]*[ ]?[%]? \(nominal\)$": lambda x: float(x.rstrip('% (nominal)')), 
				r"^[\d]+[.]?[\d]*[ ]?nce$": lambda x: float(x.rstrip(' nce')), 
				r"^[\d]+[.]?[\d]*[ ]?\(nce\)$": lambda x: float(x.rstrip(' (nce)')), 
			}
			for k, v in matches_nce.items(): 
				if re.match(k, ce_str): 
					ce = v(ce_str) * 0.01
					break
		if ce == None: 
			ce = 0.4 
			# print(ce_str)
		return ce

	def generate_ms(self, x, y, pepmass, resolution=1, charge=2):
		'''
		Input:  x   [float list denotes the x-coordinate of peaks]
				y   [float list denotes the y-coordinate of peaks]
				pepmass		[float denotes the parention]
				resolution	[float denotes the resolution of spectra]
				charge		[float denotes the charge of spectra]
		Return: ms  [float list denotes the mass spectra]
		'''
		# generate isotropic peaks (refers to Kaiyuan's codes:
		# https://github.com/lkytal/PredFull/blob/master/train_model.py)
		isotropic_peaks = []
		for delta in (0, 1, 2):
			precursor_mz = pepmass + delta / charge
			isotropic_peaks.append(int(precursor_mz // resolution))

		# prepare parameters
		pepmass = Decimal(str(pepmass)) 
		resolution = Decimal(str(resolution))
		right_bound = int(pepmass // resolution) # make pepmass as the right bound

		# init mass spectra vector: add "0" to y data
		ms = [0] * (int(Decimal(str(self.num_ms)) // resolution)) 

		# convert x, y to vector
		for idx, val in enumerate(x): 
			val = int(round(Decimal(str(val)) // resolution))
			if val >= right_bound: 
				continue
			if val in isotropic_peaks:
				continue
			ms[val] += y[idx]

		# normalize to 0-1
		if np.max(ms) - np.min(ms) == 0: 
			return ms
		ms = (ms - np.min(ms)) / (np.max(ms) - np.min(ms))

		# smooth out large values
		ms = np.sqrt(np.array(ms)) 
		return ms
		
	def __len__(self): 
		return len(self.point_sets)

	def __getitem__(self, idx): 
		X, mask = self.create_X(self.point_sets[idx], num_points=self.num_points)
		Y = self.ms_sets[idx]
		ENV = np.array([self.adduct_sets[idx], self.instru_sets[idx], self.collision_energy_sets[idx]])

		return self.ids[idx], self.smiles[idx], X, mask, ENV, Y

	def create_X(self, mol, num_points): 
		mol_block = Chem.MolToMolBlock(mol).split("\n")
		point_set, bonds = self.parse_mol_block(mol_block) # 1. x,y,z-coordinates; 2. atom type (one-hot); 
		for idx, atom in enumerate(mol.GetAtoms()): 
			point_set[idx].append(atom.GetDegree()) # 3. number of immediate neighbors who are “heavy” (nonhydrogen) atoms;
			point_set[idx].append(atom.GetExplicitValence()) # 4. valence minus the number of hydrogens;
			point_set[idx].append(atom.GetMass()/100) # 5. atomic mass; 
			point_set[idx].append(atom.GetFormalCharge()) # 6. atomic charge;
			point_set[idx].append(atom.GetNumImplicitHs()) # 7. number of implicit hydrogens;
			point_set[idx].append(int(atom.GetIsAromatic())) # 8. is aromatic; 
			point_set[idx].append(int(atom.IsInRing())) # 9. is in a ring; 

		point_set = np.array(point_set).astype(np.float32)

		# generate mask
		point_mask = np.ones_like(point_set[0])

		point_set = torch.cat((torch.Tensor(point_set), torch.zeros((num_points-point_set.shape[0], point_set.shape[1]))), dim=0)
		point_mask = torch.cat((torch.Tensor(point_mask), torch.zeros((num_points-point_mask.shape[0]))), dim=0)
		
		return point_set, point_mask
		
	
	def parse_mol_block(self, mol_block): 
		'''
		Input:  mol_block   [list denotes the lines of mol block]
		Return: points      [list denotes the atom points, (npoints, 4)]
				bonds       [list denotes the atom bonds, (npoints, 4)]
		'''
		points = []
		bonds = []
		for d in mol_block:
			if len(d) == 69: # the format of molecular block is fixed
				atom = [i for i in d.split()]
				# atom: [x, y, z, atom_type, charge, stereo_care_box, valence]
				# sdf format (atom block): https://docs.chemaxon.com/display/docs/mdl-molfiles-rgfiles-sdfiles-rxnfiles-rdfiles-formats.md
				
				if len(atom) == 16 and atom[3] in self.ENCODE_ATOM.keys(): 
					# only x-y-z coordinates
					# point = [float(atom[0]), float(atom[1]), float(atom[2])]

					# x-y-z coordinates and atom type
					point = [float(atom[0]), float(atom[1]), float(atom[2])] + self.ENCODE_ATOM[atom[3]]
					points.append(point)
				elif len(atom) == 16: # check the atom type
					raise ValueError("Error: {} is not in {}, please check the dataset.".format(atom[3], self.ENCODE_ATOM.keys()))

			elif len(d) == 12:
				bond = [int(i) for i in d.split()]
				if len(bond) == 4: 
					bonds.append(bond)
		
		points = np.array(points)

		# center the points
		points_xyz = points[:, :3]
		centroid = np.mean(points_xyz, axis=0)
		points_xyz -= centroid
		
		points = np.concatenate((points_xyz, points[:, 3:]), axis=1)
		return points.tolist(), bonds



class NISTDataset(BaseDataset): 
	def __init__(self, supp, gzsupp, num_points=200, num_ms=2000, resolution=1, data_augmentation=False):
		super(NISTDataset, self).__init__(supp, gzsupp, num_points, num_ms, resolution, data_augmentation)
		self.ENCODE_ADD = {'M+H': 0, 'M-H': 1, 'M+Na': 2, 'M+H-H2O': 3, 'M+2H': 4}
		self.MAP_INST = {'hcd': 'HCD', 
							# 'it/ion trap': 'Ion_Trap', 
							# 'it-ft/ion trap with ftms': 'FT', 
							'q-tof': 'QTOF', 
							# 'qqq': 'QqQ', 'qqit': 'QqQ',
							}
		self.ENCODE_CE = {} 

		# init
		self.init()

class GNPSDataset(BaseDataset): 
	def __init__(self, supp, gzsupp, num_points=200, num_ms=2000, resolution=1, data_augmentation=False):
		super(GNPSDataset, self).__init__(supp, gzsupp, num_points, num_ms, resolution, data_augmentation)
		self.ENCODE_ADD = {'M+H': 0, 'M-H': 1, 'M+Na': 2, 'M+H-H2O': 3, 'M+2H': 4}
		self.MAP_INST = {'esi-qtof': 'QTOF', 'lc-esi-qtof': 'QTOF', 'esi-lc-esi-qtof': 'QTOF', 'unknown': 'QTOF',
							# 'esi-orbitrap': 'Orbitrap', 'lc-esi-orbitrap': 'Orbitrap',  
							# 'lc-esi-ion trap': 'Ion_Trap', 'esi-ion trap': 'Ion_Trap',
							# 'esi-hybrid ft': 'FT', 'esi-lc-esi-itft': 'FT', 'esi-lc-esi-qft': 'FT',
							# 'esi-qqq': 'QqQ', 'esi-flow-injection qqq/ms': 'QqQ', 'esi-lc-esi-qq': 'QqQ',
							} 
		self.ENCODE_CE = {}

		# init
		self.init()

class MassBankDataset(BaseDataset):
	def __init__(self, supp, gzsupp, num_points=200, num_ms=2000, resolution=1, data_augmentation=False):
		super(MassBankDataset, self).__init__(supp, gzsupp, num_points, num_ms, resolution, data_augmentation)
		self.ENCODE_ADD = {'M+H': 0, 'M-H': 1, 'M+Na': 2, 'M+H-H2O': 3, 'M+2H': 4}
		self.MAP_INST = {'lc-esi-qft hcd': 'HCD', 'lc-esi-itft hcd': 'HCD', 
							'lc-esi-qtof n/a': 'QTOF', 'lc-esi-qtof cid': 'QTOF', 'lc-esi-qtof': 'QTOF', 'lc-q-tof/ms n/a': 'QTOF', 'lc-esi-qft n/a': 'QTOF', 'lc-esi-tof n/a': 'QTOF', 
							# 'lc-esi-itft cid': 'FT', 'lc-esi-itft': 'FT', 
							# 'unknown n/a': 'N/A', 
							# 'lc-esi-qq': 'QqQ', 'lc-esi-qqq n/a': 'QqQ',
							} 
		self.ENCODE_CE = {'scaled by m/z': 0} 

		# init
		self.init()

class HMDBDataset(BaseDataset):
	def __init__(self, supp, gzsupp, num_points=200, num_ms=2000, resolution=1, data_augmentation=False):
		super(HMDBDataset, self).__init__(supp, gzsupp, num_points, num_ms, resolution, data_augmentation)
		self.ENCODE_ADD = {'M+H': 0, 'M-H': 1} 
		self.MAP_INST = {'lc-esi-qtof': 'QTOF', 'lc-esi-qft': 'QTOF', 'esi-tof': 'QTOF', 'qtof': 'QTOF', 'lc-esi-qtof (uplc q-tof premier, waters)': 'QTOF', 'lc-esi-tof': 'QTOF', 
							# 'unknown': 'N/A', 
							# 'lc-esi-qq': 'QqQ', 'quattro_qqq': 'QqQ', 'lc-esi-qq (api3000, applied biosystems)': 'QqQ', 'qqq': 'QqQ',
							} 
		self.ENCODE_CE = {'low': 10, 'med': 20, 'high': 40}

		# init
		self.init()

class AgilentDataset(BaseDataset):
	def __init__(self, supp, gzsupp, num_points=200, num_ms=2000, resolution=1, data_augmentation=False):
		super(AgilentDataset, self).__init__(supp, gzsupp, num_points, num_ms, resolution, data_augmentation)
		self.ENCODE_ADD = {'M+H': 0, 'M-H': 1, 'M+Na': 2, 'M+H-H2O': 3, 'M+2H': 4} 
		self.MAP_INST = {'esi-qtof': 'QTOF', 'agilent qtof 6530': 'QTOF', 'q-tof': 'QTOF', } 
		self.ENCODE_CE = {}

		# init
		self.init()

class MERGEDataset(BaseDataset): 
	def __init__(self, supp, gzsupp, num_points=200, num_ms=2000, resolution=1, data_augmentation=False): 
		super(MERGEDataset, self).__init__(supp, gzsupp, num_points, num_ms, resolution, data_augmentation)
		self.ENCODE_ADD = {'M+H': 0, 'M-H': 1, 'M+Na': 2, 'M+H-H2O': 3, 'M+2H': 4}
		self.MAP_INST = {# NIST20
							'hcd': 'HCD', 
							'q-tof': 'QTOF', 
							
							# MassBank
							'lc-esi-itft': 'HCD', 'lc-esi-qft': 'HCD', 
							'unknown': 'QTOF', 'lc-esi-qtof': 'QTOF', 'lc-esi-qft': 'QTOF', 'lc-q-tof/ms': 'QTOF', 'lc-esi-tof': 'QTOF', 'esi-qtof': 'QTOF', 'q-tof': 'QTOF', 
							
							# HMDB
							'lc-esi-qtof': 'QTOF', 'lc-esi-qft': 'QTOF', 'esi-tof': 'QTOF', 'qtof': 'QTOF', 'lc-esi-qtof (uplc q-tof premier, waters)': 'QTOF', 'lc-esi-tof': 'QTOF', 
							
							# Agilent
							'esi-qtof': 'QTOF', 'agilent qtof 6530': 'QTOF', 'q-tof': 'QTOF', 
							} 

							
		self.ENCODE_CE = {# HMDB
							'low': 10, 'med': 20, 'high': 40}

		# init
		self.init()



# This dataset is for inference only. It does not need the groud truth 
# mass spectra, and in `__getitem__` it will retuen None as Y. 
class CSVDataset(BaseDataset):
	def __init__(self, supp, gzsupp, num_points=200, num_ms=2000, resolution=1, data_augmentation=False): 
		super(CSVDataset, self).__init__(supp, gzsupp, num_points, num_ms, resolution, data_augmentation)
		self.ENCODE_ADD = {'M+H': 0, 'M-H': 1, 'M+Na': 2, 'M+H-H2O': 3, 'M+2H': 4}
		self.init()
	
	def init(self): 
		# For mol file
		for i, mol in enumerate(self.gzsupp): 
			# mol_block = Chem.MolToMolBlock(mol)
			self.mol_list.append(mol)
			clean_id = mol.GetProp('clean_id')
			self.mol_idx[clean_id] = i
			
		# For data file
		for _, row in enumerate(tqdm(self.supp)): 
			clean_id = str(row['ID'])
			mol = self.mol_list[self.mol_idx[clean_id]] # concat ms with its 3D molecule by `clean_id`

			self.point_sets.append(mol)
			self.adduct_sets.append(self.ENCODE_ADD[row['Precursor_Type']])
			self.collision_energy_sets.append(row['Collision_Energy'])
			self.instru_sets.append(self.ENCODE_INST[row['Source_Instrument']])
			# additional information
			self.ids.append(str(row['ID']))
			self.smiles.append(row['SMILES'])

	def __len__(self): 
		return len(self.point_sets)

	def __getitem__(self, idx): 
		X, mask = self.create_X(self.point_sets[idx], num_points=self.num_points)
		ENV = np.array([self.adduct_sets[idx], self.instru_sets[idx], self.collision_energy_sets[idx]])

		return self.ids[idx], self.smiles[idx], X, mask, ENV