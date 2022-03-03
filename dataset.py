'''
Date: 2021-07-08 18:37:32
LastEditors: yuhhong
LastEditTime: 2022-03-03 15:16:14
'''
import torch
from torch.utils.data import Dataset

import re
import numpy as np
from tqdm import tqdm
from decimal import *

from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers

class BaseDataset(Dataset):
    def __init__(self, supp, num_points, num_bonds, num_ms, resolution, data_augmentation): 
        self.ENCODE_ATOM = {'C': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'H': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'O': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 'N': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 'F': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 'S': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 'Cl': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 'P': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 'B': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 'Br': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 'I': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}
        # self.ENCODE_INST = {'HCD': [1, 0, 0, 0, 0, 0], 'QqQ': [0, 1, 0, 0, 0, 0], 'Orbitrap': [0, 0, 1, 0, 0, 0], 'Ion_Trap': [0, 0, 0, 1, 0, 0], 'QTOF': [0, 0, 0, 0, 1, 0], 'FT': [0, 0, 0, 0, 0, 1], 'N/A': [0, 0, 0, 0, 0, 0]}
        self.ENCODE_INST = {'HCD': 0, 'QqQ': 1, 'QTOF': 2, 'FT': 3, 'N/A': 4}
        self.supp = supp
        self.num_points = num_points
        self.num_bonds = num_bonds
        self.num_ms = num_ms
        self.resolution = resolution
        
        self.ENCODE_ADD = {}
        self.ENCODE_CE = {}
        self.MAP_INST = {}

        # data for training
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
    
    # init in the GNPSDataset or NISTDataset
    def init(self):
        for _, spectrum in enumerate(tqdm(self.supp)):
            smiles = spectrum['params']['smiles']
            mol = Chem.MolFromSmiles(smiles)
            
            x = spectrum['m/z array'].tolist()
            y = spectrum['intensity array'].tolist()
            pepmass = spectrum['params']['pepmass'][0]
            ms = self.generate_ms(x, y, pepmass, resolution=self.resolution)
            ce = self.parse_collision_energy(spectrum['params']['collision_energy'].lower())
            if np.max(ms) == 0: # filt out all 0 spectra
                continue
            if spectrum['params']['precursor_type'] not in self.ENCODE_ADD.keys():
                continue

            # load the data
            self.point_sets.append(mol)
            self.ms_sets.append(ms)
            self.adduct_sets.append(self.ENCODE_ADD[spectrum['params']['precursor_type']])
            self.collision_energy_sets.append(ce)
            self.instru_sets.append(self.ENCODE_INST[self.MAP_INST[spectrum['params']['source_instrument'].lower()]])
            # additional information
            self.ids.append(spectrum['params']['clean_id'])
            self.smiles.append(smiles)

            # We only pick at most ISOMER_NUM data
            if self.data_augmentation:
                ISORMER_NUM = 4
                isomers = tuple(EnumerateStereoisomers(mol))
                if len(isomers) > ISORMER_NUM:
                    isomers = np.random.choice(isomers, size=ISORMER_NUM, replace=True)
                for m in isomers:
                    self.point_sets.append(m)
                    self.ms_sets.append(ms)
                    self.collision_energy_sets.append(ce)
        
        # check the data
        # assert len(self.point_sets) == len(self.ms_sets) == len(self.collision_energy_sets) == len(self.adduct_sets)
        assert len(self.point_sets) == len(self.ms_sets) == len(self.collision_energy_sets)

    def parse_collision_energy(self, ce): 
        if ce in self.ENCODE_CE.keys(): 
            ce = self.ENCODE_CE[ce]
        if self.is_number(ce):
            ce = float(ce)
        else:
            ce = 0
        return ce

    def is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            pass
    
        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass
        return False

    def generate_ms(self, x, y, pepmass, resolution=1):
        '''
        Input:  x   [float list denotes the x-coordinate of peaks]
                y   [float list denotes the y-coordinate of peaks]
                pepmass [float denotes the parention]
        Return: ms  [float list denotes the mass spectra]
        '''
        pepmass = Decimal(str(pepmass)) 
        resolution = Decimal(str(resolution))
        right_bound = int(pepmass // resolution)

        ms = [0] * (int(Decimal(str(self.num_ms)) // resolution)) # add "0" to y data
        for idx, val in enumerate(x): 
            val = int(round(Decimal(str(val)) // resolution))
            if val >= right_bound: 
                continue
            ms[val] += y[idx]

        ms = np.sqrt(np.array(ms)) # smooth out larger values
        # Normalization, scale the ms to [0, 1]
        max_ms = np.max(ms)
        min_ms = np.min(ms)
        if (max_ms - min_ms) != 0:
            ms = (ms - min_ms) / (max_ms - min_ms)
        return ms


    def __len__(self): 
        return len(self.point_sets)

    def __getitem__(self, idx): 
        mol = self.point_sets[idx]
        X = self.create_X(mol, num_points=self.num_points, num_bonds=self.num_bonds)
        Y = self.ms_sets[idx]
        ENV = np.array([self.adduct_sets[idx], self.instru_sets[idx], self.collision_energy_sets[idx]])

        return self.ids[idx], self.smiles[idx], X, ENV, Y

    ''' The properties from: Extended-Connectivity Fingerprints
    1. the number of immediate neighbors who are “heavy” (nonhydrogen) atoms; 
    2. the valence minus the number of hydrogens;
    3. the atomic number; 
    4. the atomic mass; 
    5. the atomic charge;
    6. the number of attached hydrogens (both implicit and explicit).
    '''
    def create_X(self, mol, num_points, num_bonds): 
        mol_block = Chem.MolToMolBlock(mol).split("\n")
        point_set, bonds = self.parse_mol_block(mol_block) # 0. x,y,z-coordinates; atom type (one-hot); 

        for idx, atom in enumerate(mol.GetAtoms()): 
            point_set[idx].append(atom.GetDegree()) # 1. number of immediate neighbors who are “heavy” (nonhydrogen) atoms;
            point_set[idx].append(atom.GetExplicitValence()) # 2. valence minus the number of hydrogens;
            point_set[idx].append(atom.GetMass()/100) # 3. atomic mass; 
            point_set[idx].append(atom.GetFormalCharge()) # 4. atomic charge;
            point_set[idx].append(atom.GetNumImplicitHs()) # 5. number of implicit hydrogens;
            point_set[idx].append(int(atom.GetIsAromatic())) # Is aromatic
            point_set[idx].append(int(atom.IsInRing())) # Is in a ring 

        point_set = np.array(point_set).astype(np.float32)
        point_set = torch.cat((torch.Tensor(point_set), torch.zeros((num_points-point_set.shape[0], point_set.shape[1]))), dim=0)
        # return point_set
        
        bond_adj = np.zeros((num_points, num_points))
        for i, j, fea, _ in bonds:
            bond_adj[i, j] = fea
            bond_adj[j, i] = fea
        return [point_set, bond_adj] # [torch.Size([num_points, 14]), torch.Size([num_bonds, 4])]
    
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
                atom = [i for i in d.split(" ") if i!= ""]
                # atom: [x, y, z, atom_type, charge, stereo_care_box, valence]
                # sdf format (atom block): https://docs.chemaxon.com/display/docs/mdl-molfiles-rgfiles-sdfiles-rxnfiles-rdfiles-formats.md
                
                if len(atom) == 16 and atom[3] in self.ENCODE_ATOM.keys(): 
                    # x-y-z coordinates and atom type
                    point = [float(atom[0]), float(atom[1]), float(atom[2])] + self.ENCODE_ATOM[atom[3]]
                    points.append(point)
                elif len(atom) == 16: # check the atom type
                    print("Error: {} is not in {}, please check the dataset.".format(atom[3], self.ENCODE_ATOM.keys()))
                    exit()

            elif len(d) == 12:
                bond = [int(i) for i in d.split(" ") if i!= ""]
                if len(bond) == 4:
                    bonds.append(bond)
        
        # normalize the point cloud
        # We normalize scale to fit points in a unit sphere
        points = np.array(points)

        points_xyz = points[:, :3]
        centroid = np.mean(points_xyz, axis=0)
        points_xyz -= centroid
        
        points = np.concatenate((points_xyz, points[:, 3:]), axis=1)
        return points.tolist(), bonds



# This dataset is for inference only. It does not need the groud truth 
# mass spectra, and it will retuen None as Y. 
class CSVDataset(BaseDataset):
    def __init__(self, supp, num_points=200, num_bonds=10, num_ms=2000, resolution=1, data_augmentation=False):
        super(CSVDataset, self).__init__(supp, num_points, num_bonds, num_ms, resolution, data_augmentation)
        self.ENCODE_ADD = {'M+H': 0, 'M-H': 1, 'M+H-H2O': 2, 'M+Na': 3, 'M+H-NH3': 4, 'M+H-2H2O': 5, 'M-H-H2O': 6, 'M+NH4': 7, 'M+H-CH4O': 8, 'M+2Na-H': 9, 'M+H-C2H6O': 10, 'M+Cl': 11, 'M+OH': 12, 'M+H+2i': 13, '2M+H': 14, '2M-H': 15, 'M-H-CO2': 16, 'M+2H': 17, 'M-H+2i': 18, 'M+H-CH2O2': 19, 'M+H-C4H8': 20, 'M+H-C2H4O2': 21, 'M+H-C2H4': 22, 'M+CHO2': 23, 'M-H-CH3': 24, 'M+H-H2O+2i': 25, 'M+H-C2H2O': 25, 'M+H-C3H6': 26, 'M+H-CH3': 27, 'M+H-3H2O': 28, 'M+H-HF': 29, 'M-2H': 30, 'M-H2O+H': 2, 'M-2H2O+H': 5}
        self.init()
    
    # init in the GNPSDataset or NISTDataset
    def init(self): 
        for _, row in enumerate(tqdm(self.supp)):
            mol = Chem.MolFromSmiles(row['SMILES'])
            self.point_sets.append(mol)

            self.adduct_sets.append(self.ENCODE_ADD[row['Precursor_Type']])
            self.collision_energy_sets.append(row['Collision_Energy'])
            self.instru_sets.append(self.ENCODE_INST[row['Source_Instrument']])
            # additional information
            self.ids.append(str(row['ID']))
            self.smiles.append(row['SMILES'])

        # check the data
        assert len(self.point_sets) == len(self.collision_energy_sets) == len(self.adduct_sets)


    def __len__(self): 
        return len(self.point_sets)

    def __getitem__(self, idx): 
        mol = self.point_sets[idx]
        X = self.create_X(mol, num_points=self.num_points, num_bonds=self.num_bonds)
        ENV = np.array([self.adduct_sets[idx], self.instru_sets[idx], self.collision_energy_sets[idx]])

        return self.ids[idx], self.smiles[idx], X, ENV