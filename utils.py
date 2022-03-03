'''
Date: 2022-05-19 17:26:44
LastEditors: yuhhong
LastEditTime: 2022-12-05 17:04:24
'''
import gzip
import pandas as pd
from tqdm import tqdm
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
# suppress rdkit warning
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from pyteomics import mgf

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import GNPSDataset, NISTDataset, MassBankDataset, MERGEDataset, HMDBDataset, AgilentDataset, CSVDataset

# -----------------------------------
# utils for Training
# -----------------------------------

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# refer: https://github.com/YuanyueLi/SpectralEntropy
def entropy_distance(a, b):
    entropy_a = entropy(a)
    entropy_b = entropy(b)
    entropy_merged = entropy(a + b)
    return (2 * entropy_merged - entropy_a - entropy_b) / np.log(4)

def entropy(intensity): 
    intensity = intensity[intensity > 0]
    return -torch.sum(intensity * torch.log(intensity))

def reg_criterion(outputs, targets): 
    # cosine similarity
    t = nn.CosineSimilarity(dim=1)
    spec_cosi = torch.mean(1 - t(outputs, targets))
    return spec_cosi

    # entropy distance
    spec_entr = torch.stack([entropy_distance(i, o) for o, i in zip(outputs, targets)])
    spec_entr = torch.mean(spec_entr)
    # return spec_entr

    # all
    return spec_cosi + 0.1 * spec_entr

# -----------------------------------
# utils for Inference
# -----------------------------------

def spec_convert(spec, resolution):
    x = []
    y = []
    for i, j in enumerate(spec):
        if j != 0: 
            x.append(str(i*resolution))
            y.append(str(j))
    # return {'m/z': np.array(x), 'intensity': np.array(y)}
    return {'m/z': ','.join(x), 'intensity': ','.join(y)}

# -----------------------------------
# utils for Data Loading
# -----------------------------------

def generate_2d_comformers(data_path, mol_path):
    # load input data
    supp = mgf.read(data_path)
    outf = gzip.open(mol_path, 'wt+')
    writer = Chem.SDWriter(outf)
    cnt = 0
    for _, spectrum in enumerate(tqdm(supp)):
        smiles = spectrum['params']['smiles']
        clean_id = spectrum['params']['clean_id']
        mol = Chem.MolFromSmiles(smiles)

        # add clean id
        mol.SetProp('clean_id', clean_id)
        writer.write(mol)
        cnt += 1

    writer.close()
    outf.close()
    return mol_path, cnt

def generate_3d_comformers(data_path, mol_path):
    # load input data
    supp = mgf.read(data_path)
    outf = gzip.open(mol_path, 'wt+')
    writer = Chem.SDWriter(outf)
    cnt = 0
    for _, spectrum in enumerate(tqdm(supp)):
        smiles = spectrum['params']['smiles']
        clean_id = spectrum['params']['clean_id']
        mol = Chem.MolFromSmiles(smiles)
        if mol == None:
            continue

        # embed molecule to 3D comformers
        mol = Chem.AddHs(mol)
        try:
            # Landrum et al. DOI: 10.1021/acs.jcim.5b00654
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            # AllChem.EmbedMolecule(mol, useRandomCoords=True)
        except: 
            continue

        # add clean id
        mol.SetProp('clean_id', clean_id)
        writer.write(mol)
        cnt += 1

    writer.close()
    outf.close()
    return mol_path, cnt

def generate_2d_comformers_csv(data_path, mol_path): 
    # load input data
    supp = pd.read_csv(data_path)
    outf = gzip.open(mol_path, 'wt+')
    writer = Chem.SDWriter(outf)
    cnt = 0
    for _, row in tqdm(supp.iterrows()): 
        smiles = row['SMILES']
        clean_id = str(row['ID'])
        mol = Chem.MolFromSmiles(smiles)
        if mol == None:
            continue

        # add clean id
        mol.SetProp('clean_id', clean_id)
        writer.write(mol)
        cnt += 1

    writer.close()
    outf.close()
    return mol_path, cnt

def generate_3d_comformers_csv(data_path, mol_path): 
    # load input data
    supp = pd.read_csv(data_path)
    outf = gzip.open(mol_path, 'wt+')
    writer = Chem.SDWriter(outf)
    cnt = 0
    for _, row in tqdm(supp.iterrows()): 
        smiles = row['SMILES']
        clean_id = str(row['ID'])
        mol = Chem.MolFromSmiles(smiles)
        if mol == None:
            continue
    
        # embed molecule to 3D comformers
        mol = Chem.AddHs(mol)
        try:
            AllChem.EmbedMolecule(mol)
        except:
            continue

        # add clean id
        mol.SetProp('clean_id', clean_id)
        writer.write(mol)
        cnt += 1

    writer.close()
    outf.close()
    return mol_path, cnt



def load_data(data_path, mol_path, num_atoms, out_dim, resolution, ion_mode, dataset, num_workers, batch_size, data_augmentation, shuffle): 
    if dataset == 'nist': 
        supp = mgf.read(data_path) # ms data
        gzsupp = Chem.ForwardSDMolSupplier(gzip.open(mol_path)) # mol data 
        dataset = NISTDataset([item for item in batch_filter(supp, num_atoms, out_dim, ion_mode, data_type='mgf')], gzsupp, 
                                num_points=num_atoms, num_ms=out_dim, resolution=resolution, data_augmentation=data_augmentation)
    elif dataset == 'gnps': 
        supp = mgf.read(data_path) # ms data
        gzsupp = Chem.ForwardSDMolSupplier(gzip.open(mol_path)) # mol data 
        dataset = GNPSDataset([item for item in batch_filter(supp, num_atoms, out_dim, ion_mode, data_type='mgf')], gzsupp, 
                                num_points=num_atoms, num_ms=out_dim, resolution=resolution, data_augmentation=data_augmentation)
    elif dataset == 'massbank':
        supp = mgf.read(data_path) # ms data
        gzsupp = Chem.ForwardSDMolSupplier(gzip.open(mol_path)) # mol data 
        dataset = MassBankDataset([item for item in batch_filter(supp, num_atoms, out_dim, ion_mode, data_type='mgf')], gzsupp, 
                                    num_points=num_atoms, num_ms=out_dim, resolution=resolution, data_augmentation=data_augmentation)
    elif dataset == 'merge': 
        supp = mgf.read(data_path) # ms data
        gzsupp = Chem.ForwardSDMolSupplier(gzip.open(mol_path)) # mol data 
        dataset = MERGEDataset([item for item in batch_filter(supp, num_atoms, out_dim, ion_mode, data_type='mgf')], gzsupp, 
                                num_points=num_atoms, num_ms=out_dim, resolution=resolution, data_augmentation=data_augmentation)
    elif dataset == 'hmdb':
        supp = mgf.read(data_path) # ms data
        gzsupp = Chem.ForwardSDMolSupplier(gzip.open(mol_path)) # mol data 
        dataset = HMDBDataset([item for item in batch_filter(supp, num_atoms, out_dim, ion_mode, data_type='mgf')], gzsupp, 
                                num_points=num_atoms, num_ms=out_dim, resolution=resolution, data_augmentation=data_augmentation)
    elif dataset == 'agilent': 
        supp = mgf.read(data_path) # ms data
        gzsupp = Chem.ForwardSDMolSupplier(gzip.open(mol_path)) # mol data 
        dataset = AgilentDataset([item for item in batch_filter(supp, num_atoms, out_dim, ion_mode, data_type='mgf')], gzsupp, 
                                num_points=num_atoms, num_ms=out_dim, resolution=resolution, data_augmentation=data_augmentation)
    elif dataset == 'merge_infer': 
        supp = pd.read_csv(data_path)
        gzsupp = Chem.ForwardSDMolSupplier(gzip.open(mol_path)) # mol data 
        dataset = CSVDataset([item for item in batch_filter(supp, num_atoms, out_dim, ion_mode, data_type='csv')], gzsupp, 
                                num_points=num_atoms, num_ms=out_dim, resolution=resolution, data_augmentation=data_augmentation)
    else:
        raise Exception("Not implemented dataset type")
    
    print('Load {} data from {}.'.format(len(dataset), data_path))
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True)
    return data_loader 

def batch_filter(supp, num_atoms=200, out_dim=2000, ion_mode='P', data_type='mgf'): 
    if ion_mode == 'P':
        ion_mode = ['p', 'positive']
    elif ion_mode == 'N':
        ion_mode = ['n', 'negative']
    else:
        ion_mode = ['p', 'positive', 'n', 'negative']
    # for training and test, we use `.mgf` files
    print('Batch filter...')
    if data_type == 'mgf':
        for _, item in tqdm(enumerate(supp)): 
            smiles = item.get('params').get('smiles')
            if item.get('m/z array').max() > out_dim: 
                continue
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)
            if len(mol.GetAtoms()) > num_atoms or len(mol.GetAtoms()) == 0: 
                continue
            if item['params']['ionmode'].lower() not in ion_mode: 
                continue
            yield item

    # for inference, we use `.csv` files
    elif data_type == 'csv':
        ATOM_LIST = ['C', 'H', 'O', 'N', 'F', 'S', 'Cl', 'P', 'B', 'Br', 'I']
        ADD_LIST = ['M+H', 'M-H', 'M+H-H2O', 'M+Na', 'M+H-NH3', 'M+H-2H2O', 'M-H-H2O', 'M+NH4', 'M+H-CH4O', 'M+2Na-H', 
                    'M+H-C2H6O', 'M+Cl', 'M+OH', 'M+H+2i', '2M+H', '2M-H', 'M-H-CO2', 'M+2H', 'M-H+2i', 'M+H-CH2O2', 'M+H-C4H8', 
                    'M+H-C2H4O2', 'M+H-C2H4', 'M+CHO2', 'M-H-CH3', 'M+H-H2O+2i', 'M+H-C2H2O', 'M+H-C3H6', 'M+H-CH3', 'M+H-3H2O', 
                    'M+H-HF', 'M-2H', 'M-H2O+H', 'M-2H2O+H']
        INST_LIST = ['HCD', 'QqQ', 'QTOF', 'FT', 'N/A']
        for _, row in supp.iterrows(): 
            mol = Chem.MolFromSmiles(row['SMILES'])
            if mol == None: 
                continue

            # check atom number
            mol = Chem.AddHs(mol)
            if len(mol.GetAtoms()) > num_atoms:
                print('{}: atom number is larger than {}'.format(row['ID'], num_atoms))
                continue
            # check atom type
            for atom in mol.GetAtoms():
                if atom.GetSymbol() not in ATOM_LIST:
                    print('{}: {} is not in the Atom List.'.format(row['ID'], atom.GetSymbol()))
                    continue
            # check precursor type
            if row['Precursor_Type'] not in ADD_LIST:
                print('{}: {} is not in the Precusor Type List.'.format(row['ID'], row['Precursor_Type']))
                continue
            # check source instrument
            if row['Source_Instrument'] not in INST_LIST: 
                print('{}: {} is not in the Intrument List.'.format(row['ID'], row['Source_Instrument']))
                continue
            yield row.to_dict()
    else:
        raise Exception("Undefied data type: %s" % data_type)