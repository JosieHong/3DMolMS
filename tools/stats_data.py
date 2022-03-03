'''
Date: 2021-12-01 13:05:05
LastEditors: yuhhong
LastEditTime: 2022-12-08 14:42:21
'''
import os
import argparse
from tqdm import tqdm

from pyteomics import mgf
from rdkit import Chem
import numpy as np



def stat(file_name, ion_mode):
    if ion_mode == 'P':
        ion_mode = ['p', 'positive']
    elif ion_mode == 'N':
        ion_mode = ['n', 'negative']
    else:
        ion_mode = ['p', 'positive', 'n', 'negative']

    smiles = set()
    atom_num = []
    bond_num = []
    
    atom_type = {}
    precursor_type = {}
    instrument = {}
    instrument_type = {}
    collision_energy = {}

    atom_type_mol = {}
    precursor_type_mol = {}
    instrument_mol = {}
    instrument_type_mol = {}
    collision_energy_mol = {}
    
    spec_cnt = 0
    with mgf.read(file_name, read_charges=False) as reader: 
        print("Got {} data from {}".format(len(reader), file_name))
        for idx, spectrum in enumerate(tqdm(reader)): 
            if spectrum['params']['ionmode'].lower() not in ion_mode:
                continue

            spec_cnt += 1
            s = spectrum['params']['smiles']
            s = Chem.CanonSmiles(s)
            smiles.add(s)

            mol = Chem.MolFromSmiles(s)
            if mol is None:
                continue
            mol = Chem.AddHs(mol)

            atom_num.append(len(mol.GetAtoms()))
            bond_num.append(len(mol.GetBonds()))

            atom_list = list(set([atom.GetSymbol() for atom in mol.GetAtoms()]))
            for atom in atom_list:
                if atom not in atom_type.keys():
                    atom_type[atom] = 1
                    atom_type_mol[atom] = set(s)
                else:
                    atom_type[atom] += 1
                    atom_type_mol[atom].add(s)

            add = spectrum['params']['precursor_type']
            if add not in precursor_type.keys():
                precursor_type[add] = 1
                precursor_type_mol[add] = set(s)
            else:
                precursor_type[add] += 1
                precursor_type_mol[add].add(s)
            
            instr = spectrum['params']['source_instrument'].lower()
            if instr not in instrument.keys():
                instrument[instr] = 1
                instrument_mol[instr] = set(s)
            else: 
                instrument[instr] += 1
                instrument_mol[instr].add(s)

            instr_type = spectrum['params']['instrument_type'].lower()
            if instr_type not in instrument_type.keys():
                instrument_type[instr_type] = 1
                instrument_type_mol[instr_type] = set(s)
            else: 
                instrument_type[instr_type] += 1
                instrument_type_mol[instr_type].add(s)
            
            ce = spectrum['params']['collision_energy'].lower()
            if ce not in collision_energy.keys():
                collision_energy[ce] = 1
                collision_energy_mol[ce] = set(s)
            else: 
                collision_energy[ce] += 1
                collision_energy_mol[ce].add(s)

        atom_num = np.array(atom_num)
        bond_num = np.array(bond_num)
        print("\n====>> Statistics Summary of {}\n".format(file_name)) 
        print("# MS: {}, # MOL: {}\n".format(spec_cnt, len(list(smiles))))
        print("# Atom: mean: {}, max: {}".format(np.mean(atom_num), np.max(atom_num)))
        print("# Bond: mean: {}, max: {}\n".format(np.mean(bond_num), np.max(bond_num)))
        
        print('# MS in different conditions:')
        print("Atom Types: {}".format(sorted(atom_type.items(), key=lambda x: x[1], reverse=True)))
        print("Precursor Types: {}".format(sorted(precursor_type.items(), key=lambda x: x[1], reverse=True)))
        print("Instruments: {}".format(sorted(instrument.items(), key=lambda x: x[1], reverse=True)))
        print("Instrument Types: {}".format(sorted(instrument_type.items(), key=lambda x: x[1], reverse=True)))
        print("Collision Energy: {}\n".format(sorted(collision_energy.items(), key=lambda x: x[1], reverse=True)))

        print('# MOL in different conditions:')
        atom_type_mol = {k:len(list(v)) for k,v in atom_type_mol.items()}
        precursor_type_mol = {k:len(list(v)) for k,v in precursor_type_mol.items()}
        instrument_mol = {k:len(list(v)) for k,v in instrument_mol.items()}
        instrument_type_mol = {k:len(list(v)) for k,v in instrument_type_mol.items()}
        collision_energy_mol = {k:len(list(v)) for k,v in collision_energy_mol.items()}
        print("Atom Types: {}".format(sorted(atom_type_mol.items(), key=lambda x: x[1], reverse=True)))
        print("Precursor Types: {}".format(sorted(precursor_type_mol.items(), key=lambda x: x[1], reverse=True)))
        print("Instruments: {}".format(sorted(instrument_mol.items(), key=lambda x: x[1], reverse=True)))
        print("Instrument Types: {}".format(sorted(instrument_type_mol.items(), key=lambda x: x[1], reverse=True)))
        print("Collision Energy: {}\n".format(sorted(collision_energy_mol.items(), key=lambda x: x[1], reverse=True)))
        print("====>>\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess the Data')
    parser.add_argument('--input', type=str, default = '',
                        help='path to input data')
    parser.add_argument('--ion_mode', type=str, default='P', choices=['P', 'N', 'ALL'], 
                        help='Ion mode used for training and test') 
    args = parser.parse_args()

    if os.path.isdir(args.input): 
        file_names = [f for f in os.listdir(args.input) if f.endswith('.mgf')]
        for file_name in file_names: 
            stat(os.path.join(args.input, file_name), ion_mode=args.ion_mode)
    else:
        stat(args.input, ion_mode=args.ion_mode)
