'''
Date: 2021-11-24 17:00:27
LastEditors: yuhhong
LastEditTime: 2022-12-09 15:35:50
'''

'''Notes
I. Filter out the data (conditions):
    1. organism
    2. instrument
    3. MS level
    4. precursor type
    5. atom 

II. Split into training and validation randomly
'''
import os
import argparse
import json
from tqdm import tqdm
from pprint import pprint

from rdkit import Chem
# ignore the warning
from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*')

import numpy as np
from pyteomics import mgf



# ------------------------------------
# Please check the conditions: 
# ------------------------------------
COND = {
    'agilent': {
        'Keep Instruments': ['esi-qtof'],
        'Keep Instruments Types': ['esi-qtof'],
        'MS Level': '2',
        'Atom Types': ['C', 'O', 'N', 'H', 'P', 'S', 'F', 'Cl', 'B', 'Br', 'I'],
        'Adduct Types': ['M+H', 'M-H', 'M+Na', 'M', 'M+NH4', 'M+Cl', 'M+COOH', 'M+H-H2O', 'M+2H'], 
    }, 
    'agilent_pos': {
        'Keep Instruments': ['esi-qtof'],
        'Keep Instruments Types': ['esi-qtof'],
        'MS Level': '2',
        'Atom Types': ['C', 'O', 'N', 'H', 'P', 'S', 'F', 'Cl', 'B', 'Br', 'I'],
        'Adduct Types': ['M+H'], 
    }, 
    'agilent_neg': {
        'Keep Instruments': ['esi-qtof'],
        'Keep Instruments Types': ['esi-qtof'],
        'MS Level': '2',
        'Atom Types': ['C', 'O', 'N', 'H', 'P', 'S', 'F', 'Cl', 'B', 'Br', 'I'],
        'Adduct Types': ['M-H'], 
    }, 
}

def filter(in_spec, conditions):
    # Record the filterout data
    filtout_ms = {'DROP_INSTRU': 0, 
                    'DROP_INSTRU_TYPE': 0, 
                    'MS_LEVEL': 0, 
                    'ATOM_TYPE': 0, 
                    'PRECURSOR_TYPE': 0,
                    'PEAK_NUM': 0,
                    'PEAK_MAX': 0}
    filtout_mol = {'DROP_INSTRU': set(), 
                    'DROP_INSTRU_TYPE': set(),
                    'MS_LEVEL': set(), 
                    'ATOM_TYPE': set(),  
                    'PRECURSOR_TYPE': set(),
                    'PEAK_NUM': set(),
                    'PEAK_MAX': set()}

    out_spec = []
    out_mol = []
    for idx, spectrum in enumerate(tqdm(in_spec)): 
        smiles = spectrum['params']['smiles'] # is has been cleaned in the `clean_up.py`
        mol = Chem.MolFromSmiles(smiles)

        # Filt by instrument type
        instrument_type = spectrum['params']['instrument_type'].lower()
        if instrument_type not in conditions['Keep Instruments Types']: 
            filtout_ms['DROP_INSTRU_TYPE'] += 1
            filtout_mol['DROP_INSTRU_TYPE'].update({smiles})
            continue

        # Filt by instrument
        instrument = spectrum['params']['source_instrument'].lower()
        if instrument not in conditions['Keep Instruments']:
            filtout_ms['DROP_INSTRU'] += 1
            filtout_mol['DROP_INSTRU'].update({smiles})
            continue

        # Filt by mslevel
        mslevel = spectrum['params']['mslevel']
        if mslevel != conditions['MS Level']:
            filtout_ms['MS_LEVEL'] += 1
            filtout_mol['MS_LEVEL'].update({smiles})
            continue

        # Filt by atom type 
        is_compound_countain_rare_atom = False 
        for i in range(mol.GetNumAtoms()):
            a = mol.GetAtomWithIdx(i).GetSymbol()
            if a not in conditions['Atom Types']:
                is_compound_countain_rare_atom = True
                break
        if is_compound_countain_rare_atom: 
            filtout_ms['ATOM_TYPE'] += 1
            filtout_mol['ATOM_TYPE'].update({smiles})
            continue

        # Filt by peak number
        if len(spectrum['m/z array']) < 5: 
            filtout_ms['PEAK_NUM'] += 1
            filtout_mol['PEAK_NUM'].update({smiles})
            continue
        # Filter by max m/z
        if np.max(spectrum['m/z array']) < 50 or np.max(spectrum['m/z array']) > 1500: 
            filtout_ms['PEAK_MAX'] += 1
            filtout_mol['PEAK_MAX'].update({smiles})
            continue

        # Filt by precursor type
        precursor_type = spectrum['params']['precursor_type']
        if precursor_type not in conditions['Adduct Types']: 
            filtout_ms['PRECURSOR_TYPE'] += 1
            filtout_mol['PRECURSOR_TYPE'].update({smiles})
            continue

        # Output the data
        out_spec.append(spectrum)
        out_mol.append(smiles)
    return out_spec, out_mol, filtout_ms, filtout_mol

def spliter(in_spec, test_mol):
    train_spec = []
    test_spec = []
    for idx, spectrum in enumerate(tqdm(in_spec)): 
        smiles = spectrum['params']['smiles'] # is has been cleaned in the `clean_up.py`
        if smiles in test_mol:
            test_spec.append(spectrum)
        else:
            train_spec.append(spectrum)
    return train_spec, test_spec



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare the data')
    parser.add_argument('--input', type=str, default = '',
                        help='path to input data')
    parser.add_argument('--output_dir', type=str, default = '',
                        help='dir to output data')
    parser.add_argument('--log', type=str, default = '',
                        help='path to log') 
    parser.add_argument('--cond', type=str, default='agilent', 
                        choices=['agilent'],
                        help='the conditions for filter')
    args = parser.parse_args()
    
    assert args.cond in COND.keys() and \
            args.cond+'_neg' in COND.keys() and \
            args.cond+'_pos' in COND.keys()

    # load the conditions
    conditions = COND[args.cond]
    conditions_neg = COND[args.cond+'_neg']
    conditions_pos = COND[args.cond+'_pos']
    print("\nPlease check the conditions!")
    print("primary conditions: ")
    pprint(conditions, compact=True)
    print("secondary conditions: ")
    pprint(conditions_neg, compact=True)
    pprint(conditions_pos, compact=True)

    # init the output file
    if not os.path.exists(args.output_dir): # Create a new directory because it does not exist
        os.makedirs(args.output_dir)
    print("Create new directory: {}".format(args.output_dir))
    
    # filter out the data
    records = {}
    with mgf.read(args.input, read_charges=True) as reader:
        print("Got {} data from {}".format(len(reader), args.input))
        primary_out, primary_mol, filtout_ms, filtout_mol = filter(reader, conditions)
        for k in filtout_mol.keys(): # convert smiles set to smiles count
            filtout_mol[k] = len(filtout_mol[k])
        records['primary'] = {'MS': filtout_ms, 'MOL': filtout_mol}
    with mgf.read(args.input, read_charges=True) as reader:
        negative_out, negative_mol, filtout_ms, filtout_mol = filter(reader, conditions_neg)
        for k in filtout_mol.keys(): # convert smiles set to smiles count
            filtout_mol[k] = len(filtout_mol[k])
        records['negative'] = {'MS': filtout_ms, 'MOL': filtout_mol}
    with mgf.read(args.input, read_charges=True) as reader:
        positive_out, positive_mol, filtout_ms, filtout_mol = filter(reader, conditions_pos)
        for k in filtout_mol.keys(): # convert smiles set to smiles count
            filtout_mol[k] = len(filtout_mol[k])
        records['positive'] = {'MS': filtout_ms, 'MOL': filtout_mol}
    # save log
    with open(args.log, 'w+') as outfile: 
        json.dump(records, outfile, indent=4)

    primary_mol = list(set(primary_mol))
    negative_mol = list(set(negative_mol))
    positive_mol = list(set(positive_mol))
    print("primary: # spec: {} # mol: {}".format(len(primary_out), len(primary_mol)))
    print("negative: # spec: {} # mol: {}".format(len(negative_out), len(negative_mol)))
    print("positive: # spec: {} # mol: {}".format(len(positive_out), len(positive_mol)))

    # split into training and test randomly
    # make sure that no overlap molecules between training and test
    # count the overlap compound
    test_mol_idx = np.random.choice(len(primary_mol), int(len(primary_mol)*0.1))
    test_mol = [s for idx, s in enumerate(primary_mol) if idx in test_mol_idx]
    print("primary: train # mol; {}, test # mol: {}".format(len(primary_mol)-len(test_mol), len(test_mol)))

    negative_test_mol = [s for s in negative_mol if s in test_mol]
    positive_test_mol = [s for s in positive_mol if s in test_mol]
    print("negative: train # mol; {}, test # mol: {}".format(len(negative_mol)-len(negative_test_mol), len(negative_test_mol)))
    print("positive: train # mol; {}, test # mol: {}".format(len(positive_mol)-len(positive_test_mol), len(positive_test_mol)))

    print("\nOutput the spectra: ")
    file_name = args.input.split('/')[-1].replace('.mgf', '')
    
    train_spec, test_spec = spliter(primary_out, test_mol)
    mgf.write(train_spec, os.path.join(args.output_dir, file_name+'_primary_train.mgf'), file_mode="w")
    mgf.write(test_spec, os.path.join(args.output_dir, file_name+'_primary_test.mgf'), file_mode="w")

    train_spec, test_spec = spliter(negative_out, negative_test_mol)
    mgf.write(train_spec, os.path.join(args.output_dir, file_name+'_negative_train.mgf'), file_mode="w")
    mgf.write(test_spec, os.path.join(args.output_dir, file_name+'_negative_test.mgf'), file_mode="w")

    train_spec, test_spec = spliter(positive_out, positive_test_mol)
    mgf.write(train_spec, os.path.join(args.output_dir, file_name+'_positive_train.mgf'), file_mode="w")
    mgf.write(test_spec, os.path.join(args.output_dir, file_name+'_positive_test.mgf'), file_mode="w")
    print("Done!\n")
