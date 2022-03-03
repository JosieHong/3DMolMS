'''
Date: 2021-11-24 17:00:27
LastEditors: yuhhong
LastEditTime: 2022-12-09 15:35:50

Filter out the data (conditions):
1. organism
2. instrument
3. MS level
4. precursor type
5. atom 

e.g. 
python cond_filter_single.py --input ../data/Agilent/ALL_Agilent_clean.mgf --output ../data/Agilent/proc/ALL_Agilent_multi.mgf --log ../data/Agilent/proc/filterout_multi.json --cond agilent
python cond_filter_single.py --input ../data/Agilent/ALL_Agilent_clean.mgf --output ../data/Agilent/proc/ALL_Agilent_multi_nega.mgf --log ../data/Agilent/proc/filterout_multi_nega.json --cond agilent_neg
python cond_filter_single.py --input ../data/Agilent/ALL_Agilent_clean.mgf --output ../data/Agilent/proc/ALL_Agilent_multi_posi.mgf --log ../data/Agilent/proc/filterout_multi_posi.json --cond agilent_pos

python cond_filter_single.py --input ../data/NIST20/ALL_NIST_clean.mgf --output ../data/NIST20/proc/ALL_NIST_algilent_neg.mgf --log ../data/NIST20/proc/filterout_agilent_neg.json --cond nist_neg
python cond_filter_single.py --input ../data/NIST20/ALL_NIST_clean.mgf --output ../data/NIST20/proc/ALL_NIST_algilent_pos.mgf --log ../data/NIST20/proc/filterout_agilent_pos.json --cond nist_pos

python cond_filter_single.py --input ../data/GNPS_lib/LDB_NEGATIVE_clean.mgf --output ../data/GNPS_lib/proc/LDB_NEGATIVE_neg.mgf --log ../data/NIST20/proc/filterout_ldb_neg.json --cond gnps_neg
python cond_filter_single.py --input ../data/GNPS_lib/LDB_POSITIVE_clean.mgf --output ../data/GNPS_lib/proc/LDB_POSITIVE_neg.mgf --log ../data/NIST20/proc/filterout_ldb_pos.json --cond gnps_pos
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
    'agilent_qtof': {
        'Drop Organisms': [], 
        'Keep Instruments': ['esi-qtof'],
        'Keep Instruments Types': ['esi-qtof'],
        'MS Level': '2',
        'Atom Types': ['C', 'O', 'N', 'H', 'P', 'S', 'F', 'Cl', 'B', 'Br', 'I'],
        'Adduct Types': ['M+H', 'M-H', 'M+Na', 'M+H-H2O', 'M+2H'], 
    }, 
    'nist_qtof': {
        'Drop Organisms': [], 
        'Keep Instruments': ['agilent qtof 6530'],
        'Keep Instruments Types': ['q-tof'], 
        'MS Level': '2',
        'Atom Types': ['C', 'O', 'N', 'H', 'P', 'S', 'F', 'Cl', 'B', 'Br', 'I'],
        'Adduct Types': ['M+H', 'M-H', 'M+Na', 'M+H-H2O', 'M+2H'], 
    }, 
    'massbank_qtof': {
        'Drop Organisms': [], 
        'Keep Instruments': ['maxis ii hd q-tof bruker', 'qtof', 
                                'lc, waters acquity uplc system; ms, waters xevo g2 q-tof'
                                'waters xevo g2 q-tof', 'uplc q-tof premier, waters', 'q-tof premier, waters',
                                'agilent 1200 rrlc; agilent 6520 qtof', 
                                'ab sciex tripletof 5600+ system (q-tof) equipped with a duospray ion source',
                                'micromass q-tof ii', 'agilent qtof', '6550 qtof (agilent technologies)', 
                                ], 
        'Keep Instruments Types': ['unknown', 'lc-esi-qtof', 'lc-esi-qft', 'lc-q-tof/ms', 'lc-esi-tof', 
                                    'esi-qtof', 'q-tof'], 
        'MS Level': '2',
        'Atom Types': ['C', 'O', 'N', 'H', 'P', 'S', 'F', 'Cl', 'B', 'Br', 'I'],
        'Adduct Types': ['M+H', 'M-H', 'M+Na', 'M+H-H2O', 'M+2H'], 
    }, 

    'nist_hcd': {
        'Drop Organisms': [], 
        'Keep Instruments': ['thermo finnigan elite orbitrap',], 
        'Keep Instruments Types': ['hcd'], 
        'MS Level': '2',
        'Atom Types': ['C', 'O', 'N', 'H', 'P', 'S', 'F', 'Cl', 'B', 'Br', 'I'],
        'Adduct Types': ['M+H', 'M-H', 'M+Na', 'M+H-H2O', 'M+2H'], 
    }, 
    'massbank_hcd': {
        'Drop Organisms': [], 
        'Keep Instruments': ['q exactive plus orbitrap thermo scientific', 
                                'ltq orbitrap xl thermo scientific',
                                'q exactive orbitrap thermo scientific',
                                'ltq orbitrap velos thermo scientific',
                                'q-exactive + thermo scientific',
                                'q-exactive thermo scientific',
                                'ltq orbitrap xl hybrid iontrap-orbitrap (thermo fisher scientific, san jose, ca, usa)',
                                'q exactive orbitrap (thermo scientific)', 
                                'q exactive thermo fisher scientific',
                                ], 
        'Keep Instruments Types': ['lc-esi-itft', 'lc-esi-qft'], 
        'MS Level': '2',
        'Atom Types': ['C', 'O', 'N', 'H', 'P', 'S', 'F', 'Cl', 'B', 'Br', 'I'],
        'Adduct Types': ['M+H', 'M-H', 'M+Na', 'M+H-H2O', 'M+2H'], 
    }, 
}

# Record the filterout data
filtout_ms = {'UNRELIABEL_ORGAN': 0, 
                'DROP_INSTRU': 0, 
                'DROP_INSTRU_TYPE': 0, 
                'MS_LEVEL': 0, 
                'ATOM_TYPE': 0, 
                'PRECURSOR_TYPE': 0,
                'COLLISION_ENERGY': 0,
                'PEAK_NUM': 0,
                'PEAK_MAX': 0,
                'CHARGE': 0}
filtout_mol = {'UNRELIABEL_ORGAN': set(), 
                'DROP_INSTRU': set(), 
                'DROP_INSTRU_TYPE': set(),
                'MS_LEVEL': set(), 
                'ATOM_TYPE': set(),  
                'PRECURSOR_TYPE': set(),
                'COLLISION_ENERGY': set(),
                'PEAK_NUM': set(),
                'PEAK_MAX': set(),
                'CHARGE': set()}
OUTPUT_CNT = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess the Data')
    parser.add_argument('--input', type=str, default = '',
                        help='path to input data')
    parser.add_argument('--output', type=str, default = '',
                        help='path to output data')
    parser.add_argument('--log', type=str, default = '',
                        help='path to log') 
    parser.add_argument('--cond', type=str, default='agilent', 
                        choices=['agilent_qtof', 'nist_qtof', 'massbank_qtof', 'nist_hcd', 'massbank_hcd'],
                        help='the conditions for filter')
    args = parser.parse_args()
    
    # load the conditions
    conditions = COND[args.cond]
    print("\nPlease check the conditions!")
    pprint(conditions, compact=True)

    # make sure the output file is empty
    print("Init the output file!\n")
    OUT_DIR = "/".join(args.output.split("/")[:-1])
    if not os.path.exists(OUT_DIR): # Create a new directory because it does not exist
        os.makedirs(OUT_DIR)
        print("Create new directory: {}".format(OUT_DIR))
    mgf.write([], args.output, file_mode="w+")
    
    with mgf.read(args.input, read_charges=True) as reader:
        print("Got {} data from {}".format(len(reader), args.input))
        for idx, spectrum in enumerate(tqdm(reader)): 
            smiles = spectrum['params']['smiles'] # is has been cleaned in the `clean_up.py`
            mol = Chem.MolFromSmiles(smiles)

            # Filt by collision energy
            if spectrum['params']['collision_energy'].startswith('ramp'): # we can not process this type collision energy now
                continue

            # Filt by organism
            organism = spectrum['params']['organism'].lower()
            if organism in conditions['Drop Organisms']:
                filtout_ms['UNRELIABEL_ORGAN'] += 1
                filtout_mol['UNRELIABEL_ORGAN'].update({smiles})
                continue

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

            # Filt by precursor type
            precursor_type = spectrum['params']['precursor_type']
            if precursor_type not in conditions['Adduct Types']: 
                filtout_ms['PRECURSOR_TYPE'] += 1
                filtout_mol['PRECURSOR_TYPE'].update({smiles})
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

            # Output the data
            mgf.write([spectrum], args.output, file_mode="a+")
            OUTPUT_CNT += 1
    print("Done!\n")

    # Save the record
    for k in filtout_mol.keys(): 
        filtout_mol[k] = len(filtout_mol[k])
    print("For #MS")
    pprint(filtout_ms)
    print("For #MOL")
    pprint(filtout_mol)
    print("Output {} MS!".format(OUTPUT_CNT))
    with open(args.log, 'w+') as outfile: 
        json.dump({'MS': filtout_ms, 'MOL': filtout_mol}, outfile, indent=4)