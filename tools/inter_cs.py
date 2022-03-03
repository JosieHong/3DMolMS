'''
Date: 2022-12-07 14:18:26
LastEditors: yuhhong
LastEditTime: 2022-12-08 12:46:45
'''
import argparse

from rdkit import Chem
# suppress rdkit warning
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from pyteomics import mgf

import re
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm

'''
python inter_cs.py --input_mgf1 ../data/Agilent/proc/ALL_Agilent_multi_nega.mgf --input_mgf2 ../data/GNPS_lib/proc/LIB_GNPS_multi.mgf

python inter_cs.py --input_mgf1 ../data/Agilent/proc/ALL_Agilent_multi_nega.mgf --input_mgf2 ../data/NIST20/proc/ALL_NIST_multi_nega.mgf
python inter_cs.py --input_mgf1 ../data/Agilent/proc/ALL_Agilent_multi_posi.mgf --input_mgf2 ../data/NIST20/proc/ALL_NIST_multi_posi.mgf

python inter_cs.py --input_mgf1 ../data/Agilent/proc/ALL_Agilent_multi_nega.mgf --input_mgf2 ../data/MassBank/proc/ALL_MB_multi_nega.mgf
python inter_cs.py --input_mgf1 ../data/Agilent/proc/ALL_Agilent_multi_posi.mgf --input_mgf2 ../data/MassBank/proc/ALL_MB_multi_posi.mgf
'''

def cosine_similarity(A, B): 
    return np.dot(A,B)/(norm(A)*norm(B))

def pad_sepc(mz, intensity, length=int(1000/2), resolution=2): 
    mz = [float(i) for i in mz]
    intensity = [float(i) for i in intensity]
    
    # init mass spectra vector: add "0" to y data
    ms = [0] * length 

    # convert x, y to vector            
    for idx, val in enumerate(mz):
        val = int(round(val / resolution))
        ms[val] += intensity[idx]
    return ms, mz, intensity

def parse_collision_energy(ce_str):     
    ce = None
    
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
        r"^ramp [\d]+[.]?[\d]*-[\d]+[.]?[\d]* (ev|v)$":  lambda x: float((float(re.split(' |-', x)[1]) + float(re.split(' |-', x)[2])) /2), 
        r"^[\d]+[.]?[\d]*-[\d]+[.]?[\d]*$": lambda x: float((float(x.split('-')[0]) + float(x.split('-')[1])) /2), 
        r"^hcd[\d]+[.]?[\d]*$": lambda x: float(x.lstrip('hcd')), 
    }
    for k, v in matches_ev.items(): 
        if re.match(k, ce_str): 
            ce = v(ce_str)
            break
    
    if ce == None: 
        ce = 0
        
    return ce

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Preprocess the Data')
    parser.add_argument('--input_mgf1', type=str, default = '',
                        help='path to input the data')
    parser.add_argument('--input_mgf2', type=str, default = '',
                        help='path to input the data')
    args = parser.parse_args()

    # extract list of (id, smiles, ce, precursor_type) from input_mgf1
    mols = []
    for idx, spec in enumerate(tqdm(mgf.read(args.input_mgf1))): 
        clean_id = spec['params']['clean_id']
        smiles = spec['params']['smiles']
        smiles = Chem.CanonSmiles(smiles)
        ce = parse_collision_energy(spec['params']['collision_energy'].lower())
        precursor_type = spec['params']['precursor_type']
        mols.append((clean_id, smiles, ce, precursor_type))
    print("Load {} molecules from {}".format(len(mols), args.input_mgf1))

    # extract the interaction between input_mgf1 and input_mgf2, (id1, id2, smiles, ce, precursor_type)
    # at the same time, save the spectra of interacted molecules from input_mgf2
    inter_mols = []
    ms2 = {}
    for idx, spec in enumerate(tqdm(mgf.read(args.input_mgf2))): 
        clean_id = spec['params']['clean_id']
        smiles = spec['params']['smiles']
        smiles = Chem.CanonSmiles(smiles)
        ce = parse_collision_energy(spec['params']['collision_energy'].lower())
        precursor_type = spec['params']['precursor_type']

        for mol in mols:
            if mol[1] == smiles and mol[2] == ce and mol[3] == precursor_type: 
                inter_mols.append((mol[0], clean_id, smiles, ce, precursor_type))
                if clean_id in ms2.keys():
                    ms2[clean_id].append({'x': spec['m/z array'].tolist(), 
                                        'y': spec['intensity array'].tolist()})
                else:
                    ms2[clean_id] = [{'x': spec['m/z array'].tolist(), 
                                        'y': spec['intensity array'].tolist()}]
    print("Get {} interacted molecules from {}".format(len(inter_mols), args.input_mgf2))

    # save the spectra of interacted molecules from input_mgf1
    inter_id_list = [mol[0] for mol in inter_mols]
    ms1 = {}
    for idx, spec in enumerate(tqdm(mgf.read(args.input_mgf1))):
        clean_id = spec['params']['clean_id']
        smiles = spec['params']['smiles']
        smiles = Chem.CanonSmiles(smiles)
        ce = parse_collision_energy(spec['params']['collision_energy'].lower())
        precursor_type = spec['params']['precursor_type']

        if clean_id in inter_id_list:
            if clean_id in ms1.keys():
                ms1[clean_id].append({'x': spec['m/z array'].tolist(), 
                                        'y': spec['intensity array'].tolist()})
            else:
                ms1[clean_id] = [{'x': spec['m/z array'].tolist(), 
                                        'y': spec['intensity array'].tolist()}]
        
    # calculate inter-cosine similarity
    inter_cs = []
    for idx, mol in enumerate(tqdm(inter_mols)): 
        # average spectra from input_mgf1
        avg_ms1 = []
        id1 = mol[0]
        for spec in ms1[id1]: 
            ms, _, _ = pad_sepc(spec['x'], spec['y'], length=int(1500/0.2), resolution=0.2)
            avg_ms1.append(ms)
        avg_ms1 = np.mean(np.array(avg_ms1), axis=0)

        # average spectra from input_mgf2
        avg_ms2 = []
        id2 = mol[1]
        for spec in ms2[id2]: 
            ms, _, _ = pad_sepc(spec['x'], spec['y'], length=int(1500/0.2), resolution=0.2)
            avg_ms2.append(ms)
        avg_ms2 = np.mean(np.array(avg_ms2), axis=0)

        inter_cs.append(cosine_similarity(avg_ms1, avg_ms2))

    # inter_cs = []
    # for idx, mol in enumerate(tqdm(inter_mols)): 
    #     # average spectra from input_mgf1
    #     avg_ms1 = []
    #     for spec in mgf.read(args.input_mgf1): 
    #         if spec['params']['clean_id'] == mol[0]:
    #             x = spec['m/z array'].tolist()
    #             y = spec['intensity array'].tolist()
    #             ms, _, _ = pad_sepc(x, y, length=int(1500/0.2), resolution=0.2)
    #             avg_ms1.append(ms)
    #     avg_ms1 = np.mean(np.array(avg_ms1), axis=0)

    #     # average spectra from input_mgf2
    #     avg_ms2 = []
    #     for spec in mgf.read(args.input_mgf2): 
    #         if spec['params']['clean_id'] == mol[1]: 
    #             x = spec['m/z array'].tolist()
    #             y = spec['intensity array'].tolist()
    #             ms, _, _ = pad_sepc(x, y, length=int(1500/0.2), resolution=0.2)
    #             avg_ms2.append(ms)
    #     avg_ms2 = np.mean(np.array(avg_ms2), axis=0)

    #     inter_cs.append(cosine_similarity(avg_ms1, avg_ms2))
    print('Average inter-cosine similarity: {}'.format(np.mean(np.array(inter_cs))))
