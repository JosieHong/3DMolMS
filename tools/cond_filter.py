'''
Date: 2021-11-24 17:00:27
LastEditors: yuhhong
LastEditTime: 2022-05-22 13:53:45

Filter Out the Data (conditions):
1. organism
2. instrument
3. MS level
4. precursor type
5. atom 
6. collision energy

e.g. 
python cond_filter.py --input ../data/GNPS/ALL_GNPS_clean.mgf --output_dir ../data/GNPS/tmp/ --log ../data/GNPS/proc/filterout.json --dataset_name gnps

python cond_filter.py --input ../data/NIST20/ALL_NIST_clean.mgf --output_dir ../data/NIST20/tmp/ --log ../data/NIST20/proc/filterout.json --dataset_name nist

python cond_filter.py --input ../data/MassBank/ALL_MB_clean.mgf --output_dir ../data/MassBank/tmp/ --log ../data/MassBank/proc/filterout.json --dataset_name massbank
'''
import os
import argparse
import json
from tqdm import tqdm
from pprint import pprint
import numpy as np

from rdkit import Chem
# ignore the warning
from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*')

from pyteomics import mgf


# Please check the conditions
# Consider the pepmass
DROP_ORGA1 = ['gnps-selleckchem-fda-part1', 'gnps-selleckchem-fda-part2', 'gnps-nih-smallmoleculepharmacologicallyactive', 'gnps-collections-pesticides-negative', 'gnps-iimn-propogated']
DROP_ORGA2 = []
DROP_ORGA3 = []

# -------------------------- GNPS -------------------------- # 
KEEP_INST1 = {
    # 'IonTRAP': ['lc-esi-ion trap', 'esi-ion trap'], 
    # 'Orbitrap': ['esi-orbitrap', 'lc-esi-orbitrap'], 
    # 'FT': ['esi-hybrid ft', 'esi-lc-esi-itft', 'lc-esi-hybrid ft', 'esi-esi-itft', 'esi-apci-itft', 'esi-lc-esi-qft'], 
    # 'QTOF': ['esi-qtof', 'lc-esi-qtof', 'esi-lc-esi-qtof', 'esi-lc-q-tof/ms', 'esi-q-tof'], 
    # 'QQQ': ['esi-qqq', 'esi-flow-injection qqq/ms', 'esi-lc-esi-qq', 'esi-lc-esi-q', 'lc-esi-qqq'], 
    'Multi': ['esi-qtof', 'lc-esi-qtof', 'esi-lc-esi-qtof', 'esi-lc-q-tof/ms', 'esi-q-tof', 
                'esi-qqq', 'esi-flow-injection qqq/ms', 'esi-lc-esi-qq', 'esi-lc-esi-q', 'lc-esi-qqq']
}
#  [('esi-orbitrap', 222727), ('esi-hybrid ft', 17658), ('esi-qtof', 16933), ('lc-esi-qtof', 10817), ('esi-qqq', 6518), ('esi-lc-esi-itft', 4757), ('esi-flow-injection qqq/ms', 3632), ('lc-esi-maxis ii hd q-tof bruker', 3589), ('lc-esi-ion trap', 3036), ('lc-esi-orbitrap', 2366), ('esi-lc-esi-qtof', 1962), ('esi-ion trap', 1889), ('esi-lc-esi-qq', 1623), ('esi-lc-esi-qft', 1075), ('esi-lc-q-tof/ms', 892), ('positive-quattro_qqq:10ev', 469), ('positive-quattro_qqq:25ev', 467), ('positive-quattro_qqq:40ev', 463), ('lc-esi-q-exactive plus', 400), ('lc-esi- impact hd', 384), ('di-esi-qtof', 337), ('-q-exactive plus orbitrap res 70k', 141), ('esi-lc-appi-qq', 140), ('esi-lc-esi-it', 120), ('lc-esi-q-exactive plus orbitrap res 70k', 119), ('-maxis hd qtof', 117), ('lc-esi-maxis hd qtof', 116), ('-q-exactive plus orbitrap res 14k', 110), ('negative-quattro_qqq:10ev', 91), ('negative-quattro_qqq:25ev', 90), ('negative-quattro_qqq:40ev', 88), ('lc-esi-q-exactive plus orbitrap res 14k', 81), ('lc-esi-hybrid ft', 54), ('esi-esi-itft', 42), ('esi-apci-itft', 42), ('di-esi-ion trap', 40), ('esi-lc-esi-q', 18), ('di-esi-hybrid ft', 16), ('esi-q-tof', 14), ('lc-esi-qqq', 12), ('esi-lc-esi-ittof', 11), ('esi-fab-ebeb', 8), ('apci-ion trap', 4), ('lc-apci-qtof', 3), ('esi-esi-fticr', 3), ('esi-uplc-esi-qtof', 3), ('esi-hplc-esi-tof', 3), ('di-esi-orbitrap', 1), ('di-esi-qqq', 1)]
# -------------------------- NIST -------------------------- # 
KEEP_INST2 = {
    # 'IonTRAP': ['it/ion trap'], 
    'HCD': ['hcd'], 
    # 'FT': ['it-ft/ion trap with ftms'], 
    # 'QQQ': ['qqq', 'qqq/triple quadrupole'], 
    # 'QTOF': ['q-tof'], 
    'Multi': ['q-tof', 'qqq', 'qqq/triple quadrupole']
}
# [('hcd', 436029), ('it-ft/ion trap with ftms', 37669), ('q-tof', 25920), ('qqq', 1535), ('it/ion trap', 1019), ('unknown', 56), ('qqit', 40)]
# -------------------------- MassBank -------------------------- #
KEEP_INST3 = {
    'HCD': ['lc-esi-qft hcd', 'lc-esi-itft hcd'], 
    # 'FT': ['lc-esi-itft cid', 'lc-esi-itft', 'lc-esi-qft n/a'], 
    # 'QTOF': ['lc-esi-qtof n/a', 'lc-esi-qtof cid', 'lc-esi-qtof', 'lc-q-tof/ms n/a', 'lc-esi-tof n/a'], 
    # 'QQQ': ['lc-esi-qq', 'lc-esi-qqq n/a'], 
    # 'Unknow': ['unknown n/a'], 
    'Multi': ['lc-esi-qtof n/a', 'lc-esi-qtof cid', 'lc-esi-qtof', 'lc-q-tof/ms n/a', 'lc-esi-tof n/a', 
                'lc-esi-qq', 'lc-esi-qqq n/a', 'unknown n/a']
}
# [('lc-esi-qft hcd', 11904), ('unknown n/a', 8075), ('lc-esi-qtof n/a', 7336), ('lc-esi-itft hcd', 6691), ('lc-esi-qtof cid', 4615), ('lc-esi-qq', 3624), ('lc-esi-qtof', 2690), ('lc-esi-itft cid', 2080), ('lc-esi-itft', 1365), ('lc-q-tof/ms n/a', 852), ('lc-esi-qft n/a', 719), ('lc-esi-qqq n/a', 488), ('lc-esi-tof n/a', 352), ('lc-esi-qit', 276), ('lc-esi-qq n/a', 266), ('lc-qtof n/a', 217), ('lc-esi-it', 198), ('lc-appi-qq', 140), ('lc-esi-qtof hcd', 120), ('esi-qtof cid', 71), ('esi-itft cid', 37), ('apci-itft cid', 34), ('lc-esi-ittof', 23), ('lc-esi-q cid', 18), ('lc-apci-itft n/a', 6), ('esi-itft', 4), ('apci-itft n/a', 3), ('lc-esi-tof cid', 3), ('esi-qtof', 1)]

KEEP_MS_LEVEL = '2'
KEEP_ATOM = ['C', 'O', 'N', 'H', 'P', 'S', 'F', 'Cl', 'B', 'Br', 'I']
KEEP_PRE_TYPE1 = ['M+H', 'M+Na', 'M-H', 'M+NH4', 'M-H2O+H', 'M+HCOO', 'M+H-H2O', 'M+K', 'M+Cl', 'M+H-2H2O'] 
# GNPS
# [('M+H', 164676), ('M+Na', 110667), ('M-H', 24296), ('M+NH4', 1033), ('M-H2O+H', 897), ('M+HCOO', 844), ('M+H-H2O', 405), ('M+K', 193), ('M-2H2O+H', 126), ('M+Cl', 52), ('M-H+CH3OH', 52), ('M-H2O-H', 33), ('M+H+CH3OH', 28), ('M+H-2H2O', 27), ('2M+Ca', 27), ('M-H+2Na', 22), ('M-H+HCOOH', 21), ('M-2H', 15), ('M+Ca-H', 11), ('M-2H+Na', 9), ('M+K-2H', 7), ('M-H-H2O', 6), ('M+2H', 4), ('M+NH3', 4), ('M+H+HCOOH', 4), ('M-2H+K', 3), ('M+H+CH3CN', 3), ('M+Li', 3), ('2M+H', 2), ('M-H+Li', 2), ('M-H+Na', 2), ('M+H-H20', 1), ('M-H2O', 1), ('M-2H2O+NH4', 1), ('M-CO2-H', 1), ('M+H-NH3', 1), ('M-C2H3O', 1), ('2M-H', 1), ('M-H2O+NH4', 1)]
KEEP_PRE_TYPE2 = ['M+H', 'M-H', 'M+H-H2O', 'M+Na', 'M+H-NH3', 'M+H-2H2O', 'M-H-H2O', 'M+NH4', 'M+Cl', 'M+H-CH4O', 'M+H-C2H6O', 'M+2Na-H', 'M+OH', 'M+K', 'M-H-NH3', 'M+2H', 'M+H-CH4', 'M+H+H2O', 'M+H-3H2O', '2M+H']
# NIST 
# [('M+H', 360767), ('M-H', 130640), ('M+H-H2O', 81991), ('M+Na', 35643), ('M+H-NH3', 17541), ('M+H-2H2O', 11758), ('M-H-H2O', 10467), ('M+NH4', 6299), ('M+Cl', 5700), ('M+H-CH4O', 5133), ('M+H-C2H6O', 3314), ('M+2Na-H', 2124), ('M+OH', 1622), ('M+K', 742), ('M-H-NH3', 688), ('M+2H', 665), ('M+H-CH4', 358), ('M+H+H2O', 283), ('M+H-3H2O', 268), ('2M+H', 237), ('M+Li', 122), ('M+H+O', 103), ('M+Na-2H', 73), ('2M-H', 57), ('3M+H', 20), ('2M+Na', 14), ('M-2H', 13), ('M-2H+K', 12), ('M-CH3', 2), ('M+H+CH3CN', 1), ('M-2H+Na', 1), ('M+NH4-H2O', 1), ('M-H+2Na', 1), ('M-H-C3H5NO2', 1)]
KEEP_PRE_TYPE3 = ['M+H', 'M-H', 'M+Na', 'M+NH4', 'M-H2O+H', 'M+K', 'M+HCOO', 'M+H-H2O']
# MassBank
# [('M+H', 31610), ('M-H', 17131), ('M+Na', 1142), ('M+NH4', 946), ('M-H2O+H', 767), ('M+K', 166), ('M+HCOO', 139), ('M-2H2O+H', 63), ('M+Cl', 46), ('M+H-H2O', 33), ('M-H2O-H', 30), ('M+CH3OH+H', 28), ('2M+Ca', 24), ('M-H+2Na', 20), ('M+K-2H', 15), ('M-2H', 12), ('M+Ca-H', 11), ('M+Na-2H', 9), ('M+2H', 8), ('M+H+CH3CN', 3), ('2M+H', 1), ('M-C2H3O', 1), ('M-CO2-H', 1), ('M+H-NH3', 1), ('M-2H2O+NH4', 1)]

output_data = {}

# Record the filterout data
filtout_ms = {'UNRELIABEL_ORGAN': 0, 
                'UNRELIABEL_INSTRU': 0, 
                'MS_LEVEL': 0, 
                'ATOM_TYPE': 0, 
                'PRECURSOR_TYPE': 0,
                'COLLISION_ENERGY': 0,
                'PEAK_NUM': 0,
                'PEAK_MAX': 0,
                'CHARGE': 0}
filtout_mol = {'UNRELIABEL_ORGAN': set(), 
                'UNRELIABEL_INSTRU': set(),
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
    parser.add_argument('--output_dir', type=str, default = '',
                        help='dir to output data')
    parser.add_argument('--log', type=str, default = '',
                        help='path to log') 
    parser.add_argument('--dataset_name', type=str, default = '',
                        help='the name of the dataset')
    # parser.add_argument('--precursor_type', type=str, default = '',
    #                     help='Precursor type') 
    args = parser.parse_args()
    
    # KEEP_PRE_TYPE = args.precursor_type
    # assert KEEP_PRE_TYPE == 'M-H' or KEEP_PRE_TYPE == 'M+H' # we only process these types now
    
    if args.dataset_name == 'gnps':
        DROP_ORGA = DROP_ORGA1
        KEEP_INST = KEEP_INST1
        KEEP_INST_LIST = [ins for ins_list in KEEP_INST.values() for ins in ins_list]
        KEEP_PRE_TYPE = KEEP_PRE_TYPE1
    elif args.dataset_name == 'nist':
        DROP_ORGA = DROP_ORGA2
        KEEP_INST = KEEP_INST2
        KEEP_INST_LIST = [ins for ins_list in KEEP_INST.values() for ins in ins_list]
        KEEP_PRE_TYPE = KEEP_PRE_TYPE2
    elif args.dataset_name == 'massbank':
        DROP_ORGA = DROP_ORGA3
        KEEP_INST = KEEP_INST3
        KEEP_INST_LIST = [ins for ins_list in KEEP_INST.values() for ins in ins_list]
        KEEP_PRE_TYPE = KEEP_PRE_TYPE3
    else:
        print("Please check the name of the dataset. It should be gnps or nist.")
        exit()

    print("\nPlease check the conditions!")
    conditions = {'Drop Organisms': DROP_ORGA, 
                    'Keep Instruments': KEEP_INST,
                    'MS Level': KEEP_MS_LEVEL,
                    'Atom Types': KEEP_ATOM,
                    'Adduct Types': KEEP_PRE_TYPE}
    pprint(conditions, compact=True)

    # make sure the output direction exists
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # make sure the output file is empty
    print("\nInit the output file!")
    output_files = [os.path.join(args.output_dir, args.dataset_name+'_'+str(key).lower()+'.mgf') for key in KEEP_INST.keys()]
    for f in output_files: 
        print(f)
        mgf.write([], f, file_mode="w+")
    print("Done!\n")
    
    with mgf.read(args.input, read_charges=True) as reader:
        print("Got {} data from {}".format(len(reader), args.input))
        for idx, spectrum in enumerate(tqdm(reader)): 
            # J0sie: remove this lines later
            if len(spectrum['m/z array']) == 0:
                continue
            
            smiles = spectrum['params']['smiles']
            mol = Chem.MolFromSmiles(smiles)
            # J0sie: remove this lines later
            smiles = Chem.MolToSmiles(mol) # normalize the smiles
            spectrum['params']['smiles'] = smiles

            # Filt by organism
            organism = spectrum['params']['organism'].lower()
            if organism in DROP_ORGA:
                filtout_ms['UNRELIABEL_ORGAN'] += 1
                filtout_mol['UNRELIABEL_ORGAN'].update({organism})
                continue

            # Filt by instrument
            instrument = spectrum['params']['source_instrument'].lower()
            if instrument not in KEEP_INST_LIST: 
                filtout_ms['UNRELIABEL_INSTRU'] += 1
                filtout_mol['UNRELIABEL_INSTRU'].update({organism})
                continue
            else: 
                for key, ins_list in KEEP_INST.items():
                    if instrument in ins_list: 
                        inst_type = key
                        break
                output_file = os.path.join(args.output_dir, args.dataset_name+'_'+inst_type.lower()+'.mgf')

            # Filt by mslevel
            mslevel = spectrum['params']['mslevel']
            if mslevel != KEEP_MS_LEVEL:
                filtout_ms['MS_LEVEL'] += 1
                filtout_mol['MS_LEVEL'].update({smiles})
                continue
            
            # Filt by atom type 
            is_compound_countain_rare_atom = False 
            for i in range(mol.GetNumAtoms()):
                a = mol.GetAtomWithIdx(i).GetSymbol()
                if a not in KEEP_ATOM:
                    is_compound_countain_rare_atom = True
                    break
            if is_compound_countain_rare_atom: 
                filtout_ms['ATOM_TYPE'] += 1
                filtout_mol['ATOM_TYPE'].update({smiles})
                continue

            # Filt by precursor type
            precursor_type = spectrum['params']['precursor_type']
            if precursor_type not in KEEP_PRE_TYPE: 
                filtout_ms['PRECURSOR_TYPE'] += 1
                filtout_mol['PRECURSOR_TYPE'].update({smiles})
                continue
            
            # Filt by charge
            charge = str(spectrum['params']['charge'])
            if charge != '1+' and charge != '1-':
                filtout_ms['CHARGE'] += 1
                filtout_mol['CHARGE'].update({smiles})
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

            # # Filt by collision energy
            # ce = spectrum['params']['collision_energy'].lower()
            # if ce not in KEEP_CE:
            #     filtout_ms['COLLISION_ENERGY'] += 1
            #     filtout_mol['COLLISION_ENERGY'].update({smiles})
            #     continue

            # Output the data
            mgf.write([spectrum], output_file, file_mode="a+")
            OUTPUT_CNT += 1
    print("Done!\n")

    # Save the record
    for k in filtout_mol.keys(): 
        filtout_mol[k] = len(filtout_mol[k])
    print("For #MS")
    print(filtout_ms)
    print("For #MOL")
    print(filtout_mol)
    print("Output {} MS!".format(OUTPUT_CNT))
    with open(args.log, 'w+') as outfile: 
        json.dump({'MS': filtout_ms, 'MOL': filtout_mol}, outfile, indent=4)