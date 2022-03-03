'''
Date: 2021-11-22 21:54:53
LastEditors: yuhhong
LastEditTime: 2022-11-04 12:08:58

Clean up the data: 
1. Generate precursor type (need to be modified for different database)
2. Generate collision energy ('Unknown', if there is no original collision energy)
~~3. Remove unreliable MS and~~
   ~~Count them by organism and instrument~~
4. Generate IDs to clean data 

e.g. 
python clean_up.py --input ../data/GNPS/ALL_GNPS_fixed.mgf --output ../data/GNPS/ALL_GNPS_clean.mgf --log ../data/GNPS/clean_up.json --dataset_name gnps
python clean_up.py --input ../data/MassBank/ALL_MB_fixed.mgf --output ../data/MassBank/ALL_MB_clean.mgf --log ../data/MassBank/clean_up.json --dataset_name massbank
python clean_up.py --input ../data/HMDB/ALL_HMDB_fixed.mgf --output ../data/HMDB/ALL_HMDB_clean.mgf --log ../data/HMDB/clean_up.json --dataset_name hmdb
python clean_up.py --input ../data/GNPS_lib/LIB_GNPS_fixed.mgf --output ../data/GNPS_lib/LIB_GNPS_clean.mgf --log ../data/GNPS_lib/clean_up.json --dataset_name gnps

python clean_up.py --input ../data/Agilent/ALL_Agilent.mgf --output ../data/Agilent/ALL_Agilent_clean.mgf --log ../data/Agilent/clean_up.json --dataset_name agilent
python clean_up.py --input ../data/NIST20/ALL_NIST.mgf --output ../data/NIST20/ALL_NIST_clean.mgf --log ../data/NIST20/clean_up.json --dataset_name nist

python clean_up.py --input ../data/GNPS_lib/LDB_NEGATIVE_titled.mgf --output ../data/GNPS_lib/LDB_NEGATIVE_clean.mgf --log ../data/GNPS_lib/clean_up.json --dataset_name gnps
python clean_up.py --input ../data/GNPS_lib/LDB_POSITIVE_titled.mgf --output ../data/GNPS_lib/LDB_POSITIVE_clean.mgf --log ../data/GNPS_lib/clean_up.json --dataset_name gnps
'''

import re
import argparse
import json
from tqdm import tqdm

from pyteomics import mgf
from rdkit import Chem
from rdkit.Chem import Descriptors
# ignore the warning
from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*')


UNRELIABLE_ORGAN = {}
UNRELIABLE_INSTR = {}
TOLERANCE_MASS = 100
CLEAN_INX = 0
PRECURSOR_MASS = {}
PRECURSOR_PATH = './precursor_mass.json'

def parse_precursor_type(precursor_type): 
    # remove the []
    if precursor_type[0] == '[':
        precursor_type = re.findall(r'\[(.*?)\]', precursor_type)[0]
    
    if precursor_type == 'M+H+': 
        precursor_type = 'M+H'
    elif precursor_type == 'M-H-':
        precursor_type = 'M-H'
    return precursor_type



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess the Data')
    parser.add_argument('--input', type=str, default = '',
                        help='path to input data')
    parser.add_argument('--output', type=str, default = '',
                        help='path to output data')
    parser.add_argument('--log', type=str, default = '',
                        help='path to log') 
    parser.add_argument('--dataset_name', type=str, default = '',
                        help='the name of the dataset, also the prefix to CLEAN_ID') 
    args = parser.parse_args()


    # load the precursor_mass
    with open(PRECURSOR_PATH) as json_file:
        PRECURSOR_MASS = json.load(json_file)

    # make sure the output is empty at the begining
    mgf.write([], args.output, file_mode="w")

    with mgf.read(args.input) as reader:
        print("Got {} data from {}".format(len(reader), args.input))
        for idx, spectrum in enumerate(tqdm(reader)): 
            # 0. filter out empty ms
            if len(spectrum['m/z array']) == 0:
                continue
            smiles = spectrum['params']['smiles']
            mol = Chem.MolFromSmiles(smiles)
            try:
                spectrum['params']['smiles'] = Chem.MolToSmiles(mol) # convert smilse to Canonical SMILES
            except:
                continue

            # 1. Generate precursor type
            if args.dataset_name == 'gnps':
                precursor_type = parse_precursor_type(spectrum['params']['name'].split(' ')[-1])
                if precursor_type not in PRECURSOR_MASS.keys():
                    continue
            elif args.dataset_name == 'nist' or args.dataset_name == 'massbank' or args.dataset_name == 'hmdb' or args.dataset_name == 'agilent':
                precursor_type = parse_precursor_type(spectrum['params']['precursor_type'])
                if precursor_type not in PRECURSOR_MASS.keys():
                    continue
            spectrum['params']['precursor_type'] = precursor_type
            

            # 2. Add unknown labels
            if 'collision_energy' not in spectrum['params'].keys():
                spectrum['params']['collision_energy'] = 'Unknown'
            # if 'organism' not in spectrum['params'].keys(): 
            #     spectrum['params']['organism'] = 'Unknown'
            # organ = spectrum['params']['organism'].lower() # not sensitive to capital words
            
            if 'source_instrument' not in spectrum['params'].keys(): 
                spectrum['params']['source_instrument'] = 'Unknown'

            if 'instrument_type' not in spectrum['params'].keys(): 
                spectrum['params']['instrument_type'] = 'Unknown'

            # 3. Modify source instrument
            if args.dataset_name == 'massbank': 
                if spectrum['params']['source_instrument'].lower().endswith('qtof agilent'):
                    spectrum['params']['source_instrument'] = 'Agilent QTOF'

            # 4. Add IDs to clean data 
            spectrum['params']['clean_id'] = args.dataset_name + '_' + str(CLEAN_INX)
            CLEAN_INX += 1 
            mgf.write([spectrum], args.output, file_mode="a+")
        
        
        # Record
        data = {}
        data['UNRELIABLE_ORGAN'] = UNRELIABLE_ORGAN
        data['UNRELIABLE_INSTR'] = UNRELIABLE_INSTR
        with open(args.log, 'w+') as outfile:
            json.dump(data, outfile, indent=4)

    print("Done!")