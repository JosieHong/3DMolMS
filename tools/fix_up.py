'''
Date: 2021-11-22 21:54:53
LastEditors: yuhhong
LastEditTime: 2022-11-02 16:11:32

Fix Up the Data: 
1. Remove None MS 
2. Fix the SMILES by molecules' name and 
   Remove MS with None SMILES 
3. Remove MS with None molecules
4. Unify the SMILES

e.g.
python fix_up.py --input ../data/GNPS/ALL_GNPS_titled.mgf --output ../data/GNPS/ALL_GNPS_fixed.mgf --log ../data/GNPS/fix_smiles.json --dataset_name gnps
python fix_up.py --input ../data/NIST20/ALL_NIST.mgf --output ../data/NIST20/ALL_NIST_fixed.mgf --log ../data/NIST20/fix_smiles.json --dataset_name nist
python fix_up.py --input ../data/MassBank/ALL_MB.mgf --output ../data/MassBank/ALL_MB_fixed.mgf --log ../data/MassBank/fix_smiles.json --dataset_name massbank
python fix_up.py --input ../data/HMDB/ALL_HMDB.mgf --output ../data/HMDB/ALL_HMDB_fixed.mgf --log ../data/HMDB/fix_smiles.json --dataset_name hmdb

python fix_up.py --input ../data/GNPS_lib/LIB_GNPS.mgf --output ../data/GNPS_lib/LIB_GNPS_fixed.mgf --log ../data/GNPS_lib/fix_smiles.json --dataset_name gnps
'''

from ast import arg
import os
import re
import argparse
import urllib.request
import json

from pyteomics import mgf
from rdkit import Chem
# ignore the warning
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

FIXED_SMILES = 0
UNFIXED_SMILES = 0
ALL_IND = 0

def search_smiles(name):
    search_item = "https://pubchem.ncbi.nlm.nih.gov/sdq/sdqagent.cgi?query={\"select\":\"*\",\"collection\":\"compound\",\"where\":{\"ands\":{\"cmpdsynonym\":\"%s\"}}}" % name
    # try to search the SMILES 3 times
    for i in range(3):
        try: 
            request = urllib.request.urlopen(search_item)
            res = json.loads(request.read().decode('utf-8')[9:-2])['SDQOutputSet'][0]['rows']
        except:
            continue
        # if get results, return it
        if len(res) != 0:
            smiles = res[0]['isosmiles']
            return smiles
    return None


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

    assert args.dataset_name == 'nist' or args.dataset_name == 'gnps' or args.dataset_name == 'massbank' or args.dataset_name == 'hmdb'

    # init
    if os.path.exists(args.log):
        with open(args.log) as json_file:
            data = json.load(json_file)
        FIXED_SMILES = data['FIXED_SMILES']
        UNFIXED_SMILES = data['UNFIXED_SMILES']
        ALL_IND = data['ALL_IND']-1 # make sure all the data will be processed
    else:
        data = {}
    # J0sie: We don't need to check that, because the same 
    # title will be overwrited. 
    # if os.path.exists(args.output):
    #     with mgf.read(args.output) as reader: 
    #         print("Got {} data from {}".format(len(reader), args.output))
    #         if len(reader) == ALL_IND-1: # have written, but CLEAN_INX didn't add one
    #             ALL_IND += 1
    #         assert len(reader) == ALL_IND # check the break point
    # print("Have processed {} data, saved in {}\n".format(ALL_IND, args.output))

    empty_mol_cnt = 0
    with mgf.read(args.input) as reader:
        print("Got {} data from {}".format(len(reader), args.input))
        for idx, spectrum in enumerate(reader): 
            # 0. Load the break point
            if idx < ALL_IND: 
                continue
            
            # begin!
            if idx == ALL_IND:
                print("Skipped {} data, which has been processed.".format(idx))
            # update the break point (make sure that the log will be updated in every enumerate)
            data['FIXED_SMILES'] = FIXED_SMILES
            data['UNFIXED_SMILES'] = UNFIXED_SMILES
            data['ALL_IND'] = idx
            with open(args.log, 'w+') as outfile:
                json.dump(data, outfile, indent=4)
            if idx % 100 == 0: 
                print(idx)
                print('==========================================')
                print(data)
                print('==========================================\n')


            # 1. Remove empty MS
            if spectrum == None: 
                # print('Empty MS!')
                continue
            # if the spectra don't have any peaks or only has one peak (precursor's peak), we remove it
            if len(spectrum['m/z array']) == 0 or len(spectrum['m/z array']) == 1: 
                continue 

            # 2. Fix the SMILES by molecules' name
            smiles = spectrum['params']['smiles']
            # remove [...] and 'fromNIST14'
            patten = r'\[.*?\]'
            smiles = re.sub(patten, '', smiles)
            smiles = smiles.replace('fromNIST14', '')
            
            if smiles == None or smiles == '' or smiles == 'N/A': 
                # name = "".join(spectrum['params']['name'].split(' ')[:-1])
                # # print("Empty SMILES: {}, {}, {}".format(idx, name, smiles))
                # new_smiles = search_smiles(name)

                new_smiles = False
                if new_smiles:
                    # update the SMILES
                    spectrum['params']['smiles'] = new_smiles
                    # print("Got {}".format(new_smiles))
                    FIXED_SMILES += 1
                else: 
                    UNFIXED_SMILES += 1
                    continue # 2. Remove MS with empty SMILES
            
            # 3. Remove MS with None molecules
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: 
                empty_mol_cnt += 1
                print('Empty MOL!', smiles)
                continue

            # 4. Unify the SMILES
            spectrum['params']['smiles'] = Chem.MolToSmiles(mol)

            # 5. Fix the title of NIST20
            if args.dataset_name == 'nist':
                spectrum['params']['title'] = str(idx)
            mgf.write([spectrum], args.output, file_mode="a+")
            # print('Save!')
    
    print("# Empty MOL: {}".format(empty_mol_cnt))
    print("Done!")