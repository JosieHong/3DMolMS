'''
Date: 2022-04-02 23:34:47
LastEditors: yuhhong
LastEditTime: 2022-04-26 14:19:21
'''
import argparse
import time
import urllib.request
import json

# from pyclassyfire import client

from rdkit import Chem
# suppress rdkit warning
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

'''
cd tools
python run_cls.py --input ../data/ClassyFire/test_ALL_MERGE_hcd.tsv --output ../data/ClassyFire/test_ALL_MERGE_hcd_cls.sdf
python run_cls.py --input ../data/ClassyFire/test_ALL_MERGE_multi3.tsv --output ../data/ClassyFire/test_ALL_MERGE_multi3_cls.sdf

python run_cls.py --input ../data/ClassyFire/train_ALL_MERGE_hcd.tsv --output ../data/ClassyFire/train_ALL_MERGE_hcd_cls.sdf
python run_cls.py --input ../data/ClassyFire/train_ALL_MERGE_multi3.tsv --output ../data/ClassyFire/train_ALL_MERGE_multi3_cls.sdf

python run_cls.py --input ../SOL/SOL_pre_test.csv --dataset SOL --output test_SOL_cls.sdf
python run_cls.py --input ../CCS/allCCS_exp_all_test.csv --dataset CCS --output test_CCS_cls.sdf

python run_cls.py --input ../RT/SMRT_test.sdf --dataset RT --output test_RT_cls.sdf
python run_cls.py --input ../Tox21/tox21_test.sdf --dataset Tox21 --output test_Tox21_cls.sdf
'''

# def run_classyfire_sdf(smiles): 
#     try:
#         query_id = client.structure_query(smiles, 'mol_cls')
#         time.sleep(20)
#         query_res = client.get_results(str(query_id), 'sdf')
#         return query_res
#     except:
#         return None

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Preprocess the Data')
    parser.add_argument('--input', type=str, default = '',
                        help='path to input the data')
    parser.add_argument('--dataset', type=str, default='MS', choices=['MS', 'CCS', 'RT', 'SOL', 'Tox21'], 
                        help='Dataset to use, [MS, CCS, RT, SOL, Tox21]')
    parser.add_argument('--output', type=str, default = '',
                        help='path to output the classification results')
    args = parser.parse_args()

    SMILES_LIST = []
    if args.dataset == 'MS':
        with open(args.input, 'r') as f:
            data = f.read().split('\n')
        for d in data:
            if d != '':
                SMILES_LIST.append(d.split('\t')[1])
    elif args.dataset == 'SOL':
        import pandas as pd

        df = pd.read_csv(args.input, index_col=0)
        SMILES_LIST = df['SMILES'].tolist()
    elif args.dataset == 'CCS':
        import pandas as pd

        df = pd.read_csv(args.input)
        SMILES_LIST = df['Structure'].tolist()
    elif args.dataset == 'RT' or args.dataset == 'Tox21':
        supp = Chem.SDMolSupplier(args.input)
        for idx, mol in enumerate(supp):
            if mol is None:
                continue
            smiles = Chem.MolToSmiles(mol)
            SMILES_LIST.append(smiles)
    else: 
        print("Not implement!")
        exit(1)
    print("Load {} data from {}".format(len(SMILES_LIST), args.input))
    
    # ClassyFire: not good to use
    # with open(args.output, 'w') as f: 
    #     for idx, smiles in enumerate(SMILES_LIST): 
    #         cls_sdf = run_classyfire_sdf(smiles)
    #         if cls_sdf == '' or cls_sdf == None:
    #             print(idx, smiles)
    #         else:
    #             f.write(cls_sdf)
    
    # GNPS: excellent!!
    mols = []
    for idx, smiles in enumerate(SMILES_LIST):
        link = "https://gnps-structure.ucsd.edu/classyfire?smiles="+ str(smiles)
        try:
            url = urllib.request.urlopen(link)
            cls_res = json.loads(url.read())
        except:
            continue

        mol = Chem.MolFromSmiles(cls_res['smiles'])
        if mol is None: 
            continue
        
        # param the cls results
        mol.SetProp('SMILES', smiles)

        if cls_res['kingdom'] == None:
            mol.SetProp('kingdom', 'None')
        else:
            mol.SetProp('kingdom', cls_res['kingdom']['name'])

        if cls_res['superclass'] == None:
            mol.SetProp('superclass', 'None')
        else:
            mol.SetProp('superclass', cls_res['superclass']['name'])

        if cls_res['class'] == None:
            mol.SetProp('class', 'None')
        else:
            mol.SetProp('class', cls_res['class']['name'])

        if cls_res['subclass'] == None:
            mol.SetProp('subclass', 'None')
        else:
            mol.SetProp('subclass', cls_res['subclass']['name'])
        
        print(idx, smiles)
        mols.append(mol)
    print("Got {} classification results!".format(len(mols)))

    # Output
    print("Write the classification results...")
    w = Chem.SDWriter(args.output)
    for idx, m in enumerate(mols):
        w.write(m)
    print("Save them to {}".format(args.output))