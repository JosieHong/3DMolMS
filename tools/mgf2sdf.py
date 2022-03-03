'''
Date: 2022-04-02 14:17:00
LastEditors: yuhhong
LastEditTime: 2022-04-02 20:53:44
'''
import argparse

from rdkit import Chem
# suppress rdkit warning
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from pyteomics import mgf

from tqdm import tqdm

'''
python ./tools/mgf2sdf.py --input_mgf ./data/MERGE/proc/ALL_MERGE_hcd.mgf --output_sdf ./data/ClassyFire/ALL_MERGE_hcd.sdf
python ./tools/mgf2sdf.py --input_mgf ./data/MERGE/proc/ALL_MERGE_multi3.mgf --output_sdf ./data/ClassyFire/ALL_MERGE_multi3.sdf
'''

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Preprocess the Data')
    parser.add_argument('--input_mgf', type=str, default = '',
                        help='path to input the data')
    parser.add_argument('--output_sdf', type=str, default = '',
                        help='path to output data')
    args = parser.parse_args()

    mols = []
    for idx, spec in enumerate(tqdm(mgf.read(args.input_mgf))):
        smiles = spec['params']['smiles']
        mol = Chem.MolFromSmiles(smiles)
        mol.SetProp('ID', spec['params']['clean_id'])
        mols.append(mol)
    print("Load {} molecules from {}".format(len(mols), args.input_mgf))

    w = Chem.SDWriter(args.output_sdf)
    for idx, m in enumerate(tqdm(mols)):
        if m is not None:
            w.write(m)
    print("Save them to {}".format(args.output_sdf))