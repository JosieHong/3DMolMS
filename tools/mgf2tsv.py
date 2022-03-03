'''
Date: 2022-04-02 16:38:50
LastEditors: yuhhong
LastEditTime: 2022-04-02 20:54:00
'''
import argparse
from rdkit import Chem
# suppress rdkit warning
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from pyteomics import mgf
from tqdm import tqdm

'''
python ./tools/mgf2tsv.py --input_mgf ./data/MERGE/proc/ALL_MERGE_hcd.mgf --output_tsv ./data/ClassyFire/ALL_MERGE_hcd.tsv
python ./tools/mgf2tsv.py --input_mgf ./data/MERGE/proc/ALL_MERGE_multi3.mgf --output_tsv ./data/ClassyFire/ALL_MERGE_multi3.tsv
'''

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Preprocess the Data')
    parser.add_argument('--input_mgf', type=str, default = '',
                        help='path to input the data')
    parser.add_argument('--output_tsv', type=str, default = '',
                        help='path to output data')
    args = parser.parse_args()

    mols = []
    ids = []
    # inchis = []
    for idx, spec in enumerate(tqdm(mgf.read(args.input_mgf))):
        mols.append(spec['params']['smiles'])
        ids.append(spec['params']['clean_id'])
        # mol = Chem.MolFromSmiles(spec['params']['smiles'])
        # inchis.append(Chem.MolToInchi(mol))
    print("Load {} molecules from {}".format(len(mols), args.input_mgf))

    # output = 'ID\tSMILES\tINCHI\n'
    # for id, m, inchi in zip(ids, mols, inchis): 
    #     output += str(id)+'\t'+str(m)+'\t'+str(inchi)+'\n'
    output = 'ID\tSMILES\n'
    for id, m in zip(ids, mols): 
        output += str(id)+'\t'+str(m)+'\n'
    with open(args.output_tsv, 'w') as f:
        f.write(output)
    print("Save them to {}".format(args.output_tsv))