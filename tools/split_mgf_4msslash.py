'''
Date: 2022-06-01 20:41:44
LastEditors: yuhhong
LastEditTime: 2022-06-03 20:31:47
'''
import os
import argparse

from pyteomics import mgf
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors

'''
Split a `.mgf` file into small files. Each small file contains N (input parameter) spectra. 

python split_mgf_4msslash.py --input /data/yuhhong/Mol_Spec/results/hmdb_ours_pred_new_0-0.mgf --output_dir /data/yuhhong/HMDB_MS_Pred/ --n 30000
python split_mgf_4msslash.py --input /data/yuhhong/Mol_Spec/results/hmdb_ours_pred_new_0-1.mgf --output_dir /data/yuhhong/HMDB_MS_Pred/ --n 30000
python split_mgf_4msslash.py --input /data/yuhhong/Mol_Spec/results/hmdb_ours_pred_new_0-2.mgf --output_dir /data/yuhhong/HMDB_MS_Pred/ --n 30000
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess the Data')
    parser.add_argument('--input', type=str, default = '',
                        help='path to input data')
    parser.add_argument('--output_dir', type=str, default = '',
                        help='directory to output the files') 
    parser.add_argument('--n', type=int, default = 30000,
                        help='max spectra contained in each output file') 
    args = parser.parse_args()

    # read the mgf file
    supp = mgf.read(args.input)
    print("Read {} data from {}".format(len(supp), args.input))

    spectra = []
    file_cnt = 0
    file_name = args.input.split('/')[-1]
    for idx, spectrum in enumerate(tqdm(supp)): 
        spectrum['intensity array'] = spectrum['intensity array'] * 1000
        spectrum['params']['peptide'] = spectrum['params']['smiles']
        mol = Chem.MolFromSmiles(spectrum['params']['smiles'])
        mol_weight = Descriptors.MolWt(mol)
        spectrum['params']['pepmass'] = mol_weight
        spectra.append(spectrum)
        
        # output N spectra
        if (idx+1) % args.n == 0: 
            output_file = os.path.join(args.output_dir, file_name[:-4]+'-'+str(file_cnt)+'.mgf')
            print("Writing {} data to{}".format(len(spectra), output_file))
            mgf.write(spectra, output_file, file_mode="w", write_charges=False)

            spectra = []
            file_cnt += 1

    # output the last spectra
    output_file = os.path.join(args.output_dir, file_name[:-4]+'-'+str(file_cnt)+'.mgf')
    print("Writing {} data to{}".format(len(spectra), output_file))
    mgf.write(spectra, output_file, file_mode="w", write_charges=False)
    
    print('Done!')
    