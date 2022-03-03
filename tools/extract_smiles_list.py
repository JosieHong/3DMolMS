'''
Date: 2021-12-31 12:09:39
LastEditors: yuhhong
LastEditTime: 2022-08-10 17:20:03
'''
import os
from pyteomics import mgf
from tqdm import tqdm
import argparse



if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Extract the SMILES from MGF file')
    parser.add_argument('--input', type=str, default = '',
                        help='path to input the data')
    parser.add_argument('--output', type=str, default = '',
                        help='path to output the data')
    parser.add_argument('--ion_mode', type=str, default='ALL', choices=['P', 'N', 'ALL'], 
                        help='Ion mode used for training and test') 
    args = parser.parse_args()

    MGF_FILE = args.input
    OUT_FILE = args.output

    if args.ion_mode == 'P':
        ion_mode = ['p', 'positive']
    elif args.ion_mode == 'N':
        ion_mode = ['n', 'negative']
    else:
        ion_mode = ['p', 'positive', 'n', 'negative']

    # keep the index of the exsit data
    SMILES_LIST = []
    # if os.path.exists(OUT_FILE):
    #     with open(OUT_FILE, 'r') as f:
    #         data =  f.readlines()
    #     for d in data:
    #         smiles = " ".join(d.split(' ')[1:])[:-1]
    #         SMILES_LIST.append(smiles)

    NEW_SMILES_LIST = []
    with mgf.read(MGF_FILE) as reader:
        print("Got {} data from {}".format(len(reader), MGF_FILE))

        for idx, spectrum in enumerate(tqdm(reader)): 
            if spectrum['params']['ionmode'].lower() not in ion_mode: 
                continue

            # filt out N/A SMILES and N/A MASS SPECTRA
            smiles = spectrum['params']['smiles']
            if smiles not in SMILES_LIST:
                NEW_SMILES_LIST.append(smiles)

    SMILES_SET = list(set(NEW_SMILES_LIST))
    SMILES_SET = SMILES_LIST + SMILES_SET
    with open(OUT_FILE, 'w') as f: 
        for idx, line in enumerate(SMILES_SET):
            f.writelines(str(idx)+"\t"+line+"\n")
    print("Save {} smiles into {}".format(len(SMILES_SET), OUT_FILE))