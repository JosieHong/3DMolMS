'''
Author: JosieHong
Date: 2021-01-13 21:43:45
LastEditAuthor: JosieHong
LastEditTime: 2022-12-08 09:33:37
Funstion: 
    split the data randomly by their SMILES
Note: 
    python random_spliter.py <test ratio> <path to .mgf file>
    
    python random_spliter.py 0.1 ../data/GNPS/proc/ALL_GNPS_multi.mgf 
    python random_spliter.py 0.1 ../data/GNPS_lib/proc/LIB_GNPS_multi.mgf 
    
    python random_spliter.py 0.1 ../data/NIST20/proc/ALL_NIST_hcd.mgf 
    python random_spliter.py 0.1 ../data/NIST20/proc/ALL_NIST_multi.mgf
    python random_spliter.py 0.1 ../data/NIST20/proc/ALL_NIST_multi_nega.mgf
    python random_spliter.py 0.1 ../data/NIST20/proc/ALL_NIST_multi_posi.mgf 

    python random_spliter.py 0.1 ../data/MassBank/proc/ALL_MB_hcd.mgf 
    python random_spliter.py 0.1 ../data/MassBank/proc/ALL_MB_multi.mgf 
    python random_spliter.py 0.1 ../data/MassBank/proc/ALL_MB_multi_nega.mgf 
    python random_spliter.py 0.1 ../data/MassBank/proc/ALL_MB_multi_posi.mgf 
 
    python random_spliter.py 0.1 ../data/HMDB/proc/ALL_HMDB_multi.mgf

    python random_spliter.py 0.1 ../data/MERGE/proc/ALL_MERGE_hcd.mgf
    python random_spliter.py 0.1 ../data/MERGE/proc/ALL_MERGE_multi3.mgf

    python random_spliter.py 0.1 ../data/Agilent/proc/ALL_Agilent_multi.mgf 
    python random_spliter.py 0.1 ../data/Agilent/proc/ALL_Agilent_multi_nega.mgf 
    python random_spliter.py 0.1 ../data/Agilent/proc/ALL_Agilent_multi_posi.mgf 
'''
import sys
import os
from tqdm import tqdm

import numpy as np

from pyteomics import mgf
from rdkit import Chem


TEST_RATIO = float(sys.argv[1])
assert TEST_RATIO < 1.0

MGF_FILE = sys.argv[2]
OUT_FILE_TRAIN = "/".join(MGF_FILE.split("/")[:-2]) + "/exp/train_" + "".join(MGF_FILE.split("/")[-1])
OUT_FILE_TEST = "/".join(MGF_FILE.split("/")[:-2]) + "/exp/test_" + "".join(MGF_FILE.split("/")[-1])
OUT_DIR = "/".join(MGF_FILE.split("/")[:-2]) + "/exp/"
if not os.path.exists(OUT_DIR): # Create a new directory because it does not exist
   os.makedirs(OUT_DIR)
   print("Create new directory: {}".format(OUT_DIR))

DATA_LIST = []
with mgf.read(MGF_FILE) as reader:
    print("Got {} data from {}".format(len(reader), MGF_FILE))
    print("Extract the molecular list...")
    for idx, spectrum in enumerate(tqdm(reader)): 
        smiles = Chem.CanonSmiles(spectrum['params']['smiles'])
        # smiles = spectrum['params']['smiles']
        DATA_LIST.append(smiles)
DATA_LIST = list(set(DATA_LIST))


Ltest = np.random.choice(DATA_LIST, int(len(DATA_LIST)*TEST_RATIO), replace=False)
Ltrain = [x for x in DATA_LIST if x not in Ltest]
# for idx, d in enumerate(DATA_LIST): 
#     if idx % 10 == 1: 
#         Ltest.append(d)
#     else:
#         Ltrain.append(d)
# print("{} train compound, {} test compound".format(len(Ltrain), len(Ltest)))

output_data_train = []
output_data_test = []
train_cleanid = []
test_cleanid = []
with mgf.read(MGF_FILE) as reader: 
    print("Split the data...")
    for idx, spectrum in enumerate(tqdm(reader)): 
        smiles = Chem.CanonSmiles(spectrum['params']['smiles'])
        # smiles = spectrum['params']['smiles']
        if smiles in Ltest:
            output_data_test.append(spectrum)
            test_cleanid.append(spectrum['params']['clean_id'])
        elif smiles in Ltrain:
            output_data_train.append(spectrum)
            train_cleanid.append(spectrum['params']['clean_id'])
        else:
            raise ValueError("{} is not in neither train nor test data. Please check the data!".format(smiles))

# save the clean_id for train and test
with open(os.path.join(OUT_DIR, 'train_'+MGF_FILE.split("/")[-1].replace('.mgf', '')+'_cleanID.txt'), 'w') as f:
    for cleanid in train_cleanid:
        # write each item on a new line
        f.write("%s\n" % cleanid)
    print('Output the test clean ID into {}'.format(os.path.join(OUT_DIR, 'train_'+MGF_FILE.split("/")[-1].replace('.mgf', '')+'_cleanID.txt')))
with open(os.path.join(OUT_DIR, 'test_'+MGF_FILE.split("/")[-1].replace('.mgf', '')+'_cleanID.txt'), 'w') as f:
    for clean_id in test_cleanid:
        # write each item on a new line
        f.write("%s\n" % clean_id)
    print('Output the test clean ID into {}'.format(os.path.join(OUT_DIR, 'test_'+MGF_FILE.split("/")[-1].replace('.mgf', '')+'_cleanID.txt')))

# makdir 
if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)

mgf.write(output_data_train, OUT_FILE_TRAIN, file_mode="w")
print("Save {}/{} data to {}".format(len(output_data_train), len(Ltrain), OUT_FILE_TRAIN))
mgf.write(output_data_test, OUT_FILE_TEST, file_mode="w")
print("Save {}/{} data to {}".format(len(output_data_test), len(Ltest), OUT_FILE_TEST))
