import sys
import os
from tqdm import tqdm

import numpy as np
from pyteomics import mgf
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


TEST_RATIO = float(sys.argv[1])
assert TEST_RATIO < 1.0

MGF_FILE = sys.argv[1]
OUT_FILE_TRAIN = "/".join(MGF_FILE.split("/")[:-2]) + "/exp/train_" + "".join(MGF_FILE.split("/")[-1])
OUT_FILE_TEST = "/".join(MGF_FILE.split("/")[:-2]) + "/exp/test_" + "".join(MGF_FILE.split("/")[-1])
OUT_DIR = "/".join(MGF_FILE.split("/")[:-2]) + "/exp/" 
if not os.path.exists(OUT_DIR): # Create a new directory because it does not exist
   os.makedirs(OUT_DIR)
   print("Create new directory: {}".format(OUT_DIR))

Scaffold_Spec = {}
reader = mgf.read(MGF_FILE)
print("Got {} data from {}".format(len(reader), MGF_FILE))
print("Extract the molecular list...")
for idx, spectrum in enumerate(tqdm(reader)): 
    smiles = spectrum['params']['smiles']
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=True)
    if scaffold in Scaffold_Spec.keys():
        Scaffold_Spec[scaffold].append(idx)
    else:
        Scaffold_Spec[scaffold] = [idx]


Ltest = []
Ltrain = []
for k, v in Scaffold_Spec.items():
    if len(Ltest) < int(len(reader)*TEST_RATIO): 
        Ltest += v
    else:
        Ltrain += v
print("# Train Spec: {}, # Test Spec: {}".format(len(Ltrain), len(Ltest)))


output_data_train = []
output_data_test = []
train_cleanid = []
test_cleanid = []
with mgf.read(MGF_FILE) as reader: 
    print("Split the data...")
    for idx, spectrum in enumerate(tqdm(reader)): 
        if idx in Ltest:
            output_data_test.append(spectrum)
            test_cleanid.append(spectrum['params']['clean_id'])
        else: 
            output_data_train.append(spectrum)
            train_cleanid.append(spectrum['params']['clean_id'])

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
print("Save {} spectra to {}".format(len(output_data_train), OUT_FILE_TRAIN))
mgf.write(output_data_test, OUT_FILE_TEST, file_mode="w")
print("Save {} spectra to {}".format(len(output_data_test), OUT_FILE_TEST))