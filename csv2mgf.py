import argparse
from tqdm import tqdm

from pyteomics import mgf
import pandas as pd
import numpy as np

from rdkit.Chem import Descriptors
from rdkit import Chem

import requests

CACTUS = "https://cactus.nci.nih.gov/chemical/structure/{0}/{1}"

def smiles_to_iupac(smiles):
    rep = "iupac_name"
    url = CACTUS.format(smiles, rep)
    response = requests.get(url)
    response.raise_for_status()
    return response.text

def smiles_to_inchi(smiles):
    rep = "stdinchi"
    url = CACTUS.format(smiles, rep)
    response = requests.get(url)
    response.raise_for_status()
    return response.text

def smiles_to_inchikey(smiles):
    rep = "stdinchikey"
    url = CACTUS.format(smiles, rep)
    response = requests.get(url)
    response.raise_for_status()
    return response.text



if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Mass Spectrum Prediction')
    parser.add_argument('--csv_file', type=str, default = './example/output_nega.csv',
                        help='path to output csv file')
    parser.add_argument('--mgf_file', type=str, default = './example/output_nega.mgf',
                        help='path to output mgf file')
    args = parser.parse_args()

    df = pd.read_csv(args.csv_file)
    print('Load {} data from {}'.format(len(df), args.csv_file))
    # columns: 
    # ID,SMILES,Precursor_Type,Instrument,Collision_Energy,Pred_M/Z,Pred_Intensity

    spectra = []
    prefix = 'pred'
    for idx, row in tqdm(df.iterrows()): 
        smiles = row['SMILES']
        mol = Chem.MolFromSmiles(smiles)

        spectrum = {
            'params': {
                'title': row['ID'], 
                'precursor_type': row['Precursor_Type'],
                'mslevel': '2',
                'pepmass': Descriptors.ExactMolWt(mol), 
                'source_instrument': row['Instrument'],
                'collision_energy': row['Collision_Energy'],
                'organism': 'DyGMNet_Agilent_12202022', 
                'smiles': smiles, 
                'iupac': smiles_to_iupac(smiles), 
                'inchi': smiles_to_inchi(smiles), 
                'inchi_key': smiles_to_inchikey(smiles), 
                'spectrumid': prefix+'_'+str(idx), 
            },
            'm/z array': np.array([float(i) for i in row['Pred_M/Z'].split(',')]),
            'intensity array': np.array([float(i)*1000 for i in row['Pred_Intensity'].split(',')])
        } 
        spectra.append(spectrum)
        
    print("Writing {} data to{}".format(len(spectra), args.mgf_file))
    mgf.write(spectra, args.mgf_file, file_mode="w", write_charges=False)
    print("Done!")