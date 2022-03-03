'''
Date: 2022-11-03 15:01:25
LastEditors: yuhhong
LastEditTime: 2022-11-03 16:45:43
'''
import re
import argparse
from tqdm import tqdm
import numpy as np

from rdkit import Chem
from pyteomics import mgf

'''
python agilent2mgf.py --input_sdf ../data/Agilent/Agilent_Combined.sdf --output_mgf ../data/Agilent/Agilent_Combined.mgf
python agilent2mgf.py --input_sdf ../data/Agilent/Agilent_Metlin.sdf --output_mgf ../data/Agilent/Agilent_Metlin.mgf
'''

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Preprocess the Data')
    parser.add_argument('--input_sdf', type=str, default = '',
                        help='path to input the structure data')
    parser.add_argument('--output_mgf', type=str, default = '',
                        help='path to output data')
    args = parser.parse_args()

    supp = Chem.SDMolSupplier(args.input_sdf)
    spectra = []
    prefix = '_'.join(args.input_sdf.split('_')[:2])
    for idx, mol in enumerate(tqdm(supp)): 
        if mol == None or not mol.HasProp('MASS SPECTRAL PEAKS'): 
            continue
        
        # ['NAME', 'CAS', 'CASNO', 'INCHI', 'SMILES', 'CHEMSPIDER', 'FORMULA', 'IUPAC_NAME', 'EXACT MASS', 'INCHIKEY', 'COLLISION ENERGY', 'IONIZATION', 'PRECURSOR TYPE', 'PRECURSOR M/Z', 'SPECTRUM TYPE', 'ION MODE', 'INSTRUMENT TYPE', 'SPLASH', 'NUM PEAKS', 'MASS SPECTRAL PEAKS']
        smiles = Chem.MolToSmiles(mol)

        precursor_type = mol.GetProp('PRECURSOR TYPE')
        if '[' in precursor_type and ']' in precursor_type: # remove the brackets
            p = re.compile(r'\[(.*?)\]', re.S)
            precursor_type = re.findall(p, precursor_type)[0]

        mz_array = []
        intensity_array = []
        raw_ms = mol.GetProp('MASS SPECTRAL PEAKS').split('\n')
        for line in raw_ms:
            mz_array.append(float(line.split()[0]))
            intensity_array.append(float(line.split()[1]))
        mz_array = np.array(mz_array)
        intensity_array = np.array(intensity_array)

        if mol.GetProp('ION MODE') == 'POSTIVE':
            charge = '1+'
        else:
            charge = '1-'

        spectrum = {
            'params': {
                'title': prefix+'_'+str(idx), 
                'precursor_type': precursor_type,
                'mslevel': '2',
                'pepmass': mol.GetProp('EXACT MASS'),
                'source_instrument': mol.GetProp('INSTRUMENT TYPE'),
                'instrument_type': mol.GetProp('INSTRUMENT TYPE'), 
                'collision_energy': mol.GetProp('COLLISION ENERGY'),
                'ionmode': mol.GetProp('ION MODE'),
                'charge': charge,
                'organism': 'Agilent', 
                'name': mol.GetProp('NAME'), 
                'smiles': smiles, 
                'inchi': mol.GetProp('INCHI'),
                'mol_mass': mol.GetProp('EXACT MASS'), 
                'spectrumid': prefix+'_'+str(idx), 
            },
            'm/z array': mz_array,
            'intensity array': intensity_array
        } 
        spectra.append(spectrum)
        
    print("Writing {} data to{}".format(len(spectra), args.output_mgf))
    mgf.write(spectra, args.output_mgf, file_mode="w", write_charges=False)
    print("Done!")