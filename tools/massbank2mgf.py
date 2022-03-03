'''
Date: 2021-10-06 21:21:06
LastEditors: yuhhong
LastEditTime: 2021-12-23 17:32:33
'''
import numpy as np
import re

from pyteomics import mgf
from tqdm import tqdm
import argparse

# ignore the warning
from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*')

'''
Convert MSP file from MassBank into MGF file
1. Get the following information from the MSP file
spectrum: {
    'params': {
        'title': <DB#>, 
        'precursor_type': <Precursor_type>,
        'mslevel': <Spectrum_type>,
        'pepmass': <PrecursorMZ>,
        'charge': can be get from <Spectrum_type>,
        'source_instrument': <Instrument>, 
        'instrument_type': <Instrument_type>,
        'ionization': <Comments> -> "ionization",
        'collision_energy': <Collision_energy>,
        'ionmode': <Ion_mode> (Positive/Negative),
        'organism': <Comments> -> "author", 
        'name': <Name>, 
        'smiles': <Comments> -> "SMILES", 
        'inchi': <Comments> -> "InChI",
        'mol_mass': <ExactMass>, 
        'spectrumid': <DB#>, 
        'fragmentation_mode': <Comments> -> "fragmentation mode"
    }
    'mass spectral peaks': <MASS SPECTRAL PEAKS>
} 
2. Save them to MGF file

Notes: 
    > python massbank2mgf.py --input_msp ../data/MassBank/MoNA-export-LC-MS-MS_Spectra.msp
'''

def preprocess_comments(comments): 
    res = re.findall(r'\"(.*?)=(.*?)\"', comments)
    # mgf.write won't wrtie {k: None} out, so we init by 'N/A'
    res_dict = {'ionization': 'N/A', 'author': 'N/A', 'SMILES': 'N/A', 'InChI': 'N/A', 'fragmentation mode': 'N/A'}
    for k, v in res:
        res_dict[k] = v
    return res_dict

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Preprocess the Data')
    parser.add_argument('--input_msp', type=str, default = '',
                        help='path to input the data')
    args = parser.parse_args()
    
    MSP_FILE = args.input_msp
    MGF_FILE = '/'.join(MSP_FILE.split('/')[:-1]) + '/ALL_MB.mgf'
    MSP_KEYS = ['Name', 'Synon', 'Synon', 'DB#', 'InChIKey', 'Precursor_type', 'Spectrum_type', 'msN_pathway', 'PrecursorMZ', 'Instrument_type', 'Instrument', 'Ion_mode', 'Collision_energy', 'Formula', 'MW', 'ExactMass', 'Comments', 'Num Peaks']

    print('MSP_FILE: {}\nMGF_FILE: {}\n'.format(MSP_FILE, MGF_FILE))

    # read the MSP file
    with open(MSP_FILE, 'r') as f:
        data = f.read().split('\n\n')
    # print("Get {}/{} data!".format(len(data), len(suppl)))

    cnt = 0
    spectrums = []
    for d in tqdm(data): 
        raw_spec = {}
        ms_peaks = ""
        d = d.split('\n')
        for line in d:
            k = line.split(': ')[0]
            if k in MSP_KEYS:
                v = "".join(line.split(': ')[1:])
                raw_spec[k] = v
            else: 
                # remove "..." from the peaks
                line = re.sub(r'\".*?\"', '', line)
                # print(ms_peaks)
                ms_peaks += line + ';'
        
        if 'PrecursorMZ' not in raw_spec.keys():
            cnt += 1
            continue
        if 'Precursor_type' not in raw_spec.keys():
            cnt += 1
            continue
        if 'Ion_mode' not in raw_spec.keys():
            cnt += 1
            continue
        # if 'Collision_energy' not in raw_spec.keys():
        #     cnt += 1
        #     continue

        # preprocess the collision energy
        raw_spec['Collision_energy'] = None if 'Collision_energy' not in raw_spec.keys() else raw_spec['Collision_energy']
        # preprocess the instrument
        raw_spec['Instrument'] = None if 'Instrument' not in raw_spec.keys() else raw_spec['Instrument']
        # preprocess the instrument type
        raw_spec['Instrument_type'] = None if 'Instrument_type' not in raw_spec.keys() else raw_spec['Instrument_type']
        # preprocess the InChIKey
        raw_spec['InChIKey'] = None if 'InChIKey' not in raw_spec.keys() else raw_spec['InChIKey']
        # preprocess the ExactMass
        raw_spec['ExactMass'] = None if 'ExactMass' not in raw_spec.keys() else raw_spec['ExactMass']
        
        # process the ion mode
        raw_spec['Ion_mode'] = 'Positive' if raw_spec['Ion_mode'] == 'P' else 'Negative'
        # process the peaks
        mz_array = np.array([float(line.strip().split()[0]) for line in ms_peaks.split(';') if line!=''])
        intensity_array = np.array([float(line.strip().split()[1]) for line in ms_peaks.split(';') if line!=''])
        # process the comments
        raw_spec['Comments'] = preprocess_comments(raw_spec['Comments'])

        # convert raw data from MSP to the format we need
        spectrum = {
            'params': {
                'title': raw_spec['DB#'], 
                'precursor_type': raw_spec['Precursor_type'],
                'mslevel': raw_spec['Spectrum_type'][-1],
                'pepmass': [float(i) for i in re.split(r'[,\s]\s*', raw_spec['PrecursorMZ'])], 
                'source_instrument': raw_spec['Instrument'],
                'instrument_type': raw_spec['Instrument_type'], 
                'ionization': raw_spec['Comments']['ionization'], 
                'collision_energy': raw_spec['Collision_energy'],
                'ionmode': raw_spec['Ion_mode'], 
                'organism': raw_spec['Comments']['author'], 
                'name': raw_spec['Name'], 
                'smiles': raw_spec['Comments']['SMILES'],
                'inchi': raw_spec['InChIKey'],
                'mol_mass': raw_spec['ExactMass'],
                'spectrumid': raw_spec['DB#'],
                'fragmentation_mode': raw_spec['Comments']['fragmentation mode']},
            'm/z array': mz_array,
            'intensity array': intensity_array}
        spectrums.append(spectrum)

    print("\nThere are {} incomplete data.".format(cnt))
    print("Writing {} data to{}".format(len(spectrums), MGF_FILE))
    mgf.write(spectrums, MGF_FILE, file_mode="w", write_charges=False)
    print("Done!")
