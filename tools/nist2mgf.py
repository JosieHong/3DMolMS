'''
Date: 2021-10-06 21:21:06
LastEditors: yuhhong
LastEditTime: 2021-12-18 23:53:09
'''
import numpy as np
import sys
import re

from pyteomics import mgf
from rdkit import Chem
from tqdm import tqdm
import argparse

# ignore the warning
from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*')

'''
Convert MSP file and SDF file from NIST20 into MGF file
1. Get mol from SDF file
2. Get the following information from the MSP file
spectrum: {
    'params': {
        'title': <Name>,
        'note': <Notes>,
        'precursor_type': <Precursor_type>,
        'mslevel': <Spectrum_type>,
        'pepmass': <PrecursorMZ>,
        'charge': can be get from <Spectrum_type>,
        'source_instrument': <Instrument>, 
        'instrument_type': <Instrument_type>,
        'ionization': <Ionization>,
        'collision_energy': <Collision_energy>,
        'ionmode': <Ion_mode> (Positive/Negative),
        'organism': <COMMENT>, 
        'name': <Name>, 
        'smiles': generate by mol,
        'inchi': <InChIKey>,
        'mol_mass': <MW>,
        'spectrumid': <ID>
    }
    'mass spectral peaks': <MASS SPECTRAL PEAKS>
}
3. Save them to MGF file

Notes: 
    > python nist2mgf.py --input_msp ../data/NIST20/hr_msms_nist.MSP --input_sdf_dir ../data/NIST20/hr_msms_nist.SDF --type hr_msms
    > python nist2mgf.py --input_msp ../data/NIST20/lr_msms_nist.MSP --input_sdf_dir ../data/NIST20/lr_msms_nist.SDF --type lr_msms
'''

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Preprocess the Data')
    parser.add_argument('--input_msp', type=str, default = '',
                        help='path to input the data')
    parser.add_argument('--input_sdf_dir', type=str, default = '',
                        help='path to input the data')
    parser.add_argument('--type', type=str, default = 'hr_msms', choices=['hr_msms', 'lr_msms'], 
                        help='data type, [hr_msms, lr_msms]')
    args = parser.parse_args()

    MSP_FILE = args.input_msp
    SDF_FILE = args.input_sdf_dir
    prefix = args.type
    MGF_FILE = MSP_FILE[:-3] + 'mgf'
    MSP_KEYS = ['Name', 'Related_CAS#', 'Notes', 'Precursor_type', 'Spectrum_type', 'msN_pathway', 'PrecursorMZ', 'Instrument_type', 'Instrument', 'Sample_inlet', 'Ionization', 'In-source_voltage', 'Collision_gas', 'Pressure', 'Collision_energy', 'Ion_mode', 'Link', 'Special_fragmentation', 'InChIKey', 'Synon', 'Formula', 'MW', 'ExactMass', 'CASNO', 'NISTNO', 'ID', 'Comment', 'Num peaks']

    print('MSP_FILE: {}\nSDF_FILE: {}\nMGF_FILE: {}\n'.format(MSP_FILE, SDF_FILE, MGF_FILE))

    # read the SDF file
    suppl = Chem.SDMolSupplier(SDF_FILE)
    # read the MSP file
    with open(MSP_FILE, 'r') as f:
        data = f.read().split('\n\n')
    print("Get {} (MSP) / {} (SDF) data!".format(len(data), len(suppl)))

    cnt = 0
    spectrums = []
    for d, mol in tqdm(zip(data, suppl)):
        if mol == None:
            cnt += 1
            continue
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
        
        # preprocess the collision energy
        raw_spec['Collision_energy'] = None if 'Collision_energy' not in raw_spec.keys() else raw_spec['Collision_energy']
        # preprocess the ionization
        raw_spec['Ionization'] = None if 'Ionization' not in raw_spec.keys() else raw_spec['Ionization']
        # preprocess the instrument
        raw_spec['Instrument'] = None if 'Instrument' not in raw_spec.keys() else raw_spec['Instrument']
        # preprocess the instrument type
        raw_spec['Instrument_type'] = None if 'Instrument_type' not in raw_spec.keys() else raw_spec['Instrument_type']
        # preprocess the InChIKey
        raw_spec['InChIKey'] = None if 'InChIKey' not in raw_spec.keys() else raw_spec['InChIKey']
        # process the ion mode
        # raw_spec['Ion_mode'] = 'Positive' if raw_spec['Ion_mode'] == 'P' else 'Negative'
        # process the peaks
        mz_array = np.array([float(line.strip().split()[0]) for line in ms_peaks.split(';') if line!=''])
        intensity_array = np.array([float(line.strip().split()[1]) for line in ms_peaks.split(';') if line!=''])
        
        # convert raw data from MSP to the format we need
        spectrum = {
            'params': {
                'title': prefix + '_'+raw_spec['ID'], 
                'note': raw_spec['Notes'],
                'precursor_type': raw_spec['Precursor_type'],
                'mslevel': raw_spec['Spectrum_type'][-1],
                'pepmass': [float(i) for i in raw_spec['PrecursorMZ'].split(', ')], 
                'source_instrument': raw_spec['Instrument'],
                'instrument_type': raw_spec['Instrument_type'], 
                'ionization': raw_spec['Ionization'], 
                'collision_energy': raw_spec['Collision_energy'],
                'ionmode': raw_spec['Ion_mode'], 
                'organism': raw_spec['Comment'], 
                'name': raw_spec['Name'], 
                'smiles': Chem.MolToSmiles(mol),
                'inchi': raw_spec['InChIKey'],
                'mol_mass': raw_spec['MW'],
                'spectrumid': raw_spec['ID']},
            'm/z array': mz_array,
            'intensity array': intensity_array}
        spectrums.append(spectrum)

    print("There are {} empty molecules.".format(cnt))
    print("Writing {} data to{}".format(len(spectrums), MGF_FILE))
    mgf.write(spectrums, MGF_FILE, file_mode="w", write_charges=False)
    print("Done!")
