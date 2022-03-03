import os
from tqdm import tqdm
import argparse
import xml.etree.ElementTree as ET
from pyteomics import mgf

from rdkit import Chem
# ignore the warning
from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*')

'''
Convert XML file from HMDB into MGF file
1. Get the following infromation from the SDF file
Properties_Dict:
{'HMDB_ID':
    'Generic_name': GENERIC_NAME,
    'Mol_Mass': Mol_Weight,
    'SMILES': SMILES_String,
    'HMDB_ID': HMDB_ID,
    'INCHI_KEY': INCHI_KEY
}
2. Get the following information from the XML file
spectrum: {
    'params': {
        'title': HMDB_ID, 
        'precursor_type': 'Unknown',
        'mslevel': '2',
        'pepmass': Properties_Dict['Mol_Mass'],
        'source_instrument': instrument,
        'collision_energy': collision_energy,
        'ionmode': ionization_mode (Positive/Negative/Unknown),
        'organism': references, 
        'name': Properties_Dict['Generic_name'], 
        'smiles': Properties_Dict['SMILES'], 
        'inchi': Properties_Dict['INCHI_KEY'],
        'mol_mass': Properties_Dict['Mol_Mass'], 
        'spectrumid': HMDB_ID
    }, 
    'mass spectral peaks': peaks
} 
3. Save them to MGF file

Notes: 
    > python hmdb2mgf.py --input_sdf ../data/HMDB/structures.sdf --input_xml_dir ../data/HMDB/hmdb_experimental_msms_spectra/ --output_mgf ../data/HMDB/ALL_HMDB.mgf
'''

def get_peaks_from_xml(xml_path):
    """
    Retrieve msms peaks and metadata from XML file
    
    Required: 
    xml_path(path):Location of the spectrum xml file

    Output: 
    peaks(tupple list): peaks and intensitites
    charge(str): Assumed charge state based on ionization mode
    collision_energy(str): low, med, high collision energy used
    instrument(str): 
    """
    tree = ET.parse(xml_path)
    root = tree.getroot() 
    mz_array = []
    intensity_array = []

    ionization_mode = root.find('ionization-mode').text
    if ionization_mode == 'Positive' or ionization_mode== 'positive':
        charge = '1+'
        ionization_mode = 'Positive'
        adduct = 'M+H'
    elif ionization_mode == 'Negative' or ionization_mode== 'negative':
        charge = '1-'
        ionization_mode = 'Negative'
        adduct = 'M-H'
    elif ionization_mode == 'N/A' or ionization_mode == 'n/a': 
        charge = 'undefined'
        adduct = 'unknown' 
    collision_energy = root.find('collision-energy-level').text
    instrument = root.find('instrument-type').text
    for ms_ms_peak in root.findall('ms-ms-peaks/ms-ms-peak'):
        peak_charge = ms_ms_peak.find('mass-charge').text
        peak_intensity = ms_ms_peak.find('intensity').text
        mz_array.append(peak_charge)
        intensity_array.append(peak_intensity)
    references = root.find('references')
    if len(references.attrib) == 0: 
        references = 'unknown'
    else: 
        references = references.find('ref-text').text
    return mz_array, intensity_array, charge, instrument, adduct, ionization_mode, collision_energy, references

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Preprocess the Data')
    parser.add_argument('--input_sdf', type=str, default = '',
                        help='path to input the structure data')
    parser.add_argument('--input_xml_dir', type=str, default = '',
                        help='dir to input the spectra data')
    parser.add_argument('--output_mgf', type=str, default = '',
                        help='path to output data')
    args = parser.parse_args()

    structure_path = args.input_sdf
    structure_list = Chem.SDMolSupplier(structure_path)
    Properties_Dict = {}
    print("Extract the structure information from {}".format(structure_path))
    for mol in tqdm(structure_list): 
        try: 
            HMDB_ID = mol.GetProp('HMDB_ID')
            try: 
                SMILES_String = mol.GetProp('SMILES')
                INCHI_KEY = mol.GetProp('INCHI_KEY')
                GENERIC_NAME = mol.GetProp('GENERIC_NAME')
                Mol_Weight = mol.GetProp('MOLECULAR_WEIGHT')
                Properties_Dict[HMDB_ID] = {'Generic_name': GENERIC_NAME,'Mol_Mass': float(Mol_Weight),'SMILES': SMILES_String,'HMDB_ID': HMDB_ID,'INCHI_KEY': INCHI_KEY}
            except: 
                pass
        except: 
            pass
    print("Got {} mol!".format(len(Properties_Dict)))

    xml_list = os.listdir(args.input_xml_dir)
    print("Got {} spectra from {}".format(len(xml_list), args.input_xml_dir))
    spectra = []
    miss_cnt = 0
    charge_cnt = 0
    for spectrum_xml in tqdm(xml_list): 
        xml_path = os.path.join(args.input_xml_dir, spectrum_xml)
        HMDB_ID = spectrum_xml.split("_")[0]
        if HMDB_ID not in Properties_Dict.keys(): 
            miss_cnt += 1
            continue
        spectrum_number = spectrum_xml.rsplit("_")[-2]
        mz_array, intensity_array, charge, instrument, adduct, ionization_mode, collision_energy, references = get_peaks_from_xml(xml_path)
        if charge == 'undefined': 
            charge_cnt += 1
            continue 
        if collision_energy == None: 
            collision_energy = 'unknown'
        if instrument == None:
            instrument = 'unknown'

        if charge == '1+':
            charge_mass = 1.0072
        elif charge == '1-':
            charge_mass = -1.0072
        spectrum = {
            'params': {
                'title': spectrum_number, 
                'precursor_type': adduct,
                'mslevel': '2',
                'pepmass': Properties_Dict[HMDB_ID]['Mol_Mass']+charge_mass,
                'source_instrument': instrument,
                'collision_energy': collision_energy,
                'ionmode': ionization_mode,
                'organism': references, 
                'name': Properties_Dict[HMDB_ID]['Generic_name'], 
                'smiles': Properties_Dict[HMDB_ID]['SMILES'], 
                'inchi': Properties_Dict[HMDB_ID]['INCHI_KEY'],
                'mol_mass': Properties_Dict[HMDB_ID]['Mol_Mass'], 
                'spectrumid': HMDB_ID
            },
            'm/z array': mz_array,
            'intensity array': intensity_array
        } 
        spectra.append(spectrum)

    print("Writing {} data to{}".format(len(spectra), args.output_mgf))
    mgf.write(spectra, args.output_mgf, file_mode="w", write_charges=False)
    print("Done!")
    print('Miss Structure: {}, Miss Charge: {}'.format(miss_cnt, charge_cnt))