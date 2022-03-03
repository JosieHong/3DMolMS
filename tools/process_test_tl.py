import argparse
from pyteomics import mgf
from tqdm import tqdm

'''
python process_test_tl.py --pre_input ../data/NIST20/proc/ALL_NIST_hcd.mgf --input ../data/HMDB/exp/test_ALL_HMDB_multi.mgf --output ../data/HMDB/exp/test_ALL_HMDB_multi_tl.mgf
'''

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Preprocess the Data')
    parser.add_argument('--pre_input', type=str, default = '',
                        help='path to pretrained input data')
    parser.add_argument('--input', type=str, default = '',
                        help='path to input data')
    parser.add_argument('--output', type=str, default = '',
                        help='path to output data')
    args = parser.parse_args()

    path_to_nist = args.pre_input
    pre_mol = []

    with mgf.read(path_to_nist) as reader:
        print("Got {} data from {}".format(len(reader), path_to_nist))

        for idx, spectrum in enumerate(tqdm(reader)): 
            smiles = spectrum['params']['smiles']
            if smiles not in pre_mol:
                pre_mol.append(smiles)

    print("Extract {} compounds from NIST20-HCD.".format(len(pre_mol)))


    output_spectra = []
    with mgf.read(args.input) as reader:
        print("Got {} data from {}".format(len(reader), args.input))

        for idx, spectrum in enumerate(tqdm(reader)): 
            smiles = spectrum['params']['smiles']
            if smiles not in pre_mol:
                output_spectra.append(spectrum)

    print("Output {} spectra into {}".format(len(output_spectra), args.output))
    mgf.write(output_spectra, args.output, file_mode="w")