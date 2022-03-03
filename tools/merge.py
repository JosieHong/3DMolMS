'''
Date: 2021-10-13 19:28:44
LastEditors: yuhhong
LastEditTime: 2022-11-22 15:08:36

python merge.py --input1 ../data/NIST20/proc/ALL_NIST_hcd.mgf --input2 ../data/MassBank/proc/ALL_MB_hcd.mgf --output ../data/MERGE/proc/ALL_MERGE_hcd.mgf
python merge.py --input1 ../data/NIST20/proc/ALL_NIST_multi.mgf --input2 ../data/MassBank/proc/ALL_MB_multi.mgf --output ../data/MERGE/proc/ALL_MERGE_multi2.mgf
python merge.py --input1 ../data/MERGE/proc/ALL_MERGE_multi2.mgf --input2 ../data/GNPS/proc/ALL_GNPS_multi.mgf --output ../data/MERGE/proc/ALL_MERGE_multi3.mgf

python merge.py --input1 ../data/Agilent/Agilent_Combined.mgf --input2 ../data/Agilent/Agilent_Metlin.mgf --output ../data/Agilent/ALL_Agilent.mgf
python merge.py --input1 ../data/MERGE/proc/ALL_MERGE_multi3.mgf --input2 ../data/Agilent/proc/ALL_Agilent_multi.mgf --output ../data/MERGE/proc/ALL_MERGE_multi4.mgf
'''
import argparse
import os

from pyteomics import mgf
from tqdm import tqdm


def merge_save(sourcefile1, sourcefile2, outfile):
    mgf.write([], outfile, file_mode="w+") # make sure the output file is empty

    with mgf.read(sourcefile1) as reader:
        print("Got {} data from {}".format(len(reader), sourcefile1))

        for idx, spectrum in enumerate(tqdm(reader)): 
            # filt out N/A SMILES and N/A MASS SPECTRA
            if spectrum == None:
                continue
            mgf.write([spectrum], outfile, file_mode="a")
    print("Save {} to {}.".format(sourcefile1, outfile))

    with mgf.read(sourcefile2) as reader:
        print("Got {} data from {}".format(len(reader), sourcefile2))

        for idx, spectrum in enumerate(tqdm(reader)): 
            # filt out N/A SMILES and N/A MASS SPECTRA
            if spectrum == None:
                continue
            mgf.write([spectrum], outfile, file_mode="a")
    print("Save {} to {}.".format(sourcefile2, outfile))


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Preprocess the Data')
    parser.add_argument('--input1', type=str, default = '',
                        help='path to input data')
    parser.add_argument('--input2', type=str, default = '',
                        help='path to input data')
    parser.add_argument('--output', type=str, default = '',
                        help='path to output data') 
    args = parser.parse_args()

    output_dir = "/".join(args.output.split('/')[:-1])
    os.makedirs(output_dir, exist_ok = True)

    merge_save(args.input1, args.input2, args.output)
