'''
Date: 2021-09-10 18:39:03
LastEditors: yuhhong
LastEditTime: 2022-11-02 16:27:07

Analysis the source instruments: 
1. Load the filter out MS of each instrument -> # filter out MS
2. Count the MS number of each instrument -> # MS
3. The rate of (# filter out MS) / (# MS) of each instrument

python analy_instru.py --input  ../data/GNPS/ALL_GNPS_clean.mgf --json_file ../data/GNPS/clean_up.json --log ../data/GNPS/instrument_log.json 

python analy_instru.py --input  ../data/GNPS_lib/LIB_GNPS_clean.mgf --json_file ../data/GNPS_lib/clean_up.json --log ../data/GNPS_lib/instrument_log.json 
'''
import json
import argparse

from pyteomics import mgf
from tqdm import tqdm

INSTU = {}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess the Data')
    parser.add_argument('--input', type=str, default = '',
                        help='path to input data')
    parser.add_argument('--json_file', type=str, default = '',
                        help='path to output data')
    parser.add_argument('--log', type=str, default = '',
                        help='path to log') 
    args = parser.parse_args()

    with open(args.json_file, 'r') as f:
        data = json.load(f)
        for i, n in data['UNRELIABLE_INSTR'].items():
            INSTU[i] = [n, 0]
    print(INSTU)

    with mgf.read(args.input) as reader:
        print("Got {} data from {}".format(len(reader), args.input))

        for idx, spectrum in enumerate(tqdm(reader)): 
            instrument = spectrum['params']['source_instrument'].lower()
            if instrument in INSTU.keys():
                INSTU[instrument][1] += 1
            else:
                INSTU[instrument] = [0, 1]

    for k, v in INSTU.items():
        INSTU[k].append(v[0]/(v[0]+v[1]))
    print(INSTU)

    with open(args.log, 'w+') as outfile:
        json.dump(INSTU, outfile, indent=4)