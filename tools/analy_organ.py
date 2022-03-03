'''
Date: 2021-09-10 18:39:03
LastEditors: yuhhong
LastEditTime: 2022-11-02 16:29:57

Analysis the source organisms: 
1. Load the filter out MS of each organism -> # filter out MS
2. Count the MS number of each organism -> # MS
3. The rate of (# filter out MS) / (# MS) of each organism

python analy_organ.py --input  ../data/GNPS/ALL_GNPS_clean.mgf --json_file ../data/GNPS/clean_up.json --log ../data/GNPS/organism_log.json 

python analy_organ.py --input  ../data/GNPS_lib/LIB_GNPS_clean.mgf --json_file ../data/GNPS_lib/clean_up.json --log ../data/GNPS_lib/organism_log.json 
'''
import json
import argparse

from pyteomics import mgf
from tqdm import tqdm

ORGAN = {}

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
        for i, n in data['UNRELIABLE_ORGAN'].items():
            ORGAN[i] = [n, 0]
    print(ORGAN)

    with mgf.read(args.input) as reader:
        print("Got {} data from {}".format(len(reader), args.input))

        for idx, spectrum in enumerate(tqdm(reader)): 
            instrument = spectrum['params']['organism'].lower()
            if instrument in ORGAN.keys():
                ORGAN[instrument][1] += 1
            else:
                ORGAN[instrument] = [0, 1]

    for k, v in ORGAN.items():
        ORGAN[k].append(v[0]/(v[0]+v[1]))
    print(ORGAN)

    with open(args.log, 'w+') as outfile:
        json.dump(ORGAN, outfile, indent=4)