import os
import sys
import numpy as np
import pandas as pd
from decimal import *

from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt

import matplotlib.pyplot as plt

'''
python ./tools/vis_ms.py ./results/molnet_merge_hcd_hr02_0624.csv ./vis/
'''

def ms2vec(x, y, pepmass, resolution=1, num_ms=2000): 
    # normalize to 0-1
    x = x / np.max(x)

    pepmass = Decimal(str(pepmass)) 
    resolution = Decimal(str(resolution))
    right_bound = int(pepmass // resolution)

    ms = [0] * (int(Decimal(str(num_ms)) // resolution)) # add "0" to y data
    # ms = [0] * (int(Decimal(str(x[-1])) // resolution)+1) # add "0" to y data
    for idx, val in enumerate(x): 
        val = int(round(Decimal(str(val)) // resolution))
        if val >= right_bound: 
            continue
        ms[val] += float(y[idx])
    return np.array(ms)



if __name__ == "__main__": 

    res_path = sys.argv[1]
    res_df = pd.read_csv(res_path, sep='\t', 
                            converters={'M/Z': lambda x: [float(i) for i in x.strip("[]").split() if i != ''],
                                        'Intensity': lambda x: [float(i) for i in x.strip("[]").split() if i != ''], 
                                        'Pred M/Z': lambda x: [float(i) for i in x.strip("[]").split() if i != ''],
                                        'Pred Intensity': lambda x: [float(i) for i in x.strip("[]").split() if i != '']})
    out_dir = sys.argv[2]

    for i, row in res_df.iterrows(): 
        acc = row['Accuracy']
        smiles = row['SMILES'] 

        if acc > 0.9: 
            target = ms2vec(row['M/Z'], row['Intensity'], ExactMolWt(Chem.MolFromSmiles(smiles)))
            pred = ms2vec(row['Pred M/Z'], row['Pred Intensity'], ExactMolWt(Chem.MolFromSmiles(smiles)))
            print(target, pred)
            
            x = range(len(target))
            plt.figure(figsize=(32,18))
            plt.bar(x, -pred, color='red', alpha=0.8, label='prediction')
            plt.bar(x, target, color='blue',alpha=0.8, label='target')
            plt.hlines(y=0, xmin=0, xmax=len(target), colors='black')
            plt.xlim(0, len(target))
            plt.legend(loc='upper right')
            plt.xlabel('m/z')
            plt.ylabel('intensity')
            plt.title("{} ({})".format(smiles, str(acc)))
            
            figure_name = (os.path.join(out_dir, "{}.png")).format(str(i))
            plt.savefig(figure_name, dpi=300, pad_inches=0.0)
            # print("Save the figure into {}, acc: {}".format(figure_name, acc))
            plt.close()
        else:
            print(i, smiles, acc)
