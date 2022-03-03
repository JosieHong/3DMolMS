'''
Date: 2021-07-08 18:37:32
LastEditors: yuhhong
LastEditTime: 2022-03-03 11:50:59
'''

import os
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
pd.set_option('display.max_columns', None)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rdkit import Chem
# suppress rdkit warning
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from dataset import CSVDataset

def spec_convert(spec, resolution):
    x = []
    y = []
    for i, j in enumerate(spec):
        if j != 0:
            x.append(i*resolution)
            y.append(j)
    return {'m/z': np.array(x), 'intensity': np.array(y)}

def eval(model, device, loader, batch_size, num_atoms, thr):
    model.eval()
    y_pred = []
    ids = []
    smiles = []
    for _, batch in enumerate(tqdm(loader, desc="Iteration")):
        id, s, x_and_adj, env = batch
        x, adj = x_and_adj
        x = x.to(device).to(torch.float32)
        x = x.permute(0, 2, 1)
        adj = adj.to(device).to(torch.float32)
        
        # encode adduct and instrument by one hot
        add = env[:, 0].to(device).to(torch.int64)
        add_oh = F.one_hot(add, num_classes=args.num_add)
        ins = env[:, 1].to(device).to(torch.int64)
        ins_oh = F.one_hot(ins, num_classes=5)
        ce = env[:, 2].to(device).to(torch.float32).unsqueeze(1)
        env = torch.cat((add_oh, ins_oh, ce), 1)

        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_atoms

        with torch.no_grad():
            pred = model(x, adj, env, idx_base) 
            pred = pred / torch.max(pred) # normalize the output
            # post process
            pred = pred.detach().cpu().apply_(lambda x: x if x > thr else 0).to(device)

        y_pred.append(pred.detach().cpu())
        ids = ids + list(id)
        smiles = smiles + list(s)

    y_pred = torch.cat(y_pred, dim = 0)
    return ids, smiles, _, y_pred, _

def batch_filter(supp, num_atoms=200, out_dim=2000, data_type='sdf'): 
    if data_type == 'mgf':
        for _, item in enumerate(supp):
            smiles = item.get('params').get('smiles')
            if item.get('m/z array').max() > out_dim: 
                continue
            mol = Chem.MolFromSmiles(smiles)
            if len(mol.GetAtoms()) > num_atoms or len(mol.GetAtoms()) == 0: 
                continue
            if len(item['m/z array']) == 0 or len(item['m/z array']) == 1: # j0sie: tmp
                continue
            yield item

    elif data_type == 'sdf': # J0sie: need check 
        for _, item in enumerate(supp):
            mol = item
            if mol is None:
                continue
            if not mol.HasProp("MASS SPECTRAL PEAKS"):
                continue
            if mol.GetProp("SPECTRUM TYPE") != "MS2":
                continue
            yield item

    elif data_type == 'csv':
        ATOM_LIST = ['C', 'H', 'O', 'N', 'F', 'S', 'Cl', 'P', 'B', 'Br', 'I']
        ADD_LIST = ['M+H', 'M-H', 'M+H-H2O', 'M+Na', 'M+H-NH3', 'M+H-2H2O', 'M-H-H2O', 'M+NH4', 'M+H-CH4O', 'M+2Na-H', 
                    'M+H-C2H6O', 'M+Cl', 'M+OH', 'M+H+2i', '2M+H', '2M-H', 'M-H-CO2', 'M+2H', 'M-H+2i', 'M+H-CH2O2', 'M+H-C4H8', 
                    'M+H-C2H4O2', 'M+H-C2H4', 'M+CHO2', 'M-H-CH3', 'M+H-H2O+2i', 'M+H-C2H2O', 'M+H-C3H6', 'M+H-CH3', 'M+H-3H2O', 
                    'M+H-HF', 'M-2H', 'M-H2O+H', 'M-2H2O+H']
        INST_LIST = ['HCD', 'QqQ', 'QTOF', 'FT', 'N/A']
        for _, row in supp.iterrows(): 
            mol = Chem.MolFromSmiles(row['SMILES'])
            # check atom number
            if len(mol.GetAtoms()) > num_atoms:
                print('{}: atom number is larger than {}'.format(row['ID'], num_atoms))
                continue
            # check atom type
            for atom in mol.GetAtoms():
                if atom.GetSymbol() not in ATOM_LIST:
                    print('{}: {} is not in the Atom List.'.format(row['ID'], atom.GetSymbol()))
                    continue
            # check precursor type
            if row['Precursor_Type'] not in ADD_LIST:
                print('{}: {} is not in the Precusor Type List.'.format(row['ID'], row['Precursor_Type']))
                continue
            # check source instrument
            if row['Source_Instrument'] not in INST_LIST: 
                print('{}: {} is not in the Intrument List.'.format(row['ID'], row['Source_Instrument']))
                continue
            yield row.to_dict()
        

def load_data(data_path, num_workers, batch_size, data_augmentation, shuffle): 
    supp = pd.read_csv(data_path)
    dataset = CSVDataset([item for item in batch_filter(supp, args.num_atoms, args.out_dim, data_type='csv')], num_points=args.num_atoms, num_ms=args.out_dim, resolution=args.resolution, data_augmentation=data_augmentation)
    
    print('Load {} data from {}.'.format(len(dataset), data_path))
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle, 
        num_workers=num_workers,
        drop_last=True)
    return data_loader 


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Mass Spectrum Prediction')
    parser.add_argument('--test_data_path', type=str, default = '',
                        help='path to test data')
    parser.add_argument('--model_path', type=str, default='', 
                        help='Model path')
    parser.add_argument('--result_path', type=str, default='', 
                        help='Output the resutls path')

    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Size of batch)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')

    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')

    parser.add_argument('--num_atoms', type=int, default=120, 
                        help='Max atom number of molecules, which is also the dimension of inputs')
    parser.add_argument('--num_add', type=int, default=32, 
                        help='Type number of adducts')
    parser.add_argument('--out_dim', type=int, default=1500,
                        help='output dimensionality (default: 1500)')
    parser.add_argument('--resolution', type=float, default=0.2, 
                        help='resolution of the spectra (default: 0.2)')
    parser.add_argument('--post_threshold', type=float, default=0.01, 
                        help='the threshold of postprocess (default: 0.01)')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    assert args.result_path != ''
    assert args.model_path != ''

    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    valid_loader = load_data(data_path=args.test_data_path, num_workers=args.num_workers, batch_size=args.batch_size, data_augmentation=False, shuffle=False)
    
    model = torch.load(args.model_path)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'{str(model)} #Params: {num_params}')

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    print('Evaluating...')
    ids, smiles, _, y_pred, _ = eval(model, device, valid_loader, args.batch_size, args.num_atoms, thr=args.post_threshold)

    pred_list = [spec_convert(spec, args.resolution) for spec in y_pred.tolist()]
    pred_mz = [pred['m/z'] for pred in pred_list]
    pred_intensity = [pred['intensity'] for pred in pred_list]

    result_dir = "".join(args.result_path.split('/')[:-1])
    if result_dir != '':
        os.makedirs(result_dir, exist_ok = True)
    df = pd.DataFrame({'ID': ids, 'SMILES': smiles, 'Pred_M/Z': pred_mz, 'Pred_Intensity': pred_intensity})
    print(df.head())
    print('Save {} test results to {}'.format(len(df), args.result_path))
    df.to_csv(args.result_path, index=None)