'''
Date: 2021-07-08 18:37:32
LastEditors: yuhhong
LastEditTime: 2022-12-09 23:41:20
'''
import os
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
pd.set_option('display.max_columns', None)

import torch
import torch.nn.functional as F

# suppress rdkit warning
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from utils import load_data, generate_3d_comformers_csv, generate_2d_comformers_csv, spec_convert



def eval(model, device, loader, batch_size, num_points): 
    model.eval()
    y_pred = []
    ids = []
    smiles = []
    adducts = []
    instruments = []
    collision_energies = []
    for _, batch in enumerate(tqdm(loader, desc="Iteration")): 
        id, s, x, mask, env = batch
        x = x.to(device).to(torch.float32)
        x = x.permute(0, 2, 1)
        
        # save parameters for return
        adducts += env[:, 0].detach().cpu().tolist()
        instruments += env[:, 1].detach().cpu().tolist()
        collision_energies += env[:, 2].detach().cpu().tolist()
        ids += list(id)
        smiles += list(s)

        # encode adduct and instrument by one hot
        add = env[:, 0].to(device).to(torch.int64)
        add_oh = F.one_hot(add, num_classes=args.num_add)
        ins = env[:, 1].to(device).to(torch.int64)
        ins_oh = F.one_hot(ins, num_classes=5)
        ce = env[:, 2].to(device).to(torch.float32).unsqueeze(1)
        env = torch.cat((add_oh, ins_oh, ce), 1)

        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        
        with torch.no_grad():
            pred = model(x, env, idx_base) 
            pred = pred / torch.max(pred) # normalize the output
        
        # recover sqrt spectra to original spectra
        pred = torch.pow(pred, 2)
        # post process
        pred = pred.detach().cpu().apply_(lambda x: x if x > 0.001 else 0).to(device)
        y_pred.append(pred.detach().cpu())

    y_pred = torch.cat(y_pred, dim = 0)
    return ids, smiles, _, y_pred, _, adducts, instruments, collision_energies



if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Mass Spectrum Prediction')
    parser.add_argument('--mol_type', type=str, default='3d', choices=['2d', '3d'], 
                        help='2D or 3D molecules?')
    parser.add_argument('--test_data_path', type=str, default = '',
                        help='path to test data (.mgf)')
    parser.add_argument('--test_mol_path', type=str, default = '',
                        help='path to test data (.sdf)')
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
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    assert args.result_path != ''
    assert args.model_path != ''

    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    if args.test_mol_path == '':
        if args.mol_type == '3d': 
            # generate 3d comformers
            test_mol_path = args.test_data_path[:-4] + '_3d.sdf.gz'
            if not os.path.exists(test_mol_path): 
                print("Generate 3D comformers for test data...")
                test_mol_path, write_cnt = generate_3d_comformers_csv(args.test_data_path, test_mol_path) 
                print("Write {} 3D molecules to {}\n".format(write_cnt, test_mol_path))
        else:
            # generate 2d comformers
            test_mol_path = args.test_data_path[:-4] + '_2d.sdf.gz'
            if not os.path.exists(test_mol_path): 
                print("Generate 2D comformers for test data...")
                test_mol_path, write_cnt = generate_2d_comformers_csv(args.test_data_path, test_mol_path) 
                print("Write {} 2D molecules to {}\n".format(write_cnt, test_mol_path))
    else:
        test_mol_path = args.test_mol_path
        assert os.path.exists(test_mol_path)
            
    valid_loader = load_data(data_path=args.test_data_path, mol_path=test_mol_path, 
                            num_atoms=args.num_atoms, out_dim=args.out_dim, resolution=args.resolution, dataset='merge_infer', 
                            num_workers=args.num_workers, batch_size=args.batch_size, data_augmentation=False, shuffle=False)

    model = torch.jit.load(args.model_path, map_location=device)
    
    model.device = device
    num_params = sum(p.numel() for p in model.parameters())
    print(f'#Params: {num_params}')
    model.to(device)

    print('Evaluating...')
    ids, smiles, _, y_pred, _, adducts, instruments, collision_energies = eval(model, device, valid_loader, args.batch_size, args.num_atoms)

    # convert integer to string
    DECODE_ADD = {0: 'M+H', 1: 'M-H', 2: 'M-H2O+H', 3: 'M+Na', 4: 'M+H-NH3', 5: 'M-2H2O+H', 6: 'M-H-H2O', 7: 'M+NH4', 8: 'M+H-CH4O', 9: 'M+2Na-H', 10: 'M+H-C2H6O', 11: 'M+Cl', 12: 'M+OH', 13: 'M+H+2i', 14: '2M+H', 15: '2M-H', 16: 'M-H-CO2', 17: 'M+2H', 18: 'M-H+2i', 19: 'M+H-CH2O2', 20: 'M+H-C4H8', 21: 'M+H-C2H4O2', 22: 'M+H-C2H4', 23: 'M+CHO2', 24: 'M-H-CH3', 25: 'M+H-C2H2O', 26: 'M+H-C3H6', 27: 'M+H-CH3', 28: 'M+H-3H2O', 29: 'M+H-HF', 30: 'M-2H'}
    DECODE_INS = {0: 'HCD', 1: 'QqQ', 2: 'QTOF', 3: 'FT', 4: 'N/A'}
    adducts = [DECODE_ADD[add] for add in adducts]
    instruments = [DECODE_INS[ins] for ins in instruments]

    pred_list = [spec_convert(spec, args.resolution) for spec in y_pred.tolist()]
    pred_mz = [pred['m/z'] for pred in pred_list]
    pred_intensity = [pred['intensity'] for pred in pred_list]

    result_dir = "".join(args.result_path.split('/')[:-1])
    if result_dir != '':
        os.makedirs(result_dir, exist_ok = True)

    # output .csv file
    df = pd.DataFrame({'ID': ids, 'SMILES': smiles, 'Precursor_Type': adducts, 'Instrument': instruments, 
                        'Collision_Energy': collision_energies, 'Pred_M/Z': pred_mz, 'Pred_Intensity': pred_intensity})

    print('Save the test results to {}'.format(args.result_path))
    df.to_csv(args.result_path, index=None)