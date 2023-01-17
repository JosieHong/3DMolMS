'''
Date: 2021-07-08 18:37:32
LastEditors: yuhhong
LastEditTime: 2022-12-11 15:40:10
'''

import os
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn.functional as F

from models.pointnet import PointNet_MS
from models.dgcnn import DGCNN_MS
from models.molnet import MolNet_MS
from models.schnet import SchNet_MS
from metrics import get_metrics
from utils import load_data, generate_3d_comformers, generate_2d_comformers, spec_convert



def eval(model, device, loader, batch_size, num_points): 
    model.eval()
    y_true = []
    y_pred = []
    ids = []
    smiles = []
    acc = []
    for _, batch in enumerate(tqdm(loader, desc="Iteration")):
        id, s, x, mask, env, y = batch
        x = x.to(device).to(torch.float32)
        x = x.permute(0, 2, 1)
        
        # encode adduct and instrument by one hot
        add = env[:, 0].to(device).to(torch.int64)
        add_oh = F.one_hot(add, num_classes=args.num_add)
        ins = env[:, 1].to(device).to(torch.int64)
        ins_oh = F.one_hot(ins, num_classes=2)
        ce = env[:, 2].to(device).to(torch.float32).unsqueeze(1)
        env = torch.cat((add_oh, ins_oh, ce), 1)

        y = y.to(device)

        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

        with torch.no_grad():
            pred = model(x, env, idx_base) 
            pred = pred / torch.max(pred) # normalize the output
            
        # recover sqrt spectra to original spectra
        y = torch.pow(y, 2)
        pred = torch.pow(pred, 2)
        # post process
        pred = pred.detach().cpu().apply_(lambda x: x if x > 0.001 else 0).to(device)

        y_true.append(y.detach().cpu())
        y_pred.append(pred.detach().cpu())
        acc = acc + F.cosine_similarity(y, pred, dim=1).tolist()
        ids = ids + list(id)
        smiles = smiles + list(s)

    y_true = torch.cat(y_true, dim = 0)
    y_pred = torch.cat(y_pred, dim = 0)
    return ids, smiles, y_true, y_pred, acc



if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Mass Spectrum Prediction (Eval)')
    parser.add_argument('--model', type=str, default='molnet', choices=['pointnet', 'dgcnn', 'molnet', 'schnet'],
                        help='Model to use, [pointnet, dgcnn, molnet, schnet]')
    parser.add_argument('--dataset', type=str, default='gnps', choices=['nist', 'gnps', 'massbank', 'merge', 'hmdb', 'agilent'], 
                        help='Dataset to use, [nist, gnps, massbank, merge, hmdb, agilent]')
    parser.add_argument('--ion_mode', type=str, default='P', choices=['P', 'N', 'ALL'], 
                        help='Ion mode used for training and test') 
    parser.add_argument('--mol_type', type=str, default='3d', choices=['2d', '3d'], 
                        help='2D or 3D molecules?')
    parser.add_argument('--test_data_path', type=str, required=True, 
                        help='path to test data (.mgf)')

    parser.add_argument('--resume_path', type=str, required=True,
                        help='Pretrained model path')
    parser.add_argument('--result_path', type=str, required=True, 
                        help='Output the resutls path')

    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Size of batch)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')

    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')

    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate (default: 0.3)')
    parser.add_argument('--in_channels', type=int, default=21, 
                        help='Channels of inputs (default: 21)')
    parser.add_argument('--num_atoms', type=int, default=300, 
                        help='Max atom number of each input molecule (default: 300)')
    parser.add_argument('--num_add', type=int, default=5, 
                        help='Type number of adducts (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=2048, 
                        help='Dimension of embeddings (default: 2048)')
    parser.add_argument('--out_dim', type=int, default=1500, 
                        help='Output dimensionality (default: 1500)')
    parser.add_argument('--resolution', type=float, default=1, 
                        help='Resolution of the output spectra (default: 1)')
    parser.add_argument('--k', type=int, default=5, 
                        help='Number of nearest neighbors to use (default: 5)')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    if args.mol_type == '3d': 
        # generate 3d comformers
        test_mol_path = args.test_data_path[:-4] + '_3d.sdf.gz'
        if not os.path.exists(test_mol_path): 
            print("Generate 3D comformers for test data...")
            test_mol_path, write_cnt = generate_3d_comformers(args.test_data_path, test_mol_path) 
            print("Write {} 3D molecules to {}\n".format(write_cnt, test_mol_path))
    else:
        # generate 2d comformers
        test_mol_path = args.test_data_path[:-4] + '_2d.sdf.gz'
        if not os.path.exists(test_mol_path): 
            print("Generate 2D comformers for test data...")
            test_mol_path, write_cnt = generate_2d_comformers(args.test_data_path, test_mol_path) 
            print("Write {} 2D molecules to {}\n".format(write_cnt, test_mol_path))

    valid_loader = load_data(data_path=args.test_data_path, mol_path=test_mol_path, 
                            num_atoms=args.num_atoms, out_dim=args.out_dim, resolution=args.resolution, ion_mode=args.ion_mode, dataset=args.dataset, 
                            num_workers=args.num_workers, batch_size=args.batch_size, data_augmentation=False, shuffle=False)
    
    if args.model == 'pointnet':
        model = PointNet_MS(args)
    elif args.model == 'dgcnn':
        model = DGCNN_MS(args)
    elif args.model == 'molnet':
        model = MolNet_MS(args)
    elif args.model == 'schnet':
        model = SchNet_MS(args)
    num_params = sum(p.numel() for p in model.parameters())
    # print(f'{str(model)} #Params: {num_params}')
    print(f'#Params: {num_params}')
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    model.load_state_dict(torch.load(args.resume_path, map_location=device)['model_state_dict'])

    print('Evaluating...')
    ids, smiles, y_true, y_pred, acc = eval(model, device, valid_loader, batch_size=args.batch_size, num_points=args.num_atoms)
    print("Validation: Acc: {} +- {}".format(np.sum(acc)/len(acc), np.std(acc)))

    ms_list = [spec_convert(spec, args.resolution) for spec in y_true.tolist()]
    ms_mz = [ms['m/z'] for ms in ms_list]
    ms_intensity = [ms['intensity'] for ms in ms_list]

    pred_list = [spec_convert(spec, args.resolution) for spec in y_pred.tolist()]
    pred_mz = [pred['m/z'] for pred in pred_list]
    pred_intensity = [pred['intensity'] for pred in pred_list]
    
    result_dir = "".join(args.result_path.split('/')[:-1])
    os.makedirs(result_dir, exist_ok = True)
    print('Save the test results to {}'.format(args.result_path))
    res_df = pd.DataFrame({'ID': ids, 'SMILES': smiles, 'Accuracy': acc, 'M/Z': ms_mz, 'Intensity': ms_intensity, 'Pred M/Z': pred_mz, 'Pred Intensity': pred_intensity})
    res_df.to_csv(args.result_path, sep='\t')

    res_df['Dataset'] = res_df.apply(lambda x: x['ID'].split('_')[0], axis=1)
    print('\nmean accuracy:')
    print(res_df.groupby(by='Dataset')['Accuracy'].mean())
    print('\nmean std:')
    print(res_df.groupby(by='Dataset')['Accuracy'].std())