'''
Date: 2022-05-20 22:59:30
LastEditors: yuhhong
LastEditTime: 2022-05-28 17:57:18
'''

import os
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn.functional as F
from pyteomics import mgf

from models.pointnet import PointNet_MS
from models.dgcnn import DGCNN_MS
from models.molnet import MolNet_MS
from models.schnet import SchNet_MS
from metrics import get_metrics
from utils import load_data, generate_3d_comformers_csv, generate_2d_comformers_csv, spec_convert

from rdkit import Chem
from rdkit.Chem import Descriptors
import requests



CACTUS = "https://cactus.nci.nih.gov/chemical/structure/{0}/{1}"

def smiles_to_iupac(smiles):
    rep = "iupac_name"
    url = CACTUS.format(smiles, rep)
    try: 
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except:
        print("Sorry, your structure identifier could not be resolved (the request returned a HTML 404 status message)")
        return ""

def smiles_to_inchi(smiles):
    rep = "stdinchi"
    url = CACTUS.format(smiles, rep)
    try: 
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except:
        print("Sorry, your structure identifier could not be resolved (the request returned a HTML 404 status message)")
        return ""

def smiles_to_inchikey(smiles):
    rep = "stdinchikey"
    url = CACTUS.format(smiles, rep)
    try: 
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except:
        print("Sorry, your structure identifier could not be resolved (the request returned a HTML 404 status message)")
        return ""

def inference(model, device, loader, batch_size, num_atoms): 
    model.eval()
    y_pred = []
    ids = []
    smiles = []
    for _, batch in enumerate(tqdm(loader, desc="Iteration")):
        id, s, x, mask, env = batch
        x = x.to(device).to(torch.float32)
        x = x.permute(0, 2, 1)
        
        # encode adduct and instrument by one hot
        add = env[:, 0].to(device).to(torch.int64)
        add_oh = F.one_hot(add, num_classes=args.num_add)
        ins = env[:, 1].to(device).to(torch.int64)
        ins_oh = F.one_hot(ins, num_classes=2)
        ce = env[:, 2].to(device).to(torch.float32).unsqueeze(1)
        env = torch.cat((add_oh, ins_oh, ce), 1)

        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_atoms

        with torch.no_grad():
            pred = model(x, env, idx_base) 
            pred = pred / torch.max(pred) # normalize the output
            # post process
            # pred = pred.detach().cpu().apply_(lambda x: x if x > 0.001 else 0).to(device)

        pred = torch.pow(pred, 2)
        pred = pred.detach().cpu().apply_(lambda x: x if x > 0.001 else 0).to(device) # post process
        
        y_pred.append(pred.detach().cpu())
        ids = ids + list(id)
        smiles = smiles + list(s)

    y_pred = torch.cat(y_pred, dim = 0)
    return ids, smiles, y_pred



if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Mass Spectrum Prediction')
    parser.add_argument('--model', type=str, default='molnet', choices=['pointnet', 'dgcnn', 'molnet', 'schnet'],
                        help='Model to use, [pointnet, dgcnn, molnet, schnet]')
    parser.add_argument('--dataset', type=str, default='merge', choices=['merge'],
                        help='Dataset to use, only [merge] has been supported here')
    parser.add_argument('--ion_mode', type=str, default='P', choices=['P', 'N', 'ALL'], 
                        help='Ion mode used for training and test') 
    parser.add_argument('--mol_type', type=str, default='3d', choices=['2d', '3d'], 
                        help='2D or 3D molecules?')
    parser.add_argument('--test_data_path', type=str, default = '',
                        help='path to test data (.csv)')
    parser.add_argument('--test_mol_path', type=str, default = '',
                        help='path to test molecular comformers data (.sdf.gz)')

    parser.add_argument('--resume_path', type=str, required=True, 
                        help='Pretrained model path')
    parser.add_argument('--result_path', type=str, required=True, 
                        help='Output the resutls path (.csv/.mgf)')
    parser.add_argument('--iupac', type=bool, default=False,
                        help='output IUPAC of molecules')
    parser.add_argument('--inchi', type=bool, default=False,
                        help='output InChI of molecules')
    parser.add_argument('--inchi_key', type=bool, default=False,
                        help='output InChI Key of molecules')

    parser.add_argument('--batch_size', type=int, default=1, 
                        help='Size of batch)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')

    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')

    parser.add_argument('--dropout', type=float, default=0.3,
                        help='dropout rate')
    parser.add_argument('--in_channels', type=int, default=21, 
                        help='Channels of inputs')
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
        if args.test_mol_path == '':
            test_mol_path = args.test_data_path[:-4] + '_3d.sdf.gz'
        else:
            test_mol_path = args.test_mol_path
        if not os.path.exists(test_mol_path): 
            print("Generate 3D comformers for test data...")
            test_mol_path, write_cnt = generate_3d_comformers_csv(args.test_data_path, test_mol_path) 
            print("Write {} 3D molecules to {}\n".format(write_cnt, test_mol_path))
    else:
        # generate 2d comformers
        if args.test_mol_path == '':
            test_mol_path = args.test_data_path[:-4] + '_2d.sdf.gz'
        else:
            test_mol_path = args.test_mol_path
        if not os.path.exists(test_mol_path): 
            print("Generate 2D comformers for test data...")
            test_mol_path, write_cnt = generate_2d_comformers_csv(args.test_data_path, test_mol_path) 
            print("Write {} 2D molecules to {}\n".format(write_cnt, test_mol_path))

    dataset = args.dataset + '_infer'
    valid_loader = load_data(data_path=args.test_data_path, mol_path=test_mol_path, 
                            num_atoms=args.num_atoms, out_dim=args.out_dim, resolution=args.resolution, ion_mode=args.ion_mode, dataset=dataset, 
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
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() and args.cuda else torch.device("cpu")
    print(f'Device: {device}')
    model.to(device)

    model.load_state_dict(torch.load(args.resume_path, map_location=device)['model_state_dict'])

    print('Inference...')
    ids, smiles, y_pred = inference(model, device, valid_loader, args.batch_size, args.num_atoms)

    pred_list = [spec_convert(spec, args.resolution) for spec in y_pred.tolist()]
    pred_mz = [pred['m/z'] for pred in pred_list]
    pred_intensity = [pred['intensity'] for pred in pred_list]
    
    result_dir = "".join(args.result_path.split('/')[:-1])
    os.makedirs(result_dir, exist_ok = True)

    data_df = pd.read_csv(args.test_data_path)
    data_df['ID'] = data_df['ID'].apply(lambda x: str(x))
    res_df = pd.DataFrame({'ID': ids, 'SMILES': smiles, 'Pred M/Z': pred_mz, 'Pred Intensity': pred_intensity})
    res_df = res_df.merge(data_df, how='left', on=['ID', 'SMILES'])
    
    if args.result_path[-3:] == 'csv':
        res_df.to_csv(args.result_path, sep='\t')
    elif args.result_path[-3:] == 'mgf':
        spectra = []
        prefix = 'pred'

        # save results to mgf file
        for idx, row in res_df.iterrows(): 
            smiles = row['SMILES']
            mol = Chem.MolFromSmiles(smiles)
            spectrum = {
                'params': {
                    'title': row['ID'], 
                    'precursor_type': row['Precursor_Type'],
                    'mslevel': '2',
                    'pepmass': Descriptors.ExactMolWt(mol), 
                    'source_instrument': row['Source_Instrument'],
                    'collision_energy': row['Collision_Energy'],
                    'organism': '3DMolMS_v1.0', 
                    'smiles': smiles, 
                    # 'iupac': smiles_to_iupac(smiles), 
                    # 'inchi': smiles_to_inchi(smiles), 
                    # 'inchi_key': smiles_to_inchikey(smiles), 
                    'spectrumid': prefix+'_'+str(idx), 
                },
                'm/z array': np.array([float(i) for i in row['Pred M/Z'].split(',')]),
                'intensity array': np.array([float(i)*1000 for i in row['Pred Intensity'].split(',')])
            } 
            if args.iupac:
                spectrum['params']['iupac'] = smiles_to_iupac(smiles)
                spectrum['params']['inchi'] = smiles_to_inchi(smiles)
                spectrum['params']['inchi_key'] = smiles_to_inchikey(smiles)

            spectra.append(spectrum)
        mgf.write(spectra, args.result_path, file_mode="w", write_charges=False)
    else:
        raise Exception("Not implemented output format. Please choose `.csv` or `.mgf`.")

    print('Save the test results to {}'.format(args.result_path))