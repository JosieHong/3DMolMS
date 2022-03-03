'''
Date: 2022-10-05 14:27:59
LastEditors: yuhhong
LastEditTime: 2022-12-11 15:40:36
'''
import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, MultiStepLR, ReduceLROnPlateau

from models.pointnet import PointNet_MS
from models.dgcnn import DGCNN_MS
from models.molnet import MolNet_MS
from utils import load_data, generate_3d_comformers, generate_2d_comformers, get_lr, reg_criterion



def train(model, device, loader, optimizer, batch_size, num_points): 
    accuracy = 0
    with tqdm(total=len(loader)) as bar:
        for step, batch in enumerate(loader):
            _, _, x, mask, env, y = batch
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

            optimizer.zero_grad()
            model.train()
            pred = model(x, env, idx_base) 
            loss = reg_criterion(pred, y)
            loss.backward()

            bar.set_description('Train')
            bar.set_postfix(lr=get_lr(optimizer), loss=loss.item())
            bar.update(1)

            optimizer.step()

            # recover sqrt spectra to original spectra
            y = torch.pow(y, 2)
            pred = torch.pow(pred, 2)
            accuracy += F.cosine_similarity(pred, y, dim=1).mean().item()

    return accuracy / (step + 1)

def eval(model, device, loader, batch_size, num_points): 
    model.eval()
    y_true = []
    y_pred = []
    ids = []
    smiles = []
    acc = []
    with tqdm(total=len(loader)) as bar:
        for _, batch in enumerate(loader):
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
                
            bar.set_description('Eval')
            bar.update(1)
    
            # recover sqrt spectra to original spectra
            y = torch.pow(y, 2)
            pred = torch.pow(pred, 2)
            # post process
            pred = pred.detach().cpu().apply_(lambda x: x if x > 0.01 else 0).to(device)

            acc = acc + F.cosine_similarity(y, pred, dim=1).tolist()
            ids = ids + list(id)
            smiles = smiles + list(s)
            
    return ids, smiles, y_true, y_pred, acc



if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Mass Spectrum Prediction (Train)')
    parser.add_argument('--model', type=str, default='molnet', choices=['pointnet', 'dgcnn', 'molnet'],
                        help='Model to use, [pointnet, dgcnn, molnet]')
    parser.add_argument('--dataset', type=str, default='gnps', choices=['nist', 'gnps', 'massbank', 'merge', 'hmdb', 'agilent'], 
                        help='Dataset to use, [nist, gnps, massbank, merge, hmdb, agilent]')
    parser.add_argument('--ion_mode', type=str, default='P', choices=['P', 'N', 'ALL'], 
                        help='Ion mode used for training and test') 
    parser.add_argument('--mol_type', type=str, default='3d', choices=['2d', '3d'], 
                        help='2D or 3D molecules?')
    parser.add_argument('--train_data_path', type=str, required=True,
                        help='Path to training data (.mgf)')
    parser.add_argument('--test_data_path', type=str, required=True, 
                        help='Path to test data (.mgf)')

    parser.add_argument('--log_dir', type=str, default="", 
                        help='Tensorboard log directory')
    parser.add_argument('--checkpoint_path', type=str, default = '', 
                        help='Path to save checkpoint')
    parser.add_argument('--resume_path', type=str, default='', 
                        help='Path to pretrained model')
    parser.add_argument('--data_augmentation', action='store_true', 
                        help='Whether to use data augmentation')
    parser.add_argument('--transfer', action='store_true', 
                        help='Whether to load the pretrained encoder')
    parser.add_argument('--ex_model_path', type=str, default='',
                        help='Path to export the whole model (structure & weights)')
                        
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=100, 
                        help='Number of episode to train ')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers (default: 0)')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help='l2')

    parser.add_argument('--device', type=int, default=0,
                        help='Which gpu to use if any (default: 0)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='Enables CUDA training')

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
        train_mol_path = args.train_data_path[:-4] + '_3d.sdf.gz'
        test_mol_path = args.test_data_path[:-4] + '_3d.sdf.gz'

        if not os.path.exists(train_mol_path): 
            print("Generate 3D comformers for training data...")
            train_mol_path, write_cnt = generate_3d_comformers(args.train_data_path, train_mol_path) 
            print("Write {} 3D molecules to {}\n".format(write_cnt, train_mol_path))
        if not os.path.exists(test_mol_path): 
            print("Generate 3D comformers for test data...")
            test_mol_path, write_cnt = generate_3d_comformers(args.test_data_path, test_mol_path) 
            print("Write {} 3D molecules to {}\n".format(write_cnt, test_mol_path))
    else:
        # generate 2d comformers
        train_mol_path = args.train_data_path[:-4] + '_2d.sdf.gz'
        test_mol_path = args.test_data_path[:-4] + '_2d.sdf.gz'

        if not os.path.exists(train_mol_path): 
            print("Generate 2D comformers for training data...")
            train_mol_path, write_cnt = generate_2d_comformers(args.train_data_path, train_mol_path) 
            print("Write {} 2D molecules to {}\n".format(write_cnt, train_mol_path))
        if not os.path.exists(test_mol_path): 
            print("Generate 2D comformers for test data...")
            test_mol_path, write_cnt = generate_2d_comformers(args.test_data_path, test_mol_path) 
            print("Write {} 2D molecules to {}\n".format(write_cnt, test_mol_path))

    print("Loading the data...")
    train_loader = load_data(data_path=args.train_data_path, mol_path=train_mol_path, 
                            num_atoms=args.num_atoms, out_dim=args.out_dim, resolution=args.resolution, ion_mode=args.ion_mode, dataset=args.dataset, 
                            num_workers=args.num_workers, batch_size=args.batch_size, data_augmentation=False, shuffle=True)
    valid_loader = load_data(data_path=args.test_data_path, mol_path=test_mol_path, 
                            num_atoms=args.num_atoms, out_dim=args.out_dim, resolution=args.resolution, ion_mode=args.ion_mode, dataset=args.dataset, 
                            num_workers=args.num_workers, batch_size=args.batch_size, data_augmentation=False, shuffle=False)
    
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    print(f'Device: {device}')
    
    if args.model == 'pointnet':
        model = PointNet_MS(args)
    elif args.model == 'dgcnn':
        model = DGCNN_MS(args)
    elif args.model == 'molnet':
        model = MolNet_MS(args)
    num_params = sum(p.numel() for p in model.parameters())
    # print(f'{str(model)} #Params: {num_params}')
    print(f'#Params: {num_params}')
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    best_valid_acc = 0
    if args.transfer:
        print("Load the pretrained encoder...")
        state_dict = torch.load(args.resume_path, map_location=device)['model_state_dict']
        encoder_dict = {}
        for name, param in state_dict.items():
            if not name.startswith("decoder.fc"):
                encoder_dict[name] = param
        model.load_state_dict(encoder_dict, strict=False)
    elif args.resume_path != '':
        print("Load the checkpoints...")
        model.load_state_dict(torch.load(args.resume_path, map_location=device)['model_state_dict'])
        optimizer.load_state_dict(torch.load(args.resume_path, map_location=device)['optimizer_state_dict'])
        scheduler.load_state_dict(torch.load(args.resume_path, map_location=device)['scheduler_state_dict'])
        best_valid_acc = torch.load(args.resume_path)['best_val_acc']

    if args.checkpoint_path != '':
        checkpoint_dir = "/".join(args.checkpoint_path.split('/')[:-1])
        os.makedirs(checkpoint_dir, exist_ok = True)

    if args.log_dir != '':
        writer = SummaryWriter(log_dir=args.log_dir)

    early_stop_step = 10
    early_stop_patience = 0
    for epoch in range(1, args.epochs + 1): 
        print("\n=====Epoch {}".format(epoch))
        train_acc = train(model, device, train_loader, optimizer, batch_size=args.batch_size, num_points=args.num_atoms)
        
        _, _, _, _, acc = eval(model, device, valid_loader, batch_size=args.batch_size, num_points=args.num_atoms)
        valid_acc = np.sum(acc)/(len(acc))

        print("Train: Acc: {}, \nValidation: Acc: {}".format(train_acc, valid_acc))

        if args.log_dir != '':
            writer.add_scalar('valid/mae', valid_acc, epoch)
            writer.add_scalar('train/mae', train_acc, epoch)

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc

            if args.checkpoint_path != '':
                print('Saving checkpoint...')
                checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'best_val_acc': best_valid_acc, 'num_params': num_params}
                torch.save(checkpoint, args.checkpoint_path)

            early_stop_patience = 0
            print('Early stop patience reset')
        else:
            early_stop_patience += 1
            print('Early stop count: {}/{}'.format(early_stop_patience, early_stop_step))

        scheduler.step(valid_acc) # ReduceLROnPlateau
        print(f'Best cosine similarity so far: {best_valid_acc}')

        if early_stop_patience == early_stop_step:
            print('Early stop!')
            break

    if args.log_dir != '':
        writer.close()
    
    if args.ex_model_path != '': # export the model
        print('Export the model...')
        model_scripted = torch.jit.script(model) # Export to TorchScript
        model_scripted.save(args.ex_model_path) # Save

    