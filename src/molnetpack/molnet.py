import os
import numpy as np
import pandas as pd
import yaml
import pickle
from pyteomics import mgf
import gdown
import zipfile

import torch
from torch.utils.data import DataLoader

from rdkit import Chem
# ignore the warning
from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import Descriptors

import matplotlib.pyplot as plt
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
from PIL import Image
from PIL import ImageFilter
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from pathlib import Path

from molnet_yuhhong.model import MolNet_MS
from molnet_yuhhong.dataset import Mol_Dataset
from molnet_yuhhong.data_utils import csv2pkl_wfilter, nce2ce, precursor_calculator
from molnet_yuhhong.data_utils import filter_spec, mgf2pkl, ms_vec2dict
from molnet_yuhhong.utils import pred_step



class MolNet(): 
    def __init__(self, device, seed): 
        print('Version information here')
        self.device = device

        # get the current file's directory
        self.current_path = Path(__file__).parent

        # hard code in ./config/preprocess_etkdgv3.yml
        data_config_path = self.current_path / Path('./config/preprocess_etkdgv3.yml')
        with open(data_config_path, 'r') as f: 
            self.data_config = yaml.load(f, Loader=yaml.FullLoader) 
        
        # hard code in ./config/molnet.yml
        msms_config_path = self.current_path / Path('./config/molnet.yml')
        with open(msms_config_path, 'r') as f: 
            self.msms_config = yaml.load(f, Loader=yaml.FullLoader)
        
        # hard code in ./config/molnet_ccs.yml
        ccs_config_path = self.current_path / Path('./config/molnet_ccs.yml')
        with open(ccs_config_path, 'r') as f: 
            self.ccs_config = yaml.load(f, Loader=yaml.FullLoader)
        
        # do not load these while initializing
        self.pkl_dict = None # pkl format data
        self.valid_loader = None # PyTorch data loader

        self.msms_model = None # PyTorch model
        self.msms_res_df = None # pandas dataframe
        
        self.ccs_model = None # PyTorch model
        self.ccs_res_df = None # pandas dataframe

        self.init_random_seed(seed)

    def init_random_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        return

    def load_data(self, path_to_test_data): 
        test_format = path_to_test_data.split('.')[-1]
        if test_format == 'csv': # convert csv file into pkl 
            self.pkl_dict = csv2pkl_wfilter(path_to_test_data, self.data_config['encoding'])
        elif test_format == 'mgf': # convert mgf file into pkl 
            origin_spectra = mgf.read(path_to_test_data)
            filter_spectra, _ = filter_spec(origin_spectra, 
                                            self.data_config['all'], 
                                            type2charge=self.data_config['encoding']['type2charge'])
            self.pkl_dict = mgf2pkl(filter_spectra, self.data_config['encoding'])
        elif test_format == 'pkl': # load pkl directly
            with open(path_to_test_data, 'rb') as file: 
                self.pkl_dict = pickle.load(file)
        else: 
            raise ValueError('Unsupported format: {}'.format(test_format))
        print('\nLoad {} data from {}'.format(len(self.pkl_dict), path_to_test_data))

        valid_set = Mol_Dataset(self.pkl_dict)
        self.valid_loader = DataLoader(
                            valid_set,
                            batch_size=1, 
                            shuffle=False, 
                            num_workers=0, 
                            drop_last=True)

    def pred_msms(self, path_to_results): 
        # create model
        self.msms_model = MolNet_MS(self.msms_config['model']).to(self.device)
        num_params = sum(p.numel() for p in self.msms_model.parameters())

        # load the best checkpoint
        checkpoint_path = str(self.current_path / Path(self.msms_config['test']['local_path']))
        if not os.path.exists(checkpoint_path): 
            checkpoint_dir = str(self.current_path / Path('/'.join(self.msms_config['test']['local_path'].split('/')[:-1])))
            os.makedirs(checkpoint_dir, exist_ok=True) 

            checkpoint_zip_path = checkpoint_path + '.zip'
            print('Download the checkpoints from Google Drive to {}'.format(checkpoint_zip_path))
            gdown.download(self.msms_config['test']['google_drive_link'], checkpoint_zip_path, fuzzy=True)
            
            print('Unzip {}'.format(checkpoint_zip_path))
            with zipfile.ZipFile(checkpoint_zip_path, 'r') as zip_ref:
                zip_ref.extractall(checkpoint_dir)
        
        self.msms_model.load_state_dict(torch.load(checkpoint_path, map_location=self.device)['model_state_dict'])
    
        # pred
        id_list, pred_list = pred_step(self.msms_model, self.device, self.valid_loader, 
                                        batch_size=1, num_points=self.msms_config['model']['max_atom_num'])
        pred_list = [ms_vec2dict(spec, float(self.msms_config['model']['resolution'])) for spec in pred_list.tolist()]
        pred_mz = [pred['m/z'] for pred in pred_list]
        pred_intensity = [pred['intensity'] for pred in pred_list]

        # output the results
        result_dir = "".join(path_to_results.split('/')[:-1])
        os.makedirs(result_dir, exist_ok = True)

        decoding_precursor_type = {','.join(map(str, v)): k for k, v in self.data_config['encoding']['precursor_type'].items()}
        ce_list = []
        add_list = []
        smiles_list = []
        for d in self.pkl_dict:
            precursor_type = decoding_precursor_type[','.join(map(str, map(int, d['env'][1:])))]
            smiles = d['smiles']
            mass = Descriptors.ExactMolWt(Chem.MolFromSmiles(smiles))
            precurosr_mz = precursor_calculator(precursor_type, mass)
            ce_list.append(nce2ce(d['env'][0], precurosr_mz, int(self.data_config['encoding']['type2charge'][precursor_type])))
            add_list.append(precursor_type)
            smiles_list.append(smiles)

        self.msms_res_df = pd.DataFrame({'ID': id_list, 'SMILES': smiles_list, 
                            'Collision Energy': ce_list, 'Precursor Type': add_list, 
                            'Pred M/Z': pred_mz, 'Pred Intensity': pred_intensity})
        spectra = []
        prefix = 'pred'
        for idx, row in self.msms_res_df.iterrows(): 
            spectrum = {
                'params': {
                    'title': row['ID'], 
                    'mslevel': '2', 
                    'organism': '3DMolMS_v1.1', 
                    'spectrumid': prefix+'_'+str(idx), 
                    'smiles': row['SMILES'], 
                    'collision_energy': row['Collision Energy'],
                    'precursor_type': row['Precursor Type'],
                },
                'm/z array': np.array([float(i) for i in row['Pred M/Z'].split(',')]),
                'intensity array': np.array([float(i)*1000 for i in row['Pred Intensity'].split(',')])
            } 
            spectra.append(spectrum)
        
        mgf.write(spectra, path_to_results, file_mode="w", write_charges=False)
        print('\nSaved the results to {}'.format(path_to_results))

        return spectra # mgf

    def pred_ccs(self,): 
        # create model

        # load the best checkpoint

        # pred
        pass # csv
    
    def plot_msms(self, dir_to_img): 
        os.makedirs(dir_to_img, exist_ok = True)

        img_dpi = 300
        y_max = 1
        x_max = None # varies in different MS/MS
        bin_width = 0.4 # please adujst it for good looking
        figsize = (9, 4)

        for idx, row in self.msms_res_df.iterrows(): 
            fig, ax = plt.subplots(figsize=figsize)
            mz_values = np.array([float(i) for i in row['Pred M/Z'].split(',')])
            x_max = np.max(mz_values)
            plt.bar(mz_values, 
                    np.array([float(i)*y_max for i in row['Pred Intensity'].split(',')]), 
                    width=bin_width, color='k')
            plt.xlim(0, x_max)
            plt.title('ID: '+row['ID'])
            plt.xlabel('M/Z')
            plt.ylabel('Relative intensity')

            # plot the molecules 
            mol = Chem.MolFromSmiles(row['SMILES'])
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
            mol_img = Draw.MolToImage(mol, size=(800, 800))
            # make the backgrounf transparent
            alpha_img = mol_img.convert('L')
            alpha_img = Image.fromarray(255 - np.array(alpha_img))
            mol_img.putalpha(alpha_img)
            imagebox = OffsetImage(mol_img, zoom=72./img_dpi) # https://stackoverflow.com/questions/48639369/does-adding-images-in-pyplot-lowers-their-resolution
            mol_ab = AnnotationBbox(imagebox, (x_max*0.28, y_max*0.64), frameon=False, xycoords='data')
            ax.add_artist(mol_ab)

            plt.savefig(os.path.join(dir_to_img, row['ID']), dpi=img_dpi, bbox_inches='tight')
            plt.close()
        print('\nSaved the plotted MS/MS to {}'.format(dir_to_img))
        return 