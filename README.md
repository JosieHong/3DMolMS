# 3DMolMS

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa] (free for academic use) 

**3D** **Mol**ecular Network for **M**ass **S**pectra Prediction (3DMolMS) is a deep neural network model to predict the MS/MS spectra of compounds from their 3D conformations. The encoder for molecular representation learned in MS/MS prediction could also be transferred to other molecular-related tasks enhancing the performance, such as retention time and collisional cross section prediction. 

[[paper on Bioinformatics]](https://academic.oup.com/bioinformatics/article/39/6/btad354/7186501) | [[online service on GNPS]](https://spectrumprediction.gnps2.org)



## Updates 

- 2023.10.30 (v1.10): enlarging training set by MoNA and Waters QTOF datasets. 

- 2023.10.22 (v1.02): pretraining on QM9-mu dataset + ETKDG algorithm. We establish a dataset from QM9-mu (dipole moment) with the generated conformations using ETKDG for pretraining 3DMolMS. It helps the model learning knowledge of molecular 3D conformations and pretraining enhances the performance on MS/MS slightly (~0.01 cosine similarity). 

- 2023.09.14 (v1.01): data augmentation by flipping atomic coordinates. Notably, this model is sensitive to the geometric structure of molecules. For tasks insensitive to geometric structure, e.g. mass spectrometry is chirally blind, please use data augmentation. However, for the tasks sensitive to geometric structure, e.g. different enantiomers with varying retention times, avoid data augmentation. 

- 2023.06.30 (v1.00): initial version. 



## Usage

Step 0: Setup the anaconda environment by the following commands: 

```bash
conda create -n molnet 
conda activate molnet
# For RDKit
# https://www.rdkit.org/docs/GettingStartedInPython.html
conda install -c conda-forge rdkit

# For PyTorch 1.11.0
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

pip install -r requirements.txt
```

Step 1: Generate custom test data. If you already have test data, please convert it into a supported format, i.e. csv, mgf, or [customed pkl](https://github.com/JosieHong/3DMolMS/blob/main/molmspack/data_utils/all2pkl.py). Here is an input example of csv format (`./demo_input.csv`): 

```
ID,SMILES,Precursor_Type,Collision_Energy
demo_0,O=S(=O)(O)CC(O)CN1CCN(CCO)CC1,[M+H]+,20
demo_1,O=S(=O)(O)CC(O)CN1CCN(CCO)CC1,[M+H]+,40
demo_2,NC(CCCCn1cccc2nc(NCCCC(N)C(=O)O)nc1-2)C(=O)O,[M+H]+,20
demo_3,NC(CCCCn1cccc2nc(NCCCC(N)C(=O)O)nc1-2)C(=O)O,[M+H]+,40
demo_4,O=S(=O)(O)CC(O)CN1CCN(CCO)CC1,[M-H]-,20
demo_5,O=S(=O)(O)CC(O)CN1CCN(CCO)CC1,[M-H]-,40
```

Please notice that the unsupported input will be filtered out automatically when loading the dataset. The supported inputs are shown in the following table. 

| Item             | Supported input                                           |
|------------------|-----------------------------------------------------------|
| Atom number      | <=300                                                     |
| Atom types       | 'C', 'O', 'N', 'H', 'P', 'S', 'F', 'Cl', 'B', 'Br', 'I'   |
| Precursor types  | '[M+H]+', '[M-H]-', '[M+H-H2O]+', '[M+Na]+'               |
| Collision energy | any number                                                |

Step 2: Download the model weights (`molnet_qtof_etkdgv3.pt.zip`) from [Google Drive](https://drive.google.com/drive/folders/1fWx3d8vCPQi-U-obJ3kVL3XiRh75x5Ce?usp=drive_link). If you have trained your own model, please ignore this step. 

Step 3: Predict the MS/MS or evaluate 3DMolMS. 

IF you want to use 3DMolMS on certain molecules that do not know the experimental MS/MS, please use the following commands: 

```bash
python pred.py --test_data ./demo/demo_input.csv \
--model_config_path ./config/molnet.yml \
--data_config_path ./config/preprocess_etkdgv3.yml \
--resume_path ./check_point/molnet_qtof_etkdgv3.pt \
--result_path ./demo/demo_output.csv
```

IF you have the experimental MS/MS and want to evaluate 3DMolMS, please use the following command: 

```bash
python eval.py --test_data ./demo/demo_input.mgf \
--model_config_path ./config/molnet.yml \
--data_config_path ./config/preprocess_etkdgv3.yml \
--resume_path ./check_point/molnet_qtof_etkdgv3.pt
```



## Train your own model

Please set up the environment as shown in step 0 from the above section. 

Step 1: Download the pretrained model (`molnet_pre_etkdgv3.pt.zip`) from [Google Drive](https://drive.google.com/drive/folders/1fWx3d8vCPQi-U-obJ3kVL3XiRh75x5Ce?usp=drive_link) or pretrain the model by yourself. The details of pretraining the model on [QM9](https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904)-mu are demonstrated at [./docs/PRETRAIN.md](https://github.com/JosieHong/3DMolMS/blob/main/docs/PRETRAIN.md). 

Step 2: Gather the datasets separately, unzip and put them in `./data/`. In the latest version, we use 4 datasets to train the model: (1) Agilent DPCL is provided by [Agilent Technologies](https://www.agilent.com/). (2) [NIST20](https://www.nist.gov/programs-projects/nist23-updates-nist-tandem-and-electron-ionization-spectral-libraries) is academically available with a License. (3) [MoNA](https://mona.fiehnlab.ucdavis.edu/downloads) is publicly available. (4) Waters QTOF is our own experimental dataset. The structure of data directory is: 

```bash
|- data
  |- origin
    |- Agilent_Combined.sdf
    |- Agilent_Metlin.sdf
    |- hr_msms_nist.SDF
    |- MoNA-export-All_LC-MS-MS_QTOF.sdf
    |- waters_qtof.mgf
```

Step 3: Use the following commands to preprocess the datasets. Please input the dataset you use at `--dataset` and choose the instrument type in `qtof` and `orbitrap`. `--maxmin_pick` means using the MaxMin algorithm in picking training molecules, otherwise, the random choice is applied. The settings of datasets are in `./preprocess_etkdgv3.yml`. 

```bash
python preprocess.py --dataset agilent nist mona waters --instrument_type qtof --data_config_path ./config/preprocess_etkdgv3.yml --mgf_dir ./data/mgf_debug/
```

Step 4: Use the following commands to train the model. The settings of model and training are in `./config/molnet.yml`. 

```bash
python train.py --train_data ./data/qtof_etkdgv3_train.pkl \
--test_data ./data/qtof_etkdgv3_test.pkl \
--model_config_path ./config/molnet.yml \
--data_config_path ./config/preprocess_etkdgv3.yml \
--checkpoint_path ./check_point/molnet_qtof_etkdgv3.pt \
--transfer \
--resume_path ./check_point/molnet_pre_etkdgv3.pt
```

In addition, 3DMolMS can be used in molecular properties prediction and generating refer libraries for molecular identification. We give the retention time prediction and cross-collision section prediction as two examples of molecular properties prediction. Please see the details in [./docs/PROP_PRED.md](https://github.com/JosieHong/3DMolMS/blob/main/docs/PROP_PRED.md). The examples of generating refer libraries can be found in [./docs/GEN_REFER_LIB.md](https://github.com/JosieHong/3DMolMS/blob/main/docs/GEN_REFER_LIB.md). 



## Citation

If you feel this work useful, please cite: 

```
@article{hong20233dmolms,
  title={3DMolMS: prediction of tandem mass spectra from 3D molecular conformations},
  author={Hong, Yuhui and Li, Sujun and Welch, Christopher J and Tichy, Shane and Ye, Yuzhen and Tang, Haixu},
  journal={Bioinformatics},
  volume={39},
  number={6},
  pages={btad354},
  year={2023},
  publisher={Oxford University Press}
}
```

Friendly links to other MS/MS prediction methods: 

- Goldman, Samuel, et al. "Prefix-tree decoding for predicting mass spectra from molecules." arXiv preprint arXiv:2303.06470 (2023).
- Young, Adamo, Bo Wang, and Hannes Röst. "MassFormer: Tandem mass spectrum prediction with graph transformers." arXiv preprint arXiv:2111.04824 (2021). 
- Wang, Fei, et al. "CFM-ID 4.0: more accurate ESI-MS/MS spectral prediction and compound identification." Analytical chemistry 93.34 (2021): 11692-11700.
- Wei, Jennifer N., et al. "Rapid prediction of electron–ionization mass spectrometry using neural networks." ACS central science 5.4 (2019): 700-708. 

---

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg