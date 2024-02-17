# 3DMolMS

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa] (free for academic use) 

**3D** **Mol**ecular Network for **M**ass **S**pectra Prediction (3DMolMS) is a deep neural network model to predict the MS/MS spectra of compounds from their 3D conformations. The encoder for molecular representation learned in MS/MS prediction could also be transferred to other molecular-related tasks enhancing the performance, such as retention time and collisional cross section prediction. 

[[paper on Bioinformatics]](https://academic.oup.com/bioinformatics/article/39/6/btad354/7186501) | [[online service on GNPS]](https://spectrumprediction.gnps2.org)

**Contents:**

* [Updates](#updates)
* [Usage for MS/MS Prediction](#usage-for-msms-prediction)
* [Train Your Own Model](#train-your-own-model)
* Using 3DMolMS on Other Tasks
  * [Molecular Properties Prediction](docs/PROP_PRED.md)
  * [Generate reference library for molecular identification](docs/GEN_REFER_LIB.md)



## Updates 

- 2023.10.30 (v1.10): enlarging training set by MoNA and Waters QTOF datasets. 

- 2023.10.22 (v1.02): pretraining on QM9-mu dataset + ETKDG algorithm. We establish a dataset from QM9-mu (dipole moment) with the generated conformations using ETKDG for pretraining 3DMolMS. It helps the model learning knowledge of molecular 3D conformations and pretraining enhances the performance on MS/MS slightly (~0.01 cosine similarity). 

- 2023.09.14 (v1.01): data augmentation by flipping atomic coordinates. Notably, this model is sensitive to the geometric structure of molecules. For tasks insensitive to geometric structure, e.g. mass spectrometry is chirally blind, please use data augmentation. However, for the tasks sensitive to geometric structure, e.g. different enantiomers with varying retention times, avoid data augmentation. 

- 2023.06.30 (v1.00): initial version. 



## Usage for MS/MS Prediction

Step 0: Clone this repository and setup the anaconda environment by the following commands: 

```bash
git clone https://github.com/JosieHong/3DMolMS.git
cd 3DMolMS



conda create -n molnet 
conda activate molnet
# For RDKit
# https://www.rdkit.org/docs/GettingStartedInPython.html
conda install -c conda-forge rdkit

# For PyTorch 1.11.0
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

pip install -r requirements.txt
# or
# conda install --file requirements.txt
```

Step 1: Prepare the test set. The following formats are supported: csv, mgf, or [customed pkl](molmspack/data_utils/all2pkl.py). 

Here is an input example from MoNA of csv format (see the whole file at `./demo_input.csv`): 

```
ID,SMILES,Precursor_Type,Collision_Energy
demo_0,C/C(=C\CNc1nc[nH]c2ncnc1-2)CO,[M+H]+,40 V
demo_1,C/C(=C\CNc1nc[nH]c2ncnc1-2)CO,[M+H]+,20 V
demo_2,C/C(=C\CNc1nc[nH]c2ncnc1-2)CO,[M+H]+,10 V
```

Here is an input example from MoNA of mgf format (see the whole file at `./demo_input.mgf`): 

```
BEGIN IONS
TITLE=demo_0
CHARGE=1+
PRECURSOR_TYPE=[M+H]+
PRECURSOR_MZ=220.1193
MOLMASS=219.11201003600002
MS_LEVEL=MS2
IONMODE=P
SOURCE_INSTRUMENT=Agilent 6530 Q-TOF
INSTRUMENT_TYPE=LC-ESI-QTOF
COLLISION_ENERGY=40 V
SMILES=C/C(=C\CNc1nc[nH]c2ncnc1-2)CO
INCHI_KEY=UZKQTCBAMSWPJD-FARCUNLSSA-N
41.0399 6.207207
43.0192 49.711712
43.0766 1.986987
......
```

Please notice that the unsupported input will be filtered out automatically when loading the dataset. The supported inputs are shown in the following table. 

| Item             | Supported input                                               |
|------------------|---------------------------------------------------------------|
| Atom number      | <=300                                                         |
| Atom types       | 'C', 'O', 'N', 'H', 'P', 'S', 'F', 'Cl', 'B', 'Br', 'I', 'Na' |
| Precursor types  | '[M+H]+', '[M-H]-', '[M+H-H2O]+', '[M+Na]+'                   |
| Collision energy | any number                                                    |

Step 2: Predict the MS/MS spectra using the following command: 

```bash
python pred.py \
--test_data ./demo/demo_input.csv \
--model_config_path ./config/molnet.yml \
--data_config_path ./config/preprocess_etkdgv3.yml \
--resume_path ./check_point/molnet_qtof_etkdgv3.pt \
--result_path ./demo/demo_output.mgf \
--save_img_dir ./img/
```

Arguments: 

- `--resume_path` is the path of model's checkpoint. In the first running, the checkpoint (`./checkpoint/molnet_qtof_etkdgv3.pt`) will be downloaded from [Google Drive](https://drive.google.com/drive/folders/1fWx3d8vCPQi-U-obJ3kVL3XiRh75x5Ce?usp=drive_link). You can also set the resume path as the path to your own model. 
- `--result_path` is the path to save the predicted MS/MS. It should end with `.mgf` or `.csv`. We recommend you use `.mgf` because mgf is a more common format for MS/MS.  
- `--save_img_dir` is an optional argument denoting the path to save the figures of predicted MS/MS. One of the plots is shown here: 

<p align="center">
  <img src='img/demo_0.png' width='600'>
</p> 

## Train Your Own Model

Please set up the environment as shown in step 0 from the above section. 

Step 1: Download the pretrained model (`molnet_pre_etkdgv3.pt.zip`) from [Google Drive](https://drive.google.com/drive/folders/1fWx3d8vCPQi-U-obJ3kVL3XiRh75x5Ce?usp=drive_link) or pretrain the model by yourself. The details of pretraining the model on [QM9](https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904)-mu are demonstrated at [./docs/PRETRAIN.md](docs/PRETRAIN.md). 

Step 2: Gather the datasets separately, unzip and put them in `./data/`. In the latest version, we use 4 datasets to train the model: (1) Agilent DPCL is provided by [Agilent Technologies](https://www.agilent.com/). (2) [NIST20](https://www.nist.gov/programs-projects/nist23-updates-nist-tandem-and-electron-ionization-spectral-libraries) is academically available with a License. (3) [MoNA](https://mona.fiehnlab.ucdavis.edu/downloads) is publicly available. (4) Waters QTOF is our own experimental dataset. The structure of data directory is: 

```bash
|- data
  |- origin
    |- Agilent_Combined.sdf
    |- Agilent_Metlin.sdf
    |- hr_msms_nist.SDF
    |- MoNA-export-All_LC-MS-MS_QTOF.sdf
    |- MoNA-export-All_LC-MS-MS_Orbitrap.sdf
    |- waters_qtof.mgf
    |- ALL_GNPS_cleaned.csv
    |- ALL_GNPS_cleaned.mgf
```

Step 3: Use the following commands to preprocess the datasets. Please input the dataset you use at `--dataset` and choose the instrument type in `qtof` and `orbitrap`. `--maxmin_pick` means using the MaxMin algorithm in picking training molecules, otherwise, the random choice is applied. The settings of datasets are in `./preprocess_etkdgv3.yml`. 

```bash
python preprocess.py --dataset agilent nist mona waters gnps --instrument_type qtof orbitrap --data_config_path ./config/preprocess_etkdgv3.yml --mgf_dir ./data/mgf_debug/
```

Step 4: Use the following commands to train the model. The settings of model and training are in `./config/molnet.yml`. 

```bash
python train.py --train_data ./data/qtof_etkdgv3_train.pkl \
--test_data ./data/qtof_etkdgv3_test.pkl \
--model_config_path ./config/molnet.yml \
--data_config_path ./config/preprocess_etkdgv3.yml \
--checkpoint_path ./check_point/molnetv2_qtof_etkdgv3.pt \
--transfer \
--resume_path ./check_point/molnetv2_pre_etkdgv3.pt
```

In addition, 3DMolMS can be used in molecular properties prediction and generating refer libraries for molecular identification. We give the retention time prediction and cross-collision section prediction as two examples of molecular properties prediction. Please see the details in [./docs/PROP_PRED.md](docs/PROP_PRED.md). The examples of generating refer libraries can be found in [./docs/GEN_REFER_LIB.md](docs/GEN_REFER_LIB.md). 



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