# 3DMolMS

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa] (free for academic use) 

**3D** **Mol**ecular Network for **M**ass **S**pectra Prediction (3DMolMS) is a deep neural network model to predict the MS/MS spectra of compounds from their 3D conformations. The encoder for molecular representation learned in MS/MS prediction could also be transferred to other molecular-related tasks enhancing the performance, such as retention time and collisional cross section prediction. 

[[paper on Bioinformatics]](https://academic.oup.com/bioinformatics/article/39/6/btad354/7186501) | [[online service on GNPS]](https://spectrumprediction.gnps2.org)



## Updates 

- 2023.09.14 (v1.1): data augmentation by flipping atomic coordinates (Notably, this model is sensitive to the geometric structure of molecules. If you will use it for any task that is non-sensitive to geometric structure, e.g. mass spectrometry is chirally blind, please use data augmentation.) 

- 2023.06.30 (v1.0): initial version



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

Step 1: Generate custom test data. If you already have test data, please convert it into a supported format, e.g. csv, mgf or pkl. Here is an input example of csv format (`./demo_input.csv`): 

```
ID,SMILES,Precursor_Type,Source_Instrument,Collision_Energy,Charge
0,O=S(=O)(O)CC(O)CN1CCN(CCO)CC1,[M+H]+,QTOF,20,1
1,O=S(=O)(O)CC(O)CN1CCN(CCO)CC1,[M+H]+,QTOF,40,1
2,NC(CCCCn1cccc2nc(NCCCC(N)C(=O)O)nc1-2)C(=O)O,[M+H]+,QTOF,20,1
3,NC(CCCCn1cccc2nc(NCCCC(N)C(=O)O)nc1-2)C(=O)O,[M+H]+,QTOF,40,1
4,O=S(=O)(O)CC(O)CN1CCN(CCO)CC1,[M-H]-,QTOF,20,1
5,O=S(=O)(O)CC(O)CN1CCN(CCO)CC1,[M-H]-,QTOF,40,1
6,NC(CCCCn1cccc2nc(NCCCC(N)C(=O)O)nc1-2)C(=O)O,[M-H]-,QTOF,20,1
7,NC(CCCCn1cccc2nc(NCCCC(N)C(=O)O)nc1-2)C(=O)O,[M-H]-,QTOF,40,1
```

Step 2: Download the pre-trained model from [[Google Drive]](https://drive.google.com/drive/folders/1fWx3d8vCPQi-U-obJ3kVL3XiRh75x5Ce?usp=drive_link). If you have trained your own model, please ignore this step. 

Step 3: Test the model by the following commands: 

```bash
python pred.py --test_data <path to test data (.csv, .mgf, or .pkl)> \
--model_config_path ./config/molnet.yml \
--data_config_path ./config/preprocess_etkdg.yml \
--resume_path <path to pretrained model> \
--result_path <path to save the results (.csv or .mgf file)> 

# e.g.
python pred.py --test_data ./demo/demo_input.csv \
--model_config_path ./config/molnet.yml \
--data_config_path ./config/preprocess_etkdg.yml \
--resume_path ./check_point/molnet_qtof_etkdg.pt \
--result_path ./demo/demo_output.csv
```



## Train your own model

Please set up the environment as shown in step 0 from the above section. 

Step 1: Download the datasets separately, unzip and put them in `./data/`. [[NIST20]](https://www.nist.gov/programs-projects/nist23-updates-nist-tandem-and-electron-ionization-spectral-libraries) is academically available with a License, and please contact us for Agilent DPCL. The structure of data directory is: 

```bash
|- data
  |- origin
	  |- Agilent_Combined.sdf
	  |- Agilent_Metlin.sdf
	  |- hr_msms_nist.SDF
```

Step 2: Use the following commands to preprocess the datasets. The settings of datasets are in `./preprocess_etkdg.yml`. 

```bash
python preprocess.py --dataset qtof
```

Step 3: Use the following commands to train the model. The settings of model and training are in `./config/molnet.yml`. 

```bash
python train.py --train_data ./data/qtof_etkdg_train.pkl \
--test_data ./data/qtof_etkdg_test.pkl \
--model_config_path ./config/molnet.yml \
--data_config_path ./config/preprocess_etkdg.yml \
--checkpoint_path ./check_point/molnet_qtof_etkdg.pt 
```



## Transfer the molecular representation into other tasks

We give the retention time prediction and collisional cross section prediction as two examples of transferring the molecular representation. Please see the details in `./TRANSFER_LEARNING.md`. 



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

---

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg