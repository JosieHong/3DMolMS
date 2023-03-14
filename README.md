<!--
 * @Date: 2022-03-03 16:18:45
 * @LastEditors: yuhhong
 * @LastEditTime: 2022-12-11 01:00:20
-->
# 3DMolMS: Prediction of Tandem Mass Spectra from Three Dimensional Molecular Conformations

👏 The online service of 3DMolMS is available on [[GNPS2]](https://spectrumprediction.gnps2.org)!

## Set up

```bash
# RDKit
# https://www.rdkit.org/docs/GettingStartedInPython.html
conda create -c rdkit -n <env-name> rdkit
conda activate <env-name>

# Pytorch 1.7.1
# Please choose the proper cuda version from their official website:
# https://pytorch.org/get-started/previous-versions/
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch

pip install lxml tqdm pandas pyteomics
```



## Dataset

Here is an example input, `./example/input.csv`. At least, the following columns are required: 

```csv
ID,SMILES,Precursor_Type,Ion_Mode,Source_Instrument,Collision_Energy
0,O=S(=O)(O)CC(O)CN1CCN(CCO)CC1,M+H,P,QTOF,10
1,O=S(=O)(O)CC(O)CN1CCN(CCO)CC1,M+H,P,QTOF,20
2,O=S(=O)(O)CC(O)CN1CCN(CCO)CC1,M+H,P,QTOF,40
3,O=S(=O)(O)CC(O)CN1CCN(CCO)CC1,M+H,P,QTOF,80
4,NC(CCCCn1cccc2nc(NCCCC(N)C(=O)O)nc1-2)C(=O)O,M+H,P,QTOF,10
5,NC(CCCCn1cccc2nc(NCCCC(N)C(=O)O)nc1-2)C(=O)O,M+H,P,QTOF,20
```

Please use the following script to preprocess:

```bash
python preprocess.py --input <path to input csv file> --output <path to output csv file>

# e.g.
python preprocess.py --input ./example/input.csv --output ./example/pre_input.csv 
```

All the items are case sensitive. The unit of `Collision_Energy` is `eV`. If the collision energy is unknow, please set it 0. 

The following item will be removed in the preprocessing: 

- Contains atomic types other than: `['C', 'H', 'O', 'N', 'F', 'S', 'Cl', 'P', 'B', 'Br', 'I']`; 

- Precursor type is not in: `['M+H', 'M-H']`; 

  ```bash
  # Some other precursor types are also supported, but they may not get high-accurate, 
  # because we don't have much training data for these types. 
  ['M+H', 'M-H', 'M+Na', 'M+H-H2O', 'M+2H']
  ```

- Instrument is not in: `['QTOF']`; 



## Inference using the released models

Released pretrained models are [3DMolMS_Release](https://drive.google.com/drive/folders/1fWx3d8vCPQi-U-obJ3kVL3XiRh75x5Ce?usp=sharing). 

- `molnet_agilent_pos.pt` is trained on positive ion mode Agilent QTOF spectra from Agilent DPCL and NIST20. 

- `molnet_agilent_neg.pt` is trained on negative ion mode Agilent QTOF spectra from Agilent DPCL and NIST20. 

```bash
python pred.py --model molnet --dataset merge --num_atoms 300 --resolution 0.2 --ion_mode <P/N> \
	--test_data_path <path to input .csv file> \
	--resume_path <path to pretrained model> \
	--result_path <path to output .csv/.mgf file> \
	--iupac <(option)True/False, output IUPAC in .mgf file> \
	--inchi <(option)True/False, output InChI in .mgf file> \
	--inchi_key <(option)True/False, output InChI Key in .mgf file> 
```

Please note that: 

- The batch size is 1 in default settings. It can be changed to other numbers by `--batch_size <integer>`. 
If it is changed to other numbers, the model could infer faster but some data will be dropped if the dataset number can not 
be evenly divisible by the batch size. 

- If you would like to use certain 3D conformers, you could use the setting `--test_mol_path <path to test molecular comformers data (.sdf.gz)>`.

- Outputting IUPAC, InChI, and InChI Key will take minutes for searching them through [CACTUS](https://cactus.nci.nih.gov/). 

Here are two sample commands: 

```bash
python pred.py --model molnet --dataset merge --num_atoms 300 --resolution 0.2 --ion_mode P \
	--test_data_path ./example/pre_input.csv \
	--resume_path ./release/molnet_agilent_pos.pt \
	--result_path ./example/output_pos.csv
python pred.py --model molnet --dataset merge --num_atoms 300 --resolution 0.2 --ion_mode N \
	--test_data_path ./example/pre_input.csv \
	--resume_path ./release/molnet_agilent_neg.pt \
	--result_path ./example/output_neg.mgf
```



## Other experimental commands

All the data preparing, training, and evaluation commands are in `EXP.md`.

