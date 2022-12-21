<!--
 * @Date: 2022-03-03 16:18:45
 * @LastEditors: yuhhong
 * @LastEditTime: 2022-12-11 01:00:20
-->
# DyMGNet_MS: Prediction of Tandem Mass Spectra from 3D Conformers

This is the implementation of using DyGMNet (Dynamic Molecular Graph Network) to predict tandem mass spectra from molecular 3D conformers. 



## Set Up

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
ID,SMILES,Precursor_Type,Source_Instrument,Collision_Energy
0,O=S(=O)(O)CC(O)CN1CCN(CCO)CC1,M+H,QTOF,10
1,O=S(=O)(O)CC(O)CN1CCN(CCO)CC1,M+H,QTOF,20
2,O=S(=O)(O)CC(O)CN1CCN(CCO)CC1,M+H,QTOF,40
3,O=S(=O)(O)CC(O)CN1CCN(CCO)CC1,M+H,QTOF,80
```

Please use the following script to preprocess:

```bash
python preprocess.py --input <path to input csv file> --output <path to output csv file>

# e.g.
python preprocess.py --input ./example/input_posi.csv --output ./example/pre_input_posi.csv 
python preprocess.py --input ./example/input_nega.csv --output ./example/pre_input_nega.csv 
```

All the items are case sensitive. The unit of `Collision_Energy` is `eV`. If the collision energy is unknow, please set it 0. 

The following item will be removed in the preprocessing: 

- Contains atomic types other than: `['C', 'H', 'O', 'N', 'F', 'S', 'Cl', 'P', 'B', 'Br', 'I']`; 

- Precursor type is not in: `['M+H', 'M-H']`; 

  ```bash
  # Some other precursor types are also supported, but they may not get high-accurat, 
  # because we don't have much training data for these types. 
  ['M+H', 'M-H', 'M+H-H2O', 'M+Na', 'M+H-NH3', 'M+H-2H2O', 'M-H-H2O', 'M+NH4', 'M+H-CH4O', 'M+2Na-H', 
    'M+H-C2H6O', 'M+Cl', 'M+OH', 'M+H+2i', '2M+H', '2M-H', 'M-H-CO2', 'M+2H', 'M-H+2i', 'M+H-CH2O2', 
    'M+H-C4H8', 'M+H-C2H4O2', 'M+H-C2H4', 'M+CHO2', 'M-H-CH3', 'M+H-H2O+2i', 'M+H-C2H2O', 'M+H-C3H6', 
    'M+H-CH3', 'M+H-3H2O', 'M+H-HF', 'M-2H', 'M-H2O+H', 'M-2H2O+H']
  ```

- Instrument is not in: `['QTOF']`; 



## Inference Using the Released Models

Released PyTorch JIT models are [DyMGNet_Release](https://drive.google.com/drive/folders/1fWx3d8vCPQi-U-obJ3kVL3XiRh75x5Ce?usp=sharing). 

- `dymgnet_posi.pt` is trained on `[M+H]` ion mode MS/MS from Agilent. 

- `dymgnet_nega.pt` is trained on `[M-H]` ion mode MS/MS from Agilent.

```bash
python inference.py --model_path <path to pretrained model> \
	--test_data_path <path to input csv file> \
	--result_path <path to output csv file>

# e.g.
python inference.py --mol_type 3d --batch_size 2 \
  --model_path ./release/dymgnet_posi.pt \
	--test_data_path ./example/pre_input_posi.csv \
	--result_path ./example/output_posi.csv

python inference.py --mol_type 3d --batch_size 2 \
  --model_path ./release/dymgnet_nega.pt \
	--test_data_path ./example/pre_input_nega.csv \
	--result_path ./example/output_nega.csv
```

The following script can be used to convert the output csv file into mgf file: 

```bash
python csv2mgf.py --csv_file <path to output csv file> --mgf_file <path to output mgf file>

# e.g.
python csv2mgf.py --csv_file ./example/output_posi.csv --mgf_file ./example/output_posi.mgf
python csv2mgf.py --csv_file ./example/output_nega.csv --mgf_file ./example/output_nega.mgf
```



<!-- ## Update Later

Now we support both `.mgf` and `.csv` format data. To evaluate the model (which means we know the experimental mass spectra), please input the `.mgf` data. For only inference (which means we don't know the experimental mass spectra), please input the `.csv` data. 

### MGF Dataset

Here is an example input, `./example/input.mgf`. At least, the following attributes are required:

```mgf
BEGIN IONS
TITLE=<title>
PEPMASS=<or 'PrecursorMZ' in NIST20 library, 'EXACT MASS' in Agilent library>
PRECURSOR_TYPE=<precursor type>
SOURCE_INSTRUMENT=<source instrument>
COLLISION_ENERGY=<collision energy>
SMILES=<SMILES>
<m/z and intensity>
END IONS
```

Please do the preprocess by: 

```bash
python preprocess.py --input <path to input mgf file> --output <path to output mgf file>

# e.g.
python preprocess.py --input ./example/input.mgf --output ./example/pre_input.mgf 
``` 

## Train & Eval-->

