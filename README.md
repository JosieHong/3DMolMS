# Mol3DNet: Prediction of Tandem Mass Spectra from 3D Conformers



## Set Up

```bash
# RDKit
# https://www.rdkit.org/docs/GettingStartedInPython.html
conda create -c rdkit -n rdkit-env rdkit
conda activate rdkit-env

# Pytorch 1.7.1
# Please choose the proper cuda version from their official website:
# https://pytorch.org/get-started/previous-versions/
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch

pip install -r requirements.txt
```



## Dataset

Here is an example input, `input.csv`. At least, the following columns are required: 

```csv
ID,SMILES,Precursor_Type,Source_Instrument,Collision_Energy
0,O=S(=O)(O)CC(O)CN1CCN(CCO)CC1,M+H,HCD,17
1,O=S(=O)(O)CC(O)CN1CCN(CCO)CC1,M+H,HCD,28
2,O=S(=O)(O)CC(O)CN1CCN(CCO)CC1,M+H,HCD,38
3,O=S(=O)(O)CC(O)CN1CCN(CCO)CC1,M+H,HCD,48
4,O=S(=O)(O)CC(O)CN1CCN(CCO)CC1,M+H,HCD,63
5,NC(CCCCn1cccc2nc(NCCCC(N)C(=O)O)nc1-2)C(=O)O,M+H,HCD,83
```

All the items are case sensitive. The unit of `Collision_Energy` is `eV`. If the collision energy is unknow, please set it 0. 

The following item will be removed, when loading the data: 

- Contains atomic types other than: `['C', 'H', 'O', 'N', 'F', 'S', 'Cl', 'P', 'B', 'Br', 'I']`; 

- Precursor type is not in: 

  ```python
  ['M+H', 'M-H', 'M+H-H2O', 'M+Na', 'M+H-NH3', 'M+H-2H2O', 'M-H-H2O', 'M+NH4', 'M+H-CH4O', 'M+2Na-H', 
  'M+H-C2H6O', 'M+Cl', 'M+OH', 'M+H+2i', '2M+H', '2M-H', 'M-H-CO2', 'M+2H', 'M-H+2i', 'M+H-CH2O2', 
  'M+H-C4H8', 'M+H-C2H4O2', 'M+H-C2H4', 'M+CHO2', 'M-H-CH3', 'M+H-H2O+2i', 'M+H-C2H2O', 'M+H-C3H6', 
  'M+H-CH3', 'M+H-3H2O', 'M+H-HF', 'M-2H', 'M-H2O+H', 'M-2H2O+H']
  ```

- Instrument is not in: `['HCD', 'QTOF']`; 



## Inference

Pretrained models are [Mol3DNet_Release](https://drive.google.com/drive/folders/1LNc8adZFj669ghk2lgkjigJFFwRoDLKs?usp=sharing). 

`pretrained_hcd_all.pth` and `pretrained_qtof_all.pth` are trained in all the data of NIST20, GNPS and MassBank. Because there is no data for validation, the model may be overfitted on the training data. 

`pretrained_hcd.pth` and `pretrained_qtof.pth` are trained in 90% the data of NIST20, GNPS and MassBank. Validation data is randomly sampled in the dataset, so this model may not be overfitted. 



```bash
python inference.py --post_threshold 0.01 \
	--model_path <path to pretrained model> \
	--test_data_path input.csv \
	--result_path output.csv
```

