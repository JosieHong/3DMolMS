# 3DMolMS

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa] (free for academic use) 

**3D** **Mol**ecular Network for **M**ass **S**pectra Prediction (3DMolMS) is a deep neural network model to predict the MS/MS spectra of compounds from their 3D conformations. This model's molecular representation, learned through MS/MS prediction tasks, can be further applied to enhance performance in other molecular-related tasks, such as predicting retention times and collision cross sections. 

[Read our paper in Bioinformatics](https://academic.oup.com/bioinformatics/article/39/6/btad354/7186501) | [Try our online service at GNPS](https://spectrumprediction.gnps2.org) | [Install from PyPI](https://pypi.org/project/molnetpack/)

## Installation

3DMolMS is available on PyPI. You can install the latest version using `pip`:

```bash
pip install molnetpack

# PyTorch must be installed separately. 
# For CUDA 11.6, install PyTorch with the following command:
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116

# For CUDA 11.7, use:
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117

# For CPU-only usage, use:
pip install torch==1.13.0+cpu torchvision==0.14.0+cpu torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cpu
```

3DMolMS can also be installed through source codes:

```bash
git clone https://github.com/JosieHong/3DMolMS.git
cd 3DMolMS

pip install .
```

## Usage

To get started quickly, you can load a CSV or MGF file to predict MS/MS and then plot the predicted results. 

```python
import torch
from molnetpack import MolNet

# Set the device to CPU for CPU-only usage:
device = torch.device("cpu")

# For GPU usage, set the device as follows (replace '0' with your desired GPU index):
# gpu_index = 0
# device = torch.device(f"cuda:{gpu_index}")

# Instantiate a MolNet object
molnet_engine = MolNet(device, seed=42) # The random seed can be any integer. 

# Load input data (here we use a CSV file as an example)
molnet_engine.load_data(path_to_test_data='./test/input_msms.csv') # Increasing the batch size if you wanna speed up.
# molnet_engine.load_data(path_to_test_data='./test/input_msms.mgf') # MGF file is also supported
# molnet_engine.load_data(path_to_test_data='./test/input_msms.pkl') # PKL file is faster. 

# Predict MS/MS
spectra1 = molnet_engine.pred_msms(path_to_results='./test/output_qtof_msms.mgf', instrument='qtof')
# You could also download the checkpoint from release and set the 'path_to_checkpoint':
# spectra = molnet_engine.pred_msms(path_to_results='./test/output_msms.mgf', path_to_checkpoint='<path to the checkpoint>')
# Instrument can be 'qtof' or 'orbitrap'. 

# Plot the predicted MS/MS with 3D molecular conformation
molnet_engine.plot_msms(dir_to_img='./img/', instrument='qtof')
```

For CCS prediction, please use the following codes after instantiating a MolNet object. 

```python
# Load input data
molnet_engine.load_data(path_to_test_data='./test/input_ccs.csv')

# Pred CCS
ccs_df = molnet_engine.pred_ccs(path_to_results='./test/output_ccs.csv')
```

For saving the molecular embeddings, please use the following codes after instantiating a MolNet object. 

```python
# Load input data
molnet_engine.load_data(path_to_test_data='./test/input_savefeat.csv')

# Inference to get the features
features = molnet_engine.save_features()

print('Titles:', ids)
print('Features shape:', features.shape)
```

The sample input files, a CSV and an MGF, are located at `./test/demo_input.csv` and `./test/demo_input.mgf`, respectively. If the input data is only expected to be used in CCS prediction, you may assign an arbitrary numerical value to the `Collision_Energy` field in the CSV file or to `COLLISION_ENERGY` in the MGF file. It's important to note that during the data loading phase, any input formats that are not supported will be automatically excluded. Below is a table outlining the types of input data that are supported: 

| Item             | Supported input                                               |
|------------------|---------------------------------------------------------------|
| Atom number      | <=300                                                         |
| Atom types       | 'C', 'O', 'N', 'H', 'P', 'S', 'F', 'Cl', 'B', 'Br', 'I', 'Na' |
| Precursor types  | '[M+H]+', '[M-H]-', '[M+H-H2O]+', '[M+Na]+', '[M+2H]2+'       |
| Collision energy | any number                                                    |

The documents for running MS/MS prediction from source codes are at [MSMS_PRED.md](docs/MSMS_PRED.md). 



## Citation

If you use 3DMolMS in your research, please cite our paper:

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
@article{hong2024enhanced,
  title={Enhanced structure-based prediction of chiral stationary phases for chromatographic enantioseparation from 3D molecular conformations},
  author={Hong, Yuhui and Welch, Christopher J and Piras, Patrick and Tang, Haixu},
  journal={Analytical Chemistry},
  volume={96},
  number={6},
  pages={2351--2359},
  year={2024},
  publisher={ACS Publications}
}
```

Thank you for considering 3DMolMS for your research needs!

## License

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg