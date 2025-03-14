# 3DMolMS

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa] (free for academic use) 

**3D** **Mol**ecular Network for **M**ass **S**pectra Prediction (3DMolMS) is a deep neural network model to predict the MS/MS spectra of compounds from their 3D conformations. This model's molecular representation, learned through MS/MS prediction tasks, can be further applied to enhance performance in other molecular-related tasks, such as predicting retention times (RT) and collision cross sections (CCS). 

[Read paper in Bioinformatics](https://academic.oup.com/bioinformatics/article/39/6/btad354/7186501) | [Try model on Konia](https://koina.wilhelmlab.org/docs#post-/3dmolms_qtof/infer) | [Install from PyPI](https://pypi.org/project/molnetpack/)

🆕 3DMolMS v1.1.10 is now available for inference on **Konia**, and **PyPI**! 

The changes log can be found at [[CHANGE_LOG.md]](./CHANGE_LOG.md). 

## Installation

3DMolMS is available on PyPI (`molnetpack`). You can install the latest version using `pip`:

```bash
pip install molnetpack

# PyTorch must be installed separately. 
# Please check the official website of PyTorch for the proper version:
# https://pytorch.org/get-started/locally/
# e.g.
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

3DMolMS can also be installed through source codes:

```bash
git clone https://github.com/JosieHong/3DMolMS.git
cd 3DMolMS

pip install .
```

## Usage

To get started quickly, you can instantiate a MolNet and load a CSV or MGF file for MS/MS prediction as: 

```python
import torch
from molnetpack import MolNet, plot_msms

# Set the device to CPU for CPU-only usage:
device = torch.device("cpu")

# For GPU usage, set the device as follows (replace '0' with your desired GPU index):
# gpu_index = 0
# device = torch.device(f"cuda:{gpu_index}")

# Instantiate a MolNet object
molnet_engine = MolNet(device, seed=42) # The random seed can be any integer. 

# Load input data (here we use a CSV file as an example)
molnet_engine.load_data(path_to_test_data='./test/input_msms.csv')
"""Load data from the specified path.
Args:
    path_to_test_data (str): Path to the test data file. Supported formats are 'csv', 'mgf', and 'pkl'.
Returns:
    None
"""

# Predict MS/MS
pred_spectra_df = molnet_engine.pred_msms(instrument='qtof')
"""Predict MS/MS spectra.
Args:
    path_to_results (Optional[str]): Path to save the prediction results. Supports '.mgf' or '.csv' formats. If None, the results won't be saved. 
    path_to_checkpoint (Optional[str]): Path to the model checkpoint. If None, the model will be downloaded from a default URL.
    instrument (str): Type of instrument used ('qtof' or 'orbitrap').
Returns:
    pd.DataFrame: DataFrame containing the predicted MS/MS results.
"""
```

We also implement a function to plot the predicted results.

```python
# Plot the predicted MS/MS with 3D molecular conformation
plot_msms(pred_spectra_df, dir_to_img='./img/')
```

The sample input files, a CSV and an MGF, are located at `./test/demo_input.csv` and `./test/demo_input.mgf`, respectively. It's important to note that during the data loading phase, any input formats that are not supported will be automatically excluded. Below is a table outlining the types of input data that are supported: 

| Item             | Supported input                                               |
|------------------|---------------------------------------------------------------|
| Atom number      | <=300                                                         |
| Atom types       | 'C', 'O', 'N', 'H', 'P', 'S', 'F', 'Cl', 'B', 'Br', 'I', 'Na' |
| Precursor types  | '[M+H]+', '[M-H]-', '[M+H-H2O]+', '[M+Na]+', '[M+2H]2+'       |
| Collision energy | any number                                                    |

Below is an example of a predicted MS/MS spectrum plot.

<p align="center">
  <img src='https://github.com/JosieHong/3DMolMS/blob/main/img/demo_0.png' width='600'>
</p> 

A more detailed documentation for various tasks using molnetpack or source code can be found in the [docs/](docs/) directory, which includes the following: 
* [./docs/](./docs/)
  * [PROP_USAGE.md](./docs/PROP_USAGE.md): Guide on using `molnetpack` for RT prediction, CCS prediction, and molecular embedding. 
  * [MSMS_PRED.md](./docs/MSMS_PRED.md): Instructions for using 3DMolMS to predict MS/MS spectra from your own CSV files via the source code. The training details can be found in the [next section](#train-your-own-model). 
  * [GEN_REFER_LIB.md](./docs/GEN_REFER_LIB.md): Instructions for using 3DMolMS to generate MS/MS reference libraries from small molecule databases, such as HMDB and RefMet, via the source code. 
  * [PROP_PRED.md](./docs/PROP_PRED.md): Instructions for training and testing 3DMolMS on RT and CCS prediction via the source code.
  * [PRETRAIN.md](./docs/PRETRAIN.md): Instructions for pretraining 3DMolMS on the QM9 dataset via the source code. 
  
## Train your own model

**Step 0**: Clone the Repository and Set Up the Environment

Clone the 3DMolMS repository and install the required packages using the following commands:

```bash
git clone https://github.com/JosieHong/3DMolMS.git
cd 3DMolMS

# Please install the packages if you have not installed them yet. 
pip install .
```

**Step 1**: Obtain the Pretrained Model

Download the pretrained model (`molnet_pre_etkdgv3.pt.zip`) from [Releases](https://github.com/JosieHong/3DMolMS/releases). You can also train the model from scratch. For details on pretraining the model on the [QM9](https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904) dataset, refer to [PRETRAIN.md](docs/PRETRAIN.md).

**Step 2**: Prepare the Datasets

Download and organize the datasets into the `./data/` directory. The current version uses four datasets:

1. Agilent DPCL, provided by [Agilent Technologies](https://www.agilent.com/).
2. [NIST20](https://www.nist.gov/programs-projects/nist23-updates-nist-tandem-and-electron-ionization-spectral-libraries), available under license for academic use.
3. [MoNA](https://mona.fiehnlab.ucdavis.edu/downloads), publicly available.
4. Waters QTOF, our own experimental dataset.

The data directory structure should look like this:

```plaintext
|- data
  |- origin
    |- Agilent_Combined.sdf
    |- Agilent_Metlin.sdf
    |- hr_msms_nist.SDF
    |- MoNA-export-All_LC-MS-MS_QTOF.sdf
    |- MoNA-export-All_LC-MS-MS_Orbitrap.sdf
    |- waters_qtof.mgf
```

**Step 3**: Preprocess the Datasets

Run the following commands to preprocess the datasets. Specify the dataset with `--dataset` and select the instrument type as `qtof`. Use `--maxmin_pick` to apply the MaxMin algorithm for selecting training molecules; otherwise, selection will be random. The dataset configurations are in `./src/molnetpack/config/preprocess_etkdgv3.yml`.

```bash
python ./src/preprocess.py --dataset agilent nist mona waters gnps \
--instrument_type qtof orbitrap \
--data_config_path ./src/molnetpack/config/preprocess_etkdgv3.yml \
--mgf_dir ./data/mgf_debug/ 
```

**Step 4**: Train the Model

Use the following commands to train the model. Configuration settings for the model and training process are located in `./src/molnetpack/config/molnet.yml`.

```bash
# Train the model from pretrain: 
# Q-TOF (Orbitrap is ignored here.): 
python ./src/train.py --train_data ./data/qtof_etkdgv3_train.pkl \
--test_data ./data/qtof_etkdgv3_test.pkl \
--model_config_path ./src/molnetpack/config/molnet.yml \
--data_config_path ./src/molnetpack/config/preprocess_etkdgv3.yml \
--checkpoint_path ./check_point/molnet_qtof_etkdgv3.pt \
--transfer --resume_path ./check_point/molnet_pre_etkdgv3.pt \
--ex_model_path ./check_point/molnet_qtof_etkdgv3_jit.pt

# Train the model from scratch
# Q-TOF: 
python ./src/train.py --train_data ./data/qtof_etkdgv3_train.pkl \
--test_data ./data/qtof_etkdgv3_test.pkl \
--model_config_path ./src/molnetpack/config/molnet.yml \
--data_config_path ./src/molnetpack/config/preprocess_etkdgv3.yml \
--checkpoint_path ./check_point/molnet_qtof_etkdgv3.pt \
--ex_model_path ./check_point/molnet_qtof_etkdgv3_jit.pt
# Orbitrap: 
python ./src/train.py --train_data ./data/orbitrap_etkdgv3_train.pkl \
--test_data ./data/orbitrap_etkdgv3_test.pkl \
--model_config_path ./src/molnetpack/config/molnet.yml \
--data_config_path ./src/molnetpack/config/preprocess_etkdgv3.yml \
--checkpoint_path ./check_point/molnet_orbitrap_etkdgv3.pt \
--ex_model_path ./check_point/molnet_orbitrap_etkdgv3_jit.pt 
```

**Step 5**: Evaluation

Let's evaluate the model trained above! 

```bash
# Predict the spectra: 
# Q-TOF: 
python ./src/pred.py \
--test_data ./data/qtof_etkdgv3_test.pkl \
--model_config_path ./src/molnetpack/config/molnet.yml \
--data_config_path ./src/molnetpack/config/preprocess_etkdgv3.yml \
--resume_path ./check_point/molnet_qtof_etkdgv3.pt \
--result_path ./result/pred_qtof_etkdgv3_test.mgf 
# Orbitrap: 
python ./src/pred.py \
--test_data ./data/orbitrap_etkdgv3_test.pkl \
--model_config_path ./src/molnetpack/config/molnet.yml \
--data_config_path ./src/molnetpack/config/preprocess_etkdgv3.yml \
--resume_path ./check_point/molnet_orbitrap_etkdgv3.pt \
--result_path ./result/pred_orbitrap_etkdgv3_test.mgf 

# Evaluate the cosine similarity between experimental spectra and predicted spectra:
# Q-TOF: 
python ./src/eval.py ./data/qtof_etkdgv3_test.pkl ./result/pred_qtof_etkdgv3_test.mgf \
./eval_qtof_etkdgv3_test.csv ./eval_qtof_etkdgv3_test.png
# Orbitrap: 
python ./src/eval.py ./data/orbitrap_etkdgv3_test.pkl ./result/pred_orbitrap_etkdgv3_test.mgf \
./eval_orbitrap_etkdgv3_test.csv ./eval_orbitrap_etkdgv3_test.png
```

**Additional application**

3DMolMS is also capable of predicting molecular properties and generating reference libraries for molecular identification. For more details, refer to [PROP_PRED.md](docs/PROP_PRED.md) and [GEN_REFER_LIB.md](docs/GEN_REFER_LIB.md) respectively. 

## Citation

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

## License

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg