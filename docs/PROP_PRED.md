# Molecular Properties Prediction using 3DMolMS

3DMolMS can be used to predict MS/MS-related properties, such as retention time (RT) and collision cross section (CCS). This file shows how to train a model for [[RT prediction]](#retention-time-prediction) and [[CCS prediction]](#cross-collision-section-prediction), and how to [[transfer these models to your own RT and CCS dataset]](#fine-tune-on-your-own-data). All the following models can be downloaded from [[release v1.1.6]](https://github.com/JosieHong/3DMolMS/releases/tag/v1.1.6). 

## Retention time prediction

Please set up the environment as shown in step 0 from `README.md`. 

Step 1: Download the retention time dataset, [[METLIN]](https://figshare.com/articles/dataset/The_METLIN_small_molecule_dataset_for_machine_learning-based_retention_time_prediction/8038913?file=18130625). The structure of data directory is: 

```bash
|- data
  |- origin
    |- SMRT_dataset.sdf
```

Step 2: Use the following commands to preprocess the datasets. The settings of datasets are in `./src/molnetpack/config/preprocess_etkdgv3.yml`. 

```bash
python ./src/preprocess_oth.py --dataset metlin 
```

Step 3: Use the following commands to train the model. The settings of model and training are in `./config/molnet_rt.yml`. If you'd like to train this model from the pre-trained model on MS/MS prediction, please download the pre-trained model from [[Google Drive]](https://drive.google.com/drive/folders/1fWx3d8vCPQi-U-obJ3kVL3XiRh75x5Ce?usp=drive_link). 

```bash
# learn from scratch
python ./src/train_rt.py --train_data ./data/metlin_etkdgv3_train.pkl \
--test_data ./data/metlin_etkdgv3_test.pkl \
--model_config_path ./src/molnetpack/config/molnet_rt.yml \
--data_config_path ./src/molnetpack/config/preprocess_etkdgv3.yml \
--checkpoint_path ./check_point/molnet_rt_etkdgv3.pt

# learn from pretrained model
python ./src/train_rt.py --train_data ./data/metlin_etkdgv3_train.pkl \
--test_data ./data/metlin_etkdgv3_test.pkl \
--model_config_path ./src/molnetpack/config/molnet_rt.yml \
--data_config_path ./src/molnetpack/config/preprocess_etkdgv3.yml \
--checkpoint_path ./check_point/molnet_rt_etkdgv3_tl.pt \
--transfer \
--resume_path ./check_point/molnet_pre_etkdgv3.pt 
```



## Cross-collision section prediction

Please set up the environment as shown in step 0 from `README.md`. 

Step 1: Download the cross-collision section dataset, [[AllCCS]](http://allccs.zhulab.cn/), manually or using `download_allccs.py`:

```bash
python ./src/download_allccs.py --user <user_name> --passw <passwords> --output ./data/origin/allccs_download.csv
```

The structure of data directory is: 

```bash
|- data
  |- origin
    |- allccs_download.csv
```

Step 2: Use the following commands to preprocess the datasets. The settings of datasets are in `./src/molnetpack/config/preprocess_etkdgv3.yml`. 

```bash
python ./src/preprocess_oth.py --dataset allccs --data_config_path ./src/molnetpack/config/preprocess_etkdgv3.yml
```

Step 3: Use the following commands to train the model. The settings of model and training are in `./src/molnetpack/config/molnet_ccs.yml`. If you'd like to train this model from the pre-trained model on MS/MS prediction, please download the pre-trained model from [[Google Drive]](https://drive.google.com/drive/folders/1fWx3d8vCPQi-U-obJ3kVL3XiRh75x5Ce?usp=drive_link). 

```bash
# learn from scratch
python ./src/train_ccs.py --train_data ./data/allccs_etkdgv3_train.pkl \
--test_data ./data/allccs_etkdgv3_test.pkl \
--model_config_path ./src/molnetpack/config/molnet_ccs.yml \
--data_config_path ./src/molnetpack/config/preprocess_etkdgv3.yml \
--checkpoint_path ./check_point/molnet_ccs_etkdgv3.pt 

# learn from pretrained model
python ./src/train_ccs.py --train_data ./data/allccs_etkdgv3_train.pkl \
--test_data ./data/allccs_etkdgv3_test.pkl \
--model_config_path ./src/molnetpack/config/molnet_ccs.yml \
--data_config_path ./src/molnetpack/config/preprocess_etkdgv3.yml \
--checkpoint_path ./check_point/molnet_ccs_etkdgv3_tl.pt \
--transfer \
--resume_path ./check_point/molnet_pre_etkdgv3.pt 
```

## Fine-tune on your own data

Step 1: Please prepare the retention time data as: 
```csv
,id,smiles,prop
0,0382_00004,NC(=O)N1c2ccccc2[C@H](O)[C@@H](O)c2ccccc21,5.79
1,0382_00005,CN(C)[C@@H]1C(=O)C(C(N)=O)=C(O)[C@@]2(O)C(=O)C3=C(O)c4c(O)ccc(Cl)c4[C@@](C)(O)[C@H]3C[C@@H]12,4.5
2,0382_00008,Cc1onc(-c2c(Cl)cccc2Cl)c1C(=O)N[C@@H]1C(=O)N2[C@@H](C(=O)O)C(C)(C)S[C@H]12,7.8
3,0382_00009,C[C@H]1c2cccc(O)c2C(O)=C2C(=O)[C@]3(O)C(O)=C(C(N)=O)C(=O)[C@@H](N(C)C)[C@@H]3[C@@H](O)[C@@H]21,6.2
4,0382_00010,C#C[C@]1(O)CC[C@H]2[C@@H]3CCc4cc(O)ccc4[C@H]3CC[C@@]21C,9.46
5,0382_00012,Cc1onc(-c2ccccc2)c1C(=O)N[C@@H]1C(=O)N2[C@@H](C(=O)O)C(C)(C)S[C@H]12,6.9
```
where `prop` column is the RT or CCS values. 

Step 2: Fine-tune the model!

```bash
python ./src/train_csv.py --data <path to csv/pkl data> \ 
--model_config_path <path to configuration> \
--checkpoint_path <path to save the checkpoint> \
--transfer \
--resume_path <path to pretrained model> \
--result_path <path to save the prediction results> \
--seed 42 # for randomly data splitting and randomly dropping out the neurons in the model

# If the data is already preprocessed, you could use pkl file directly.
```

Step 3: Predict the unlabeled data

```bash
python pred_rt.py --data <path to csv/pkl data> \
--model_config_path <path to configuration> \
--checkpoint_path <path to save the checkpoint> \
--result_path <path to save the prediction results> \
--seed 42
```
