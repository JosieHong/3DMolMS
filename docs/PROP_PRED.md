# Molecular Properties Prediction using 3DMolMS



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
python ./src/scripts/preprocess_oth.py --dataset metlin 
```

Step 3: Use the following commands to train the model. The settings of model and training are in `./config/molnet_rt.yml`. If you'd like to train this model from the pre-trained model on MS/MS prediction, please download the pre-trained model from [[Google Drive]](https://drive.google.com/drive/folders/1fWx3d8vCPQi-U-obJ3kVL3XiRh75x5Ce?usp=drive_link). 

```bash
# learn from scratch
python ./src/scripts/train_rt.py --train_data ./data/metlin_etkdgv3_train.pkl \
--test_data ./data/metlin_etkdgv3_test.pkl \
--model_config_path ./src/molnetpack/config/molnet_rt.yml \
--data_config_path ./src/molnetpack/config/preprocess_etkdgv3.yml \
--checkpoint_path ./check_point/molnet_rt_etkdgv3.pt

# learn from pretrained model
python ./src/scripts/train_rt.py --train_data ./data/metlin_etkdgv3_train.pkl \
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
python ./src/scripts/download_allccs.py --user <user_name> --passw <passwords> --output ./data/origin/allccs_download.csv
```

The structure of data directory is: 

```bash
|- data
  |- origin
    |- allccs_download.csv
```

Step 2: Use the following commands to preprocess the datasets. The settings of datasets are in `./src/molnetpack/config/preprocess_etkdgv3.yml`. 

```bash
python ./src/scripts/preprocess_oth.py --dataset allccs --data_config_path ./src/molnetpack/config/preprocess_etkdgv3.yml
```

Step 3: Use the following commands to train the model. The settings of model and training are in `./src/molnetpack/config/molnet_ccs.yml`. If you'd like to train this model from the pre-trained model on MS/MS prediction, please download the pre-trained model from [[Google Drive]](https://drive.google.com/drive/folders/1fWx3d8vCPQi-U-obJ3kVL3XiRh75x5Ce?usp=drive_link). 

```bash
# learn from scratch
python ./src/scripts/train_ccs.py --train_data ./data/allccs_etkdgv3_train.pkl \
--test_data ./data/allccs_etkdgv3_test.pkl \
--model_config_path ./src/molnetpack/config/molnet_ccs.yml \
--data_config_path ./src/molnetpack/config/preprocess_etkdgv3.yml \
--checkpoint_path ./check_point/molnet_ccs_etkdgv3.pt 

# learn from pretrained model
python ./src/scripts/train_ccs.py --train_data ./data/allccs_etkdgv3_train.pkl \
--test_data ./data/allccs_etkdgv3_test.pkl \
--model_config_path ./src/molnetpack/config/molnet_ccs.yml \
--data_config_path ./src/molnetpack/config/preprocess_etkdgv3.yml \
--checkpoint_path ./check_point/molnet_ccs_etkdgv3_tl.pt \
--transfer \
--resume_path ./check_point/molnet_pre_etkdgv3.pt 
```
