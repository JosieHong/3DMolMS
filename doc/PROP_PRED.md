<!--
 * @Date: 2023-10-03 17:36:39
 * @LastEditors: yuhhong
 * @LastEditTime: 2023-10-06 11:51:32
-->
# Molecular Properties Prediction using 3DMolMS



## Retention time prediction

Please set up the environment as shown in step 0 from `README.md`. 

Step 1: Download the retention time dataset, [[METLIN]](https://figshare.com/articles/dataset/The_METLIN_small_molecule_dataset_for_machine_learning-based_retention_time_prediction/8038913?file=18130625). The structure of data directory is: 

```bash
|- data
  |- origin
	  |- SMRT_dataset.sdf
```

Step 2: Use the following commands to preprocess the datasets. The settings of datasets are in `./preprocess_etkdg.yml`. 

```bash
nohup python preprocess_rt.py --dataset metlin > preprocess_metlin.out 
```

Step 3: Use the following commands to train the model. The settings of model and training are in `./config/molnet_rt.yml`. If you'd like to train this model from the pre-trained model on MS/MS prediction, please download the pre-trained model from [[Google Drive]](https://drive.google.com/drive/folders/1fWx3d8vCPQi-U-obJ3kVL3XiRh75x5Ce?usp=drive_link). 

```bash
# learn from scratch
python train_rt.py --train_data ./data/metlin_etkdg_train.pkl \
--test_data ./data/metlin_etkdg_test.pkl \
--model_config_path ./config/molnet_rt.yml \
--data_config_path ./config/preprocess_etkdg.yml \
--checkpoint_path ./check_point/molnet_rt_etkdg.pt

# transfer from MS/MS prediction
python train_rt.py --train_data ./data/metlin_etkdg_train.pkl \
--test_data ./data/metlin_etkdg_test.pkl \
--model_config_path ./config/molnet_rt.yml \
--data_config_path ./config/preprocess_etkdg.yml \
--checkpoint_path ./check_point/molnet_rt_etkdg_tl.pt \
--transfer \
--resume_path ./check_point/molnet_qtof_etkdg.pt 
```



## Collision energy prediction

coming soon...