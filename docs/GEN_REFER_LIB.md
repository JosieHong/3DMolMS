# Generate reference library for molecular identification



## Using molecules from HMDB

Please set up the environment as shown in step 0 from `README.md`. 

Step 1: Download the HMDB molecules dataset [[here]](https://hmdb.ca/downloads). The structure of data directory is: 

```bash
|- data
  |- hmdb
    |- structures.sdf
```

Step 2: Use the following commands to preprocess the datasets. The settings of datasets are in `./preprocess_etkdgv3.yml`. 

```bash
python hmdb2pkl.py --data_config_path ./config/preprocess_etkdgv3.yml
```

Step 3: Use the following commands to generate MS/MS. The settings of model are in `./config/molnet.yml`. 

```bash
python pred.py --test_data ./data/hmdb/hmdb_etkdgv3.pkl \
--model_config_path ./config/molnet.yml \
--data_config_path ./config/preprocess_etkdgv3.yml \
--resume_path ./check_point/molnet_qtof_etkdgv3.pt \
--result_path ./data/hmdb/molnet_v1.1_hmdb_etkdgv3.mgf
```

## Using molecules from RefMet

Please set up the environment as shown in step 0 from `README.md`. 

Step 1: Download the RefMet molecules dataset [[here]](https://www.metabolomicsworkbench.org/databases/refmet/browse.php). The structure of data directory is: 

```bash
|- data
  |- refmet
    |- refmet.csv
```

Step 2: Use the following commands to preprocess the datasets. The settings of datasets are in `./preprocess_etkdgv3.yml`. 

```bash
python refmet2pkl.py --data_config_path ./config/preprocess_etkdgv3.yml
```

Step 3: Use the following commands to generate MS/MS. The settings of model are in `./config/molnet.yml`. 

```bash
python pred.py --test_data ./data/refmet/refmet_etkdgv3.pkl \
--model_config_path ./config/molnet.yml \
--data_config_path ./config/preprocess_etkdgv3.yml \
--resume_path ./check_point/molnet_qtof_etkdgv3.pt \
--result_path ./data/refmet/molnet_v1.1_refmet_etkdgv3.mgf
```


