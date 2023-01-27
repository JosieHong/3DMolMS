# Exp

## Data Preprocess

Please download the datasets, unzip and put them in `./data/`. The structure of data directory is:

```bash
|- data
    |- Agilent
        |- Agilent_Combined.sdf
        |- Agilent_Metlin.sdf
    |- MassBank
        |- MoNA-export-LC-MS-MS_Spectra.zip
        |- MoNA-export-LC-MS-MS_Spectra.msp
    |- NIST20
        |- hr_msms_nist.MSP
        |- hr_msms_nist.SDF
        |- lr_msms_nist.MSP
        |- lr_msms_nist.SDF
```

### Agilent & NIST20 (training & validation data)

```bash
cd tools
```

1. Agilent 

```bash
# 1. convert sdf to mgf
python agilent2mgf.py --input_sdf ../data/Agilent/Agilent_Combined.sdf --output_mgf ../data/Agilent/Agilent_Combined.mgf
python agilent2mgf.py --input_sdf ../data/Agilent/Agilent_Metlin.sdf --output_mgf ../data/Agilent/Agilent_Metlin.mgf

# 2. cat all data together
cat ../data/Agilent/Agilent_Combined.mgf ../data/Agilent/Agilent_Metlin.mgf > ../data/Agilent/ALL_Agilent.mgf

# 3. clean up
python clean_up.py --input ../data/Agilent/ALL_Agilent.mgf --output ../data/Agilent/ALL_Agilent_clean.mgf --log ../data/Agilent/clean_up.json --dataset_name agilent

# 4. data prepare
python data_prepare.py --input ../data/Agilent/ALL_Agilent_clean.mgf --output_dir ../data/Agilent/proc/ --log ../data/Agilent/proc/filterout_multi.json --cond agilent

# 4. filter by the following conditions:
	# (1) organism
	# (2) instrument
	# (3) MS level
	# (4) precursor type
	# (5) atom 
	# (6) collision energy
python cond_filter_single.py --input ../data/Agilent/ALL_Agilent_clean.mgf --output ../data/Agilent/proc/ALL_Agilent_agilent.mgf --log ../data/Agilent/proc/filterout_qtof.json --cond agilent_qtof
```

2. NIST20

```bash
cd ./tools
# 1. convert MSP and SDF into mgf
# python nist2mgf.py --input_msp <path to MSP file> --input_sdf_dir <path to SDF file>  --type <hr_msms|lr_msms>
python nist2mgf.py --input_msp ../data/NIST20/hr_msms_nist.MSP --input_sdf_dir ../data/NIST20/hr_msms_nist.SDF --type hr_msms
python nist2mgf.py --input_msp ../data/NIST20/lr_msms_nist.MSP --input_sdf_dir ../data/NIST20/lr_msms_nist.SDF --type lr_msms

# 2. cat all data together
cat ../data/NIST20/hr_msms_nist.mgf ../data/NIST20/lr_msms_nist.mgf > ../data/NIST20/ALL_NIST.mgf

# 3. clean up
python clean_up.py --input ../data/NIST20/ALL_NIST.mgf --output ../data/NIST20/ALL_NIST_clean.mgf --log ../data/NIST20/clean_up.json --dataset_name nist

# 4. filter by conditions
python cond_filter_single.py --input ../data/NIST20/ALL_NIST_clean.mgf --output ../data/NIST20/proc/ALL_NIST_agilent.mgf --log ../data/NIST20/proc/filterout_qtof.json --cond nist_qtof
# python cond_filter_single.py --input ../data/NIST20/ALL_NIST_clean.mgf --output ../data/NIST20/proc/ALL_NIST_hcd.mgf --log ../data/NIST20/proc/filterout_hcd.json --cond nist_hcd

# Fragmentation Methods: 27,840 HRAM (High Res Accurate Mass) Compounds; 29,890 QTOF, HCD, IT-HRAM, QqQ Compounds; 29,444 Ion Trap Compounds (Low Res, up to MS4); 246 APCI HRAM 'Extractables and Leachables'. 
```

3. Merge them (agilent & nist20) together

```bash
cat ../data/Agilent/proc/ALL_Agilent_agilent.mgf ../data/NIST20/proc/ALL_NIST_agilent.mgf > ../data/MERGE/proc/ALL_MERGE_agilent.mgf

python scaffold_spliter.py 0.1 ../data/MERGE/proc/ALL_MERGE_agilent.mgf
python random_spliter.py 0.1 ../data/MERGE/proc/ALL_MERGE_agilent.mgf

# cat ../data/NIST20/proc/ALL_NIST_hcd.mgf > ../data/MERGE/proc/ALL_MERGE_hcd.mgf

# python scaffold_spliter.py 0.1 ../data/MERGE/proc/ALL_MERGE_hcd.mgf
# python random_spliter.py 0.1 ../data/MERGE/proc/ALL_MERGE_hcd.mgf
```

4. Export smiles for other models

```bash
python extract_smiles_list.py --input ../data/MERGE/exp/test_ALL_MERGE_agilent.mgf --output ../data/MERGE/test_ALL_MERGE_agilent_pos.txt --ion_mode P
python extract_smiles_list.py --input ../data/MERGE/exp/test_ALL_MERGE_agilent.mgf --output ../data/MERGE/test_ALL_MERGE_agilent_neg.txt --ion_mode N
```

### MassBank (test)

1. MassBank

```bash
# 1. conver msp to mgf
python massbank2mgf.py --input_msp ../data/MassBank/MoNA-export-LC-MS-MS_Spectra.msp

# 2. clean up
python clean_up.py --input ../data/MassBank/ALL_MB.mgf --output ../data/MassBank/ALL_MB_clean.mgf --log ../data/MassBank/clean_up.json --dataset_name massbank

# 3. filter by conditions
python cond_filter_single.py --input ../data/MassBank/ALL_MB_clean.mgf --output ../data/MassBank/proc/ALL_MB_qtof.mgf --log ../data/MassBank/proc/filterout_qtof.json --cond massbank_qtof
# python cond_filter_single.py --input ../data/MassBank/ALL_MB_clean.mgf --output ../data/MassBank/proc/ALL_MB_hcd.mgf --log ../data/MassBank/proc/filterout_hcd.json --cond massbank_hcd

# 4. split for fine-tune
python scaffold_spliter.py 0.5 ../data/MassBank/proc/ALL_MB_qtof.mgf
python random_spliter.py 0.5 ../data/MassBank/proc/ALL_MB_qtof.mgf

# Save 1427/481 data to ../data/MassBank/exp/train_ALL_MB_qtof.mgf
# Save 1448/480 data to ../data/MassBank/exp/test_ALL_MB_qtof.mgf
```

~~2. GNPS (test data)~~

Download GNPS subset here: 

```bash
wget https://gnps-external.ucsd.edu/gnpslibrary/LDB_POSITIVE.mgf
wget https://gnps-external.ucsd.edu/gnpslibrary/LDB_NEGATIVE.mgf
wget https://gnps-external.ucsd.edu/gnpslibrary/IQAMDB.mgf
```

```bash
cd ./tools
# 1. add title
python add_title.py ../data/GNPS_lib/raw/LDB_NEGATIVE.mgf ../data/GNPS_lib/raw/LDB_NEGATIVE_titled.mgf
python add_title.py ../data/GNPS_lib/raw/LDB_POSITIVE.mgf ../data/GNPS_lib/raw/LDB_POSITIVE_titled.mgf
# 2. clean up
python clean_up.py --input ../data/GNPS_lib/raw/LDB_NEGATIVE_titled.mgf --output ../data/GNPS_lib/LDB_NEGATIVE_clean.mgf --log ../data/GNPS_lib/clean_up.json --dataset_name gnps
python clean_up.py --input ../data/GNPS_lib/raw/LDB_POSITIVE_titled.mgf --output ../data/GNPS_lib/LDB_POSITIVE_clean.mgf --log ../data/GNPS_lib/clean_up.json --dataset_name gnps
# 3. filter by conditions
python cond_filter_single.py --input ../data/GNPS_lib/LDB_NEGATIVE_clean.mgf --output ../data/GNPS_lib/proc/LDB_NEGATIVE_neg.mgf --log ../data/NIST20/proc/filterout_ldb_neg.json --cond gnps_neg
python cond_filter_single.py --input ../data/GNPS_lib/LDB_POSITIVE_clean.mgf --output ../data/GNPS_lib/proc/LDB_POSITIVE_pos.mgf --log ../data/NIST20/proc/filterout_ldb_pos.json --cond gnps_pos
```

3. Export smiles for other models

```bash
python extract_smiles_list.py --input ../data/MassBank/exp/test_ALL_MB_qtof.mgf --output ../data/MassBank/test_ALL_MB_qtof_pos.txt --ion_mode P
python extract_smiles_list.py --input ../data/MassBank/exp/test_ALL_MB_qtof.mgf --output ../data/MassBank/test_ALL_MB_qtof_neg.txt --ion_mode N
```



### Statistics

```bash
python stats_data.py --input ../data/MERGE/proc/ALL_MERGE_agilent.mgf --ion_mode P
python stats_data.py --input ../data/MERGE/proc/ALL_MERGE_agilent.mgf --ion_mode N
python stats_data.py --input ../data/MERGE/proc/ALL_MERGE_agilent.mgf --ion_mode ALL

python replicates.py --input1 ../data/Agilent/proc/ALL_Agilent_agilent.mgf --input2 ../data/NIST20/proc/ALL_NIST_agilent.mgf --ion_mode P --output ../results/replicated_agilent_pos.csv
python replicates.py --input1 ../data/Agilent/proc/ALL_Agilent_agilent.mgf --input2 ../data/NIST20/proc/ALL_NIST_agilent.mgf --ion_mode N --output ../results/replicated_agilent_neg.csv
python replicates.py --input1 ../data/Agilent/proc/ALL_Agilent_agilent.mgf --input2 ../data/NIST20/proc/ALL_NIST_agilent.mgf --ion_mode ALL --output ../results/replicated_agilent.csv

python stats_data.py --input ../data/Agilent/proc/ALL_Agilent_agilent.mgf --ion_mode P
python stats_data.py --input ../data/Agilent/proc/ALL_Agilent_agilent.mgf --ion_mode N

python stats_data.py --input ../data/NIST20/proc/ALL_NIST_agilent.mgf --ion_mode P
python stats_data.py --input ../data/NIST20/proc/ALL_NIST_agilent.mgf --ion_mode N

python stats_data.py --input ../data/MassBank/proc/ALL_MB_qtof.mgf --ion_mode P
python stats_data.py --input ../data/MassBank/proc/ALL_MB_qtof.mgf --ion_mode N

python replicates.py --input1 ../data/MERGE/proc/ALL_MERGE_agilent.mgf --input2 ../data/MassBank/proc/ALL_MB_qtof.mgf --ion_mode P --output ../results/replicated_mb_pos.csv
python replicates.py --input1 ../data/MERGE/proc/ALL_MERGE_agilent.mgf --input2 ../data/MassBank/proc/ALL_MB_qtof.mgf --ion_mode N --output ../results/replicated_mb_neg.csv
python replicates.py --input1 ../data/MERGE/proc/ALL_MERGE_agilent.mgf --input2 ../data/MassBank/proc/ALL_MB_qtof.mgf --ion_mode ALL --output ../results/replicated_mb.csv
```



## Train

### 1. Train on merged MS/MS (Agilent & NIST20)

```bash
# pretrain
python train.py --model molnet --dataset merge --epochs 300 --resolution 0.2 --batch_size 64 --ion_mode ALL \
    --train_data_path ./data/MERGE/exp/train_ALL_MERGE_agilent.mgf \
	--test_data_path ./data/MERGE/exp/test_ALL_MERGE_agilent.mgf \
	--log_dir ./logs \
	--checkpoint_path ./check_point/molnet_agilent.pt \
	--resume_path ./check_point/molnet_agilent.pt \
	--device 1 

# positive
python train.py --model molnet --dataset merge --epochs 300 --resolution 0.2 --batch_size 64 --ion_mode P \
    --train_data_path ./data/MERGE/exp/train_ALL_MERGE_agilent.mgf \
	--test_data_path ./data/MERGE/exp/test_ALL_MERGE_agilent.mgf \
	--log_dir ./logs \
	--transfer \
	--checkpoint_path ./check_point/molnet_agilent_pos.pt \
	--resume_path ./check_point/molnet_agilent.pt \
	--device 1 
python eval.py --model molnet --dataset merge --resolution 0.2 --ion_mode P \
	--test_data_path ./data/MERGE/exp/test_ALL_MERGE_agilent.mgf \
	--resume_path ./check_point/molnet_agilent_pos.pt \
	--result_path ./results/molnet_agilent_pos.csv

# negative
python train.py --model molnet --dataset merge --epochs 300 --resolution 0.2 --batch_size 64 --ion_mode N \
    --train_data_path ./data/MERGE/exp/train_ALL_MERGE_agilent.mgf \
	--test_data_path ./data/MERGE/exp/test_ALL_MERGE_agilent.mgf \
	--log_dir ./logs \
	--transfer \
	--checkpoint_path ./check_point/molnet_agilent_neg.pt \
	--resume_path ./check_point/molnet_agilent.pt \
	--device 1 
python eval.py --model molnet --dataset merge --resolution 0.2 --ion_mode N \
	--test_data_path ./data/MERGE/exp/test_ALL_MERGE_agilent.mgf \
	--resume_path ./check_point/molnet_agilent_neg.pt \
	--result_path ./results/molnet_agilent_neg.csv 
```

### 2. Eval on MoNA

```bash
python eval.py --model molnet --dataset merge --resolution 0.2 --ion_mode P \
	--test_data_path ./data/MassBank/exp/test_ALL_MB_qtof.mgf \
	--resume_path ./check_point/molnet_agilent_pos.pt \
	--result_path ./results/molnet_mb_test_pos.csv

python eval.py --model molnet --dataset merge --resolution 0.2 --ion_mode N \
	--test_data_path ./data/MassBank/exp/test_ALL_MB_qtof.mgf \
	--resume_path ./check_point/molnet_agilent_neg.pt \
	--result_path ./results/molnet_mb_test_neg.csv
```

### 3. Fine-tune on MoNA

```bash
python train.py --model molnet --dataset merge --epochs 300 --resolution 0.2 --batch_size 64 --ion_mode P \
    --train_data_path ./data/MassBank/exp/train_ALL_MB_qtof.mgf \
	--test_data_path ./data/MassBank/exp/test_ALL_MB_qtof.mgf \
	--log_dir ./logs \
	--transfer \
	--checkpoint_path ./check_point/molnet_agilent_mb_pos.pt \
	--resume_path ./check_point/molnet_agilent_pos.pt \
	--device 1 
python eval.py --model molnet --dataset merge --resolution 0.2 --ion_mode P \
	--test_data_path ./data/MassBank/exp/test_ALL_MB_qtof.mgf \
	--resume_path ./check_point/molnet_agilent_mb_pos.pt \
	--result_path ./results/molnet_mb_finetune_pos.csv

python train.py --model molnet --dataset merge --epochs 300 --resolution 0.2 --batch_size 64 --ion_mode N \
    --train_data_path ./data/MassBank/exp/train_ALL_MB_qtof.mgf \
	--test_data_path ./data/MassBank/exp/test_ALL_MB_qtof.mgf \
	--log_dir ./logs \
	--transfer \
	--checkpoint_path ./check_point/molnet_agilent_mb_neg.pt \
	--resume_path ./check_point/molnet_agilent_neg.pt \
	--device 1 
python eval.py --model molnet --dataset merge --resolution 0.2 --ion_mode N \
	--test_data_path ./data/MassBank/exp/test_ALL_MB_qtof.mgf \
	--resume_path ./check_point/molnet_agilent_mb_neg.pt \
	--result_path ./results/molnet_mb_finetune_neg.csv
```



## Comparision Methods

### 1. PointNet + MSDecoder

```bash
python train.py --model pointnet --dataset merge --epochs 300 --resolution 0.2 --batch_size 64 --ion_mode P \
    --train_data_path ./data/MERGE/exp/train_ALL_MERGE_agilent.mgf \
	--test_data_path ./data/MERGE/exp/test_ALL_MERGE_agilent.mgf \
	--log_dir ./logs \
	--checkpoint_path ./check_point/pointnet_agilent_pos.pt \
	--resume_path ./check_point/pointnet_agilent_pos.pt \
	--device 1 
python eval.py --model pointnet --dataset merge --resolution 0.2 --ion_mode P \
	--test_data_path ./data/MERGE/exp/test_ALL_MERGE_agilent.mgf \
	--resume_path ./check_point/pointnet_agilent_pos.pt \
	--result_path ./results/pointnet_agilent_pos.csv

python train.py --model pointnet --dataset merge --epochs 300 --resolution 0.2 --batch_size 64 --ion_mode N \
    --train_data_path ./data/MERGE/exp/train_ALL_MERGE_agilent.mgf \
	--test_data_path ./data/MERGE/exp/test_ALL_MERGE_agilent.mgf \
	--log_dir ./logs \
	--checkpoint_path ./check_point/pointnet_agilent_neg.pt \
	--resume_path ./check_point/pointnet_agilent_neg.pt \
	--device 1 
python eval.py --model pointnet --dataset merge --resolution 0.2 --ion_mode N \
	--test_data_path ./data/MERGE/exp/test_ALL_MERGE_agilent.mgf \
	--resume_path ./check_point/pointnet_agilent_neg.pt \
	--result_path ./results/pointnet_agilent_neg.csv
```

### 2. DGCNN + MSDecoder

```bash
python train.py --model dgcnn --dataset merge --epochs 300 --resolution 0.2 --batch_size 64 --ion_mode P \
    --train_data_path ./data/MERGE/exp/train_ALL_MERGE_agilent.mgf \
	--test_data_path ./data/MERGE/exp/test_ALL_MERGE_agilent.mgf \
	--log_dir ./logs \
	--checkpoint_path ./check_point/dgcnn_agilent_pos.pt \
	--resume_path ./check_point/dgcnn_agilent_pos.pt \
	--device 1 
python eval.py --model dgcnn --dataset merge --resolution 0.2 --ion_mode P \
	--test_data_path ./data/MERGE/exp/test_ALL_MERGE_agilent.mgf \
	--resume_path ./check_point/dgcnn_agilent_pos.pt \
	--result_path ./results/dgcnn_agilent_pos.csv

python train.py --model dgcnn --dataset merge --epochs 300 --resolution 0.2 --batch_size 64 --ion_mode N \
    --train_data_path ./data/MERGE/exp/train_ALL_MERGE_agilent.mgf \
	--test_data_path ./data/MERGE/exp/test_ALL_MERGE_agilent.mgf \
	--log_dir ./logs \
	--checkpoint_path ./check_point/dgcnn_agilent_neg.pt \
	--resume_path ./check_point/dgcnn_agilent_neg.pt \
	--device 1 
python eval.py --model dgcnn --dataset merge --resolution 0.2 --ion_mode N \
	--test_data_path ./data/MERGE/exp/test_ALL_MERGE_agilent.mgf \
	--resume_path ./check_point/dgcnn_agilent_neg.pt \
	--result_path ./results/dgcnn_agilent_neg.csv
```

### 3. SchNet + MSDecoder

```bash
python train.py --model schnet --dataset merge --epochs 300 --resolution 0.2 --batch_size 64 --ion_mode P \
    --train_data_path ./data/MERGE/exp/train_ALL_MERGE_agilent.mgf \
	--test_data_path ./data/MERGE/exp/test_ALL_MERGE_agilent.mgf \
	--log_dir ./logs \
	--checkpoint_path ./check_point/schnet_agilent_pos.pt \
	--resume_path ./check_point/schnet_agilent_pos.pt \
	--device 1 
python eval.py --model schnet --dataset merge --resolution 0.2 --ion_mode P \
	--test_data_path ./data/MERGE/exp/test_ALL_MERGE_agilent.mgf \
	--resume_path ./check_point/schnet_agilent_pos.pt \
	--result_path ./results/schnet_agilent_pos.csv

python train.py --model schnet --dataset merge --epochs 300 --resolution 0.2 --batch_size 64 --ion_mode N \
    --train_data_path ./data/MERGE/exp/train_ALL_MERGE_agilent.mgf \
	--test_data_path ./data/MERGE/exp/test_ALL_MERGE_agilent.mgf \
	--log_dir ./logs \
	--transfer \
	--checkpoint_path ./check_point/schnet_agilent_neg.pt \
	--resume_path ./check_point/schnet_agilent.pt \
	--device 1 
python eval.py --model schnet --dataset merge --resolution 0.2 --ion_mode N \
	--test_data_path ./data/MERGE/exp/test_ALL_MERGE_agilent.mgf \
	--resume_path ./check_point/schnet_agilent_neg.pt \
	--result_path ./results/schnet_agilent_neg.csv
```

