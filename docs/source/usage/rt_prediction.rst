Retention time prediction
=========================

3DMolMS can be used to predict MS/MS-related properties, such as retention time (RT) and collision cross section (CCS). This guide shows how to train a model for RT prediction and CCS prediction, and how to transfer these models to your own RT and CCS dataset.

All models mentioned can be downloaded from `release v1.2.0 <https://github.com/JosieHong/3DMolMS/releases/tag/v1.2.0>`_.

Setup
-----

Please set up the environment as shown in the :doc:`../sourcecode` page.

**Step 1**: Data preparation
----------------------------

Download the retention time dataset, `METLIN <https://figshare.com/articles/dataset/The_METLIN_small_molecule_dataset_for_machine_learning-based_retention_time_prediction/8038913?file=18130625>`_. The structure of data directory is:

.. code-block:: text

   |- data
     |- origin
       |- SMRT_dataset.sdf

**Step 2**: Preprocessing
-------------------------

Use the following commands to preprocess the datasets. The settings of datasets are in ``./src/molnetpack/config/preprocess_etkdgv3.yml``.

.. code-block:: bash

   python ./src/preprocess_oth.py --dataset metlin

**Step 3**: Training
--------------------

Use the following commands to train the model. The settings of model and training are in ``./src/molnetpack/config/molnet_rt.yml``. 

Learning from scratch:

.. code-block:: bash

   python ./src/train_rt.py --train_data ./data/metlin_etkdgv3_train.pkl \
   --test_data ./data/metlin_etkdgv3_test.pkl \
   --model_config_path ./src/molnetpack/config/molnet_rt.yml \
   --data_config_path ./src/molnetpack/config/preprocess_etkdgv3.yml \
   --checkpoint_path ./check_point/molnet_<version>_rt_etkdgv3.pt

If you'd like to train this model from the pre-trained model on MS/MS prediction, please download the pre-trained model from `release v1.2.0 <https://github.com/JosieHong/3DMolMS/releases/tag/v1.2.0>`_.

Learning from MS/MS model:

.. code-block:: bash

   python ./src/train_rt.py --train_data ./data/metlin_etkdgv3_train.pkl \
   --test_data ./data/metlin_etkdgv3_test.pkl \
   --model_config_path ./src/molnetpack/config/molnet_rt_tl.yml \
   --data_config_path ./src/molnetpack/config/preprocess_etkdgv3.yml \
   --checkpoint_path ./check_point/molnet_<version>_rt_etkdgv3_tl.pt \
   --transfer \
   --resume_path ./check_point/molnet_<version>_qtof_etkdgv3.pt

python ./src/train_rt.py --train_data ./data/metlin_etkdgv3_train.pkl \
   --test_data ./data/metlin_etkdgv3_test.pkl \
   --model_config_path ./src/molnetpack/config/molnet_rt_tl.yml \
   --data_config_path ./src/molnetpack/config/preprocess_etkdgv3.yml \
   --checkpoint_path ./check_point/molnet_v1.11_rt_etkdgv3_tl.pt \
   --transfer \
   --resume_path ./check_point/molnet_v1.11_qtof_etkdgv3.pt