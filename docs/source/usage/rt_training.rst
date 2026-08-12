Retention time model training
==============================

3DMolMS predicts MS/MS-related molecular properties such as retention time (RT). This guide shows how to train an RT model, from scratch or by transfer learning from the ChEMBL-pretrained encoder.

The released RT model is at `release v1.4.0 <https://github.com/JosieHong/3DMolMS/releases>`_.

Setup
-----

Please set up the environment as shown in the :doc:`../sourcecode` page.

**Step 1**: Data preparation
----------------------------

Download the `METLIN-SMRT <https://figshare.com/articles/dataset/The_METLIN_small_molecule_dataset_for_machine_learning-based_retention_time_prediction/8038913?file=18130625>`_ retention time dataset as CSV (columns ``id``, ``smiles``, ``rt``):

.. code-block:: text

   |- data
     |- origin
       |- SMRT_dataset.csv

**Step 2**: Build the training pickles
--------------------------------------

``scripts/build_rt_ccs_dataset.py`` featurises the molecules with the shared encoding config, attaches the covalent bond graph the released encoder requires, and splits on non-stereochemical InChIKey skeletons (80/10/10):

.. code-block:: bash

   python scripts/build_rt_ccs_dataset.py --task rt --workers 8

**Step 3**: Training
--------------------

Model and training settings are in ``molnetpack/config/molnet_rt_tl.yml``; edit its ``train:`` section for your own runs (see the :doc:`../configuration` guide).

*Using the command-line script:*

.. code-block:: bash

   # From scratch:
   python scripts/train_rt_ccs.py --task rt --gpu 0

   # Transfer learning from the ChEMBL-pretrained encoder
   # (how the released model was trained; see the pretraining guide):
   python scripts/train_rt_ccs.py --task rt --gpu 0 \
   --pretrain ./check_point/molnet_pre_geobond.pt --freeze_encoder

*Using the Python API:*

.. code-block:: python

   import torch
   from molnetpack import MolNet

   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   molnet_engine = MolNet(device, seed=42)

   # From scratch:
   molnet_engine.train(
       task='rt',
       train_data='./data/rt_bond_train.pkl',
       valid_data='./data/rt_bond_val.pkl',
       checkpoint_path='./check_point/molnet_rt.pt',
   )

   # Transfer learning: only the encoder weights are loaded from resume_path;
   # the head starts fresh. The encoder is frozen by default — pass
   # freeze_encoder=False for a full fine-tune.
   molnet_engine.train(
       task='rt',
       train_data='./data/rt_bond_train.pkl',
       valid_data='./data/rt_bond_val.pkl',
       checkpoint_path='./check_point/molnet_rt_tl.pt',
       resume_path='./check_point/molnet_pre_geobond.pt',
       transfer=True,
       use_scaler=True,
   )
