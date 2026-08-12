Collision cross section model training
======================================

3DMolMS predicts MS/MS-related molecular properties such as collision cross section (CCS). This guide shows how to train a CCS model, from scratch or by transfer learning from the ChEMBL-pretrained encoder.

The released CCS model is at `release v1.4.0 <https://github.com/JosieHong/3DMolMS/releases>`_.

Setup
-----

Please set up the environment as shown in the :doc:`../sourcecode` page.

**Step 1**: Data preparation
----------------------------

Download the `AllCCS <http://allccs.zhulab.cn/>`_ dataset, manually or with ``download_allccs.py``:

.. code-block:: bash

   python scripts/download_allccs.py --user <user_name> --passw <password> --output ./data/origin/allccs_download.csv

.. code-block:: text

   |- data
     |- origin
       |- allccs_download.csv

**Step 2**: Build the training pickles
--------------------------------------

``scripts/build_rt_ccs_dataset.py`` keeps only experimental CCS measurements (AllCCS also ships predicted values, which are not used as training targets), featurises the molecules with the shared encoding config, attaches the covalent bond graph, and splits on non-stereochemical InChIKey skeletons (80/10/10). The CCS adduct one-hot uses its own six-adduct set, defined in the ``encoding`` section of ``molnet_ccs_tl.yml``:

.. code-block:: bash

   python scripts/build_rt_ccs_dataset.py --task ccs --workers 8

**Step 3**: Training
--------------------

Model and training settings are in ``molnetpack/config/molnet_ccs_tl.yml``; edit its ``train:`` section for your own runs (see the :doc:`../configuration` guide).

*Using the command-line script:*

.. code-block:: bash

   # From scratch:
   python scripts/train_rt_ccs.py --task ccs --gpu 0

   # Transfer learning from the ChEMBL-pretrained encoder
   # (how the released model was trained; see the pretraining guide):
   python scripts/train_rt_ccs.py --task ccs --gpu 0 \
   --pretrain ./check_point/molnet_pre_geobond.pt --freeze_encoder

*Using the Python API:*

.. code-block:: python

   import torch
   from molnetpack import MolNet

   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   molnet_engine = MolNet(device, seed=42)

   # From scratch:
   molnet_engine.train(
       task='ccs',
       train_data='./data/ccs_bond_train.pkl',
       valid_data='./data/ccs_bond_val.pkl',
       checkpoint_path='./check_point/molnet_ccs.pt',
   )

   # Transfer learning: only the encoder weights are loaded from resume_path;
   # the head starts fresh. The encoder is frozen by default — pass
   # freeze_encoder=False for a full fine-tune.
   molnet_engine.train(
       task='ccs',
       train_data='./data/ccs_bond_train.pkl',
       valid_data='./data/ccs_bond_val.pkl',
       checkpoint_path='./check_point/molnet_ccs_tl.pt',
       resume_path='./check_point/molnet_pre_geobond.pt',
       transfer=True,
   )
