Fine-tune on your own data
==========================

This section shows how to fine-tune a regression model (retention time or CCS) on your own measurements.

Setup
-----

Please set up the environment as shown in the :doc:`../sourcecode` page.

**Step 1**: Data preparation
----------------------------

Prepare a CSV with an ``ID``, a ``SMILES`` and a target column per molecule, split into train and test files. Convert each split into a training pickle with ``molecules_to_records`` (which generates the 3D conformation and the covalent bond graph), then attach the target under the task's key — ``rt`` or ``ccs``:

.. code-block:: python

   import pickle
   import numpy as np
   import pandas as pd
   import yaml
   from molnetpack import molecules_to_records, config_path

   cfg = yaml.safe_load(open(config_path("encoding_etkdgv3.yml")))["encoding"]

   for split in ("train", "test"):
       df = pd.read_csv(f"<path_to_{split}.csv>")   # columns: ID, SMILES, RT
       records = molecules_to_records(df, cfg)
       targets = dict(zip(df["ID"], df["RT"]))
       for r in records:
           r["rt"] = float(targets[r["title"]])
           # RT uses no experimental covariates: env is a single placeholder column.
           r["env"] = np.zeros(1, dtype=np.float32)
       pickle.dump(records, open(f"<path_to_{split}.pkl>", "wb"))

For CCS, the target key is ``ccs`` and ``env`` must be the adduct one-hot from the CCS config instead of a placeholder:

.. code-block:: python

   ccs_layout = yaml.safe_load(open(config_path("molnet_ccs_tl.yml")))["encoding"]["precursor_type"]
   # per record, with the adduct string for that measurement:
   r["ccs"] = float(targets[r["title"]])
   r["env"] = np.asarray(ccs_layout[adduct], dtype=np.float32)

**Step 2**: Training
--------------------

Fine-tune from the ChEMBL-pretrained encoder, ``molnet_pre_geobond.pt`` — download it from
the `GitHub release <https://github.com/JosieHong/3DMolMS/releases>`_ into ``./check_point/``,
or produce it with the :doc:`pretraining pipeline <../advanced_usage/pretrain>`. The released
v1.4.0 models all warm-start from it. ``transfer=True`` loads only the encoder weights; the regression head starts fresh. The encoder is frozen by default (head-only training); pass ``freeze_encoder=False`` to train everything:

.. code-block:: python

   import torch
   from molnetpack import MolNet

   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   molnet_engine = MolNet(device, seed=42)

   molnet_engine.train(
       task='rt',
       train_data='<path_to_train.pkl>',
       valid_data='<path_to_test.pkl>',
       checkpoint_path='<path_to_save_checkpoint>',
       resume_path='./check_point/molnet_pre_geobond.pt',
       transfer=True,
       use_scaler=True,
   )

**Step 3**: Running prediction
------------------------------

Predict unlabeled data.

*Using the command-line script:*

.. code-block:: bash

   python scripts/predict.py --task rt \
   --test_data <path_to_csv_or_pkl> \
   --resume_path <path_to_checkpoint> \
   --result_path <path_to_results.csv>

*Using the Python API:*

.. code-block:: python

   # After training, the model is ready immediately — no reload needed.
   # To use an existing checkpoint instead, pass it explicitly:
   molnet_engine.load_data('<path_to_csv_or_pkl>')
   rt_df = molnet_engine.pred_rt(
       path_to_results='<path_to_results.csv>',
       path_to_checkpoint='<path_to_checkpoint>',
   )
