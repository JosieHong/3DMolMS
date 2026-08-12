Pretraining the encoder on ChEMBL
=================================

The released v1.4.0 models all warm-start from a ChEMBL-pretrained encoder,
``molnet_pre_geobond.pt``, available from the
`GitHub release <https://github.com/JosieHong/3DMolMS/releases>`_ — download and unzip it
into ``./check_point/`` to fine-tune without running this pipeline. The rest of this page
describes how that checkpoint is produced.

The encoder is pretrained with a geometric self-supervised task before fine-tuning on MS/MS, RT or CCS. The task is coordinate denoising: Gaussian noise is added to the atom coordinates, and the encoder's rotation-invariant per-atom features must reconstruct each atom's clean local bond geometry (squared bond lengths and bond angles, both O(3)-invariant). This teaches the encoder the bond lengths and angles that the bond-graph aggregation exposes, without any labels.

Setup
-----

Please set up the environment as shown in the :doc:`../sourcecode` page.

**Step 1**: Build the conformer set
-----------------------------------

``scripts/chembl2pkl.py`` downloads ChEMBL structures (or reads a local SDF), generates ETKDGv3 conformers, and featurises them with the same encoding config as every downstream task, so the pretrained weights transfer without a featurisation mismatch:

.. code-block:: bash

   # quick test: a few thousand molecules from the EBI FTP
   python scripts/chembl2pkl.py --output ./data/chembl_ssl.pkl --limit 5000

   # full run
   python scripts/chembl2pkl.py --output ./data/chembl_ssl.pkl

**Step 2**: Attach the bond graph
---------------------------------

``scripts/build_chembl_dataset.py`` adds the covalent bonded-neighbour indices (k=6, matching the fine-tuning encoder) and writes the train/valid pickles used by the pretrainer:

.. code-block:: bash

   python scripts/build_chembl_dataset.py

**Step 3**: Pretrain
--------------------

.. code-block:: bash

   python scripts/pretrain_geo.py --gpu 0 --epochs 50 --batch 128 --sigma 0.2 \
   --ckpt ./check_point/molnet_pre_geobond.pt

``--sigma`` is the coordinate-noise standard deviation in Ångström.

**Step 4**: Use the pretrained encoder
--------------------------------------

Pass the checkpoint as the transfer source when training a task model — ``--pretrain`` in the CLI trainers, or ``resume_path=... , transfer=True`` in :meth:`molnetpack.MolNet.train`. Only the encoder weights are loaded; task heads always start fresh, and the encoder settings are validated against the checkpoint's embedded config.

.. code-block:: bash

   python scripts/train_msms_release.py --pretrain ./check_point/molnet_pre_geobond.pt \
   --train_data ./data/qtof_all_train.pkl --val_data ./data/qtof_all_val.pkl \
   --test_data ./data/qtof_all_test.pkl --ckpt ./check_point/molnet_qtof_tl.pt --gpu 0
