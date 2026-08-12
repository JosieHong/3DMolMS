Train your own model for MS/MS prediction
=========================================

Setup
-----

Please set up the environment as shown in the :doc:`../sourcecode` page.

**Step 1**: Prepare the datasets
--------------------------------

Download and organize the source spectral libraries. The released models are trained on:

1. Agilent DPCL, provided by `Agilent Technologies <https://www.agilent.com/>`_.
2. `NIST20/NIST23 <https://www.nist.gov/programs-projects/nist23-updates-nist-tandem-and-electron-ionization-spectral-libraries>`_, available under license for academic use.
3. `MoNA <https://mona.fiehnlab.ucdavis.edu/downloads>`_, publicly available.
4. `GNPS <https://gnps.ucsd.edu/>`_, publicly available.
5. Waters QTOF, our own experimental dataset.

Convert each source to MGF (see ``molnetpack.sdf2mgf`` for SDF sources) and place the files under ``./data/origin/mgf/``, or point ``--raw_dir`` at your own location.

**Step 2**: Build the training pickles
--------------------------------------

``scripts/build_msms_dataset.py`` filters, deduplicates, annotates and splits the spectra into train/validation/test pickles. Instruments are grouped by analyser: ``qtof`` pools Q-TOF and QQQ spectra, ``orbi`` pools Orbitrap spectra. The split unit is the non-stereochemical InChIKey skeleton, so no structure leaks between partitions.

.. code-block:: bash

    # Q-TOF group:
    python scripts/build_msms_dataset.py --group qtof
    # Orbitrap group:
    python scripts/build_msms_dataset.py --group orbi

The outputs are ``data/qtof_all_{train,val,test}.pkl`` (or ``data/orbi_hcd_*.pkl`` for the Orbitrap group). Run with ``--stats_only`` first to see what would be kept, and ``--help`` for the filter knobs (collision-energy caps, CASMI exclusion, worker count). Molecule featurisation comes from the shared ``molnetpack/config/encoding_etkdgv3.yml``; the bins and the collision-energy scale come from the ``model`` block of ``molnetpack/config/molnet.yml``. The collision energy is stored as an NCE *fraction* (0.35 for 35%), declared by ``ce_scale`` in the model config.

**Step 3**: Train the model
---------------------------

Model architecture and training hyperparameters live in ``molnetpack/config/molnet.yml``; edit the ``train:`` section for your own runs (see the :doc:`../configuration` guide for which file and section owns what).

*Using the command-line script:*

.. code-block:: bash

  python scripts/train_msms_release.py \
  --train_data ./data/qtof_all_train.pkl \
  --val_data ./data/qtof_all_val.pkl \
  --test_data ./data/qtof_all_test.pkl \
  --ckpt ./check_point/molnet_qtof.pt \
  --gpu 0

Pass ``--pretrain <checkpoint>`` to initialise the encoder from a pretrained model (see :doc:`../advanced_usage/pretrain`).

*Using the Python API:*

.. code-block:: python

  import torch
  from molnetpack import MolNet

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  molnet_engine = MolNet(device, seed=42)

  # Train from scratch:
  molnet_engine.train(
      task='msms',
      train_data='./data/qtof_all_train.pkl',
      valid_data='./data/qtof_all_val.pkl',
      checkpoint_path='./check_point/molnet_qtof.pt',
  )

  # Or fine-tune from a pretrained encoder (frozen by default;
  # pass freeze_encoder=False to train everything):
  molnet_engine.train(
      task='msms',
      train_data='./data/qtof_all_train.pkl',
      valid_data='./data/qtof_all_val.pkl',
      checkpoint_path='./check_point/molnet_qtof_tl.pt',
      resume_path='./check_point/molnet_pre_geobond.pt',
      transfer=True,
  )

**Step 4**: Evaluation
----------------------

*Using the command-line scripts:*

.. code-block:: bash

  # Predict the spectra for the held-out test set:
  python scripts/predict.py --task msms \
  --test_data ./data/qtof_all_test.pkl \
  --resume_path ./check_point/molnet_qtof.pt \
  --result_path ./result/pred_qtof_test.mgf

  # Cosine similarity between experimental and predicted spectra:
  python scripts/eval.py ./data/qtof_all_test.pkl ./result/pred_qtof_test.mgf \
  --result_path ./eval_qtof_test.csv --plot_path ./eval_qtof_test.png

*Using the Python API:*

.. code-block:: python

  # After training, the model is immediately ready — no checkpoint reload needed.
  molnet_engine.load_data('./data/qtof_all_test.pkl')
  pred_df = molnet_engine.pred_msms(
      path_to_results='./result/pred_qtof_test.mgf',
      instrument='qtof',
  )

  results_df = molnet_engine.evaluate(
      test_pkl='./data/qtof_all_test.pkl',
      pred_mgf='./result/pred_qtof_test.mgf',
      result_path='./eval_qtof_test.csv',
      plot_path='./eval_qtof_test.png',
  )
