PyPI package: training
======================

``MolNet`` can train all task types (MS/MS, RT, CCS) directly from Python, without the command-line scripts. For CLI-based training, see :doc:`usage/index`.

.. autofunction:: molnetpack.MolNet.train
   :no-index:

.. autofunction:: molnetpack.MolNet.evaluate
   :no-index:

MS/MS model training
--------------------

Fine-tune from a pretrained checkpoint:

.. code-block:: python

   import torch
   from molnetpack import MolNet

   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   molnet_engine = MolNet(device, seed=42)

   best_cosine = molnet_engine.train(
       task='msms',
       train_data='./data/qtof_all_train.pkl',
       valid_data='./data/qtof_all_test.pkl',
       checkpoint_path='./check_point/molnet_qtof_tl.pt',
       resume_path='./check_point/molnet_pre_geobond.pt',
       transfer=True,
   )

   # The trained model is ready for inference immediately — no reload needed
   molnet_engine.load_data('./examples/demo_input.csv')
   pred_df = molnet_engine.pred_msms(instrument='qtof')

``transfer=True`` loads **only the encoder weights** from ``resume_path`` (a pretraining
or task checkpoint) — head weights, including the layer that consumes the experimental
condition, always start fresh. By default the transferred encoder is frozen and only the
head trains; pass ``freeze_encoder=False`` for a full fine-tune where everything trains.
Consistency between the checkpoint's embedded config and yours is validated on load, and
mismatches on meaning-changing keys are refused with an explanation.

Evaluate predictions against ground truth:

.. code-block:: python

   results_df = molnet_engine.evaluate(
       test_pkl='./data/qtof_all_test.pkl',
       pred_mgf='./result/pred_qtof_test.mgf',
       result_path='./eval_qtof_test.csv',
       plot_path='./eval_qtof_test.png',
   )

Retention time model training
-----------------------------

Fine-tune from the ChEMBL-pretrained encoder (how the released model was trained):

.. code-block:: python

   import torch
   from molnetpack import MolNet

   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   molnet_engine = MolNet(device, seed=42)

   best_mae = molnet_engine.train(
       task='rt',
       train_data='./data/rt_bond_train.pkl',
       valid_data='./data/rt_bond_val.pkl',
       checkpoint_path='./check_point/molnet_rt_tl.pt',
       resume_path='./check_point/molnet_pre_geobond.pt',
       transfer=True,
       use_scaler=True,
   )

CCS model training
------------------

Fine-tune from the ChEMBL-pretrained encoder:

.. code-block:: python

   import torch
   from molnetpack import MolNet

   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   molnet_engine = MolNet(device, seed=42)

   best_mae = molnet_engine.train(
       task='ccs',
       train_data='./data/ccs_bond_train.pkl',
       valid_data='./data/ccs_bond_val.pkl',
       checkpoint_path='./check_point/molnet_ccs_tl.pt',
       resume_path='./check_point/molnet_pre_geobond.pt',
       transfer=True,
   )
