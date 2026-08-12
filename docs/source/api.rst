API reference
=============

The high-level entry point
--------------------------

.. autoclass:: molnetpack.MolNet
   :members: load_data, load_dataframe, load_smiles, get_data,
             pred_msms, pred_rt, pred_ccs, pred_all, save_features,
             train, evaluate, load_checkpoint, generate_spectra_from_df

.. autofunction:: molnetpack.plot_msms

.. autofunction:: molnetpack.config_path

Models
------

.. autoclass:: molnetpack.MolNetMS

.. autoclass:: molnetpack.MolNetScalar
   :members: predict, set_scaler, fit_scaler

.. autoclass:: molnetpack.model.Encoder

.. autoclass:: molnetpack.molconv.MolConv

Datasets
--------

.. autoclass:: molnetpack.MolMSDataset

.. autoclass:: molnetpack.MolScalarDataset

.. autoclass:: molnetpack.MolRTDataset

.. autoclass:: molnetpack.MolCCSDataset

.. autoclass:: molnetpack.MolInferenceDataset

Checkpoint management
---------------------

.. automodule:: molnetpack.checkpoints
   :members: checkpoint_dir, resolve_checkpoint_path, ensure_checkpoint,
             validate_checkpoint_config, validate_encoder_config, load_weights

Data conversion and encoding
----------------------------

.. autofunction:: molnetpack.molecules_to_records

.. autofunction:: molnetpack.mgf2pkl

.. autofunction:: molnetpack.sdf2pkl_with_cond

.. autofunction:: molnetpack.sdf2mgf

.. autofunction:: molnetpack.filter_spec

.. autofunction:: molnetpack.filter_mol

.. autofunction:: molnetpack.check_atom

.. autofunction:: molnetpack.data_utils.encoding.conformation_array

.. autofunction:: molnetpack.data_utils.encoding.bond_graph_array

.. autofunction:: molnetpack.data_utils.encoding.generate_ms

.. autofunction:: molnetpack.data_utils.encoding.mz_to_bin

.. autofunction:: molnetpack.data_utils.encoding.parse_collision_energy

.. autofunction:: molnetpack.data_utils.encoding.precursor_calculator

.. autofunction:: molnetpack.data_utils.encoding.normalize_adduct

.. autofunction:: molnetpack.data_utils.encoding.skeleton_key

Deprecated aliases
------------------

Pre-v1.4.0 names remain importable and forward to the current ones: ``MolNet_MS`` →
:class:`molnetpack.MolNetMS`, ``MolNet_Oth`` → :class:`molnetpack.MolNetScalar`,
``Mol_Dataset`` → :class:`molnetpack.MolInferenceDataset`, ``MolMS_Dataset`` /
``MolRT_Dataset`` / ``MolCCS_Dataset`` → the corresponding dataset classes above,
``csv2pkl_wfilter`` → :func:`molnetpack.molecules_to_records`, ``MolConv2`` →
:class:`molnetpack.molconv.MolConv`; the modules ``molnetpack.utils`` and
``molnetpack.data_utils.utils`` forward to ``molnetpack.steps`` and
``molnetpack.data_utils.encoding``.
