Tandem mass spectra prediction
==============================

This guide explains how to predict tandem mass spectra (MS/MS) from the command line.

Setup
-----

Please set up the environment as shown in the :doc:`../sourcecode` page.

**Step 1**: Input preparation
-----------------------------

Prepare the test set as a CSV, MGF, or PKL file. A minimal CSV needs ``ID``, ``SMILES``, ``Precursor_Type``, and ``Collision_Energy``:

.. code-block:: text

   ID,SMILES,Precursor_Type,Collision_Energy
   demo_0,C/C(=C\CNc1nc[nH]c2ncnc1-2)CO,[M+H]+,40 V

The ``Collision_Energy`` column accepts ``20 V`` or ``NCE=35%``; with an optional ``Collision_Energy_Unit`` column (``eV`` / ``NCE``) you can give plain numbers instead. See :doc:`../supported_formats` for the MGF layout, the PKL structure, the supported atom and precursor types, and what the predicted spectra contain (no peaks above the precursor m/z). Unsupported molecules are skipped automatically on load.

**Step 2**: Running prediction
------------------------------

``scripts/predict.py`` is a thin wrapper over ``molnetpack.MolNet``; the bundled model and encoding configs are used automatically:

.. code-block:: bash

  python scripts/predict.py --task msms \
  --test_data ./examples/demo_input.csv \
  --result_path ./examples/output_msms.mgf \
  --instrument qtof

Arguments
~~~~~~~~~

* ``--instrument``: ``qtof`` (default) or ``orbitrap``.
* ``--resume_path``: optional custom checkpoint. By default the released weights are downloaded on first use from the `GitHub release <https://github.com/JosieHong/3DMolMS/releases>`_ into the per-user cache directory. The checkpoint's embedded config is validated against the loaded config; a mismatch on any meaning-changing key (encoder settings, binning, collision-energy scale) is refused with an explanation instead of predicting from the wrong model.
* ``--result_path``: where to save the prediction. Use ``.mgf`` (recommended for MS/MS) or ``.csv``.
* ``--batch_size``: inference batch size (default 1).
