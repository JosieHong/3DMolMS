Generating a reference library for molecular identification
===========================================================

3DMolMS can generate a reference library of small-molecule MS/MS spectra for identification through MS/MS searching.

Using molecules from HMDB
-------------------------

Setup
~~~~~

Please set up the environment as shown in the :doc:`../sourcecode` page.

**Step 1**: Data preparation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Download the HMDB structures from `HMDB Downloads <https://hmdb.ca/downloads>`_:

.. code-block:: text

   |- data
     |- hmdb
       |- structures.sdf

**Step 2**: Preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~

``scripts/hmdb2pkl.py`` uses the shared encoding config (``encoding_etkdgv3.yml``) automatically. The conformation source is a flag: ``--conf_type origin`` (default) uses the coordinates shipped in the SDF, while ``--conf_type etkdgv3`` regenerates conformers the same way the released models were trained — a consideration if you want the library geometry to match the training distribution:

.. code-block:: bash

   # coordinates from the SDF (historical default):
   python scripts/hmdb2pkl.py

   # or regenerate ETKDGv3 conformers:
   python scripts/hmdb2pkl.py --conf_type etkdgv3

**Step 3**: MS/MS generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Predict a spectrum for every molecule/adduct/collision-energy combination. The released checkpoint downloads automatically on first use:

.. code-block:: bash

   for i in {0..21}; do
     python scripts/predict.py --task msms \
     --test_data ./data/hmdb/hmdb_origin_$i.pkl \
     --result_path ./data/hmdb/molnet_hmdb_$i.mgf
   done

(The pickle filenames carry the conformation type, e.g. ``hmdb_etkdgv3_$i.pkl`` when ``--conf_type etkdgv3`` was used.)

Using molecules from RefMet
---------------------------

Setup
~~~~~

Please set up the environment as shown in the :doc:`../sourcecode` page.

**Step 1**: Data preparation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Download the RefMet dataset from `RefMet Browse <https://www.metabolomicsworkbench.org/databases/refmet/browse.php>`_:

.. code-block:: text

   |- data
     |- refmet
       |- refmet.csv

**Step 2**: Preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~

``scripts/refmet2pkl.py`` also uses the shared encoding config automatically and generates ETKDGv3 conformers from the SMILES:

.. code-block:: bash

   python scripts/refmet2pkl.py

**Step 3**: MS/MS generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python scripts/predict.py --task msms \
   --test_data ./data/refmet/refmet_etkdgv3.pkl \
   --result_path ./data/refmet/molnet_refmet.mgf
