Supported formats
==================

3DMolMS reads molecules for **inference** from three file types — **CSV**, **MGF**,
and **PKL** — through ``MolNet.load_data``. **SDF** is supported for **training /
reference-set preparation** via the preprocessing scripts. Predictions are written
back as **MGF** (MS/MS) or **CSV** (RT / CCS).

Every input must satisfy the model's molecular limits:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Item
     - Supported values
   * - Atom count
     - ≤ 300 (including hydrogens)
   * - Atom types
     - C, O, N, H, P, S, F, Cl, B, Br, I
   * - Precursor types
     - ``[M+H]+``, ``[M-H]-``, ``[M+H-H2O]+``, ``[M+Na]+``, ``[M+2H]2+``
   * - Collision energy
     - any number (e.g. ``40 V`` or ``40``)

Molecules that fall outside these limits — too many atoms, an unlisted element, an
unparseable SMILES, or an unlisted precursor type — are **silently skipped** during
loading.

Input formats
-------------

Which fields are required depends on the task:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Task
     - Requires
   * - MS/MS (``pred_msms``)
     - SMILES + precursor type + collision energy
   * - CCS (``pred_ccs``)
     - SMILES + precursor type
   * - RT (``pred_rt``) / features (``save_features``)
     - SMILES

CSV
~~~

A header row followed by one molecule per line. Column names are **case-sensitive**:

.. code-block:: text

   ID,SMILES,Precursor_Type,Collision_Energy
   demo_0,C/C(=C\CNc1nc[nH]c2ncnc1-2)CO,[M+H]+,80 V

- ``ID`` — molecule identifier, used as the result title. **Required.**
- ``SMILES`` — the molecule structure. **Required.**
- ``Precursor_Type`` — adduct; one of the supported precursor types. Needed for
  MS/MS and CCS.
- ``Collision_Energy`` — e.g. ``40 V`` (the unit is optional). Needed for MS/MS.

Omit the columns a task does not use — ``ID,SMILES`` alone is enough for RT or
``save_features``. See ``examples/demo_input.csv``.

The ``Collision_Energy`` column accepts free-text values in either unit — absolute
(``20 V``) or normalized (``NCE=35%``) — per row. Alternatively, give plain numbers and
add an optional ``Collision_Energy_Unit`` column (``eV`` or ``NCE``) stating how to read
them; rows with an unrecognised unit are skipped.

MGF
~~~

One ``BEGIN IONS`` … ``END IONS`` block per molecule (keys are case-insensitive):

.. code-block:: text

   BEGIN IONS
   TITLE=demo_0
   SMILES=C/C(=C\CNc1nc[nH]c2ncnc1-2)CO
   PRECURSOR_TYPE=[M+H]+
   COLLISION_ENERGY=80 V
   END IONS

Each block must state ``TITLE``, ``SMILES``, ``PRECURSOR_TYPE`` and
``COLLISION_ENERGY``; blocks missing one of them are skipped with a warning. All other
fields are ignored on input — including peak lists and any stated ``PRECURSOR_MZ``
(the precursor m/z is computed from the SMILES and adduct, exactly as for CSV input),
so records exported from a spectral library load unchanged. See
``examples/demo_input.mgf``.

SDF
~~~

Used for **preparing training or reference sets in bulk** (e.g. METLIN for RT, HMDB
for a reference library) via the preprocessing scripts (``scripts/build_msms_dataset.py``,
``scripts/build_rt_ccs_dataset.py``, ``scripts/hmdb2pkl.py``, ``scripts/refmet2pkl.py``).
These read each SDF molecule
block and its properties (SMILES and task labels such as retention time) and emit a
PKL. SDF is **not** a direct ``MolNet.load_data`` inference input — convert it to a
PKL first.

PKL
~~~

The preprocessed, ready-to-run format: a pickled ``list`` of dicts. It is the
fastest input because the 3D conformation has already been computed (CSV and MGF
inputs are converted to this on load).

.. code-block:: python

   [
     {
       "title":         "demo_0",            # str  — molecule id
       "smiles":        "C/C(=C\\CNc1...)CO", # str
       "mol":           np.ndarray,           # [max_atom_num, 21] — 3D conformation
       "neighbor_idx":  np.ndarray,           # [max_atom_num, k] — covalent bond graph
       "neighbor_mask": np.ndarray,           # [max_atom_num, k] — real-bond mask
       "env":           np.ndarray,           # collision-energy + precursor-type context
       "spec":          np.ndarray,           # binned reference spectrum (MS/MS training only)
     },
     ...
   ]

- ``mol`` — the 3D point cloud: dims 0–2 are the centered ``xyz`` coordinates,
  3–20 are per-atom attributes and the atom-type one-hot.
- ``env`` — the normalized collision energy plus the precursor-type one-hot
  (present for MS/MS and CCS).
- ``neighbor_idx`` / ``neighbor_mask`` — the covalent bond graph. The released
  encoder aggregates over bonded neighbours, so these are **required**; pickles
  built by pre-v1.4.0 preprocessing lack them and are refused with an error.
- ``spec`` — the binned reference spectrum; needed only for training / evaluation,
  not for prediction. RT / CCS training pickles carry an ``rt`` / ``ccs`` target
  value instead.

Output formats
--------------

MS/MS — MGF
~~~~~~~~~~~

``pred_msms`` writes one ``BEGIN IONS`` block per molecule with the predicted
m/z–intensity peak list, alongside ``TITLE``, ``SMILES``, ``PRECURSOR_TYPE`` and
``COLLISION_ENERGY``. Predicted spectra never contain peaks above the precursor m/z —
in particular, no isotope envelope. The released models may predict a surviving
precursor ion at the precursor m/z itself (prominent at low collision energy); models
trained with the package's own preprocessing (``generate_ms``) have the precursor and
isotope peaks removed from their targets and do not predict them.

``MolNet.pred_all`` additionally embeds the RT and CCS predictions in each ion block as
``PRED_RT`` (seconds) and ``PRED_CCS`` (Å²) — named to make clear they are predictions,
not measurements — and, for CSV output, as ``Pred RT`` / ``Pred CCS`` columns:

.. code-block:: text

   BEGIN IONS
   TITLE=demo_0
   SMILES=C/C(=C\CNc1nc[nH]c2ncnc1-2)CO
   PRECURSOR_TYPE=[M+H]+
   COLLISION_ENERGY=79.95
   PRED_RT=227.74
   PRED_CCS=147.9
   51.00000 122.1
   53.00000 134.3
   ...
   END IONS

RT / CCS — CSV
~~~~~~~~~~~~~~

``pred_rt`` and ``pred_ccs`` return a ``pandas.DataFrame`` (and write a CSV) with
one row per molecule; the prediction column is ``Pred RT`` or ``Pred CCS``:

.. code-block:: text

   ,ID,SMILES,Precursor Type,Pred CCS
   0,demo_0,C/C(=C\CNc1nc[nH]c2ncnc1-2)CO,[M+H]+,154.62
