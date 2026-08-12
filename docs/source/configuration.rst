Configuration guide
===================

3DMolMS is configured by four YAML files bundled with the package. This page explains which
file owns what, and which one to edit for the most common case.

**The short answer: to train on your own data, edit the** ``train:`` **section of your task's
config and nothing else.**

The four files
--------------

Resolve any bundled config by name with :func:`molnetpack.config_path`; the same directory is
used by ``MolNet`` and all the scripts.

``encoding_etkdgv3.yml`` — the shared data encoding
    Used by **all tasks** and by the preprocessing scripts. Defines how molecules are
    featurised (conformer method, atom-type one-hots, padding size), how spectra are binned,
    and the MS/MS adduct one-hot layout. You normally never edit this file: changing it
    changes what every model's inputs *mean*, which requires re-running preprocessing and
    retraining every model. It also carries per-source filter sections (``nist_qtof:``,
    ``hmdb:``, ...) used when building datasets.

``molnet.yml`` — the MS/MS task
    Three sections: ``model:``, ``train:``, ``release:`` (below).

``molnet_rt_tl.yml`` — the RT task
    Same three sections. RT has no experimental covariates, so there is no adduct encoding;
    ``env`` is a single placeholder column.

``molnet_ccs_tl.yml`` — the CCS task
    The three sections plus an ``encoding:`` block holding the **CCS-specific adduct one-hot
    layout**. CCS deliberately uses its own six-adduct set (AllCCS is dominated by adducts the
    MS/MS data barely has), so this does not reuse the shared config's five-way MS/MS layout.

The section rule
----------------

``model:``
    The architecture and output-space definition. This is what a checkpoint *is*. Never edit
    it to "tune": a released checkpoint embeds a copy of this block and the loader refuses a
    checkpoint whose copy contradicts yours (see :doc:`checkpoints`). Edit it only to define a
    deliberately new architecture you will train from scratch.

``train:``
    Optimisation hyperparameters: epochs, learning rate, weight decay, batch size, early-stop
    patience. **This is the section to edit when training on your own data.** Nothing here
    changes what the weights mean, so it is never validated against checkpoints.

``release:``
    Locations and download URLs of the released checkpoints, plus evaluation filters. Only
    relevant when you publish your own release artifacts.

The collision-energy scale (``ce_scale``)
-----------------------------------------

``molnet.yml`` declares ``ce_scale: fraction``: the v1.4.0 MS/MS models were trained with the
normalised collision energy as a *fraction* (0.35 for 35%), while the package's data
converters historically store it as a *percent* (35.0). ``MolNet`` converts at inference
according to this key. Feeding the wrong scale is a silent 100× error in collision energy.
That is why ``ce_scale`` is one of the validated checkpoint keys, and why a pre-v1.4.0
checkpoint (trained on the percent scale) must be loaded with a config that says
``ce_scale: percent``.

Consistency is enforced at startup
----------------------------------

A few sizes are necessarily duplicated between the shared encoding config and the task model
configs: ``resolution``, ``max_mz``, ``max_atom_num``, and the widths behind ``in_dim`` and
``add_num``. Molecules are featurised and spectra binned with the *encoding* values, while
the models are sized with the *model* values, so a drift between the two is a
silent-mismatch bug (spectra binned on one grid, output head sized for another).

``MolNet`` therefore cross-checks them when it is constructed and refuses to start on any
disagreement, naming both values and the consequence. If you edit one side deliberately,
update the other to match.
