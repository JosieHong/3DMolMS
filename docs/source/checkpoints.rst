How checkpoints work
====================

Each released 3DMolMS checkpoint carries, besides its weights, the metadata needed to load
it *correctly*, and the loader refuses combinations it cannot verify. This page explains the
download cache, what a checkpoint contains, why loading can fail on purpose, and how transfer
learning treats checkpoints.

Download and cache
------------------

The released weights are downloaded automatically on first use (from the URLs in each task
config's ``release:`` section) into a per-user cache directory following OS conventions:
``~/.cache/molnetpack`` on Linux, ``~/Library/Caches/molnetpack`` on macOS. To use a shared
location (e.g. on a server), pass it explicitly:

.. code-block:: python

   molnet_engine = MolNet(device, seed=42, checkpoint_dir="/shared/molnetpack")

An explicit ``checkpoint_dir`` is authoritative. Without one, a checkpoint already present in
the legacy in-package location (editable/dev checkouts) is reused to avoid a multi-GB
re-download. A custom checkpoint can always be passed directly via ``path_to_checkpoint=`` on
the ``pred_*`` methods.

What a checkpoint contains
--------------------------

Besides ``model_state_dict``, a v1.4.0 release checkpoint embeds:

* ``config`` — a copy of the ``model:`` block it was trained under (the "birth certificate");
* ``task``, ``version``, ``encoder_mode``, and the source-checkpoint name;
* the validation metric (``val_cos`` or ``val_mae``);
* for RT/CCS: ``mu`` / ``sd``, the train-set statistics the targets were standardised with.

Checkpoints saved by ``MolNet.train`` embed the ``config`` block too, so your own trained
models are self-describing in the same way.

Validation on load: why loading can refuse
-------------------------------------------

Several config keys change what the weights *mean* without changing any tensor shape:
``resolution`` and ``max_mz`` reinterpret every output bin, ``ce_scale`` rescales the
collision energy 100×, ``k`` changes which neighbour slots exist. A mismatch on any of them
loads cleanly and predicts wrong numbers with no error. That is the failure class the
validation exists to kill.

On every load, the checkpoint's embedded ``config`` is compared against the config you are
loading with. Three outcomes:

* **No embedded config** (pre-v1.4.0 checkpoint): a warning. Such checkpoints cannot be
  verified at all.
* **All critical keys present on both sides and equal**: the checkpoint loads.
* **Any critical key differing, or recorded on only one side**: loading is refused with an
  error naming the keys and both values. "Cannot verify" is treated as a refusal, not a
  shrug: a key recorded on only one side is exactly how a silent mismatch slips through.

If a refusal is wrong for your case, load with the config from the era the checkpoint was
built in, or, after verifying how the model was actually trained, re-embed the current
config with ``scripts/make_release_checkpoints.py --refresh_meta``.

Transfer learning
-----------------

``MolNet.train(..., resume_path=..., transfer=True)`` loads **only the** ``encoder.*``
**weights** from the source checkpoint. Head weights are never transferred (the head's first
layer consumes the experimental-condition columns, whose layout is task-specific), and any
auxiliary heads on a pretraining source are ignored the same way. Validation is scoped
accordingly: only the encoder-defining keys (``in_dim``, ``emb_dim``, ``k``, ``chirality``,
``max_atom_num``) must match, since a pretraining source and a fine-tune target are *expected*
to differ on task keys.

The transferred encoder is frozen by default (head-only training); pass
``freeze_encoder=False`` for a full fine-tune. The flag is only meaningful together with
``transfer=True`` and is refused otherwise.

Publishing a release (maintainers)
----------------------------------

``scripts/make_release_checkpoints.py`` converts raw training checkpoints into release
artifacts: it renames experiment-side attribute keys to the shipped layout and embeds the
current ``model:`` config plus the metadata above. When a validated config key is added
*after* conversion, ``--refresh_meta`` re-embeds the current config into the existing release
checkpoints without touching the weights. Do this only after verifying the new key's value is
true of how the model was actually trained, then rebuild the release zips.
