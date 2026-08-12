"""PyTorch ``Dataset`` classes for the 3DMolMS tasks.

Training datasets: :class:`MolMSDataset` (MS/MS), :class:`MolScalarDataset` (scalar targets,
with :class:`MolRTDataset` / :class:`MolCCSDataset` as task-specific subclasses),
:class:`MolInferenceDataset` (formerly ``Mol_Dataset``) serves inference for all tasks.
"""

import logging
import os
import pickle

import numpy as np
from torch.utils.data import Dataset

from .spectrum_heads import precursor_bin

logger = logging.getLogger(__name__)

_DEFAULT_DATA_CONFIG = os.path.join(
    os.path.dirname(__file__), "config", "encoding_etkdgv3.yml"
)


def _load_records(x, mode):
    """Resolve the (path | in-memory data) input convention shared by several datasets.

    Returns ``(records, source_label)`` where ``source_label`` names the origin for messages.
    """
    if mode == "path":
        with open(x, "rb") as f:
            return pickle.load(f), str(x)
    if mode == "data":
        return x, "in-memory data"
    raise ValueError(f"Unsupported mode: {mode!r} (expected 'path' or 'data')")


def _require_bond_graph(data, path):
    """The encoder aggregates over the covalent bond graph. Fail loudly if it is absent.

    MolConv itself refuses to run without a neighbour graph; this check exists to fail
    EARLIER, at load time, with a message that names the fix instead of surfacing
    mid-epoch from inside a forward pass.
    """
    if data and "neighbor_idx" not in data[0]:
        raise KeyError(
            f"{path} has no 'neighbor_idx' -- it was built by a pre-v1.4.0 preprocessing run. "
            f"The models aggregate over the covalent bond graph and cannot run without it. "
            f"Re-run the dataset build: python scripts/build_msms_dataset.py ..."
        )


def _add_masks(data):
    """Mark real atoms: a row of all zeros in the padded ``mol`` array is padding."""
    for d in data:
        d["mask"] = (~np.all(d["mol"] == 0, axis=1)).astype(bool)
    return data


def _filter_precursor_type(data, encoded_precursor_type):
    """Keep records whose adduct one-hot (``env[1:]``) matches the encoded filter string."""
    return [
        d for d in data
        if ",".join(str(int(i)) for i in d["env"][1:]) == encoded_precursor_type
    ]


def _add_precursor_bins(data, resolution, max_mz, data_config_path):
    # Precursor bin index: the reverse head indexes bins downward from it and the output mask
    # zeroes everything above it, so it is a required model input, not a convenience.
    n_bins = int(max_mz / resolution)
    for d in data:
        if "prec_idx" not in d:
            d["prec_idx"] = precursor_bin(
                d["smiles"], d["env"], resolution, n_bins, data_config_path
            )
    return data


class MolMSDataset(Dataset):
    """MS/MS spectra.

    Note on augmentation: earlier versions doubled the dataset by mirroring the x coordinate. The
    encoder is E(3)-invariant in every shipped setting and reflection is an element of E(3), so
    the mirrored copy produced a bit-identical embedding -- measured max|f(x) - f(flip)| = 0.0.
    It doubled epoch time to train on exact duplicates, and has been removed. Mirroring is only
    informative with chirality=True (the SE(3) mode), which no released model uses.
    """

    def __init__(self, x, precursor_type=False, mode="path",
                 data_config_path=_DEFAULT_DATA_CONFIG, resolution=0.2, max_mz=1500):
        data, source = _load_records(x, mode)

        if precursor_type:
            data = _filter_precursor_type(data, precursor_type)

        _require_bond_graph(data, source)
        self.data = _add_precursor_bins(
            _add_masks(data), resolution, max_mz, data_config_path
        )
        logger.info("Loaded %d records from %s", len(self.data), source)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        return (
            d["title"],
            d["mol"],
            d["mask"],
            d["neighbor_idx"],
            d["neighbor_mask"],
            np.int64(d["prec_idx"]),
            d["spec"],
            d["env"],
        )


class MolInferenceDataset(Dataset):
    """Molecules for INFERENCE (no reference spectrum), shared by all three tasks.

    It deliberately does NOT return `env`. The same loaded file is used to predict MS/MS, CCS and
    RT, but those three models do not share an experimental-condition layout: MS/MS expects
    [collision energy] + a 5-way adduct one-hot, CCS a 6-way adduct set of its own, RT a single
    placeholder. Emitting one of those layouts here would silently mis-encode the other two -- an
    adduct shifted by one index still has the right width, so nothing would raise. The caller
    supplies `env` for the task it is running; see `MolNet._task_env`.
    """

    def __init__(self, data, precursor_type=False,
                 data_config_path=_DEFAULT_DATA_CONFIG, resolution=0.2, max_mz=1500):
        if precursor_type:
            data = _filter_precursor_type(data, precursor_type)

        _require_bond_graph(data, "in-memory data")
        self.data = _add_precursor_bins(
            _add_masks(data), resolution, max_mz, data_config_path
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        return (
            d["title"],
            d["mol"],
            d["mask"],
            d["neighbor_idx"],
            d["neighbor_mask"],
            np.int64(d["prec_idx"]),
        )


class MolScalarDataset(Dataset):
    """Scalar-target training records (retention time, CCS, ...).

    :param path: Path to the PKL file.
    :param target_key: Record key holding the regression target (e.g. ``'rt'``, ``'ccs'``).
    """

    def __init__(self, path, target_key):
        self.target_key = target_key
        self.data, source = _load_records(path, "path")
        logger.info("Loaded %d records from %s", len(self.data), source)
        _require_bond_graph(self.data, source)
        _add_masks(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        return (
            d["title"],
            d["mol"],
            d["mask"],
            d["neighbor_idx"],
            d["neighbor_mask"],
            np.asarray(d["env"], dtype=np.float32),
            d[self.target_key],
        )


class MolRTDataset(MolScalarDataset):
    """Retention time. `env` is a single placeholder column -- SMRT is one chromatographic
    method, so there are no experimental covariates to encode; retention is predicted from
    structure alone."""

    def __init__(self, path):
        super().__init__(path, target_key="rt")


class MolCCSDataset(MolScalarDataset):
    """Collision cross-section.

    `env` is an adduct one-hot over the CCS adduct set, which is NOT the MS/MS set: AllCCS
    measurements are dominated by adducts the MS/MS data barely has. The layout is recorded in
    the checkpoint so it cannot drift away from the data.
    """

    def __init__(self, path):
        super().__init__(path, target_key="ccs")


# Deprecated aliases: inference input is shared by all tasks, not MS/MS-specific;
# the underscored names are the pre-v1.4.0 (non-PEP 8) spellings.
Mol_Dataset = MolInferenceDataset
MolMS_Dataset = MolMSDataset
MolRT_Dataset = MolRTDataset
MolCCS_Dataset = MolCCSDataset
