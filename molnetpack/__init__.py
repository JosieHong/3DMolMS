"""3DMolMS: predicting MS/MS spectra, retention time and collision cross-section
from 3D molecular conformations.

The public API is re-exported here: the high-level :class:`MolNet` entry point, the model and
dataset classes, and the preprocessing helpers from :mod:`molnetpack.data_utils`.

Logging: the package logs through the standard :mod:`logging` module under the ``"molnetpack"``
logger. A plain-format INFO-level stream handler is attached by default so progress messages
stay visible in command-line use; reconfigure or silence it with
``logging.getLogger("molnetpack")``.

Deprecated names (kept as aliases): ``MolNet_Oth`` → :class:`MolNetScalar`,
``Mol_Dataset`` → :class:`MolInferenceDataset`, ``csv2pkl_wfilter`` →
:func:`molecules_to_records`, ``eval_step_oth`` → :func:`molnetpack.steps.pred_step_scalar`;
the modules ``molnetpack.utils`` and ``molnetpack.data_utils.utils`` forward to
``molnetpack.steps`` and ``molnetpack.data_utils.encoding``.
"""

import logging as _logging

from .paths import config_path
from .molnet import MolNet, plot_msms

from .model import MolNetMS, MolNetScalar, MolNet_MS, MolNet_Oth
from .dataset import (
    MolMSDataset,
    MolScalarDataset,
    MolRTDataset,
    MolCCSDataset,
    MolInferenceDataset,
    MolMS_Dataset,
    MolRT_Dataset,
    MolCCS_Dataset,
    Mol_Dataset,
)

from .data_utils import filter_mol, sdf2pkl_with_cond, conformation_array
from .data_utils import sdf2mgf, filter_spec, mgf2pkl, check_atom
from .data_utils import molecules_to_records, csv2pkl_wfilter, nce2ce, precursor_calculator
from .data_utils import ms_vec2dict

from .steps import get_lr

from ._version import __version__

# Default log handler: plain messages at INFO, so command-line progress output stays
# visible (see the module docstring). Attached once, only if the user has not already
# configured the "molnetpack" logger.
_logger = _logging.getLogger("molnetpack")
if not _logger.handlers:
    _handler = _logging.StreamHandler()
    _handler.setFormatter(_logging.Formatter("%(message)s"))
    _logger.addHandler(_handler)
    _logger.setLevel(_logging.INFO)
    _logger.propagate = False

__all__ = [
    "config_path",
    "MolNet",
    "plot_msms",
    "MolNetMS",
    "MolNet_MS",
    "MolNetScalar",
    "MolNet_Oth",
    "MolMSDataset",
    "MolScalarDataset",
    "MolRTDataset",
    "MolCCSDataset",
    "MolMS_Dataset",
    "MolRT_Dataset",
    "MolCCS_Dataset",
    "MolInferenceDataset",
    "Mol_Dataset",
    "filter_mol",
    "sdf2pkl_with_cond",
    "conformation_array",
    "sdf2mgf",
    "filter_spec",
    "mgf2pkl",
    "check_atom",
    "molecules_to_records",
    "csv2pkl_wfilter",
    "nce2ce",
    "precursor_calculator",
    "ms_vec2dict",
    "get_lr",
    "__version__",
]
