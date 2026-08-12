"""Deprecated module alias — the encoding helpers live in :mod:`molnetpack.data_utils.encoding`.

This shim exists so ``from molnetpack.data_utils.utils import ...`` keeps working; new code
should import from ``molnetpack.data_utils.encoding`` directly.
"""

from .encoding import (  # noqa: F401
    mz_to_bin,
    ms_vec2dict,
    generate_ms,
    ce2nce,
    nce2ce,
    parse_collision_energy,
    conformation_array,
    bond_graph_array,
    skeleton_key,
    normalize_adduct,
    precursor_calculator,
)
