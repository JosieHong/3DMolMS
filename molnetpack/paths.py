"""Locations of resources bundled with the installed package."""

import warnings
from pathlib import Path

_CONFIG_DIR = Path(__file__).parent / "config"

# v1.4.0 renames: the "preprocess_*" files are the shared molecule/spectrum ENCODING used by
# every task, and were renamed to say so. Old names keep resolving, with a warning.
# (preprocess_hmdb.yml was not renamed but REMOVED: it duplicated the shared encoding
# byte-for-byte except conf_type, which is now a --conf_type flag on hmdb2pkl.py.)
_RENAMED_CONFIGS = {
    "preprocess_etkdgv3.yml": "encoding_etkdgv3.yml",
}


def config_path(filename):
    """Absolute path of a config YAML bundled with molnetpack.

    Scripts should resolve bundled configs through this helper (e.g.
    ``config_path("molnet.yml")``) rather than relative to their own location,
    so they work from any clone and against an installed package alike.

    :param filename: Config file name, e.g. ``"encoding_etkdgv3.yml"``.
    :return: Absolute path as a string.
    :raises FileNotFoundError: if no such bundled config exists (the message
        lists the available names).
    """
    if filename in _RENAMED_CONFIGS:
        renamed = _RENAMED_CONFIGS[filename]
        warnings.warn(
            f"config {filename!r} was renamed to {renamed!r} in v1.4.0; update the caller",
            DeprecationWarning,
            stacklevel=2,
        )
        filename = renamed
    path = _CONFIG_DIR / filename
    if not path.is_file():
        available = ", ".join(sorted(p.name for p in _CONFIG_DIR.glob("*.yml")))
        raise FileNotFoundError(
            f"No bundled config named {filename!r}. Available: {available}"
        )
    return str(path)
