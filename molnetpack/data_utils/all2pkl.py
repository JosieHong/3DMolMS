"""Converters from raw inputs (MGF spectra, CSV/DataFrame rows, SDF molecules) to the
pickle record format consumed by the datasets in :mod:`molnetpack.dataset`.

pkl format for training::

    [
        {'title': <str>, 'smiles': <str>,
         'mol': <numpy array>, 'env': <numpy array>, 'spec': <numpy array>},
        ...
    ]

pkl format for prediction is identical but without ``'spec'``. In both, ``'env'`` is
``np.array([normalized collision energy] + one-hot encoding of the precursor type)``,
and every record carries the covalent bond graph as ``'neighbor_idx'`` / ``'neighbor_mask'``.
"""

import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

from .encoding import (
    bond_graph_array,
    conformation_array,
    precursor_calculator,
    parse_collision_energy,
    generate_ms,
)
from .._compat import deprecated_alias

RDLogger.DisableLog("rdApp.*")

logger = logging.getLogger(__name__)


def _padded_mol_array(xyz_arr, atom_type, encoder):
    """Concatenate coordinates+attributes with the atom-type one-hot and pad to max_atom_num."""
    atom_type_one_hot = np.array([encoder["atom_type"][atom] for atom in atom_type])
    mol_arr = np.concatenate([xyz_arr, atom_type_one_hot], axis=1)
    return np.pad(
        mol_arr,
        ((0, encoder["max_atom_num"] - xyz_arr.shape[0]), (0, 0)),
        constant_values=0,
    )


_ABSOLUTE_UNITS = {"ev", "v", "ce", "absolute"}
_NORMALIZED_UNITS = {"nce", "%", "nce (%)", "nce%", "normalized"}


def _resolve_ce_string(row):
    """Collision-energy cell plus the optional ``Collision_Energy_Unit`` column, as the
    free-text form ``parse_collision_energy`` understands.

    Without a unit column the cell is passed through as-is (``"20 V"``, ``"NCE=35%"``, ...).
    With one, the cell holds a plain number and the unit says how to read it. Returns None
    for an unrecognised unit, so the caller skips the row instead of guessing.
    """
    cell = str(row["Collision_Energy"]).strip()
    unit = ""
    if "Collision_Energy_Unit" in row and not pd.isna(row["Collision_Energy_Unit"]):
        unit = str(row["Collision_Energy_Unit"]).strip().lower()
    if not unit:
        return cell
    try:
        value = float(cell)
    except ValueError:
        # the cell already carries its own unit ("20 V"); the unit column is redundant here
        return cell
    if unit in _ABSOLUTE_UNITS:
        return f"{value} eV"
    if unit in _NORMALIZED_UNITS:
        return f"NCE={value}%"
    return None


def _require_generated_conformation(encoder):
    if encoder["conf_type"] == "origin":
        raise ValueError(
            "conf_type 'origin' is not supported here; conformations must be generated"
        )


# used in training and evaluation ------------------------------------------------------------
def mgf2pkl(spectra, encoder):
    """Convert filtered MGF spectra (see :func:`filter_spec`) into training records."""
    _require_generated_conformation(encoder)
    data = []
    for spectrum in tqdm(spectra):
        # mol array. Conformation generation has known limitations
        # (e.g. https://github.com/rdkit/rdkit/issues/5145); skip unsolvable molecules.
        good_conf, xyz_arr, atom_type = conformation_array(
            x=spectrum["params"]["smiles"], conf_type=encoder["conf_type"]
        )
        if not good_conf:
            continue
        mol_arr = _padded_mol_array(xyz_arr, atom_type, encoder)

        # spec array. After binning, some spectra do not have enough peaks.
        good_spec, spec_arr = generate_ms(
            x=spectrum["m/z array"],
            y=spectrum["intensity array"],
            precursor_mz=float(spectrum["params"]["precursor_mz"]),
            resolution=encoder["resolution"],
            max_mz=encoder["max_mz"],
            charge=int(encoder["type2charge"][spectrum["params"]["precursor_type"]]),
        )
        if not good_spec:
            continue

        # env array
        ce, nce = parse_collision_energy(
            ce_str=spectrum["params"]["collision_energy"],
            precursor_mz=float(spectrum["params"]["precursor_mz"]),
            charge=int(encoder["type2charge"][spectrum["params"]["precursor_type"]]),
        )
        if ce is None and nce is None:
            continue
        precursor_type_one_hot = encoder["precursor_type"][
            spectrum["params"]["precursor_type"]
        ]
        env_arr = np.array([nce] + precursor_type_one_hot)

        # Covalent bond graph. The encoder aggregates over bonded neighbours and refuses to
        # run without the graph, so every record must carry it.
        nidx, nmask = bond_graph_array(
            spectrum["params"]["smiles"], max_atom_num=encoder["max_atom_num"]
        )
        if nidx is None:
            continue

        data.append(
            {
                "title": spectrum["params"]["title"],
                "smiles": spectrum["params"]["smiles"],
                "mol": mol_arr,
                "neighbor_idx": nidx,
                "neighbor_mask": nmask,
                "spec": spec_arr,
                "env": env_arr,
            }
        )
    return data


# used in prediction ------------------------------------------------------------------------------
def molecules_to_records(csv_path, encoder):
    """Convert CSV rows (or a DataFrame) into prediction records, filtering unusable molecules.

    This function is only used in prediction, so the records carry no spectra.

    ``csv_path`` may be a path to a CSV file or an already-loaded pandas DataFrame.

    The ``Collision_Energy`` column accepts free-text values in either unit (``"20 V"`` or
    ``"NCE=35%"``). Alternatively, give plain numbers and add an optional
    ``Collision_Energy_Unit`` column (``eV`` or ``NCE``) that says how to read them, per row.

    (Formerly ``csv2pkl_wfilter``, which remains available as an alias.)
    """
    _require_generated_conformation(encoder)
    df = csv_path if isinstance(csv_path, pd.DataFrame) else pd.read_csv(csv_path)
    data = []
    for _, row in df.iterrows():
        # filter 1: conformation generation has known limitations
        # (e.g. https://github.com/rdkit/rdkit/issues/5145); skip unsolvable molecules.
        good_conf, xyz_arr, atom_type = conformation_array(
            x=row["SMILES"], conf_type=encoder["conf_type"]
        )
        if not good_conf:
            logger.debug("Cannot generate conformation: %s (%s)", row["SMILES"], row["ID"])
            continue
        # filter 2: molecule size
        if xyz_arr.shape[0] > encoder["max_atom_num"]:
            logger.debug(
                "Atom count %d exceeds the limit %d (%s)",
                xyz_arr.shape[0], encoder["max_atom_num"], row["ID"],
            )
            continue
        # filter 3: unsupported atom types
        unsupported = set(atom_type) - set(encoder["atom_type"])
        if unsupported:
            logger.debug("Unsupported atom type(s) %s (%s)", sorted(unsupported), row["ID"])
            continue

        mol_arr = _padded_mol_array(xyz_arr, atom_type, encoder)

        # env array
        if "Collision_Energy" in row.keys():
            # filter 4: unsupported precursor type
            if row["Precursor_Type"] not in encoder["precursor_type"]:
                logger.debug("Unsupported precursor type: %s", row["Precursor_Type"])
                continue
            precursor_mz = precursor_calculator(
                row["Precursor_Type"],
                mass=Descriptors.MolWt(Chem.MolFromSmiles(row["SMILES"])),
            )
            ce_str = _resolve_ce_string(row)
            if ce_str is None:
                logger.debug("Unknown Collision_Energy_Unit: %r (%s)",
                             row["Collision_Energy_Unit"], row["ID"])
                continue
            ce, nce = parse_collision_energy(
                ce_str=ce_str,
                precursor_mz=precursor_mz,
                charge=int(encoder["type2charge"][row["Precursor_Type"]]),
            )
            if ce is None and nce is None:
                continue
        if "Precursor_Type" in row.keys():
            precursor_type_one_hot = encoder["precursor_type"][row["Precursor_Type"]]

        if (
            "Collision_Energy" in row.keys() and "Precursor_Type" in row.keys()
        ):  # (msms prediction)
            env_arr = np.array([nce] + precursor_type_one_hot)
        elif "Precursor_Type" in row.keys():  # only precursor types (ccs prediction)
            env_arr = np.array([0.0] + precursor_type_one_hot)
        elif (
            "Collision_Energy" in row.keys()
        ):  # only collision energies (didn't use this so far)
            env_arr = np.array([nce] + [0.0] * len(encoder["precursor_type"]))
        else:  # no env (rt prediction: input zeros)
            env_arr = np.zeros(1 + len(encoder["precursor_type"]))

        # covalent bond graph -- required by the released encoder (see mgf2pkl)
        nidx, nmask = bond_graph_array(row["SMILES"], max_atom_num=encoder["max_atom_num"])
        if nidx is None:
            continue

        data.append(
            {
                "title": row["ID"],
                "smiles": row["SMILES"],
                "mol": mol_arr,
                "neighbor_idx": nidx,
                "neighbor_mask": nmask,
                "env": env_arr,
            }
        )
    return data


# Deprecated name: it accepts a DataFrame as well as a CSV and returns records, not a pkl.
csv2pkl_wfilter = deprecated_alias(molecules_to_records, "csv2pkl_wfilter")


# used in generating reference library -------------------------------------------------------
def sdf2pkl_with_cond(suppl, encoder, collision_energies, precursor_types):
    """Expand SDF molecules into one prediction record per (collision energy, adduct) pair."""
    data = []
    bad_conformation = 0
    for mol in tqdm(suppl):
        # mol array
        if encoder["conf_type"] == "origin":
            x = mol  # input molecule
        else:
            x = Chem.MolToSmiles(mol, isomericSmiles=True)  # input smiles
        # Conformation generation has known limitations
        # (e.g. https://github.com/rdkit/rdkit/issues/5145); skip unsolvable molecules.
        good_conf, xyz_arr, atom_type = conformation_array(
            x=x, conf_type=encoder["conf_type"]
        )
        if not good_conf:
            bad_conformation += 1
            continue
        mol_arr = _padded_mol_array(xyz_arr, atom_type, encoder)

        # covalent bond graph -- required by the released encoder (see mgf2pkl). Computed once per
        # molecule, outside the collision-energy / adduct loop below.
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        nidx, nmask = bond_graph_array(smiles, max_atom_num=encoder["max_atom_num"])
        if nidx is None:
            bad_conformation += 1
            continue

        # env array
        for ce_str in collision_energies:
            for add in precursor_types:
                precursor_mz = precursor_calculator(add, Descriptors.ExactMolWt(mol))
                ce, nce = parse_collision_energy(
                    ce_str=ce_str,
                    precursor_mz=precursor_mz,
                    charge=int(encoder["type2charge"][add]),
                )
                if ce is None and nce is None:
                    continue
                precursor_type_one_hot = encoder["precursor_type"][add]
                env_arr = np.array([nce] + precursor_type_one_hot)

                data.append(
                    {
                        "title": "{}_{}_{}".format(
                            mol.GetProp("DATABASE_ID"), ce_str, add
                        ),
                        "smiles": smiles,
                        "mol": mol_arr,
                        "neighbor_idx": nidx,
                        "neighbor_mask": nmask,
                        "env": env_arr,
                    }
                )

    logger.info("Skipped %d molecules with bad conformations", bad_conformation)
    return data
