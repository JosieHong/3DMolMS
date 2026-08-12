"""Spectrum / molecule filters applied before pickle conversion: instrument, collision
energy, precursor type, atom count/type, peak count and precursor mass error."""

import logging
import re

import numpy as np
from tqdm import tqdm

from rdkit import Chem, RDLogger
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from molmass import Formula

RDLogger.DisableLog("rdApp.*")

logger = logging.getLogger(__name__)


def filter_spec(spectra, config, type2charge):
    """Filter MGF spectra by the criteria in ``config``.

    :param spectra: iterable of pyteomics-style spectrum dicts.
    :param config: filter thresholds (instrument type, ms level, atom limits,
        precursor types, peak count, m/z range, ppm tolerance).
    :param type2charge: mapping from precursor type to charge, written into each
        kept spectrum's params.
    :return: ``(clean_spectra, smiles_list)``.
    """
    clean_spectra = []
    smiles_list = []
    for spectrum in tqdm(spectra):
        smiles = spectrum["params"]["smiles"]
        try:
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        except Exception:
            logger.debug("Invalid SMILES: %s", smiles)
            continue
        if mol is None:
            continue

        # Filter by collision energy: ramp energies cannot be processed
        collision_energy = spectrum["params"]["collision_energy"]
        if collision_energy.startswith(("ramp", "Ramp")) or collision_energy.endswith("ramp"):
            continue

        # Filter by instrument type
        if "instrument_type" in config:
            if spectrum["params"]["instrument_type"] not in config["instrument_type"]:
                continue

        # Filter by instrument (MoNA contains too many instrument names to filter out)
        if "instrument" in config:
            if spectrum["params"]["source_instrument"] not in config["instrument"]:
                continue

        # Filter by mslevel
        if "ms_level" in config:
            if spectrum["params"]["ms_level"] != config["ms_level"]:
                continue

        # Filter by atom number and atom type
        if check_atom(mol, config, in_type="molh") < 0:
            continue

        # Filter by precursor type
        precursor_type = spectrum["params"]["precursor_type"]
        if precursor_type not in config["precursor_type"]:
            continue

        # Filter by peak number
        if len(spectrum["m/z array"]) < config["min_peak_num"]:
            continue

        # Filter by max m/z
        if (
            np.max(spectrum["m/z array"]) < config["min_mz"]
            or np.max(spectrum["m/z array"]) > config["max_mz"]
        ):
            continue

        # Filter by ppm (mass error between stated precursor m/z and the theoretical
        # m/z of the adduct formula)
        try:
            f = Formula(added_formula(CalcMolFormula(mol), precursor_type))
            theo_mz = f.isotope.mass
            ppm = (
                abs(theo_mz - float(spectrum["params"]["precursor_mz"]))
                / theo_mz
                * 10**6
            )
        except Exception:
            # invalid formula, unsupported precursor type, or invalid precursor m/z
            continue
        if ppm > config["ppm_tolerance"]:
            continue

        # add charge
        spectrum["params"]["charge"] = type2charge[precursor_type]
        clean_spectra.append(spectrum)
        smiles_list.append(smiles)
    return clean_spectra, smiles_list


_ELEMENT_COUNT_RE = re.compile(
    r"(He|Li|Be|Ne|Na|Mg|Al|Si|Cl|Ar|Ca|Sc|Ti|Cr|Mn|Fe|Co|Ni|Cu|Zn|Ga|Ge|As|Se|Br"
    r"|Kr|Rb|Sr|Zr|Nb|Mo|Tc|Ru|Rh|Pd|Ag|Cd|In|Sn|Sb|Te|Xe|Cs|Ba|La|Hf|Ta|Re|Os|Ir"
    r"|Pt|Au|Hg|Tl|Pb|Bi|Po|At|Rn|Fr|Ra|Ac|Rf|Db|Sg|Bh|Hs|Mt|Ds|Rg|Cn|Nh|Fl|Mc|Lv"
    r"|Ts|Og|[A-Z])(\d\d|\d|)"
)


def f_str2dict(f):
    """Parse a molecular formula string into an {element: count} dict."""
    return {
        symbol: int(count) if count else 1
        for symbol, count in _ELEMENT_COUNT_RE.findall(f)
    }


def f_dict2str(f_dict):
    """Serialise an {element: count} dict back into a formula string (counts < 1 dropped)."""
    f = ""
    for symbol, count in f_dict.items():
        if count > 1:
            f += symbol + str(count)
        elif count == 1:
            f += symbol
    return f


def added_formula(f, precursor_type):
    """Apply the atom gains/losses of an adduct to a neutral molecular formula."""
    f_dict = f_str2dict(str(f))
    if precursor_type == "[M+H]+":
        f_dict["H"] += 1
    elif precursor_type == "[M+Na]+":
        f_dict["Na"] = f_dict.get("Na", 0) + 1
    elif precursor_type == "[M-H]-":
        f_dict["H"] -= 1
    elif precursor_type in ("[M+H-H2O]+", "[M-H2O+H]+"):
        f_dict["H"] -= 1
        f_dict["O"] -= 1
    elif precursor_type == "[M+2H]2+":
        f_dict["H"] += 2
    elif precursor_type == "[2M+H]+":
        f_dict = {k: int(v * 2) for k, v in f_dict.items()}
        f_dict["H"] += 1
    elif precursor_type == "[2M-H]-":
        f_dict = {k: int(v * 2) for k, v in f_dict.items()}
        f_dict["H"] -= 1
    else:
        raise ValueError("Unsupported precursor type: {}".format(precursor_type))

    return f_dict2str(f_dict)


def filter_mol(suppl, config):
    """Filter SDF molecules by atom count and atom type.

    :return: ``(clean_suppl, smiles_list)``.
    """
    clean_suppl = []
    smiles_list = []
    exceed_atom_num = 0
    unsupported_atom_type = 0
    for mol in tqdm(suppl):
        if mol is None:
            continue
        mol = Chem.AddHs(mol)

        # Filter by atom number and atom type
        flag_atom = check_atom(mol, config, in_type="molh")
        if flag_atom < 0:
            if flag_atom == -1:
                exceed_atom_num += 1
            elif flag_atom == -2:
                unsupported_atom_type += 1
            continue

        clean_suppl.append(mol)
        smiles_list.append(Chem.MolToSmiles(mol))
    logger.debug(
        "Filtered out %d molecules over the atom limit and %d with unsupported atom types",
        exceed_atom_num, unsupported_atom_type,
    )
    return clean_suppl, smiles_list


def check_atom(x, config, in_type="smiles"):
    """Check a molecule against the configured atom-count range and atom-type set.

    :param in_type: ``'smiles'`` (SMILES string), ``'mol'`` (RDKit Mol, hydrogens
        added here) or ``'molh'`` (RDKit Mol that already has explicit hydrogens).
    :return: ``1`` if the molecule passes, ``-1`` if the atom count is out of range,
        ``-2`` if it contains an unsupported atom type.
    """
    if in_type not in ("smiles", "mol", "molh"):
        raise ValueError(f"Unsupported in_type: {in_type!r}")
    if in_type == "smiles":
        mol = Chem.AddHs(Chem.MolFromSmiles(x))
    elif in_type == "mol":
        mol = Chem.AddHs(x)
    else:
        mol = x

    num_atoms = len(mol.GetAtoms())
    if num_atoms > config["max_atom_num"] or num_atoms < config["min_atom_num"]:
        return -1

    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in config["atom_type"]:
            return -2
    return 1
