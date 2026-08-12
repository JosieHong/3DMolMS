"""Molecule / spectrum encoding helpers: conformer generation, spectrum binning,
collision-energy parsing, adduct handling and the covalent bond graph.
"""

import logging
import re
from decimal import Decimal, ROUND_HALF_UP

import numpy as np

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdDepictor

RDLogger.DisableLog("rdApp.*")

logger = logging.getLogger(__name__)

# NCE <-> eV conversion factor per precursor charge, shared by ce2nce / nce2ce /
# parse_collision_energy: NCE = eV * 500 * factor(charge) / precursor_mz.
_NCE_CHARGE_FACTOR = {1: 1, 2: 0.9, 3: 0.85, 4: 0.8, 5: 0.75, 6: 0.75, 7: 0.75, 8: 0.75}

# Per-atom feature columns produced by `conformation_array` BEFORE the atom-type one-hot:
# 3 coordinates + 7 attributes (degree, valence, mass/100, charge, #implicit H, aromatic,
# in-ring). The model's `in_dim` must equal this plus the atom-type one-hot width.
ATOM_FEATURE_DIMS = 10


def mz_to_bin(mz, resolution):
    """Map an m/z value to its index on the round-to-nearest bin grid.

    This is THE grid definition: ``generate_ms`` (ground-truth binning) and
    ``molnetpack.steps.bin_spectrum`` (evaluation binning) both use it, so the two
    can never disagree. Rounding to the nearest grid point (not flooring) is what
    makes a 0.2 Da resolution actually usable -- see the comment in ``generate_ms``.
    """
    return int(
        (Decimal(str(mz)) / Decimal(str(resolution))).to_integral_value(rounding=ROUND_HALF_UP)
    )


def ms_vec2dict(spec, resolution=1):
    """Convert a binned spectrum vector into ``{'m/z': ..., 'intensity': ...}`` CSV strings,
    keeping only non-zero bins."""
    x, y = [], []
    for i, intensity in enumerate(spec):
        if intensity != 0:
            x.append(str(i * resolution))
            y.append(str(intensity))
    return {"m/z": ",".join(x), "intensity": ",".join(y)}


def generate_ms(x, y, precursor_mz, resolution=1, max_mz=1500, charge=1):
    """Bin a peak list into a fixed-length spectrum vector.

    :param x: m/z values of the peaks.
    :param y: intensities of the peaks.
    :param precursor_mz: precursor m/z (peaks at or above it are removed).
    :param resolution: bin width in Da.
    :param max_mz: maximum m/z covered by the vector.
    :param charge: precursor charge (used to locate the isotope peaks).
    :return: ``(ok, ms)`` where ``ms`` is the normalised, sqrt-transformed spectrum
        vector and ``ok`` is False when the binned spectrum is degenerate.
    """
    # generate isotropic peaks (refers to Kaiyuan's codes:
    # https://github.com/lkytal/PredFull/blob/master/train_model.py)
    # Isotope peaks bin to the NEAREST grid point via mz_to_bin, the same mapping used
    # for the fragment peaks below, so exclusion and binning can never disagree.
    isotropic_peaks = [
        mz_to_bin(precursor_mz + delta / charge, resolution) for delta in (0, 1, 2)
    ]

    # make precursor_mz the right bound
    right_bound = int(Decimal(str(precursor_mz)) // Decimal(str(resolution)))

    # init mass spectra vector: add "0" to y data
    ms = [0] * int(Decimal(str(max_mz)) // Decimal(str(resolution)))

    # convert x, y to vector
    for idx, val in enumerate(x):
        # Bin each peak to the NEAREST 0.2-grid point. Using floor (//) here would
        # merge the whole [X.0, X.2) interval into the X.0 bin; since most CHNO
        # fragment mass defects are < 0.2 Da, that collapses ~90% of peaks onto
        # integer m/z and wastes the 0.2 resolution. Rounding uses the full grid.
        val = mz_to_bin(val, resolution)
        if val >= right_bound:  # remove precursor peak
            continue
        if val in isotropic_peaks:
            continue
        ms[val] += y[idx]

    # normalize to 0-1
    if np.max(ms) - np.min(ms) == 0:
        logger.warning(
            "Degenerate spectrum: all binned intensities are identical "
            "(right bound %d, %d input peaks)", right_bound, len(x)
        )
        return False, np.array(ms)
    ms = (ms - np.min(ms)) / (np.max(ms) - np.min(ms))

    # smooth out large values
    ms = np.sqrt(np.array(ms))
    return True, ms


def ce2nce(ce, precursor_mz, charge):
    """Convert absolute collision energy (eV) to normalised collision energy (percent)."""
    return ce * 500 * _NCE_CHARGE_FACTOR[charge] / precursor_mz


def nce2ce(nce, precursor_mz, charge):
    """Convert normalised collision energy (percent) to absolute collision energy (eV)."""
    return nce * precursor_mz / (500 * _NCE_CHARGE_FACTOR[charge])


def parse_collision_energy(ce_str, precursor_mz, charge=1):
    """Parse a free-text collision-energy string into ``(ce_eV, nce_percent)``.

    Recognises the eV and NCE spellings found in NIST, MassBank, GNPS and CASMI; whichever
    of the two values is not stated is derived via the NCE formula. Returns ``(None, None)``
    when the string cannot be parsed.
    """
    ce = None
    nce = None

    # match collision energy (eV)
    matches_ev = {
        # NIST20
        r"^[\d]+[.]?[\d]*$": lambda x: float(x),
        r"^[\d]+[.]?[\d]*[ ]?eV$": lambda x: float(x.rstrip(" eV")),
        r"^[\d]+[.]?[\d]*[ ]?ev$": lambda x: float(x.rstrip(" ev")),
        r"^[\d]+[.]?[\d]*[ ]?v$": lambda x: float(x.rstrip(" v")),
        r"^[\d]+[.]?[\d]*[ ]?V$": lambda x: float(x.rstrip(" V")),
        r"^NCE=[\d]+[.]?[\d]*% [\d]+[.]?[\d]*eV$": lambda x: float(
            x.split()[1].rstrip("eV")
        ),
        r"^nce=[\d]+[.]?[\d]*% [\d]+[.]?[\d]*ev$": lambda x: float(
            x.split()[1].rstrip("ev")
        ),
        # MassBank
        r"^hcd[\d]+[.]?[\d]*$": lambda x: float(x.lstrip("hcd")),
        r"^[\d]+HCD$": lambda x: float(x.rstrip("HCD")),  # 35HCD
    }
    for pattern, extract in matches_ev.items():
        if re.match(pattern, ce_str):
            ce = extract(ce_str)
            break

    # match collision energy (NCE)
    matches_nce = {
        # MassBank
        r"^[\d]+[.]?[\d]*[ ]?[%]? \(nominal\)$": lambda x: float(
            x.rstrip("% (nominal)")
        ),
        r"^[\d]+[.]?[\d]*[ ]?nce$": lambda x: float(x.rstrip(" nce")),
        r"^[\d]+[.]?[\d]*[ ]?\(nce\)$": lambda x: float(x.rstrip(" (nce)")),
        # Accepts decimals and lowercase, e.g. 'NCE=27.5%' and 'nce=30%'.
        r"^[Nn][Cc][Ee]=[\d]+[.]?[\d]*[ ]?\%$": lambda x: float(
            re.search(r"([\d]+[.]?[\d]*)", x).group(1)
        ),
        r"^[\d]+[.]?[\d]*\([Nn][Cc][Ee]\)$": lambda x: float(
            x.split("(")[0]
        ),  # 90(NCE)
        r"^HCD \(NCE [\d]+[.]?[\d]*%\)$": lambda x: float(
            x.split(" ")[-1].rstrip("%)")
        ),  # HCD (NCE 40%)
        # CASMI
        r"^[\d]+[.]?[\d]*[ ]?\(nominal\)$": lambda x: float(
            x.rstrip("(nominal)").rstrip(" ")
        ),
    }
    for pattern, extract in matches_nce.items():
        if re.match(pattern, ce_str):
            # NCE is kept on the PERCENT scale (35.0 for 35%), matching the scale the eV branch
            # below produces via ce*500*charge_factor/precursor_mz.
            nce = extract(ce_str)
            break

    # Strings that state BOTH, e.g. NIST's "NCE=35% 25eV": use the two STATED values directly
    # instead of deriving one from the other. The instrument setting is ground truth; the
    # formula is only an approximation of it.
    m_both = re.match(
        r"^\s*[Nn][Cc][Ee]=([\d]+[.]?[\d]*)%\s+([\d]+[.]?[\d]*)\s*[eE]?[vV]\s*$", ce_str
    )
    if m_both:
        nce = float(m_both.group(1))
        ce = float(m_both.group(2))

    # unknown collision energy
    if ce_str == "Unknown":
        ce = 40

    if ce is not None and nce is not None:
        return ce, nce
    if ce is not None:
        nce = ce2nce(ce, precursor_mz, charge)
    elif nce is not None:
        ce = nce2ce(nce, precursor_mz, charge)
    else:
        return None, None
    return ce, nce


def conformation_array(x, conf_type):
    """Embed a molecule and return its per-atom feature array.

    :param x: a SMILES string, or an RDKit Mol when ``conf_type`` is ``'origin'``.
    :param conf_type: conformation source: ``'etkdg'``, ``'etkdgv3'``, ``'2d'``,
        ``'mmff'`` (ETKDGv3 + MMFF94 optimisation) or ``'origin'`` (use the input
        molecule's existing conformer).
    :return: ``(ok, xyz_arr, atom_type)`` where ``xyz_arr`` is an ``[n_atoms, 10]``
        array of centered coordinates plus atom attributes and ``atom_type`` the
        list of element symbols; ``ok`` is False when embedding failed.
    """
    if conf_type == "etkdg":
        mol_from_smiles = Chem.AddHs(Chem.MolFromSmiles(x))
        AllChem.EmbedMolecule(mol_from_smiles)

    elif conf_type == "etkdgv3":
        mol_from_smiles = Chem.AddHs(Chem.MolFromSmiles(x))
        ps = AllChem.ETKDGv3()
        ps.randomSeed = 0xF00D
        ps.maxIterations = 1000  # prevent rare infinite loops on difficult ring systems
        AllChem.EmbedMolecule(mol_from_smiles, ps)

    elif conf_type == "2d":
        mol_from_smiles = Chem.AddHs(Chem.MolFromSmiles(x))
        rdDepictor.Compute2DCoords(mol_from_smiles)

    elif conf_type == "origin":
        mol_from_smiles = Chem.AddHs(x)

    elif conf_type == "mmff":
        # ETKDGv3 for initial geometry, then MMFF94 force-field optimisation.
        mol_from_smiles = Chem.AddHs(Chem.MolFromSmiles(x))
        ps = AllChem.ETKDGv3()
        ps.randomSeed = 0xF00D
        ps.maxIterations = 1000
        if AllChem.EmbedMolecule(mol_from_smiles, ps) == -1:
            return False, None, None
        AllChem.MMFFOptimizeMolecule(mol_from_smiles)

    elif conf_type == "omega":
        raise ValueError("OMEGA conformation will be supported soon. ")
    else:
        raise ValueError("Unsupported conformation type. {}".format(conf_type))

    # get the x,y,z-coordinates of atoms; embedding can fail and leave no conformer
    try:
        conf = mol_from_smiles.GetConformer()
    except ValueError:
        return False, None, None
    xyz_arr = conf.GetPositions()
    # center the x,y,z-coordinates
    centroid = np.mean(xyz_arr, axis=0)
    xyz_arr -= centroid

    # concatenate with atom attributes
    xyz_arr = xyz_arr.tolist()
    for i, atom in enumerate(mol_from_smiles.GetAtoms()):
        xyz_arr[i] += [atom.GetDegree()]
        xyz_arr[i] += [atom.GetExplicitValence()]
        xyz_arr[i] += [atom.GetMass() / 100]
        xyz_arr[i] += [atom.GetFormalCharge()]
        xyz_arr[i] += [atom.GetNumImplicitHs()]
        xyz_arr[i] += [int(atom.GetIsAromatic())]
        xyz_arr[i] += [int(atom.IsInRing())]
    xyz_arr = np.array(xyz_arr)

    # get the atom types of atoms
    atom_type = [atom.GetSymbol() for atom in mol_from_smiles.GetAtoms()]
    return True, xyz_arr, atom_type


# Adduct strings are written in inconsistent orders across sources, and the SAME source can mix
# them: AllCCS contains both "[M-H2O+H]+" (177 rows) and "[M+H-2H2O]+" (58), while this repo's
# preprocessing config uses "[M+H-H2O]+". These name identical species, so a hardcoded list
# silently drops whichever spelling it does not happen to contain. Normalise before matching.
_ADDUCT_ALIASES = {
    "[M-H2O+H]+": "[M+H-H2O]+",
    "[M-2H2O+H]+": "[M+H-2H2O]+",
    "[M-3H2O+H]+": "[M+H-3H2O]+",
    "[M-H2O-H]-": "[M-H-H2O]-",
    "[M-NH3+H]+": "[M+H-NH3]+",
}


def bond_graph_array(smiles, max_atom_num=300, k=6):
    """Covalent bond graph as (neighbor_idx, neighbor_mask), matching the MS/MS encoder's input.

    CONTRACT: unused neighbour slots (atoms with degree < k) are SELF-POINTING, i.e.
    neighbor_idx[i, s] = i with neighbor_mask[i, s] = False. The mask zeroes their contribution
    after aggregation, but the Gram / distance / LayerNorm statistics are computed over ALL k slots,
    so a padded slot pointing at some OTHER atom would perturb the real slots' normalisation.

    Returns (None, None) if the molecule is unparseable or exceeds max_atom_num.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    mol = Chem.AddHs(mol)
    n = mol.GetNumAtoms()
    if n > max_atom_num or n < 2:
        return None, None
    idx = np.tile(np.arange(max_atom_num, dtype=np.int64)[:, None], (1, k))  # self-padded
    msk = np.zeros((max_atom_num, k), dtype=bool)
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        for slot, nb in enumerate([x.GetIdx() for x in atom.GetNeighbors()][:k]):
            idx[i, slot] = nb
            msk[i, slot] = True
    return idx, msk


def skeleton_key(smiles):
    """Non-stereochemical InChIKey first block — the split unit used everywhere in this repo.

    Splitting on the SMILES STRING instead lets stereoisomers and tautomer-variant spellings of the
    same compound land in different partitions, which leaks structure between train and test.
    """
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol).split("-")[0]
    except Exception:
        return None


def normalize_adduct(adduct):
    """Canonicalise an adduct string so equivalent spellings compare equal.

    Only reorders KNOWN aliases; anything unrecognised is returned unchanged (stripped), so an
    unexpected form fails a membership test loudly rather than being silently rewritten.
    """
    a = (adduct or "").strip().replace(" ", "")
    return _ADDUCT_ALIASES.get(a, a)


# Precursor m/z as a function of the neutral monoisotopic mass, per adduct.
_PRECURSOR_MZ_CALCULATORS = {
    "[M+H]+": lambda mass: mass + 1.007276,
    "[M+Na]+": lambda mass: mass + 22.989218,
    "[2M+H]+": lambda mass: 2 * mass + 1.007276,
    "[M-H]-": lambda mass: mass - 1.007276,
    "[M+H-H2O]+": lambda mass: mass - 17.0038370665,
    "[M+2H]2+": lambda mass: mass / 2 + 1.007276,
}


def precursor_calculator(precursor_type, mass):
    """Precursor m/z for a given adduct type and neutral monoisotopic mass.

    :raises ValueError: when the adduct type is not supported.
    """
    try:
        return _PRECURSOR_MZ_CALCULATORS[precursor_type](mass)
    except KeyError:
        raise ValueError(
            "Unsupported precursor type: {} (supported: {})".format(
                precursor_type, ", ".join(sorted(_PRECURSOR_MZ_CALCULATORS))
            )
        ) from None
