"""SDF → MGF conversion.

mgf format::

    [{
        'params': {
            'title': prefix_<index>,
            'precursor_type': <precursor_type (e.g. [M+NH4]+ and [M+H]+)>,
            'precursor_mz': <precursor m/z>,
            'molmass': <isotopic mass>,
            'ms_level': <ms_level>,
            'ionmode': <POSITIVE|NEGATIVE>,
            'source_instrument': <source_instrument>,
            'instrument_type': <instrument_type>,
            'collision_energy': <collision energy>,
            'smiles': <smiles>,
            'inchi_key': <inchi_key>,
        },
        'm/z array': mz_array,
        'intensity array': intensity_array
    }, ...]
"""

import logging

import numpy as np
from tqdm import tqdm

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

logger = logging.getLogger(__name__)

_REQUIRED_PROPS = (
    "MASS SPECTRAL PEAKS",
    "PRECURSOR TYPE",
    "PRECURSOR M/Z",
    "SPECTRUM TYPE",
    "COLLISION ENERGY",
    "ION MODE",
    "EXACT MASS",
    "INSTRUMENT TYPE",
)


def sdf2mgf(path, prefix):
    """Read an SDF file and convert each annotated molecule into an MGF spectrum dict.

    Molecules missing any required spectral property are skipped.

    :param path: path to the SDF file.
    :param prefix: title prefix; each spectrum is titled ``<prefix>_<index>``.
    :return: list of pyteomics-style spectrum dicts.
    """
    supp = Chem.SDMolSupplier(path)
    logger.info("Read %d records from %s", len(supp), path)

    spectra = []
    for idx, mol in enumerate(tqdm(supp)):
        if mol is None or not all(mol.HasProp(p) for p in _REQUIRED_PROPS):
            continue

        mz_array = []
        intensity_array = []
        for line in mol.GetProp("MASS SPECTRAL PEAKS").split("\n"):
            mz, intensity = line.split()[:2]
            mz_array.append(float(mz))
            intensity_array.append(float(intensity))

        spectrum = {
            "params": {
                "title": f"{prefix}_{idx}",
                "precursor_type": mol.GetProp("PRECURSOR TYPE"),
                "precursor_mz": mol.GetProp("PRECURSOR M/Z"),
                "molmass": mol.GetProp("EXACT MASS"),
                "ms_level": mol.GetProp("SPECTRUM TYPE"),
                "ionmode": mol.GetProp("ION MODE"),
                "source_instrument": (
                    mol.GetProp("INSTRUMENT") if mol.HasProp("INSTRUMENT") else "Unknown"
                ),
                "instrument_type": mol.GetProp("INSTRUMENT TYPE"),
                "collision_energy": mol.GetProp("COLLISION ENERGY"),
                "smiles": Chem.MolToSmiles(mol, isomericSmiles=True),
                "inchi_key": (
                    mol.GetProp("INCHIKEY") if mol.HasProp("INCHIKEY") else "Unknown"
                ),
            },
            "m/z array": np.array(mz_array),
            "intensity array": np.array(intensity_array),
        }
        spectra.append(spectrum)
    return spectra
