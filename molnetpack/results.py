"""Result assembly, saving, evaluation and plotting for MS/MS / RT / CCS predictions."""

import logging
import os
import pickle

import numpy as np
import pandas as pd
from pyteomics import mgf
from rdkit import Chem
from rdkit.Chem import Descriptors

from .data_utils import nce2ce, precursor_calculator
from .steps import bin_spectrum, cosine_similarity

# matplotlib and PIL are imported lazily inside the plotting helpers, so headless
# prediction-only use never pays their import cost.

logger = logging.getLogger(__name__)


def precursor_decoder(data_config):
    """Invert ``encoding.precursor_type``: one-hot (as a comma string) -> adduct name."""
    return {
        ",".join(map(str, v)): k
        for k, v in data_config["encoding"]["precursor_type"].items()
    }


def assemble_msms_results(records, id_list, pred_dicts, data_config):
    """Build the ``pred_msms`` result DataFrame from loaded records and predictions."""
    decoding = precursor_decoder(data_config)
    ce_list, add_list, smiles_list = [], [], []
    for d in records:
        adduct = decoding[",".join(map(str, map(int, d["env"][1:])))]
        smiles = d["smiles"]
        mass   = Descriptors.ExactMolWt(Chem.MolFromSmiles(smiles))
        charge = int(data_config["encoding"]["type2charge"][adduct])
        ce_list.append(nce2ce(d["env"][0], precursor_calculator(adduct, mass), charge))
        add_list.append(adduct)
        smiles_list.append(smiles)

    return pd.DataFrame({
        "ID":               id_list,
        "SMILES":           smiles_list,
        "Collision Energy": ce_list,
        "Precursor Type":   add_list,
        "Pred M/Z":         [p["m/z"]       for p in pred_dicts],
        "Pred Intensity":   [p["intensity"] for p in pred_dicts],
    })


def spectra_from_dataframe(df, version, instrument=None):
    """Convert a ``pred_msms`` result DataFrame into pyteomics-style spectrum dicts.

    When the DataFrame carries ``Pred RT`` / ``Pred CCS`` columns (see
    :meth:`molnetpack.MolNet.pred_all`), they are emitted as the ``RTINSECONDS`` and
    ``CCS`` ion parameters — the fields RT/CCS-aware library tools read.
    """
    spectra = []
    for idx, row in df.iterrows():
        params = {
            "title":            row["ID"],
            "mslevel":          "2",
            "organism":         f"3DMolMS_{version}",
            "spectrumid":       f"pred_{idx}",
            "smiles":           row["SMILES"],
            "collision_energy": row["Collision Energy"],
            "precursor_type":   row["Precursor Type"],
            "instrument_type":  instrument,
        }
        if "Pred RT" in row and pd.notna(row["Pred RT"]):
            params["rtinseconds"] = round(float(row["Pred RT"]), 2)
        if "Pred CCS" in row and pd.notna(row["Pred CCS"]):
            params["ccs"] = round(float(row["Pred CCS"]), 1)
        spectra.append({
            "params": params,
            "m/z array":       np.array([float(v) for v in row["Pred M/Z"].split(",") if v]),
            "intensity array": np.array(
                [float(v) * 1000 for v in row["Pred Intensity"].split(",") if v]),
        })
    return spectra


def save_msms_results(res_df, path, version, instrument):
    """Save a ``pred_msms`` result DataFrame as ``.mgf`` or ``.csv``."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if path.endswith(".mgf"):
        mgf.write(spectra_from_dataframe(res_df, version, instrument),
                  path, file_mode="w", write_charges=False)
    elif path.endswith(".csv"):
        res_df.to_csv(path, index=False)
    else:
        raise ValueError("result path must end with .mgf or .csv")
    logger.info("Saved results to %s", path)


def save_csv(df, path):
    """Save a result DataFrame as CSV, creating parent directories as needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved results to %s", path)


def evaluate_predictions(test_pkl, pred_mgf):
    """Compare predicted MS/MS spectra against ground-truth spectra.

    :param test_pkl: Path to the ground-truth PKL file (from preprocessing).
    :param pred_mgf: Path to the predicted spectra MGF file (from ``pred_msms``).
    :return: DataFrame with per-spectrum cosine similarity and metadata.
    """
    with open(test_pkl, "rb") as f:
        gt_spectra = pickle.load(f)
    pred_spectra = list(mgf.read(pred_mgf))

    gt_by_title   = {s["title"]: s["spec"]  for s in gt_spectra}
    pred_by_title = {s["params"]["title"]: s for s in pred_spectra}

    rows = []
    for title, gt_vec in gt_by_title.items():
        if title not in pred_by_title:
            continue
        pred = pred_by_title[title]
        binned = bin_spectrum(pred["m/z array"], pred["intensity array"])
        if binned is None or len(gt_vec) != len(binned):
            continue
        # gt_vec is sqrt-normalised (stored by generate_ms); binned is
        # intensity-space (pred_step squares model output).  Square gt_vec
        # so both vectors are in intensity space, consistent with the
        # cosine(pred², y²) metric used during training.
        sim = cosine_similarity(np.array(gt_vec) ** 2, binned)
        if sim is None:
            continue
        rows.append({
            "title":              title,
            "smiles":             pred["params"].get("smiles", ""),
            "collision_energy":   pred["params"].get("collision_energy", ""),
            "precursor_type":     pred["params"].get("precursor_type", "Unknown"),
            "cosine_similarity":  sim,
        })

    df = pd.DataFrame(rows)
    logger.info("Evaluated %d matched spectra", len(df))
    logger.info("Overall mean cosine similarity: %.4f", df["cosine_similarity"].mean())
    logger.info(
        "Mean by precursor type:\n%s",
        df.groupby("precursor_type")["cosine_similarity"].mean().to_string(),
    )
    return df


def plot_similarity_hist(similarities, path):
    """Save a histogram of cosine similarities as PNG."""
    from matplotlib import pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.hist(similarities, bins=50, edgecolor="black")
    plt.title("Cosine Similarity Distribution")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    logger.info("Saved histogram to %s", path)


def plot_msms(msms_res_df, dir_to_img):
    """Plot MS/MS spectra with inset 2-D molecular structures.

    :param msms_res_df: DataFrame returned by :meth:`MolNet.pred_msms`.
    :type msms_res_df: pandas.DataFrame
    :param dir_to_img: Directory where PNG files will be saved (one per spectrum).
    :type dir_to_img: str
    """
    from PIL import Image
    from matplotlib import pyplot as plt
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    from rdkit.Chem import Draw

    os.makedirs(dir_to_img, exist_ok=True)
    img_dpi, y_max, bin_width = 300, 1, 0.4

    for _, row in msms_res_df.iterrows():
        mz_values  = np.array([float(v) for v in row["Pred M/Z"].split(",")])
        intensities = np.array([float(v) * y_max for v in row["Pred Intensity"].split(",")])

        fig, ax = plt.subplots(figsize=(9, 4))
        plt.bar(mz_values, intensities, width=bin_width, color="k")
        plt.xlim(0, np.max(mz_values))
        plt.title("ID: " + row["ID"])
        plt.xlabel("M/Z")
        plt.ylabel("Relative intensity")

        # 2-D depiction; RDKit computes 2-D coordinates automatically, so there
        # is no need to embed/optimize a 3-D conformer just to draw it.
        mol = Chem.MolFromSmiles(row["SMILES"])
        if mol is not None:
            mol_img = Draw.MolToImage(mol, size=(800, 800))
            alpha   = Image.fromarray(255 - np.array(mol_img.convert("L")))
            mol_img.putalpha(alpha)
            imagebox = OffsetImage(mol_img, zoom=72.0 / img_dpi)
            ax.add_artist(AnnotationBbox(
                imagebox, (np.max(mz_values) * 0.28, y_max * 0.64),
                frameon=False, xycoords="data",
            ))

        out_path = os.path.join(dir_to_img, f"{row['ID']}.png")
        plt.savefig(out_path, dpi=img_dpi, bbox_inches="tight")
        plt.close()

    logger.info("Saved plots to %s", dir_to_img)
