"""Training, inference and evaluation step functions shared by :class:`MolNet` and the scripts.

(Formerly ``molnetpack.utils``, which remains importable as an alias.)
"""

import logging
from decimal import Decimal

import numpy as np
from tqdm import tqdm

import torch

import torch.nn as nn
import torch.nn.functional as F

from ._compat import deprecated_alias
from .data_utils.encoding import mz_to_bin

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared low-level helpers
# ---------------------------------------------------------------------------

def get_lr(optimizer):
    """Return the learning rate of the optimizer's first parameter group."""
    return optimizer.param_groups[0]["lr"]


def make_idx_base(batch_size, num_points, device):
    """Per-molecule index offset used to gather neighbours from a flattened batch."""
    return torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points


# ---------------------------------------------------------------------------
# Inference steps  (used by MolNet.pred_*)
# ---------------------------------------------------------------------------

def _to_device(x, mask, nidx, nmask, device):
    """Move the point cloud and its bond graph to the device.

    The bond graph must travel with the molecule everywhere: MolConv refuses to run
    without it — as of v1.4.0 the encoder has no internal neighbour selection to fall
    back on.
    """
    return (
        x.to(device=device, dtype=torch.float).permute(0, 2, 1),
        mask.to(device=device),
        nidx.to(device=device, dtype=torch.long),
        nmask.to(device=device),
    )


def pred_step(model, device, loader, batch_size, num_points, env):
    """`env` is the experimental condition for the WHOLE dataset, [N, add_num], in this task's
    layout. It is supplied by the caller rather than read from the loader, because the inference
    dataset is shared by three models that do not agree on what env means -- see Mol_Dataset."""
    model.eval()
    id_list, pred_list = [], []
    row = 0

    with tqdm(total=len(loader), desc="Predict") as bar:
        for batch in loader:
            ids, x, mask, nidx, nmask, prec_idx = batch
            x, mask, nidx, nmask = _to_device(x, mask, nidx, nmask, device)
            env_b = env[row:row + x.size(0)].to(device=device, dtype=torch.float)
            row += x.size(0)
            prec_idx = prec_idx.to(device=device, dtype=torch.long)
            idx_base = make_idx_base(x.size(0), num_points, device)

            with torch.no_grad():
                pred = model(x, mask, env_b, idx_base, prec_idx=prec_idx,
                             neighbor_idx=nidx, neighbor_mask=nmask)
                # Normalize each spectrum by its own max so batched inference
                # (batch_size > 1) matches single-molecule results. The previous
                # global torch.max(pred) forced batch_size == 1.
                pred = pred / pred.amax(dim=1, keepdim=True).clamp(min=1e-12)
                pred = torch.pow(pred, 2)
                # Suppress sub-threshold noise peaks.
                pred = torch.where(pred > 0.01, pred, torch.zeros_like(pred))

            id_list += list(ids)
            pred_list.append(pred.cpu())
            bar.update(1)

    return id_list, torch.cat(pred_list, dim=0)


def pred_step_scalar(model, device, loader, batch_size, num_points, env):
    """`env` is [N, add_num] in this task's own layout -- see `pred_step` and `Mol_Dataset`."""
    if batch_size != 1:
        raise ValueError("batch_size should be 1 for prediction")
    model.eval()
    id_list, pred_list = [], []

    row = 0
    with tqdm(total=len(loader), desc="Predict") as bar:
        for batch in loader:
            # the shared inference dataset carries a precursor bin index the scalar models ignore
            ids, x, mask, nidx, nmask, _prec_idx = batch
            x, mask, nidx, nmask = _to_device(x, mask, nidx, nmask, device)
            env_b = env[row:row + x.size(0)].to(device=device, dtype=torch.float)
            row += x.size(0)
            idx_base = make_idx_base(x.size(0), num_points, device)

            with torch.no_grad():
                pred = model.predict(x, mask, env_b, idx_base,
                                     neighbor_idx=nidx, neighbor_mask=nmask)

            id_list += list(ids)
            pred_list.append(pred)
            bar.update(1)

    return id_list, torch.cat(pred_list, dim=0)


# Deprecated name: this function PREDICTS scalar targets (its bar even says "Predict");
# `eval_step` is the actual validation loop.
eval_step_oth = deprecated_alias(pred_step_scalar, "eval_step_oth")


def pred_feat(model, device, loader, batch_size, num_points):
    """Extract encoder embeddings for every molecule in the loader."""
    if batch_size != 1:
        raise ValueError("batch_size should be 1 for prediction")
    model.eval()
    id_list, pred_list = [], []

    with tqdm(total=len(loader), desc="Features") as bar:
        for batch in loader:
            ids, x, mask, nidx, nmask = batch[0], batch[1], batch[2], batch[3], batch[4]
            x, mask, nidx, nmask = _to_device(x, mask, nidx, nmask, device)
            idx_base = make_idx_base(x.size(0), num_points, device)

            with torch.no_grad():
                pred = model(x, idx_base, mask, neighbor_idx=nidx, neighbor_mask=nmask)

            id_list += list(ids)
            pred_list.append(pred)
            bar.update(1)

    return id_list, torch.cat(pred_list, dim=0)


# ---------------------------------------------------------------------------
# Training steps  (used by MolNet.train)
# ---------------------------------------------------------------------------

def _unpack_train_batch(batch, task, device):
    """Unpack one training batch into (x, mask, nidx, nmask, y, env, prec_idx).

    Layouts, set by the Dataset classes:
        msms    (title, mol, mask, nidx, nmask, prec_idx, spec, env)
        rt/ccs  (title, mol, mask, nidx, nmask, env, y)
    """
    if task == "msms":
        _, x, mask, nidx, nmask, prec_idx, y, env = batch
        prec_idx = prec_idx.to(device=device, dtype=torch.long)
    else:
        _, x, mask, nidx, nmask, env, y = batch
        prec_idx = None
    x, mask, nidx, nmask = _to_device(x, mask, nidx, nmask, device)
    y   = y.to(device=device, dtype=torch.float)
    env = env.to(device=device, dtype=torch.float)
    return x, mask, nidx, nmask, y, env, prec_idx


def _msms_batch_cosine(pred, y):
    """Batch-mean cosine between thresholded squared prediction and squared target.

    The prediction is max-normalised, zeroed below the 0.01 noise threshold and squared —
    approximating the inference post-processing in `pred_step` (which thresholds after
    squaring) so train/valid metrics are comparable with predicted spectra.
    """
    pred_max = pred.max(dim=1, keepdim=True).values.clamp(min=1e-8)
    pred_m = pred / pred_max
    pred_m = torch.where(pred_m > 0.01, pred_m, torch.zeros_like(pred_m))
    return F.cosine_similarity(
        torch.pow(pred_m, 2), torch.pow(y, 2), dim=1
    ).mean().item()


def train_step(model, device, loader, optimizer, batch_size, num_points, task):
    """One full training epoch. Returns the average metric for the epoch."""
    model.train()
    cos = nn.CosineSimilarity(dim=1)
    metric_sum = 0.0
    num_batches = 0

    with tqdm(total=len(loader), desc="Train") as bar:
        for batch in loader:
            x, mask, nidx, nmask, y, env, prec_idx = _unpack_train_batch(batch, task, device)
            idx_base = make_idx_base(x.size(0), num_points, device)

            optimizer.zero_grad()
            pred = model(x, mask, env, idx_base,
                         neighbor_idx=nidx, neighbor_mask=nmask,
                         **({"prec_idx": prec_idx} if task == "msms" else {}))

            if task == "msms":
                loss = torch.mean(1 - cos(pred, y))
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    metric_sum += _msms_batch_cosine(pred, y)
            else:
                y_scaled = model.scale(y) if model.scaler is not None else y
                loss = nn.MSELoss()(pred, y_scaled)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    pred_u = model.unscale(pred) if model.scaler is not None else pred
                    metric_sum += torch.abs(pred_u - y).mean().item()

            num_batches += 1
            bar.set_postfix(lr=f"{get_lr(optimizer):.2e}", loss=f"{loss.item():.4f}")
            bar.update(1)

    if num_batches == 0:
        raise ValueError("training loader is empty")
    return metric_sum / num_batches


def eval_step(model, device, loader, batch_size, num_points, task):
    """One full validation epoch. Returns the average metric for the epoch."""
    model.eval()
    metric_sum = 0.0
    num_batches = 0

    with tqdm(total=len(loader), desc="Eval") as bar:
        for batch in loader:
            x, mask, nidx, nmask, y, env, prec_idx = _unpack_train_batch(batch, task, device)
            idx_base = make_idx_base(x.size(0), num_points, device)

            with torch.no_grad():
                pred = model(x, mask, env, idx_base,
                         neighbor_idx=nidx, neighbor_mask=nmask,
                             **({"prec_idx": prec_idx} if task == "msms" else {}))

                if task == "msms":
                    metric_sum += _msms_batch_cosine(pred, y)
                else:
                    pred_u = model.unscale(pred) if model.scaler is not None else pred
                    metric_sum += torch.abs(pred_u - y).mean().item()

            num_batches += 1
            bar.update(1)

    if num_batches == 0:
        raise ValueError("validation loader is empty")
    return metric_sum / num_batches


def collect_targets(loader):
    """Collect all training targets; used to fit the output scaler for RT/CCS.

    The target is the LAST element of the scalar-task batch layout
    (title, mol, mask, nidx, nmask, env, y).
    """
    all_targets = [batch[-1].cpu().numpy() for batch in loader]
    return np.concatenate(all_targets, axis=0).reshape(-1, 1)


# ---------------------------------------------------------------------------
# Evaluation helpers  (used by MolNet.evaluate)
# ---------------------------------------------------------------------------

def bin_spectrum(mz_array, intensity_array, resolution=0.2, max_mz=1500):
    """Bin a sparse (mz, intensity) spectrum into a fixed-length vector.

    Uses the same round-to-nearest grid as ``generate_ms`` (via ``mz_to_bin``), so spectra
    binned here are directly comparable with ground-truth vectors from preprocessing.
    """
    n_bins = int(Decimal(str(max_mz)) // Decimal(str(resolution)))
    binned = [0.0] * n_bins

    for mz, intensity in zip(mz_array, intensity_array):
        idx = mz_to_bin(mz, resolution)
        if 0 <= idx < n_bins:
            binned[idx] += intensity

    return binned if sum(binned) > 0 else None


def cosine_similarity(vec1, vec2):
    """Cosine similarity between two vectors, or None when either has zero norm."""
    a, b  = np.array(vec1), np.array(vec2)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else None
