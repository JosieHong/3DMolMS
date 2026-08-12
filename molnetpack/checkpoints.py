"""Checkpoint management: cache-directory resolution, download, architecture
validation and weight loading. Task-agnostic — :class:`molnetpack.MolNet` decides
*which* checkpoint a task needs; this module handles getting and loading it."""

import logging
import os
import zipfile
from pathlib import Path

import platformdirs
import requests
import torch

logger = logging.getLogger(__name__)

# Model-config keys that change what the weights MEAN, not merely how they are trained. A
# mismatch on any of these produces wrong numbers rather than an error, because none of them
# changes a tensor shape: `k` changes which neighbour slots exist, and
# resolution/max_mz reinterpret every output bin.
CRITICAL_CONFIG_KEYS = (
    "in_dim", "add_num", "max_atom_num", "emb_dim", "k",
    "chirality", "resolution", "max_mz", "ce_scale",
)

# The subset of critical keys that define what the ENCODER weights mean. Transfer learning
# loads only the encoder, so only these must match between the pretraining source and the
# fine-tune target; task-side keys (add_num, resolution, max_mz, ce_scale) legitimately
# differ across tasks and are deliberately not compared on transfer.
ENCODER_CRITICAL_CONFIG_KEYS = (
    "in_dim", "emb_dim", "k", "chirality", "max_atom_num",
)


def _model_block(saved):
    """Accept both config shapes: a flat model block, or a full config with a 'model' section
    (the SSL pretraining checkpoints embed the latter)."""
    if isinstance(saved, dict) and "model" in saved and "in_dim" not in saved:
        return saved["model"]
    return saved


def checkpoint_dir(override=None):
    """Directory where downloaded checkpoints are cached.

    :param override: Explicit cache directory (e.g. a shared cache on a server), as
        passed to ``MolNet(checkpoint_dir=...)``. Default: a per-user cache directory
        following OS conventions (``~/.cache/molnetpack`` on Linux,
        ``~/Library/Caches/molnetpack`` on macOS, ``%LOCALAPPDATA%\\molnetpack\\Cache``
        on Windows).

    Checkpoints are deliberately kept out of the installed package
    directory, which may be read-only and is wiped on upgrade/reinstall.
    """
    return Path(override) if override else Path(platformdirs.user_cache_dir("molnetpack"))


def resolve_checkpoint_path(rel, package_dir, cache_dir=None):
    """Resolve a config-relative checkpoint path to the location it should live at.

    An explicit ``cache_dir`` is authoritative — no fallback to a possibly-stale
    in-package checkpoint. Without one, a checkpoint already present in the legacy
    in-package location (editable/dev checkouts) is reused to avoid a multi-GB
    re-download; otherwise the per-user cache directory is used.
    """
    if cache_dir is None:
        legacy_path = Path(package_dir) / rel
        if legacy_path.exists():
            return str(legacy_path)
    return str(checkpoint_dir(cache_dir) / os.path.basename(rel))


def ensure_checkpoint(checkpoint_path, url, task_name):
    """Download and extract the checkpoint zip if ``checkpoint_path`` does not exist yet.

    :param url: Download URL, or falsy when the config intentionally ships without one
        (non-default training configs) — then the checkpoint must be provided locally.
    :raises RuntimeError: when no URL is configured or the download/extract fails.
    """
    if os.path.exists(checkpoint_path):
        return
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    zip_path = checkpoint_path + ".zip"

    if not url:
        raise RuntimeError(
            f"No download URL is configured for task '{task_name}'. "
            f"Place the checkpoint at '{checkpoint_path}' manually, or pass "
            f"path_to_checkpoint=... ."
        )

    logger.info("Downloading checkpoint from %s", url)
    try:
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(os.path.dirname(checkpoint_path))
        os.remove(zip_path)  # don't keep the archive around in the cache
    except (requests.RequestException, zipfile.BadZipFile) as e:
        # Remove partial/corrupt downloads so the next attempt starts clean.
        if os.path.exists(zip_path):
            os.remove(zip_path)
        raise RuntimeError(
            f"Failed to download or extract checkpoint from {url}: {e}"
        ) from e


def validate_checkpoint_config(ckpt, current_model_config, checkpoint_path):
    """Refuse a checkpoint whose training config contradicts the config we just built from.

    Checkpoints WITHOUT any embedded config only warn: they cannot be verified at all.
    Checkpoints WITH a config are held to the full standard: every critical key must be
    present on both sides and equal. A key that only one side records is an error too —
    skipping it would silently waive exactly the check the key was added for.
    """
    saved = _model_block(ckpt.get("config"))
    if saved is None:
        logger.warning(
            "%s carries no config, so its encoder mode and adduct layout cannot be "
            "verified against yours. If it was trained with different settings the "
            "predictions will be wrong WITHOUT any error.", checkpoint_path
        )
        return

    missing_in_ckpt = [k for k in CRITICAL_CONFIG_KEYS
                       if k in current_model_config and k not in saved]
    missing_in_cfg = [k for k in CRITICAL_CONFIG_KEYS
                      if k in saved and k not in current_model_config]
    if missing_in_ckpt or missing_in_cfg:
        lines = []
        if missing_in_ckpt:
            lines.append(
                f"    recorded by your config but not by the checkpoint: {missing_in_ckpt}"
            )
        if missing_in_cfg:
            lines.append(
                f"    recorded by the checkpoint but not by your config: {missing_in_cfg}"
            )
        detail = "\n".join(lines)
        raise ValueError(
            f"{checkpoint_path} and the loaded config do not record the same critical keys:\n"
            f"{detail}\n"
            f"Consistency cannot be verified, so loading is refused rather than guessed. Either "
            f"load with the config from the era this checkpoint was built in, or — after "
            f"verifying how the model was actually trained — re-embed the current config with "
            f"scripts/make_release_checkpoints.py --refresh_meta."
        )

    mismatched = {
        key: (saved[key], current_model_config[key])
        for key in CRITICAL_CONFIG_KEYS
        if key in saved and key in current_model_config
        and saved[key] != current_model_config[key]
    }
    if mismatched:
        detail = "\n".join(
            f"    {k}: checkpoint={was!r} but config={now!r}"
            for k, (was, now) in mismatched.items()
        )
        raise ValueError(
            f"{checkpoint_path} was trained with a different architecture:\n{detail}\n"
            f"These keys do not change any tensor shape, so the weights would load cleanly and "
            f"predict from the wrong model. Load the matching config, or the matching checkpoint."
        )


def validate_encoder_config(ckpt, current_model_config, checkpoint_path):
    """Transfer-scoped consistency check: only the encoder-defining keys must agree.

    Same three regimes as :func:`validate_checkpoint_config`, restricted to
    ``ENCODER_CRITICAL_CONFIG_KEYS`` — a pretraining source and a fine-tune target are
    EXPECTED to differ on task-side keys (add_num, resolution, max_mz, ce_scale), so those
    must not block a transfer.
    """
    saved = _model_block(ckpt.get("config"))
    if saved is None:
        logger.warning(
            "%s carries no config; its encoder settings cannot be verified against yours. "
            "If it was pretrained with a different encoder the transferred weights will be "
            "wrong WITHOUT any error.", checkpoint_path
        )
        return

    missing = [k for k in ENCODER_CRITICAL_CONFIG_KEYS
               if (k in current_model_config) != (k in saved)]
    if missing:
        raise ValueError(
            f"{checkpoint_path} and the loaded config do not record the same encoder keys: "
            f"{missing}. Consistency cannot be verified, so the transfer is refused rather "
            f"than guessed."
        )
    mismatched = {
        k: (saved[k], current_model_config[k])
        for k in ENCODER_CRITICAL_CONFIG_KEYS
        if k in saved and k in current_model_config and saved[k] != current_model_config[k]
    }
    if mismatched:
        detail = "\n".join(
            f"    {k}: checkpoint={was!r} but config={now!r}"
            for k, (was, now) in mismatched.items()
        )
        raise ValueError(
            f"{checkpoint_path} was pretrained with a different encoder:\n{detail}\n"
            f"Encoder weights transferred across these settings load cleanly but mean the "
            f"wrong thing. Pretrain and fine-tune with the same encoder settings."
        )


def load_weights(model, checkpoint_path, device, optimizer=None, scheduler=None, transfer=False,
                 current_model_config=None, freeze_encoder=None):
    """Load a checkpoint into model (and optionally optimizer/scheduler).

    ``transfer=True`` loads ONLY the ``encoder.*`` weights and freezes them (pretraining →
    fine-tune). Head weights are deliberately NOT transferred: the head's first layer consumes
    ``[emb_dim + add_num]`` inputs, so its weight columns encode the SOURCE task's env layout
    (collision energy + adduct one-hot). Across tasks that layout differs in width and meaning,
    so those weights — and any auxiliary heads such as the SSL distance head — are left at
    fresh initialisation. This also means an SSL pretraining checkpoint and a task checkpoint
    are equally valid transfer sources.

    When ``current_model_config`` is given, consistency with the checkpoint's embedded config
    is validated first — the full critical-key check for plain loads, the encoder-scoped check
    for transfers.

    ``freeze_encoder`` selects between the two fine-tuning regimes and is only meaningful
    with ``transfer=True``: True (the default when unspecified) freezes the transferred
    encoder and trains the head alone; False leaves everything trainable (full fine-tune
    from the pretrained encoder). Passing it without ``transfer=True`` is an error rather
    than a silent no-op.

    Returns the best validation metric stored in the checkpoint, or None.
    """
    if freeze_encoder is not None and not transfer:
        raise ValueError(
            "freeze_encoder only applies to transfer=True; without a transfer there is "
            "nothing this flag would do, so passing it is refused rather than ignored."
        )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if transfer:
        if current_model_config is not None:
            validate_encoder_config(ckpt, current_model_config, checkpoint_path)
        encoder_dict = {
            k: v for k, v in ckpt["model_state_dict"].items()
            if k.startswith("encoder.")
        }
        if not encoder_dict:
            raise ValueError(
                f"{checkpoint_path} contains no 'encoder.*' weights; it is not a usable "
                f"transfer source."
            )
        model.load_state_dict(encoder_dict, strict=False)
        # Freeze the encoder parameters in the model (not the checkpoint tensors).
        if freeze_encoder is None or freeze_encoder:
            for name, param in model.named_parameters():
                if name.startswith("encoder."):
                    param.requires_grad = False
        else:
            logger.info(
                "freeze_encoder=False: encoder loaded from transfer source but NOT frozen "
                "(full fine-tune)"
            )
        return None

    if current_model_config is not None:
        validate_checkpoint_config(ckpt, current_model_config, checkpoint_path)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt.get("best_val_acc") or ckpt.get("best_val_mae")
