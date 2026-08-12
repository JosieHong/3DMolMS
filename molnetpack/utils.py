"""Deprecated module alias — the step functions live in :mod:`molnetpack.steps`.

This shim exists so ``from molnetpack.utils import ...`` keeps working; new code
should import from ``molnetpack.steps`` directly.
"""

from .steps import (  # noqa: F401
    get_lr,
    make_idx_base,
    pred_step,
    pred_step_scalar,
    eval_step_oth,
    pred_feat,
    train_step,
    eval_step,
    collect_targets,
    bin_spectrum,
    cosine_similarity,
)
