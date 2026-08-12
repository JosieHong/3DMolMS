"""Task registry and training loop shared by :meth:`molnetpack.MolNet.train`.

The :data:`TASKS` registry holds everything task-specific in one place; the
:func:`fit` loop is task-agnostic and reads whatever it needs from a
:class:`TaskSpec` entry.
"""

import logging
from dataclasses import dataclass

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from .dataset import MolMSDataset, MolRTDataset, MolCCSDataset
from .steps import train_step, eval_step
from ._version import __version__

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskSpec:
    """Everything task-specific in one place; adding a task means adding one entry
    to ``TASKS`` instead of editing several parallel dicts."""

    metric_label: str        # validation metric name used in log lines
    higher_is_better: bool   # direction of "improved" (also sets the scheduler mode)
    scheduler_patience: int  # ReduceLROnPlateau patience (epochs)
    early_stop_patience: int  # default early-stop patience (config may override)
    best_key: str            # checkpoint key the best validation metric is stored under
    model_attr: str          # MolNet attribute holding this task's model
    config_attr: str         # MolNet attribute holding this task's config

    @property
    def scheduler_mode(self):
        return "max" if self.higher_is_better else "min"

    @property
    def best_init(self):
        return 0.0 if self.higher_is_better else float("inf")


TASKS = {
    "msms": TaskSpec("cosine", True, 5, 10, "best_val_acc", "msms_model", "msms_config"),
    "rt":   TaskSpec("MAE", False, 20, 60, "best_val_mae", "rt_model", "rt_config"),
    "ccs":  TaskSpec("MAE", False, 20, 60, "best_val_mae", "ccs_model", "ccs_config"),
}


def build_loaders(task, train_path, valid_path, config, data_config, data_config_path,
                  precursor_type="All"):
    """Construct the train/valid DataLoaders for ``task``.

    :param data_config: The preprocessing/encoding config (supplies the adduct
        one-hot layout used to encode the msms precursor-type filter).
    :param data_config_path: Path to that config; the MS/MS dataset needs it to
        compute precursor bin indices.
    """
    batch_size = config["train"]["batch_size"]
    num_workers = config["train"]["num_workers"]

    if task == "msms":
        # Build the encoded precursor-type filter string
        encoder = {
            k: ",".join(str(int(i)) for i in v)
            for k, v in data_config["encoding"]["precursor_type"].items()
        }
        encoder["All"] = False
        encoded = encoder[precursor_type]
        ms_kwargs = dict(
            precursor_type=encoded,
            data_config_path=data_config_path,
            resolution=float(config["model"]["resolution"]),
            max_mz=float(config["model"]["max_mz"]),
        )
        train_set = MolMSDataset(train_path, **ms_kwargs)
        valid_set = MolMSDataset(valid_path, **ms_kwargs)
    elif task == "rt":
        train_set = MolRTDataset(train_path)
        valid_set = MolRTDataset(valid_path)
    else:  # ccs
        train_set = MolCCSDataset(train_path)
        valid_set = MolCCSDataset(valid_path)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, drop_last=False)
    return train_loader, valid_loader


def build_optimizer(model, spec, train_config):
    """AdamW + ReduceLROnPlateau configured from the task spec and train config."""
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_config["lr"],
        weight_decay=train_config["weight_decay"],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=spec.scheduler_mode,
        factor=0.5,
        patience=spec.scheduler_patience,
    )
    return optimizer, scheduler


def fit(model, task, spec, config, train_loader, valid_loader, optimizer, scheduler,
        device, checkpoint_path=""):
    """Run the epoch loop with early stopping; save the best checkpoint if requested.

    Returns the best validation metric achieved.
    """
    batch_size = config["train"]["batch_size"]
    num_points = config["model"]["max_atom_num"]
    num_params = sum(p.numel() for p in model.parameters())

    best_metric    = spec.best_init
    early_patience = 0
    early_limit    = config["train"].get("early_stop_patience", spec.early_stop_patience)
    label          = spec.metric_label

    for epoch in range(1, config["train"]["epochs"] + 1):
        logger.info("===== Epoch %d", epoch)
        train_metric = train_step(model, device, train_loader, optimizer,
                                  batch_size, num_points, task)
        valid_metric = eval_step(model, device, valid_loader,
                                 batch_size, num_points, task)
        logger.info("Train %s: %.4f  |  Valid %s: %.4f",
                    label, train_metric, label, valid_metric)

        improved = (valid_metric > best_metric if spec.higher_is_better
                    else valid_metric < best_metric)
        if improved:
            best_metric    = valid_metric
            early_patience = 0
            logger.info("Early stop patience reset")
            if checkpoint_path:
                logger.info("Saving checkpoint...")
                # The RT scaler must travel with the weights it standardised.
                extra = {"scaler": model.scaler} if task == "rt" else {}
                torch.save(
                    {
                        "version":            __version__,
                        "epoch":              epoch,
                        # Embed the model config so the checkpoint is self-describing and
                        # passes consistency validation on resume / reload.
                        "config":             config["model"],
                        "model_state_dict":   model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "num_params":         num_params,
                        spec.best_key:        best_metric,
                        **extra,
                    },
                    checkpoint_path,
                )
        else:
            early_patience += 1
            logger.info("Early stop count: %d/%d", early_patience, early_limit)

        scheduler.step(valid_metric)
        logger.info("Best %s so far: %.4f", label, best_metric)

        if early_patience >= early_limit:
            logger.info("Early stop!")
            break

    return best_metric
