"""High-level entry point: the :class:`MolNet` facade for loading data, predicting
MS/MS spectra / RT / CCS, training and evaluation.

The heavy lifting lives in focused modules — :mod:`molnetpack.checkpoints` (download,
validation, weight loading), :mod:`molnetpack.training` (task registry + epoch loop) and
:mod:`molnetpack.results` (result assembly, saving, evaluation, plotting) — this class
orchestrates them behind one stable interface.
"""

import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pyteomics import mgf

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError as e:
    raise ImportError(
        "PyTorch is required by molnetpack but is not installed. "
        "See https://pytorch.org/get-started/ for installation instructions."
    ) from e

from rdkit import RDLogger

from . import checkpoints, results, training
from .model import MolNetMS, MolNetScalar, _build_encoder
from .dataset import MolInferenceDataset
from .data_utils import molecules_to_records, filter_spec, mgf2pkl, ms_vec2dict
from .data_utils.encoding import ATOM_FEATURE_DIMS
from .steps import pred_step, pred_step_scalar, pred_feat, collect_targets
from .results import plot_msms as plot_msms  # re-export: public API, historic location
from ._version import __version__

RDLogger.DisableLog("rdApp.*")

logger = logging.getLogger(__name__)


class MolNet:
    """Facade over the 3DMolMS models: data loading, checkpoint management,
    prediction (MS/MS, RT, CCS), training and evaluation."""

    def __init__(
        self,
        device,
        seed,
        data_config_path=None,
        msms_config_path=None,
        ccs_config_path=None,
        rt_config_path=None,
        checkpoint_dir=None,
    ):
        """
        :param device: PyTorch device.
        :param seed: Random seed.
        :param data_config_path: Path to preprocessing/encoding config YAML.
            Defaults to the bundled ``encoding_etkdgv3.yml``.
        :param msms_config_path: Path to MS/MS model config YAML.
            Defaults to the bundled ``molnet.yml``.
        :param ccs_config_path: Path to CCS model config YAML.
            Defaults to the bundled ``molnet_ccs_tl.yml``.
        :param rt_config_path: Path to RT model config YAML.
            Defaults to the bundled ``molnet_rt_tl.yml``.
        :param checkpoint_dir: Directory where downloaded checkpoints are cached
            (e.g. a shared cache on a server). Defaults to the per-user OS cache
            directory; an explicit value is authoritative and skips the legacy
            in-package fallback.
        """
        self.version = __version__
        logger.info("MolNetPack version: %s", self.version)

        self.device = device
        self.current_path = Path(__file__).parent
        self.checkpoint_dir = checkpoint_dir

        # Configs (shared across inference and training). The resolved data-config PATH is kept
        # because the adduct one-hot layout lives in it, and the datasets need it to turn an
        # adduct into a precursor m/z bin.
        self.data_config_path = str(
            Path(data_config_path) if data_config_path
            else self.current_path / "config" / "encoding_etkdgv3.yml"
        )
        self.data_config  = self._load_config("encoding_etkdgv3.yml", data_config_path)
        self.msms_config  = self._load_config("molnet.yml",              msms_config_path)
        self.ccs_config   = self._load_config("molnet_ccs_tl.yml",       ccs_config_path)
        self.rt_config    = self._load_config("molnet_rt_tl.yml",        rt_config_path)
        self._validate_config_consistency()

        # Inference state
        self.pkl_dict     = None
        self.valid_loader = None

        # Models (set by pred_* or train)
        self.msms_model   = None
        self.ccs_model    = None
        self.rt_model     = None
        self.encoder      = None

        # Cached result DataFrames
        self.qtof_msms_res_df     = None
        self.orbitrap_msms_res_df = None
        self.ccs_res_df           = None
        self.rt_res_df            = None

        # Tracks which MS/MS weights are currently loaded so pred_msms can
        # reload when the instrument or custom checkpoint changes.
        self._loaded_msms_instrument = None
        self._loaded_msms_ckpt       = None

        # DataLoader batch size (set by load_data / load_dataframe).
        self.batch_size   = 1

        self._init_random_seed(seed)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_config(self, default_filename, override_path=None):
        path = (Path(override_path) if override_path
                else self.current_path / "config" / default_filename)
        with open(path) as f:
            return yaml.safe_load(f)

    def _validate_config_consistency(self):
        """Fail loudly when the shared data-encoding config and the task model configs disagree.

        Several sizes are necessarily duplicated across the config files: molecules are
        featurised, padded and binned with the ENCODING values, while the models are sized and
        interpreted with the MODEL values. Nothing else ties the two together — a drift is the
        same silent-mismatch bug class the checkpoint validation guards against, one config
        layer up (spectra binned on one grid, model heads sized for another).
        """
        enc = self.data_config["encoding"]
        atom_width = len(next(iter(enc["atom_type"].values())))
        adduct_width = len(next(iter(enc["precursor_type"].values())))
        expected_in_dim = ATOM_FEATURE_DIMS + atom_width
        problems = []

        for task in ("msms", "rt", "ccs"):
            model = self._task_config(task)["model"]
            if int(model["max_atom_num"]) != int(enc["max_atom_num"]):
                problems.append(
                    f"{task}: model.max_atom_num={model['max_atom_num']} but "
                    f"encoding.max_atom_num={enc['max_atom_num']} — molecules are padded to "
                    f"the encoding value, so the model would see the wrong point count"
                )
            if int(model["in_dim"]) != expected_in_dim:
                problems.append(
                    f"{task}: model.in_dim={model['in_dim']} but the encoding produces "
                    f"{expected_in_dim} features ({ATOM_FEATURE_DIMS} coordinate/attribute "
                    f"columns + {atom_width}-wide atom one-hot)"
                )

        msms = self.msms_config["model"]
        if float(msms["resolution"]) != float(enc["resolution"]):
            problems.append(
                f"msms: model.resolution={msms['resolution']} but "
                f"encoding.resolution={enc['resolution']} — spectra would be binned on a "
                f"different grid than the output head predicts on"
            )
        if float(msms["max_mz"]) != float(enc["max_mz"]):
            problems.append(
                f"msms: model.max_mz={msms['max_mz']} but encoding.max_mz={enc['max_mz']} — "
                f"the spectrum vector length would not match the output head"
            )
        if int(msms["add_num"]) != 1 + adduct_width:
            problems.append(
                f"msms: model.add_num={msms['add_num']} but the encoding defines "
                f"1 collision-energy column + a {adduct_width}-wide adduct one-hot "
                f"= {1 + adduct_width}"
            )

        ccs_layout = self.ccs_config.get("encoding", {}).get("precursor_type")
        if ccs_layout:
            ccs_width = len(next(iter(ccs_layout.values())))
            if int(self.ccs_config["model"]["add_num"]) != ccs_width:
                problems.append(
                    f"ccs: model.add_num={self.ccs_config['model']['add_num']} but its own "
                    f"encoding.precursor_type vectors are {ccs_width} wide"
                )
        if (not self.rt_config.get("encoding", {}).get("precursor_type")
                and int(self.rt_config["model"]["add_num"]) != 1):
            problems.append(
                f"rt: model.add_num={self.rt_config['model']['add_num']} but the RT config "
                f"defines no adduct encoding, so env is a single placeholder column"
            )

        if problems:
            raise ValueError(
                "Configuration files are inconsistent:\n  - " + "\n  - ".join(problems)
                + "\nThese values are duplicated between the shared data-encoding config and "
                "the task model configs and must agree; see the header comments in the config "
                "files for which file owns what."
            )

    @staticmethod
    def _init_random_seed(seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def _task_config(self, task):
        return getattr(self, training.TASKS[task].config_attr)

    def _task_model_attr(self, task):
        return training.TASKS[task].model_attr

    def _task_env(self, task):
        """Re-encode the experimental condition in `task`'s own layout.

        The loaded pickle carries the MS/MS env: [collision energy] + a 5-way adduct one-hot. The
        scalar models do not share that layout -- CCS uses a 6-way adduct set of its own (see
        `encoding.precursor_type` in molnet_ccs_tl.yml) and RT uses a single placeholder column,
        since SMRT is one chromatographic method with no covariates to encode. Feeding the MS/MS
        env to the CCS model would shift every adduct index by one and mis-encode all of them.
        """
        config = self._task_config(task)
        add_num = int(config["model"]["add_num"])
        env = torch.zeros(len(self.pkl_dict), add_num, dtype=torch.float)

        if task == "msms":
            # The pickle is written in the MS/MS layout already, so use it verbatim.
            for i, record in enumerate(self.pkl_dict):
                env[i] = torch.tensor(np.asarray(record["env"], dtype=np.float32))
            # The package converters write env[0] as NCE on the PERCENT scale (35.0 for 35%),
            # but the v1.4.0 release models were trained with NCE as a FRACTION (0.35) -- see
            # scripts/build_msms_dataset.py: ce_to_nce. `ce_scale` in the model config names
            # the convention the checkpoint expects; feeding the wrong one is a silent 100x
            # error in collision energy (measured on caffeine: the spectrum degrades into
            # small high-energy fragments). Pre-v1.4.0 checkpoints expect 'percent' -- load
            # them with a config that says so.
            if config["model"].get("ce_scale", "percent") == "fraction":
                env[:, 0] /= 100.0
            return env

        layout = config.get("encoding", {}).get("precursor_type")
        if layout is None:
            return env  # RT: a single placeholder column, nothing to encode
        # decode each record's adduct from the MS/MS one-hot, then re-encode in this task's layout
        decoding = results.precursor_decoder(self.data_config)
        unsupported = set()
        for i, record in enumerate(self.pkl_dict):
            adduct = decoding[",".join(map(str, map(int, record["env"][1:])))]
            if adduct not in layout:
                unsupported.add(adduct)
                continue
            env[i] = torch.tensor(layout[adduct], dtype=torch.float)
        if unsupported:
            logger.warning(
                "%s are not in the %s adduct set %s; those molecules get an all-zero adduct "
                "encoding, which the model never saw in training. Their predictions are "
                "unreliable.", sorted(unsupported), task.upper(), sorted(layout)
            )
        return env

    def _build_model(self, task, config):
        model_cls = MolNetMS if task == "msms" else MolNetScalar
        return model_cls(config["model"]).to(self.device)

    def _load_weights(self, model, checkpoint_path, optimizer=None, scheduler=None, transfer=False,
                      current_model_config=None, freeze_encoder=None):
        """Load a checkpoint into model (and optionally optimizer/scheduler).

        Returns the best validation metric stored in the checkpoint, or None.
        """
        return checkpoints.load_weights(
            model, checkpoint_path, self.device,
            optimizer=optimizer, scheduler=scheduler, transfer=transfer,
            current_model_config=current_model_config, freeze_encoder=freeze_encoder,
        )

    # ------------------------------------------------------------------
    # Checkpoint resolution (task-specific; generic parts live in `checkpoints`)
    # ------------------------------------------------------------------

    @staticmethod
    def _release_section(config):
        """The ``release:`` config section (checkpoint locations/URLs). Accepts the
        pre-v1.4.0 spelling ``test:`` from user-supplied configs."""
        return config.get("release") or config.get("test") or {}

    def _msms_checkpoint_rel_path(self, instrument):
        key = "local_path_qtof" if instrument == "qtof" else "local_path_orbitrap"
        return self._release_section(self.msms_config)[key]

    def _get_checkpoint_path(self, task_name, instrument=None):
        task_map = {
            "msms":      lambda: self._msms_checkpoint_rel_path(instrument),
            "ccs":       lambda: self._release_section(self.ccs_config)["local_path"],
            "rt":        lambda: self._release_section(self.rt_config)["local_path"],
            "save_feat": lambda: self._msms_checkpoint_rel_path(instrument),
        }
        rel = task_map[task_name]()
        return checkpoints.resolve_checkpoint_path(rel, self.current_path, self.checkpoint_dir)

    def _checkpoint_url(self, task_name, instrument=None):
        if task_name == "ccs":
            return self._release_section(self.ccs_config).get("github_release_url")
        if task_name == "rt":
            return self._release_section(self.rt_config).get("github_release_url")
        if instrument == "qtof":
            return self._release_section(self.msms_config).get("github_release_url_qtof")
        return self._release_section(self.msms_config).get("github_release_url_orbitrap")

    def load_checkpoint(self, task_name, path_to_checkpoint=None, instrument=None):
        """Download (if needed), validate and load the checkpoint for ``task_name``
        into the corresponding model attribute.

        :param task_name: One of ``'msms'``, ``'rt'``, ``'ccs'``, ``'save_feat'``.
        :param path_to_checkpoint: Optional path to a custom checkpoint.
        :param instrument: ``'qtof'`` or ``'orbitrap'`` (MS/MS checkpoints only).
        """
        checkpoint_path = path_to_checkpoint or self._get_checkpoint_path(task_name, instrument)
        checkpoints.ensure_checkpoint(
            checkpoint_path, self._checkpoint_url(task_name, instrument), task_name,
        )
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        current = self.msms_config["model"] if task_name == "save_feat" \
            else self._task_config(task_name)["model"]
        checkpoints.validate_checkpoint_config(ckpt, current, checkpoint_path)
        if task_name == "save_feat":
            # A bare Encoder extracts embeddings; keep only its sub-tree of the full
            # model's state dict, with the "encoder." prefix stripped.
            state = {k[len("encoder."):]: v
                     for k, v in ckpt["model_state_dict"].items()
                     if k.startswith("encoder.")}
            self.encoder.load_state_dict(state)
        else:
            model = getattr(self, self._task_model_attr(task_name))
            model.load_state_dict(ckpt["model_state_dict"])
        # RT/CCS targets are standardised with TRAIN-set statistics stored alongside the weights.
        if task_name in ("rt", "ccs") and "mu" in ckpt and "sd" in ckpt:
            model.mu, model.sd = float(ckpt["mu"]), float(ckpt["sd"])

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(self, path_to_test_data, batch_size=1):
        """Load input molecules from a CSV, MGF, or PKL file.

        :param path_to_test_data: Path to the input file. Supported formats: ``csv``,
            ``mgf``, ``pkl``.
        :type path_to_test_data: str
        :param batch_size: DataLoader batch size for inference (default ``1``).
        :type batch_size: int
        """
        loaders = {
            "csv": self._load_csv,
            "mgf": self._load_mgf,
            "pkl": self._load_pkl,
        }
        ext = str(path_to_test_data).rsplit(".", 1)[-1].lower()
        if ext not in loaders:
            raise ValueError(f"Unsupported format: .{ext}")

        pkl_dict = loaders[ext](path_to_test_data)
        self._set_data(pkl_dict, batch_size, source=path_to_test_data)

    def _load_csv(self, path):
        return molecules_to_records(path, self.data_config["encoding"])

    def _load_mgf(self, path):
        clean_spectra, _ = filter_spec(
            mgf.read(path),
            self.data_config["all"],
            self.data_config["encoding"]["type2charge"],
        )
        return mgf2pkl(clean_spectra, self.data_config["encoding"])

    @staticmethod
    def _load_pkl(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def load_dataframe(self, df, batch_size=1):
        """Load input molecules directly from a pandas DataFrame (no temp file).

        :param df: DataFrame with columns ``ID``, ``SMILES``, ``Precursor_Type``,
            ``Collision_Energy`` (same schema as the CSV input).
        :type df: pandas.DataFrame
        :param batch_size: DataLoader batch size for inference (default ``1``).
        :type batch_size: int
        """
        pkl_dict = molecules_to_records(df, self.data_config["encoding"])
        self._set_data(pkl_dict, batch_size, source="<DataFrame>")

    def load_smiles(self, smiles, precursor_type="[M+H]+", collision_energy="20 V",
                    ids=None, batch_size=1):
        """Load one or more molecules from SMILES strings.

        :param smiles: A single SMILES string or a list of SMILES strings.
        :param precursor_type: Adduct type (str) or per-molecule list.
        :param collision_energy: Collision energy (e.g. ``"20 V"``) or per-molecule list.
        :param ids: Optional list of IDs; defaults to ``mol_0``, ``mol_1`` ...
        :param batch_size: DataLoader batch size for inference (default ``1``).
        """
        if isinstance(smiles, str):
            smiles = [smiles]
        n = len(smiles)

        def _broadcast(v):
            return v if isinstance(v, (list, tuple)) else [v] * n

        df = pd.DataFrame({
            "ID":               ids or [f"mol_{i}" for i in range(n)],
            "SMILES":           smiles,
            "Precursor_Type":   _broadcast(precursor_type),
            "Collision_Energy": _broadcast(collision_energy),
        })
        self.load_dataframe(df, batch_size=batch_size)

    def _set_data(self, pkl_dict, batch_size, source):
        self.pkl_dict = pkl_dict
        self.batch_size = batch_size
        logger.info("Loaded %d records from %s", len(self.pkl_dict), source)
        # The shared loader stays at batch_size=1 so pred_ccs / pred_rt (which
        # require it) are unaffected; pred_msms builds its own batched loader.
        self.valid_loader = DataLoader(
            MolInferenceDataset(self.pkl_dict),
            batch_size=1, shuffle=False, num_workers=0, drop_last=False,
        )

    def get_data(self):
        """Return the currently loaded records (list of dicts), or None."""
        return self.pkl_dict

    # ------------------------------------------------------------------
    # Inference  (pred_msms / pred_ccs / pred_rt / save_features)
    # ------------------------------------------------------------------

    def save_features(self, checkpoint_path=None, instrument="qtof"):
        """Extract encoder embeddings for loaded molecules.

        :param checkpoint_path: Optional path to a custom checkpoint.
        :type checkpoint_path: str, optional
        :param instrument: ``'qtof'`` or ``'orbitrap'``.
        :type instrument: str
        :return: ``(id_list, features)`` where features is a numpy array of shape ``(N, emb_dim)``.
        :rtype: tuple
        """
        cfg = self.msms_config["model"]
        self.encoder = _build_encoder(cfg, bond_dim=int(cfg.get("bond_dim", 0))).to(self.device)
        self.load_checkpoint("save_feat", checkpoint_path, instrument)
        ids, features = pred_feat(
            self.encoder, self.device, self.valid_loader,
            batch_size=1, num_points=self.msms_config["model"]["max_atom_num"],
        )
        return ids, features.cpu().detach().numpy()

    def pred_msms(self, path_to_results=None, path_to_checkpoint=None, instrument="qtof"):
        """Predict MS/MS spectra for loaded molecules.

        :param path_to_results: Optional path to save results (``.mgf`` or ``.csv``).
        :type path_to_results: str, optional
        :param path_to_checkpoint: Optional path to a custom checkpoint.
        :type path_to_checkpoint: str, optional
        :param instrument: ``'qtof'`` or ``'orbitrap'``.
        :type instrument: str
        :return: DataFrame with columns ID, SMILES, Collision Energy, Precursor Type,
            Pred M/Z, Pred Intensity.
        :rtype: pandas.DataFrame
        """
        if instrument not in ("qtof", "orbitrap"):
            raise ValueError('instrument must be "qtof" or "orbitrap"')

        # (Re)load weights when the model is uninitialized, or when the
        # instrument / custom checkpoint differs from what is currently loaded.
        # Without this, the cached model silently keeps stale weights after an
        # instrument switch within the same MolNet instance.
        if (self.msms_model is None
                or self._loaded_msms_instrument != instrument
                or self._loaded_msms_ckpt != path_to_checkpoint):
            if self.msms_model is None:
                self.msms_model = MolNetMS(self.msms_config["model"]).to(self.device)
            self.load_checkpoint("msms", path_to_checkpoint, instrument)
            self._loaded_msms_instrument = instrument
            self._loaded_msms_ckpt       = path_to_checkpoint

        loader = self.valid_loader
        if self.batch_size > 1:
            loader = DataLoader(
                MolInferenceDataset(self.pkl_dict),
                batch_size=self.batch_size, shuffle=False, num_workers=0, drop_last=False,
            )
        id_list, pred_tensor = pred_step(
            self.msms_model, self.device, loader,
            batch_size=self.batch_size, num_points=self.msms_config["model"]["max_atom_num"],
            env=self._task_env("msms"),
        )
        pred_dicts = [
            ms_vec2dict(spec, float(self.msms_config["model"]["resolution"]))
            for spec in pred_tensor.tolist()
        ]
        res_df = results.assemble_msms_results(
            self.pkl_dict, id_list, pred_dicts, self.data_config,
        )
        if instrument == "qtof":
            self.qtof_msms_res_df = res_df
        else:
            self.orbitrap_msms_res_df = res_df

        if path_to_results:
            results.save_msms_results(res_df, path_to_results, self.version, instrument)
        return res_df

    def pred_ccs(self, path_to_results=None, path_to_checkpoint=None):
        """Predict CCS values for loaded molecules.

        :param path_to_results: Optional path to save results as CSV.
        :type path_to_results: str, optional
        :param path_to_checkpoint: Optional path to a custom checkpoint.
        :type path_to_checkpoint: str, optional
        :return: DataFrame with columns ID, SMILES, Precursor Type, Pred CCS.
        :rtype: pandas.DataFrame
        """
        if self.ccs_model is None:
            self.ccs_model = MolNetScalar(self.ccs_config["model"]).to(self.device)
            self.load_checkpoint("ccs", path_to_checkpoint)

        id_list, pred_tensor = pred_step_scalar(
            self.ccs_model, self.device, self.valid_loader,
            batch_size=1, num_points=self.ccs_config["model"]["max_atom_num"],
            env=self._task_env("ccs"),
        )
        decoding = results.precursor_decoder(self.data_config)
        add_list    = [decoding[",".join(map(str, map(int, d["env"][1:])))] for d in self.pkl_dict]
        smiles_list = [d["smiles"] for d in self.pkl_dict]

        self.ccs_res_df = pd.DataFrame({
            "ID": id_list, "SMILES": smiles_list,
            "Precursor Type": add_list, "Pred CCS": pred_tensor.squeeze().tolist(),
        })
        if path_to_results:
            results.save_csv(self.ccs_res_df, path_to_results)
        return self.ccs_res_df

    def pred_rt(self, path_to_results=None, path_to_checkpoint=None):
        """Predict retention times for loaded molecules.

        :param path_to_results: Optional path to save results as CSV.
        :type path_to_results: str, optional
        :param path_to_checkpoint: Optional path to a custom checkpoint.
        :type path_to_checkpoint: str, optional
        :return: DataFrame with columns ID, SMILES, Pred RT.
        :rtype: pandas.DataFrame
        """
        if self.rt_model is None:
            self.rt_model = MolNetScalar(self.rt_config["model"]).to(self.device)
            self.load_checkpoint("rt", path_to_checkpoint)

        id_list, pred_tensor = pred_step_scalar(
            self.rt_model, self.device, self.valid_loader,
            batch_size=1, num_points=self.rt_config["model"]["max_atom_num"],
            env=self._task_env("rt"),
        )
        smiles_list = [d["smiles"] for d in self.pkl_dict]

        self.rt_res_df = pd.DataFrame({
            "ID": id_list, "SMILES": smiles_list,
            "Pred RT": pred_tensor.squeeze().tolist(),
        })
        if path_to_results:
            results.save_csv(self.rt_res_df, path_to_results)
        return self.rt_res_df

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        task,
        train_data,
        valid_data,
        checkpoint_path="",
        resume_path="",
        transfer=False,
        freeze_encoder=None,
        precursor_type="All",
        use_scaler=False,
    ):
        """Train a model and store it on this MolNet instance.

        After training the model is ready for immediate use via ``pred_msms``,
        ``pred_rt``, or ``pred_ccs`` — no checkpoint reload needed.

        :param task: One of ``'msms'``, ``'rt'``, ``'ccs'``.
        :type task: str
        :param train_data: Path to training PKL file.
        :type train_data: str
        :param valid_data: Path to validation PKL file.
        :type valid_data: str
        :param checkpoint_path: Where to save the best checkpoint. Empty string disables saving.
        :type checkpoint_path: str
        :param resume_path: Resume from or transfer-learn from this checkpoint.
        :type resume_path: str
        :param transfer: If ``True``, fine-tune: load only the pretrained encoder weights from
            ``resume_path`` (e.g. an SSL pretraining checkpoint). Head weights — including the
            env-consuming first layer, whose columns encode the source task's collision-energy
            and adduct layout — start fresh.
        :type transfer: bool
        :param freeze_encoder: Fine-tuning regime, only meaningful with ``transfer=True``:
            ``True`` (the default when unspecified) freezes the transferred encoder and trains
            the head alone; ``False`` trains everything (full fine-tune). Passing it without
            ``transfer=True`` raises a ``ValueError``.
        :type freeze_encoder: bool, optional
        :param precursor_type: Filter training data by precursor type (``msms`` task only).
            One of ``'All'``, ``'[M+H]+'``, ``'[M-H]-'``.
        :type precursor_type: str
        :param use_scaler: Fit a StandardScaler on training targets (``rt`` task only).
        :type use_scaler: bool
        :return: Best validation metric achieved during training.
        :rtype: float
        """
        if task not in training.TASKS:
            raise ValueError(f"Unknown task: {task} (expected one of {tuple(training.TASKS)})")
        if freeze_encoder is not None and not transfer:
            raise ValueError(
                "freeze_encoder only applies to transfer=True; without a transfer there is "
                "nothing this flag would do, so passing it is refused rather than ignored."
            )
        spec = training.TASKS[task]
        config = self._task_config(task)

        train_loader, valid_loader = training.build_loaders(
            task, train_data, valid_data, config,
            self.data_config, self.data_config_path, precursor_type,
        )

        model = self._build_model(task, config)
        logger.info("%s  #params: %s", model.__class__.__name__,
                    f"{sum(p.numel() for p in model.parameters()):,}")
        optimizer, scheduler = training.build_optimizer(model, spec, config["train"])

        # Resume / transfer
        if resume_path:
            if transfer:
                logger.info("Loading encoder weights (frozen) for transfer learning...")
                self._load_weights(model, resume_path, transfer=True,
                                   current_model_config=config["model"],
                                   freeze_encoder=freeze_encoder)
                if use_scaler and task == "rt":
                    model.fit_scaler(collect_targets(train_loader))
            else:
                logger.info("Resuming from %s ...", resume_path)
                self._load_weights(model, resume_path, optimizer, scheduler,
                                   current_model_config=config["model"])
                if task == "rt":
                    ckpt = torch.load(resume_path, map_location=self.device, weights_only=False)
                    model.set_scaler(ckpt.get("scaler"))
        elif use_scaler and task == "rt":
            model.fit_scaler(collect_targets(train_loader))

        if checkpoint_path:
            os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)

        best_metric = training.fit(
            model, task, spec, config, train_loader, valid_loader,
            optimizer, scheduler, self.device, checkpoint_path,
        )

        # Reload best checkpoint weights before storing so pred_* methods use the
        # best-validation model, not the final (possibly worse) last-epoch model.
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_weights(model, checkpoint_path, current_model_config=config["model"])
        setattr(self, self._task_model_attr(task), model)
        logger.info("Training complete. Best %s: %.4f", spec.metric_label, best_metric)
        return best_metric

    # ------------------------------------------------------------------
    # Evaluation (ground truth vs. predictions)
    # ------------------------------------------------------------------

    def evaluate(self, test_pkl, pred_mgf, result_path="", plot_path=""):
        """Compare predicted MS/MS spectra against ground-truth spectra.

        :param test_pkl: Path to the ground-truth PKL file (from preprocessing).
        :type test_pkl: str
        :param pred_mgf: Path to the predicted spectra MGF file (from ``pred_msms``).
        :type pred_mgf: str
        :param result_path: Optional path to save per-spectrum results as CSV.
        :type result_path: str
        :param plot_path: Optional path to save a cosine similarity histogram PNG.
        :type plot_path: str
        :return: DataFrame with per-spectrum cosine similarity and metadata.
        :rtype: pandas.DataFrame
        """
        df = results.evaluate_predictions(test_pkl, pred_mgf)
        if result_path:
            results.save_csv(df, result_path)
        if plot_path:
            results.plot_similarity_hist(df["cosine_similarity"].tolist(), plot_path)
        return df

    def generate_spectra_from_df(self, df, instrument=None):
        """Convert a ``pred_msms`` result DataFrame into pyteomics-style spectrum dicts."""
        return results.spectra_from_dataframe(df, self.version, instrument)
