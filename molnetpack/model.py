"""3DMolMS model architectures.

* :class:`Encoder` — the shared MolConv point-cloud encoder.
* :class:`MolNetMS` — encoder + bidirectional spectrum head for MS/MS prediction.
* :class:`MolNetScalar` — encoder + scalar head for regression tasks (RT, CCS, pretraining).
  (Formerly ``MolNet_Oth``, which remains available as an alias.)
"""

from decimal import Decimal
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .molconv import MolConv
from .spectrum_heads import mask_prediction_by_mass, reverse_prediction


# ----------------------------------------
# >>>          shared helpers          <<<
# ----------------------------------------

def _init_weights(module: nn.Module) -> None:
    """Initialise all submodules: Kaiming for convs, unit/zero for norms, Xavier for linears."""
    for m in module.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(
                m.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu"
            )
        elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()


def _build_encoder(config: dict, bond_dim: int = 0) -> "Encoder":
    """Construct the shared :class:`Encoder` from a model config block.

    ``encoder_version`` is no longer a knob — the legacy v1 layer was removed in v1.4.0
    and the parameter with it. A stale config still declaring v1 is refused here rather
    than silently building the current encoder from a config that means something else.
    """
    version = config.get("encoder_version")
    if version is not None and int(version) != 2:
        raise ValueError(
            f"encoder_version={version} is not supported. v1 (MolConv1, absolute-position "
            f"Gram) was removed in v1.4.0: it is not E(3)-invariant (output shifted 11-18% "
            f"under a pure translation), it cannot take a bond graph, and no released "
            f"checkpoint uses it. Remove the key from the config, or use a pre-v1.4.0 "
            f"release to reproduce a v1 checkpoint."
        )
    return Encoder(
        in_dim=int(config["in_dim"]),
        layers=config["encode_layers"],
        emb_dim=int(config["emb_dim"]),
        point_num=int(config["max_atom_num"]),
        k=config["k"],
        chirality=bool(config.get("chirality", False)),
        bond_dim=bond_dim,
    )


def _default_idx_base(x: torch.Tensor) -> torch.Tensor:
    """Per-molecule index offset for the neighbour gather, computed from the batch shape."""
    batch_size, _, num_points = x.size()
    return (
        torch.arange(0, batch_size, device=x.device, dtype=torch.long).view(-1, 1, 1)
        * num_points
    )


def _append_env(x: torch.Tensor, env: Optional[torch.Tensor], add_num: int) -> torch.Tensor:
    """Concatenate the encoded experimental condition onto the pooled embedding.

    ``env`` arrives as ``[B, add_num]`` from the datasets, but a single-column env is easy to
    hand in as ``[B]``; accept both rather than silently producing a 3-D tensor (an
    ``unsqueeze(env, 1)`` on an already-2-D env did exactly that historically, which is why the
    RT model — ``add_num`` 1 — could not run through this path).
    """
    if add_num <= 0:
        return x
    if env is None:
        raise ValueError(f"this model was built with add_num={add_num}, so env is required")
    if env.dim() == 1:
        env = env.unsqueeze(1)
    if env.shape[1] != add_num:
        raise ValueError(
            f"env has {env.shape[1]} columns but the model expects add_num={add_num}. "
            f"A mismatch here means the checkpoint and the data disagree about the adduct "
            f"one-hot layout, which would silently mis-encode every adduct."
        )
    return torch.cat((x, env), 1)


# ----------------------------------------
# >>>           encoder part           <<<
# ----------------------------------------
class Encoder(nn.Module):
    """Stack of MolConv layers with masked max+average pooling to a molecule embedding."""

    def __init__(self, in_dim, layers, emb_dim, point_num, k, chirality=False,
                 bond_dim=0):
        super().__init__()
        self.emb_dim = emb_dim
        self.bond_dim = int(bond_dim)
        # k may be a single int (same k every layer) or a per-layer list of length len(layers).
        ks = [int(v) for v in k] if isinstance(k, (list, tuple)) else [int(k)] * len(layers)
        if len(ks) != len(layers):
            raise ValueError(f"k list len {len(ks)} != #layers {len(layers)}")
        # chirality=True -> SE(3) (reflection-sensitive, chiral tasks); False -> E(3) (default).
        # Only the first layer sees xyz, so it carries the flag.
        self.hidden_layers = nn.ModuleList(
            [
                MolConv(
                    in_dim=in_dim,
                    out_dim=layers[0],
                    point_num=point_num,
                    k=ks[0],
                    remove_xyz=True,
                    chirality=chirality,
                    bond_dim=self.bond_dim,
                )
            ]
        )
        for i in range(1, len(layers)):
            self.hidden_layers.append(
                MolConv(
                    in_dim=layers[i - 1],
                    out_dim=layers[i],
                    point_num=point_num,
                    k=ks[i],
                    remove_xyz=False,
                    bond_dim=self.bond_dim,
                )
            )

        self.conv = nn.Sequential(
            nn.Conv1d(emb_dim, emb_dim, kernel_size=1, bias=False),
            nn.LayerNorm((emb_dim, point_num)),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(
        self,
        x: torch.Tensor,
        idx_base: torch.Tensor,
        mask: torch.Tensor,
        return_per_atom: bool = False,
        neighbor_idx: Optional[torch.Tensor] = None,
        neighbor_mask: Optional[torch.Tensor] = None,
        bond_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # neighbor_idx [B, N, k] / neighbor_mask [B, N, k]: the fixed bonded-neighbour graph,
        # passed to every layer. REQUIRED — MolConv refuses to run without a graph.
        # bond_feat [B, N, k, bond_dim]: explicit per-edge bond descriptor (same graph every layer).
        xs = []
        for i, hidden_layer in enumerate(self.hidden_layers):
            inp = x if i == 0 else xs[-1]
            xs.append(hidden_layer(inp, idx_base, mask, neighbor_idx, neighbor_mask, bond_feat))

        x = torch.cat(xs, dim=1)  # torch.Size([batch_size, emb_dim, point_num])
        x = self.conv(x)

        # per-atom features before pooling: [batch_size, emb_dim, point_num]
        per_atom = x

        # Apply the mask: set padding points to -inf for max pooling and zero for average pooling.
        mask_expanded = mask.unsqueeze(1).expand_as(x)  # [batch_size, emb_dim, point_num]
        x_masked_max = x.masked_fill(~mask_expanded, float("-inf"))
        x_masked_avg = x.masked_fill(~mask_expanded, 0.0)

        # Max pooling along the point dimension
        max_pooled = torch.max(x_masked_max, dim=2)[0]  # [batch_size, emb_dim]

        # Average pooling along the point dimension, normalised by the number of
        # valid (non-padding) points; clamp avoids division by zero.
        valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=0.1)
        avg_pooled = x_masked_avg.sum(dim=2) / valid_counts  # [batch_size, emb_dim]

        pooled = max_pooled + avg_pooled
        if return_per_atom:
            return per_atom, pooled  # [B, emb_dim, N], [B, emb_dim]
        return pooled


# ----------------------------------------
# >>>           decoder part           <<<
# ----------------------------------------
class FCResBlock(nn.Module):
    """Fully-connected residual block: three Linear+LayerNorm stages with a resized skip."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.linear1 = nn.Linear(in_dim, out_dim, bias=False)
        self.bn1 = nn.LayerNorm(out_dim)

        self.linear2 = nn.Linear(out_dim, out_dim, bias=False)
        self.bn2 = nn.LayerNorm(out_dim)

        self.linear3 = nn.Linear(out_dim, out_dim, bias=False)
        self.bn3 = nn.LayerNorm(out_dim)

        self.dp = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = self.bn1(self.linear1(x))
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.bn2(self.linear2(x))
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.bn3(self.linear3(x))

        x = x + F.interpolate(identity.unsqueeze(1), size=x.size()[1]).squeeze()

        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.dp(x)
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} ({self.in_dim} -> {self.out_dim})"


class MSDecoder(nn.Module):
    """MLP trunk of stacked :class:`FCResBlock` s; dropout only on the last three blocks."""

    def __init__(self, in_dim, layers, out_dim, dropout):
        super().__init__()
        self.blocks = nn.ModuleList([FCResBlock(in_dim=in_dim, out_dim=layers[0])])
        for i in range(len(layers) - 1):
            if len(layers) - i > 3:
                self.blocks.append(FCResBlock(in_dim=layers[i], out_dim=layers[i + 1]))
            else:
                self.blocks.append(
                    FCResBlock(in_dim=layers[i], out_dim=layers[i + 1], dropout=dropout)
                )

        self.fc = nn.Linear(layers[-1], out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.fc(x)


# -----------------------------------
# >>>           3DMolMS           <<<
# -----------------------------------
class MolNetMS(nn.Module):
    """MolConv encoder + BIDIRECTIONAL spectrum head.

    As of v1.4.0 the plain MLP-to-bins decoder is gone. The head is a shared trunk feeding three
    projections -- forward, reverse (indexed downward from the precursor) and a sigmoid gate --
    followed by a hard mask that zeroes every bin above the precursor. See `spectrum_heads` for
    the equations and the measurement (+0.068 cosine over the plain decoder).

    Because the reverse head and the mask are both defined RELATIVE TO THE PRECURSOR, `forward`
    now requires `prec_idx`: the precursor m/z of each molecule expressed as a bin index. Use
    `spectrum_heads.precursor_bin` to compute it; the datasets in `molnetpack.dataset` supply it.
    """

    def __init__(self, config: dict, prec_mass_offset: int = 10, out_relu: bool = False):
        super().__init__()
        self.add_num = config["add_num"]

        self.encoder = _build_encoder(config, bond_dim=int(config.get("bond_dim", 0)))
        out_dim = int(Decimal(str(config["max_mz"])) // Decimal(str(config["resolution"])))
        hidden = int(config["decode_layers"][-1])
        if hidden < 1:
            raise ValueError("decode_layers must end in a positive width")
        # The trunk is MSDecoder with its final projection widened to `hidden` instead of n_bins,
        # so the three heads share one representation.
        self.trunk = MSDecoder(
            in_dim=int(config["emb_dim"] + config["add_num"]),
            layers=config["decode_layers"],
            out_dim=hidden,
            dropout=config["dropout"],
        )
        self.forw = nn.Linear(hidden, out_dim)
        self.rev = nn.Linear(hidden, out_dim)
        self.gate = nn.Linear(hidden, out_dim)
        self.offset = int(prec_mass_offset)
        # No output nonlinearity by default. A ReLU here zeroes the gradient wherever the logit is
        # negative, which stalled training when it was tried; the released models set this False.
        self.out_relu = bool(out_relu)

        _init_weights(self)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        env: Optional[torch.Tensor] = None,
        idx_base: Optional[torch.Tensor] = None,
        neighbor_idx: Optional[torch.Tensor] = None,
        neighbor_mask: Optional[torch.Tensor] = None,
        bond_feat: Optional[torch.Tensor] = None,
        prec_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Input:
                x:      point set, torch.Size([batch_size, in_dim, atom_num])
                mask:	 mask for padding points, torch.Size([batch_size, atom_num])
                env:    experimental condition (collision energy + adduct one-hot)
                idx_base:   idx offset for neighbour gather
                neighbor_idx/neighbor_mask: bonded-neighbour graph (REQUIRED)
                bond_feat: explicit per-edge bond features [B, N, k, bond_dim] (BOND MODE)
                prec_idx:  [B] long, precursor m/z as a bin index. REQUIRED -- the reverse head
                           and the mask are both defined relative to it.
        """
        if prec_idx is None:
            raise ValueError(
                "MolNetMS requires prec_idx (precursor m/z as a bin index). The reverse head "
                "indexes bins downward from the precursor and the output mask zeroes bins above "
                "it, so there is no meaningful default. Use spectrum_heads.precursor_bin, or the "
                "datasets in molnetpack.dataset, which supply it."
            )
        if idx_base is None:
            idx_base = _default_idx_base(x)
        x = self.encoder(x, idx_base, mask, neighbor_idx=neighbor_idx,
                         neighbor_mask=neighbor_mask, bond_feat=bond_feat)  # [batch_size, emb_dim]
        x = _append_env(x, env, self.add_num)

        # bidirectional head (spectrum_heads eqs 6-10)
        s = self.trunk(x)
        g = torch.sigmoid(self.gate(s))
        y = g * self.forw(s) + (1.0 - g) * reverse_prediction(self.rev(s), prec_idx, self.offset)
        y = mask_prediction_by_mass(y, prec_idx, self.offset)
        return F.relu(y) if self.out_relu else y


# -------------------------------------------------------------------------
# >>>                           MolNetScalar                           <<<
# 1) This is the model for scalar regression tasks, including pretrain,
# retention time prediction, collision cross-section prediction...
# 2) The difference from MolNetMS is the configuration parameters
# controlling output dimension, and MolNetScalar contains a scaler.
# -------------------------------------------------------------------------
class MolNetScalar(nn.Module):
    """MolConv encoder + scalar regression head (RT, CCS, pretraining).

    Formerly named ``MolNet_Oth``; the old name remains available as an alias.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.add_num = config["add_num"]
        self.encoder = _build_encoder(config, bond_dim=int(config.get("bond_dim", 0)))
        self.decoder = MSDecoder(
            in_dim=int(config["emb_dim"] + config["add_num"]),
            layers=config["decode_layers"],
            out_dim=1,
            dropout=config["dropout"],
        )
        self.scaler = None
        # The released RT/CCS models standardise the target with plain TRAIN-set statistics
        # (fitting on all data would leak test information through the mean and variance). These
        # are restored from the checkpoint by `MolNet`; `scaler` remains for the sklearn path.
        self.mu = None
        self.sd = None

        _init_weights(self)

    def set_scaler(self, scaler):
        """Set the scaler for output normalization"""
        self.scaler = scaler

    def fit_scaler(self, targets):
        """Fit a new scaler on given targets"""
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler().fit(targets)
        return self.scaler

    def scale(self, y: torch.Tensor) -> torch.Tensor:
        """Scale the input using the internal scaler"""
        if self.scaler is None:
            return y

        # Convert to numpy, transform, and convert back to tensor
        y_np = y.cpu().detach().numpy().reshape(-1, 1)
        y_scaled = self.scaler.transform(y_np)
        return torch.tensor(y_scaled, dtype=torch.float).reshape(y.shape).to(y.device)

    def unscale(self, y_scaled: torch.Tensor) -> torch.Tensor:
        """Inverse scale the input using the internal scaler"""
        if self.scaler is None:
            return y_scaled

        # Convert to numpy, inverse transform, and convert back to tensor
        y_scaled_np = y_scaled.cpu().detach().numpy().reshape(-1, 1)
        y = self.scaler.inverse_transform(y_scaled_np)
        return (
            torch.tensor(y, dtype=torch.float)
            .reshape(y_scaled.shape)
            .to(y_scaled.device)
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        env: Optional[torch.Tensor] = None,
        idx_base: Optional[torch.Tensor] = None,
        neighbor_idx: Optional[torch.Tensor] = None,
        neighbor_mask: Optional[torch.Tensor] = None,
        bond_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Input:
                x:      	point set, torch.Size([batch_size, 14, atom_num])
                mask:	 	mask for padding points, torch.Size([batch_size, atom_num])
                env:    	experimental condition
                idx_base:   per-molecule index offset for the neighbour gather
                neighbor_idx / neighbor_mask:
                            covalent BOND graph (REQUIRED). The encoder aggregates over bonded
                            neighbours — the best-measuring neighbourhood definition (0.6202
                            val cosine on MS/MS) — and since v1.4.0 MolConv refuses to run
                            without a graph rather than silently selecting a different
                            architecture.
                bond_feat:  optional per-edge bond features (bond_dim > 0)
        """
        if idx_base is None:
            idx_base = _default_idx_base(x)
        x = self.encoder(
            x,
            idx_base,
            mask,
            neighbor_idx=neighbor_idx,
            neighbor_mask=neighbor_mask,
            bond_feat=bond_feat,
        )  # torch.Size([batch_size, emb_dim])
        x = _append_env(x, env, self.add_num)

        # decoder
        x = self.decoder(x)

        # Squeeze the trailing width-1 dimension so the output is [B], matching the shape of the
        # targets the datasets yield. Returning [B, 1] against a [B] target makes MSELoss BROADCAST
        # to [B, B] -- it compares every prediction with every other row's target, trains on a
        # meaningless objective, and only emits a UserWarning. The scalar heads always have
        # out_dim=1, so there is no case where the extra axis carries information.
        return x.squeeze(-1)  # [B], scaled output

    def predict(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        env: Optional[torch.Tensor] = None,
        idx_base: Optional[torch.Tensor] = None,
        neighbor_idx: Optional[torch.Tensor] = None,
        neighbor_mask: Optional[torch.Tensor] = None,
        bond_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get unscaled predictions for inference.

        The bond graph is forwarded through; MolConv raises if it is missing — as of
        v1.4.0 the encoder has no internal neighbour selection to fall back on.
        """
        with torch.no_grad():
            x = self.forward(
                x, mask, env, idx_base,
                neighbor_idx=neighbor_idx, neighbor_mask=neighbor_mask, bond_feat=bond_feat,
            )
            # Unscale. `scaler` is an sklearn transformer; `mu`/`sd` are the plain train-set
            # statistics the released RT/CCS models were standardised with.
            if self.scaler is not None:
                return self.unscale(x)
            if self.mu is not None and self.sd is not None:
                return x * self.sd + self.mu
            return x  # unscaled output


# Deprecated aliases: "Oth" said nothing about what the model does; MolNet_MS is the
# pre-v1.4.0 (non-PEP 8) spelling of MolNetMS.
MolNet_Oth = MolNetScalar
MolNet_MS = MolNetMS
