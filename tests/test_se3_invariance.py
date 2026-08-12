"""SE(3)-invariance tests for the MolConv encoder.

MolConv builds its first-layer Gram matrix from *relative* displacement
vectors ``(x_j - x_i)`` instead of the absolute positions ``x_j`` used by
the removed legacy v1 layer.  The inner product ``<(x_j - x_i), (x_k - x_i)>`` is the
(distance-scaled) angle at atom i, which is invariant to both rotation and
translation. Because the first layer strips the raw xyz from its output, that
invariance propagates through the whole encoder.

For the invariance to hold end-to-end MolConv also relies on (a) the supplied
neighbour graph's padded slots being self-pointing (so they stay inert), and (b)
centering coordinates on the real-atom centroid in the first layer. These tests
check the property directly, with random weights — invariance is structural, so
no trained checkpoint is required.

    pytest tests/test_se3_invariance.py
    # or, standalone:
    python tests/test_se3_invariance.py

The structural tests run in float64 to isolate the mathematical property from
float32 accumulation noise; ``test_encoder_invariant_float32`` separately
confirms the production (float32) path stays invariant to a small tolerance
(~7e-5 relative to the embedding scale).
"""

import torch

from molnetpack.model import Encoder
from conftest import graph_for
from molnetpack.utils import make_idx_base

# Real model dimensions (from config/molnet.yml).
IN_DIM = 21          # dims 0-2 are xyz, 3-20 are atom features
LAYERS = [64, 64, 128, 256, 512, 1024]
EMB_DIM = sum(LAYERS)  # 2048
POINT_NUM = 300
K = 5


def build_molecule(n_atoms=16, center=(8.0, 8.0, 8.0), spread=0.8, seed=0,
                   dtype=torch.float64):
    """A compact synthetic molecule padded to POINT_NUM, offset from the origin."""
    g = torch.Generator().manual_seed(seed)
    xyz = (torch.randn(n_atoms, 3, generator=g) * spread
           + torch.tensor(center)).to(dtype)

    feat = torch.zeros(n_atoms, IN_DIM - 3, dtype=dtype)
    types = torch.randint(0, 11, (n_atoms,), generator=g)  # 11 supported elements
    feat[torch.arange(n_atoms), types] = 1.0

    x = torch.zeros(1, IN_DIM, POINT_NUM, dtype=dtype)
    x[0, :3, :n_atoms] = xyz.t()
    x[0, 3:, :n_atoms] = feat.t()

    mask = torch.zeros(1, POINT_NUM, dtype=torch.bool)
    mask[0, :n_atoms] = True
    return x, mask, n_atoms


def random_rotation(seed=1, dtype=torch.float64):
    """A proper rotation matrix (det = +1) via QR of a random matrix."""
    g = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=g))
    q = q * torch.sign(torch.diagonal(r))  # fix QR sign ambiguity
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q.to(dtype)


def apply_se3(x, n_atoms, R=None, t=None):
    """Apply rotation R and translation t to the xyz of the real atoms only."""
    x2 = x.clone()
    xyz = x2[0, :3, :n_atoms]                          # [3, n_atoms]
    if R is not None:
        xyz = R.to(x.dtype) @ xyz
    if t is not None:
        xyz = xyz + torch.as_tensor(t, dtype=x.dtype).view(3, 1)
    x2[0, :3, :n_atoms] = xyz                          # padding xyz stays 0
    return x2


def make_encoder(dtype=torch.float64):
    torch.manual_seed(0)
    return Encoder(in_dim=IN_DIM, layers=LAYERS, emb_dim=EMB_DIM,
                   point_num=POINT_NUM, k=K).eval().to(dtype)


def _max_abs_diff(a, b):
    return (a - b).abs().max().item()


def _encoder_se3_diff(center, R, t, dtype):
    x, mask, n = build_molecule(center=center, dtype=dtype)
    idx_base = make_idx_base(1, POINT_NUM, x.device)
    x_se3 = apply_se3(x, n, R=R, t=t)
    enc = make_encoder(dtype)
    g = lambda t: graph_for(t, mask, K)
    # v1.4.0 removed the encoder's internal per-layer kNN, so a graph must be supplied.
    # Building it from the SAME coordinates that are fed in reproduces exactly what the
    # old fallback did, and keeps the test meaningful: a spatial graph is identical under
    # rotation/reflection/translation (distances are preserved) and permutes with the
    # atoms, so genuine invariance still has to hold.
    with torch.no_grad():
        return _max_abs_diff(enc(x, idx_base, mask, neighbor_idx=g(x)[0], neighbor_mask=g(x)[1]),
                             enc(x_se3, idx_base, mask,
                                 neighbor_idx=g(x_se3)[0], neighbor_mask=g(x_se3)[1]))


def test_encoder_is_se3_invariant():
    """The full MolConv encoder output is unchanged by rotation + translation."""
    diff = _encoder_se3_diff((8.0, 8.0, 8.0), random_rotation(),
                             t=[3.0, -5.0, 2.0], dtype=torch.float64)
    print(f"[encoder float64] max|Δembedding| under SE(3) = {diff:.2e}")
    assert diff < 1e-6, f"encoder is not SE(3)-invariant (max diff {diff:.2e})"


def test_encoder_invariant_at_origin():
    """Invariance holds even when the molecule is centred on the origin.

    Here the zero-padding atoms (at the origin) are closer to every real atom
    than its true neighbours, so without the row-aware kNN masking they would
    enter the graph and break invariance. This is the direct regression test for
    the padding-masking fix.
    """
    diff = _encoder_se3_diff((0.0, 0.0, 0.0), random_rotation(seed=3),
                             t=[2.0, 7.0, -4.0], dtype=torch.float64)
    print(f"[encoder@origin float64] max|Δembedding| under SE(3) = {diff:.2e}")
    assert diff < 1e-6, f"encoder not SE(3)-invariant at origin (max diff {diff:.2e})"


def test_encoder_invariant_float32():
    """The production (float32) path stays invariant to a small tolerance.

    Invariance is mathematically exact (see the float64 tests); float32 leaves a
    residual of ~2e-3 absolute, which is ~7e-5 relative to the embedding scale
    (max |value| ~ 30). A large translation is used to confirm the residual is
    accumulation noise, not distance-formula cancellation that grows with offset.
    """
    diff = _encoder_se3_diff((8.0, 8.0, 8.0), random_rotation(),
                             t=[100.0, -100.0, 50.0], dtype=torch.float32)
    print(f"[encoder float32] max|Δembedding| under SE(3) = {diff:.2e}")
    assert diff < 5e-3, f"float32 encoder residual too large (max diff {diff:.2e})"


if __name__ == "__main__":
    test_encoder_is_se3_invariant()
    test_encoder_invariant_at_origin()
    test_encoder_invariant_float32()
    print("\nAll SE(3)-invariance tests passed.")
