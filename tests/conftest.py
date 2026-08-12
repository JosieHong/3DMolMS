"""Shared test scaffolding.

`neighbour_graph_from_xyz` is TEST-ONLY on purpose. The encoder requires an explicit neighbour
graph — as of v1.4.0 it has no internal neighbour selection at all — so the geometric-invariance
tests need some graph to feed it. Real callers always have the covalent bond graph from
preprocessing; these tests use synthetic point clouds that have no chemistry, so they build a
spatial one instead.

A spatial graph is the right choice here, and not an arbitrary stand-in:

  * it is IDENTICAL under rotation, reflection and translation, because those preserve distances,
    so any change in the encoder's output is a genuine invariance failure rather than the graph
    moving underneath the test;
  * it PERMUTES with the atoms, so the permutation-invariance case stays honest.

A hardcoded index array would satisfy neither property without extra bookkeeping.

This lived briefly in `molnetpack.molconv` while the bond-vs-kNN ablation ran
(see ABLATION_NEIGHBOURHOOD.md). kNN lost on every measure and was removed from the library, so
the helper moved here rather than staying as shipped code with no shipped caller.
"""

import torch


def neighbour_graph_from_xyz(xyz, mask, k):
    """k nearest neighbours by Euclidean distance -> (neighbor_idx, neighbor_mask).

    Matches the covalent bond graph's format and padding contract: unused slots point at the atom
    itself with mask False.

    Args:
        xyz:  [B, N, 3] coordinates
        mask: [B, N] True for real atoms
        k:    neighbour slots per atom
    """
    batch, num_points, _ = xyz.shape
    d2 = torch.cdist(xyz, xyz) ** 2
    d2 = d2.masked_fill(~mask.unsqueeze(1), float("inf"))          # never select padding atoms
    eye = torch.eye(num_points, dtype=torch.bool, device=xyz.device).unsqueeze(0)
    d2 = d2.masked_fill(eye, float("inf"))                         # never select self
    _, idx = torch.topk(d2, k=k, dim=2, largest=False)
    nmask = torch.isfinite(torch.gather(d2, 2, idx)) & mask.unsqueeze(-1)
    self_idx = torch.arange(num_points, device=xyz.device).view(1, num_points, 1).expand(batch, num_points, k)
    return torch.where(nmask, idx, self_idx), nmask


def graph_for(x, mask, k):
    """Convenience for the common `[B, C, N]` layout the encoder takes."""
    return neighbour_graph_from_xyz(x[:, :3, :].transpose(1, 2), mask, k)
