# Intro to MolConv

`MolConv` is the graph-convolution layer at the heart of 3DMolMS. It turns a
molecule's **3D conformation** (a cloud of atoms, each with `xyz` coordinates and
chemical features) into a fixed-length embedding that the task heads (MS/MS, RT,
CCS) read to make their predictions.

Concretely, the encoder takes up to `max_atom_num` atoms, each a 21-dimensional
vector (the first 3 dims are `xyz`; the remaining 18 are one-hot atom features),
and returns one embedding per molecule.

## Why invariance matters

The same molecule can be written down in infinitely many ways: shift it a few
ångström, rotate it, mirror it, or list its atoms in a different order: it is
still the *same molecule* with the *same chemistry*. A good encoder must return
the **same embedding** under all of these, otherwise the prediction would depend
on arbitrary details of how the conformer happened to be stored.

| Transformation | Example | Should the embedding change? |
|---|---|---|
| **Permutation** | reorder the atom list | no |
| **Rotation** | spin the molecule | no |
| **Translation** | move it in space | no |
| **Reflection** | mirror it (its enantiomer) | usually no — but *yes* for chiral tasks |

These combine into the named symmetry groups used throughout this page: **O(3)** =
rotation + reflection; **SE(3)** = rotation + translation; **E(3)** = rotation +
reflection + translation. For the achiral 3DMolMS tasks the target is
**E(3) + permutation** invariance: ignore *where* and *how* the molecule sits in
space, but still see its shape.

## How `MolConv` achieves E(3) invariance

**In short: the layer describes geometry through relative positions and angles
rather than absolute coordinates, so the molecule's pose does not matter.**

Three design elements make it pose-independent:

1. **Relative-displacement Gram** `⟨xⱼ−xᵢ, xₗ−xᵢ⟩` — this inner product encodes the
   local angle `∠jik` between neighbours, which is unchanged by rotation and
   reflection.
2. **Center on the real-atom centroid** before computing the first-layer geometry.
   The centroid moves with any translation of the input, so the centered
   coordinates (and all distances) are identical under a shift → translation
   invariance becomes exact.
3. **A fixed, supplied neighbour graph with self-pointing padding.** Neighbours
   come from the covalent bond graph computed at preprocessing time, and unused
   neighbour slots point at the atom itself, so their relative displacement is
   exactly zero and padding stays inert instead of leaking a pose-dependent term
   through the atom-dimension normalization.

The result is **full E(3) + permutation invariance** (rotation + reflection +
translation), verified to floating-point-exact precision in `float64`. `MolConv`
is the encoder for all released checkpoints.

An earlier version of this layer (`MolConv1`) computed geometry from absolute atom
positions and was removed in v1.4.0 because it was not translation-invariant:
merely shifting a molecule in space changed its embedding.

## Neighbourhoods: the covalent bond graph

Each atom's neighbours are its **covalently bonded atoms**, supplied as a fixed
graph by preprocessing (`data_utils.bond_graph_array`). With bonded neighbours,
the layer's distance channel carries **bond lengths** and its Gram matrix **bond
angles**: real chemistry rather than incidental spatial proximity. The graph is
required: the encoder refuses to run without one.

## Chirality: E(3) (default) vs SE(3)

Reflection is the one symmetry you sometimes *don't* want. By default the encoder
is **reflection-invariant**, so a molecule and its mirror image (its
enantiomer) get the **identical** embedding; the encoder cannot tell enantiomers
apart. That is exactly right for the **achiral 3DMolMS tasks** (MS/MS, RT, CCS),
where enantiomers share the same target.

For a **chirality-dependent** task — e.g. **3DMolCSP** (chiral stationary-phase
separation), where enantiomers elute differently — set **`chirality: true`**.
It appends a **signed-volume pseudoscalar** channel `dⱼ·(d₁×d₂)` in the first
layer, which is invariant under proper rotation and translation but **flips sign
under reflection**, so the encoder becomes **SE(3)** (reflection-*sensitive*).

| | `chirality: false` (default) | `chirality: true` |
|---|---|---|
| Symmetry group | **E(3)** | **SE(3)** |
| Rotation | invariant | invariant |
| Translation | invariant | invariant |
| **Reflection** | **invariant** (exact) | **sensitive** |
| Enantiomers (mirror images) | identical prediction | **can differ** |
| Correct for | achiral tasks: MS/MS, RT, CCS | chiral tasks: e.g. 3DMolCSP |

An E(3) encoder is **provably unable** to model enantiomer differences (it must
output one value for both), so SE(3) is *required* for chiral separation, not
optional. These properties are verified on random-weight layers
(see `test_chirality_is_se3_reflection_sensitive`).

## Invariance verification

Invariance here is **architectural** (it holds for any weights), so it can be
checked with random-weight layers, with no trained checkpoint needed:

- `tests/test_encoder_invariance.py` — layer-level rotation/reflection/
  translation/permutation checks for `MolConv` and its `chirality` (SE(3)) mode.
- `tests/test_se3_invariance.py` — full-encoder SE(3) invariance in `float64` and
  `float32`.

```bash
pytest tests/test_encoder_invariance.py tests/test_se3_invariance.py -q
```

## Configuration

```yaml
model:
  chirality: false   # true -> SE(3), reflection-sensitive (chiral tasks, e.g. 3DMolCSP)
```
