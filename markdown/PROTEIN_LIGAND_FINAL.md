# Protein–ligand application notation and migration

The current feature is **FS-AN**: a 15 Å crop and physical alpha radii
0.1–5.0 Å, with the unchanged `(10,50,55)` feature layout. The September 23
final ML selection uses the new GBDT hyperparameters and the prediction
average of seeds 0/1/2; see the [current ML contract](PROTEIN_LIGAND_ML_FINAL.md)
and [workflow specification](../workflows/protein_ligand_prediction/README.md).
The following September 16 specification is a historical record. Its feature
grid and original GBDT settings are retained for the earlier ablations.

## Historical September 16 specification

Effective 2026-09-16, the user-fixed feature method is **m2+m3+m5+r4**.
There are no revision flags in the active application. See the complete
[workflow specification](../workflows/protein_ligand_prediction/README.md).

- `s`: reported interatomic scale grid `0,.2,...,9.8 Å` (50 observations).
- `r=s/2`: alpha radius; squared edge birth is compared with `r²`.
- `null`: component omitted, distinct from an all-element category.
- `λ+`: eigenvalues strictly above `1e-10`; `μ` is their mean.
- `MAD`: positive-spectrum mean absolute deviation about `μ`.
- `H`: `−Σp ln(p)` for `p=λ+/Σλ+`, in nats, without entropy normalization.
- Tensor axes: statistic × scale × protein-major element channel;
  `(10,50,55)` float32, flattened in C order to 27,500 coordinates.
- Ordinary L0 at successive scales is not a two-scale persistent Laplacian.
- General release name: **PDBbind v2020R1**, never an unqualified original
  v2020/18,904-row protocol. The final v2020R1 general training count is 18,498.

Historical refined memberships and existing NMI point targets are preserved;
structure bytes include updated v2020R1 and audited historical recoveries.
The exact final manifests have 1,105/2,764/3,772 refined training rows and
195/195/285 CASF rows. The general set excludes all three CASF sets. No sample
is removed for feature availability or geometry failure.

Migration: seven earlier repository workflow directories now reside under
`archive/protein_ligand_prediction_20260916/workflows/`. Previous dataset
results, models, labels and feature provenance reside under
`datasets/protein_ligand_prediction/archive/2026-09-16_pre_final/`.
Only `workflows/protein_ligand_prediction/` and `features/final_alpha_l0_unique/`
are active entry points. Original scientific records keep their original
paths/hashes; relocation and generated-array deletion have separate ledgers.
Archive replay may require restoring the historical layout. The archive is
not imported by the final workflow and is not bundled into package releases.

All four final models use refined-v2/ablation GBDT settings: 10,000 trees,
learning rate .002, depth seven, sqrt features, minimum split five, subsample
.8 and seed zero. Training-only StandardScaler is preserved; targets are
unscaled; there is no early stopping or additional holdout now that selection
is fixed. The archived early general-v2 .01 configuration remains historical.
CASF-2016 was involved in exploratory strategy selection; its reused scores
are not independent validation of that choice.

## 2026-09-17 native alpha backend migration

The user requested removal of the GUDHI requirement. That migration introduced workflow
1.1.0 / schema 2 with native alpha engine 1.0.0, implemented in TopoKit.
The mathematical m2+m3+m5+r4 recipe and all counts above stay fixed.
[Native alpha notes](NATIVE_ALPHA.md) define its floating-point/rational
precision contract, full coface propagation and higher-dimensional support.
Existing `final_alpha_l0` artifacts are frozen GUDHI-backed evidence; future
native runs use `final_alpha_l0_native`, with a distinct schema/source identity.
Read-only validation receipts compare the engines without rewriting existing
features, models, predictions or provenance. At that time, the 208 coincident-coordinate cases remained blocked.

## 2026-09-17 user-authorized unique-atom input policy

Workflow 1.2.0 / schema 3 and native engine 1.1.0 now keep each exact coordinate
once. Retain the first row of the cropped protein-then-ligand atom list,
including its element/component labels, before any element-channel selection.
Original ligand sites define cropping; no input file or coordinate is changed.
All removals are recorded. Point multiplicity is not a weight, and removed
atoms are not restored by edge lifting. Alpha builders default to merge with
an explicit strict error mode available. Current output roots are
`final_alpha_l0_unique`; previous native validation and GUDHI benchmarks stay
immutable. All manifests, target values, scales, summaries and ML settings
remain fixed.

## 2026-09-17 — Recovery for a nearly planar hull cell

The full native run exposed one previously untested geometry in 1tps: four
almost-coplanar ligand H sites produced a spurious Qhull hull tetrahedron.
Its exact circumsphere contains all 69 other H sites and has squared radius
about 5.002269e22, exhausting the radius-ball repair budget. Native alpha
1.1.1 adds a facet-neighbor cavity repair only AFTER all previous paths have
raised. Candidate cells are still tested against every input point; manifold,
boundary, full coface and simplex-budget checks remain active. Coordinates,
duplicate policy, cutoffs and Laplacian calculations are unchanged.

Workflow 1.2.1 retains schema 3, with recipe
`final-alpha-l0-48051949e7c7c0f2`. Its output roots are
`final_alpha_l0_unique_repair`. The existing native-1.1.0 run continues from
its frozen source until the complete feature pass ends. A narrowly scoped
reuse receipt can then copy its SUCCESSFUL tensors and JSON byte-for-byte.
No record is relabeled: original recipe, generation source and timestamp stay
intact. `source_compatibility.json` lists the exact permitted records/hashes.
An AST comparison proves that pre-existing successful native paths are
unchanged; every other package source and feature arithmetic must match the
pinned predecessor. Arbitrary recipe changes and GUDHI output reuse are
rejected. Generation, training and audits verify the compatibility inventory
and freeze its checksum. Only failed samples need new computation.

The repaired 1tps tensor matches its historical GUDHI tensor bit-for-bit.
The 73-point hull fixture also matches the complete exact-alpha complex,
including higher cofaces; its old failure and new recovery are regression
tested. Production/source/proof receipts are under
`experiments/native_alpha_repair_20260917/`. The overall benchmark is still
in progress; running source snapshots are never edited.
