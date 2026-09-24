# Protein–ligand VR atom-deletion experiment

This separate experimental workflow generates **12,960 features per complex**
and compares full-training GBDT models with **10,000 versus 30,000 trees** on
the matched refined/CASF 2007 and 2016 benchmarks. It runs on `cornell-wsl`
in the activated **`topokit` conda environment**. The fixed alpha baseline and
its completed results remain in `../protein_ligand_prediction/`.

## Exact feature recipe

1. Read protein PDB `ATOM` records with elements **C, N, O, S** and explicit
   ligand MOL2 atoms with elements **C, N, O, S, P, F, Cl, Br, I, H**. Do not
   infer hydrogens or include protein HETATM records. Keep protein atoms at
   minimum distance **at most 12 Å** to any supported original ligand atom,
   including H. Keep this crop fixed through all deletions.
2. Before forming channels, deduplicate exact coordinates across the selected
   protein rows followed by ligand rows. Keep the first atom and its original
   element/component attributes; reader order breaks ties within a component.
   A protein/ligand collision retains the protein atom. Nearby coordinates
   are never rounded or merged. Crop anchors are the original ligand sites;
   multiplicity does not change distances. Each record preserves the retained
   indices and counts. This matches the established molecular input convention.
3. Form **40 element channels**, protein outermost in C/N/O/S order and ligand
   innermost in C/N/O/S/P/F/Cl/Br/I/H order. There are no `all` or `null`
   categories. Select the union of that protein element and that ligand element.
4. Use the nine cutoffs **2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0 Å**.
   At each cutoff, connect every distinct unordered pair whose Euclidean
   distance is at most the cutoff. Include protein–protein, ligand–ligand and
   protein–ligand edges, with unit weight. Keep isolated vertices. No Delaunay,
   alpha, chemical-bond or cross-component constraint is applied. These are
   direct distance cutoffs; **do not divide by two**. A single arbitrary edge
   orientation encodes each undirected edge in the Hyperdigraph core, avoiding
   reciprocal-edge double counting.
5. For every protein atom in the channel, independently remove that atom and
   all its incident edges. Recompute the remaining degrees and obtain the
   complete ordinary L0 spectrum of the remaining graph. Repeat independently
   for every ligand atom. Each deletion starts from the original graph at that
   cutoff. Deletions are not cumulative and do not change the original crop.
6. For each deletion spectrum, compute these nine statistics in order:

   | Index | Statistic |
   | --- | --- |
   | 1 | Sum of positive eigenvalues |
   | 2 | Mean of positive eigenvalues |
   | 3 | Median of positive eigenvalues |
   | 4 | Population standard deviation of positive eigenvalues |
   | 5 | Population variance of positive eigenvalues |
   | 6 | Maximum positive eigenvalue, without normalization |
   | 7 | Minimum positive eigenvalue, without normalization |
   | 8 | Sum of squared positive eigenvalues |
   | 9 | Number of zero or near-zero eigenvalues |

   Positive means greater than `1e-10`; near-zero means absolute value at most
   `1e-10`. Values below `-1e-10` fail validation. Standard deviation and
   variance use `ddof=0`. If no positive values remain, the first eight values
   are zero and the zero count is retained. A graph with no remaining atoms
   has nine zeros. All eigenvalues are computed; no spectral truncation occurs.
7. Aggregate each statistic separately over the protein deletions and ligand
   deletions, in this exact order: **sum_protein, mean_protein, sum_ligand,
   mean_ligand**. Means divide by the number of atoms originally present on
   that side of the channel, including deletions with zero statistics.
   A missing side contributes two zero aggregates; the available side still
   contributes its own deletion results. If both sides are missing, everything
   is zero. A one-atom channel becomes an empty graph after its sole deletion.
8. Store a C-contiguous float32 tensor with axes
   **channel × scale × statistic × aggregation**, shape **(40, 9, 9, 4)**.
   Calculations use float64. C-order flattening gives **12,960 features**.
   There is no separate undeleted-graph feature block.

## Exact reuse and correctness

`loo_features.py` uses public TopoKit molecular readers, coordinate
deduplication, explicit Hyperdigraph objects and the ordinary L0 core. The
optional `L0Sweep.vertex_deleted_laplacian()` API reuses the original matrix,
removes the selected row/column and subtracts incident-edge contributions
from the remaining degrees. Merely taking a principal submatrix would be
incorrect. Every deletion is independent, and the original matrix is retained.

The application also reuses complete spectra of unchanged connected
components, equivalent deletions of graph-twin vertices, identical adjacency
matrices across cutoffs, and repeated one-sided channels. A bounded 64 MiB
cache per worker stores exact adjacency-keyed component results. These are
exact reuse operations, with no approximation or missing-atom imputation.
The explicit dense budget is 25 million entries per component/channel;
resource-limit failures remain visible and never remove samples silently.

Independent tests build induced adjacency matrices and D−A directly. The
full run additionally checks 45 molecular channel/cutoff slices from 1a30,
1w8l and 10gs using full NumPy eigensolves after every atom deletion. Saved
sum/mean relationships, missing-side zeros, tensor shapes and finite values
are audited for every complex. Higher-dimensional and persistent core routes
are unchanged and included in the package regression suite.

## Datasets and model settings

| Training manifest | Training size | Test manifest | Test size |
| --- | ---: | --- | ---: |
| `train_refined_2007.csv` | 1,105 | `test_casf_2007.csv` | 195 |
| `train_refined_2016.csv` | 3,772 | `test_casf_2016.csv` | 285 |

Reuse byte-identical existing manifests from
`datasets/protein_ligand_prediction/labels/final`. Their union contains
**4,324 unique complexes**. Each matched year's test IDs are excluded from
its training set. Cross-year membership is preserved, with no new splitting,
label editing, filtering or sample dropping. The cohort names identify the
historical benchmarks; the existing manifests separately document actual
structure release/provenance, mostly the updated v2020R1 working structures
plus the previously recovered historical files. This experiment does not
train on general v2020R1 or the 2013 refined set.

The previous complete-manifest baseline settings use 10,000 trees. The second
configuration uses 30,000 trees. Both otherwise use:

| Parameter | Value |
| --- | --- |
| Estimator | scikit-learn `GradientBoostingRegressor` |
| Learning rate | 0.002 |
| Maximum depth | 7 |
| Minimum samples to split / minimum leaf size | 5 / 1 |
| Training fraction per iteration | 0.8 |
| Features considered per split | `sqrt`, 113 of 12,960 |
| Loss / split criterion | `squared_error` / `friedman_mse` |
| Random seed | 0 |
| Feature transformation | `StandardScaler`, fit on training rows only |
| Target | Existing `label_logka` pK value, unscaled |
| Early stopping / internal validation split | Disabled / none |

These four fits use every row of their training manifest. There is no
holdout-based tree-count selection and no repeated-seed ensemble. CASF does
not enter fitting or standardization. All four results are reported; CASF is
a previously used comparison benchmark, not an untouched new external set.
RMSE, Pearson correlation (PCC) and MAE are reported separately. Undefined PCC
for constant predictions is explicitly null, not replaced with zero.

## Remote folder layout

```text
/home/dc2339/topokit_project/
├── topokit/workflows/protein_ligand_prediction_version2/
│   ├── README.md
│   ├── loo_features.py
│   ├── study_common.py
│   ├── loo_train.py
│   ├── run_study.py
│   ├── audit_real_samples.py
│   ├── launch_cornell.sh
│   └── tests/
└── datasets/protein_ligand_prediction_version2/
    ├── README.md
    ├── labels/final/                  # four unchanged manifests
    ├── structures -> ../protein_ligand_prediction/structures
    ├── features/                     # schema, samples, records, failures
    ├── models/                       # four full fits, plans and checksums
    ├── results/                      # predictions, RMSE/PCC/MAE and report
    └── experiments/
        ├── study/                    # frozen source, progress, logs, audits
        └── smoke/models/             # four explicitly marked five-tree tests
```

The structures symlink is read-only by workflow contract; input hashes are
checked before and after generation. Results, features and models remain on
Cornell WSL. Only source and documentation are maintained locally.

## Commands

Run on `cornell-wsl`. The launcher always activates the correct environment
and sets numerical-library threads to one per process:

```bash
cd /home/dc2339/topokit_project
bash topokit/workflows/protein_ligand_prediction_version2/launch_cornell.sh \
  --action smoke --workers 16 --ml-workers 4
bash topokit/workflows/protein_ligand_prediction_version2/launch_cornell.sh \
  --action run --workers 16 --ml-workers 4
```

The smoke run generates 39 real tensors under the final scientific recipe,
runs the independent molecular checks and fits five trees on 12 training/eight
test rows per year. These smoke scores are not research results. Production
reuses those tensors only after verifying full identity and input checksums.
It then completes the full feature union, fits the four complete models and
audits saved models against independently refitted training scalers,
manifest-ordered matrices, saved predictions and recomputed metrics.

For an unattended production run, use `nohup bash .../launch_cornell.sh ...`
with its output redirected to `experiments/study/logs/controller.log`.
An exclusive lock prevents duplicate controllers. Resume with the same
`--action run` command: complete features/models must verify; incomplete
model staging folders are retained as failure evidence. Failures stop the
training handoff; no sample is silently omitted. Every fit writes progress
every 500 trees. The callback always returns false and cannot enable stopping.

```bash
cat datasets/protein_ligand_prediction_version2/experiments/study/progress.json
cat datasets/protein_ligand_prediction_version2/experiments/study/*progress.json
cat datasets/protein_ligand_prediction_version2/results/RESULTS.md
bash topokit/workflows/protein_ligand_prediction_version2/launch_cornell.sh --action audit
```

Source and environment hashes are frozen at preparation. Do not edit active
Python sources during a run. Use a separate `--output` for any scientific or
code revision. Final completion requires `final_audit.json` with
`state="complete"`, `smoke=false`, `feature_count=4324`, `model_count=4`,
`failures=0`, plus the published result table.

Validation on 2026-09-18: the remote package/workflow regression passed
**1,217 tests and 240 subtests**, with 23 optional-dependency skips. The initial
1a30 pilot (580 cropped protein atoms, 49 ligand atoms, largest channel 399
atoms) took 156.57 s with repeated native object reconstruction and 81.81 s
with exact core deletion reuse, one numerical thread, approximately 118 MiB
peak RSS. This single-complex timing is not a full-dataset runtime estimate.
The end-to-end smoke passed at **22:16:39 UTC**, including all 39 tensors,
45 independent real molecular slices, four five-tree fits and their independent
model/scaler audits. These smoke scores are not benchmark results. All 4,324
complexes passed the input preflight, and all 83 frozen Python source hashes
match locally and remotely. **Production launched at 22:17:33 UTC** with
controller PID 4358 and 16 confirmed active feature workers, followed
automatically by four full model fits. Full results are pending; fresh status
and final completion are recorded in the remote receipts.
