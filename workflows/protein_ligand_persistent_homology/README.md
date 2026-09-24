# Cornell WSL protein–ligand barcode experiment

Status, 2026-09-20: the user has fixed H0, full-bin coverage counts and a
filtration maximum of 5. The implementation and local scientific tests pass;
the 48-complex remote smoke passed, and full production launched at
18:50:43 UTC on September 20 (controller PID 9243). The existing baseline, atom-deletion and HPCC experiments remain
separate. Target: `/home/dc2339/topokit_project` on `cornell-wsl`, using the
activated `topokit` conda environment.

## Confirmed molecular rules

1. Protein groups, in order: C, N, O, S, Null. Ligand groups, in order:
   C, N, O, S, P, F, Cl, Br, I, H, Null. These define 55 channels, with protein
   group as the outer index. Null means no atoms from that component; it is
   not an additional atom or a group of all elements.
2. Retain supported protein atoms within an inclusive 15 Å of any supported
   ligand atom, including explicit ligand hydrogens. No hydrogens are added.
3. Concatenate selected protein atoms before supported ligand atoms, then
   call TopoKit's exact-coordinate keep-first deduplication once, before
   splitting channels. Retain the first atom's element and component identity.
   Nearby but unequal coordinates remain distinct.
4. Construct native alpha support separately on each channel's selected
   coordinates. Mixed non-Null channels retain protein–ligand edges only;
   protein–protein and ligand–ligand edges are excluded there. Edge weights
   are one; geometric birth values determine filtration order.
5. The user's final Null-channel clarification is an explicit exception:
   **protein/Null** allows alpha protein–protein edges, and **Null/ligand**
   allows alpha ligand–ligand edges. Null/Null is empty and has zero features.
6. Keep all selected vertices, including isolates. An absent non-Null element
   does not turn its channel into an explicit Null channel. For example, C/I
   with no ligand iodine keeps isolated protein C vertices; C/Null allows
   protein C–C alpha edges. A channel with no atoms on either side is zero.
7. Generate and retain actual persistence intervals first. Postprocess those
   intervals using 101 boundaries `0, 0.05, ..., 4.95, 5`, giving 100 bins per
   channel and 5,500 features for one homology degree. No atom deletion or
   eigenvalue summary is part of this recipe.
8. Infinite deaths in a filtration stopped at the upper bound mean survival
   through the observed window. They must not be changed to deaths at 5.

## Fixed barcode and filtration definition

Compute **H0 only over GF(2)** using the public TopoKit hyperdigraph persistence
API. There is no H1 feature, eigenvalue calculation or atom deletion. Each
channel is constructed once, its barcode is computed once, and the complete
observed barcode is saved before postprocessing.

The filtration parameter is the **alpha radius in Å**, from 0 through 5.
TopoKit's native builder uses squared-radius units internally: build through
25 Å² and take the square root of each retained edge birth before passing the
edge filtration to the hyperdigraph core. There is **no separate 5 Å atom-pair
distance cap**, and no division by two. An isolated pair 8 Å apart therefore
merges at alpha radius 4 Å. Alpha births include full Delaunay coface
propagation, even though H0 needs only vertices and edges.

For each half-open bin `[left,right)`, count bars that cover the **entire bin**:
`birth <= left` and `death >= right`, with positive bar length. A bar dying
exactly at the right boundary still covers that bin; a bar dying within the
bin does not. This is `barcode_bin_counts(..., mode="cover")`, not the default
overlap mode. Example: `[0,0.075)` counts in `[0,0.05)` but not `[0.05,0.10)`.
All vertices are born at zero, so H0 counts are nonincreasing across bins.

No bin extends beyond 5. Bars surviving the observed window retain infinite
deaths in the saved barcode and contribute to `[4.95,5)`. They are not
artificially terminated or discarded at 5. A finite death exactly at 5 also
covers the final bin. Numerical endpoints use exact floating-point comparisons,
with no rounding or tolerance-based shifts.

Native hyperdigraph objects require an ambient vertex. An empty channel is
therefore encoded explicitly as an empty H0 barcode and zero features, without
inventing a vertex. One orientation per retained unordered alpha edge suffices
for this H0 recipe; no claim about an H1 construction is made.

## Existing inputs and planned model protocol

The read-only Cornell preflight found all files for **4,414 unique complexes**.
Use the existing audited manifests under
`datasets/protein_ligand_prediction/labels/final/` without changing their
membership, labels, matched-year holdouts, or row order:

| Benchmark | Refined training | Matching CASF test |
| --- | ---: | ---: |
| 2007 | 1,105 | 195 |
| 2013 | 2,764 | 195 |
| 2016 | 3,772 | 285 |

These are historical benchmark memberships with the project's working
structures (primarily v2020R1 plus recovered historical inputs), not a claim
that every coordinate file comes from the original historical release.

Planned comparisons retain the previous Cornell experiment's two GBDT
settings: 10,000 and 30,000 trees, one fit per setting per year (six fits).
Learning rate 0.002, maximum depth 7, minimum split size 5, minimum leaf size
1, subsample 0.8, square-root feature subsampling, squared-error loss,
Friedman MSE criterion, seed 0, and early stopping disabled. StandardScaler
is fitted on the training set only; targets stay on their existing pK scale.
No CASF samples enter scaler fitting, model fitting or stopping decisions.

Features, saved barcodes, provenance, models, predictions and RMSE/PCC/MAE
results will use a distinct remote dataset directory. Final scores require
complete input/feature coverage and independent scaler/model reload audits.
All **4,414 feature/barcode pairs are generated and audited**, with zero
failures. The 2,100 independent real-bin checks also passed. Full training
started at 19:00:14 UTC on September 20, with all six worker processes verified
active. Current scores: **pending full model completion and audit**. Production uses
16 feature workers and six model workers. The smoke passed all 48 complexes,
2,100 independent real bin checks and six short model reload/scaler audits.
Frozen study identity:
`edc0be35bd423140a9d0c6ed4e29d512eebbd05dea4f2f20d635603118623a7f`.
All 84 source files match locally and on Cornell. Authoritative status is
`experiments/study/progress.json`; the final production completion record is
`experiments/study/final_audit.json` (distinct from the smoke record).

## Available postprocessing

`topokit.postprocessing.barcode_bin_counts` consumes a `PersistenceResult`
and explicit bin boundaries and returns a serializable `VectorResult`.
It supports any already-computed homology degree and records its counting
mode and units. It uses sorted boundary searches and range additions, avoiding
a bars-by-bins matrix. No topology is reconstructed during postprocessing.
See [notation and endpoint rules](../../markdown/NOTATION.md).

## Execution and verification

Workflow directory:
`/home/dc2339/topokit_project/topokit/workflows/protein_ligand_persistent_homology`

Output directory:
`/home/dc2339/topokit_project/datasets/protein_ligand_persistent_homology`

```bash
bash topokit/workflows/protein_ligand_persistent_homology/launch_cornell.sh --action smoke --workers 16 --ml-workers 6
bash topokit/workflows/protein_ligand_persistent_homology/launch_cornell.sh --action run --workers 16 --ml-workers 6
```

The launcher activates the `topokit` conda environment and caps numerical
libraries at one thread per worker. Feature generation uses 16 processes;
the six independent GBDT fits use up to six processes. A controller lock
prevents duplicate launches. Resume requires matching source, recipe,
software, input and feature checksums. Failures are recorded and block training;
no sample is silently removed.

`barcodes/samples/<id>.npz` contains 56 channel offsets and float64 birth/death
arrays in alpha-radius Å, with actual positive infinity preserved.
`features/samples/<id>.npy` contains a float32 `(55,100)` tensor (integer-valued
counts); C-order flattening gives 5,500 columns. Records retain atom counts,
deduplication maps, edge/barcode counts and both artifact hashes. The frozen
schema provides all channel and bin labels. `features/records/`, `models/`,
`results/`, and `experiments/study/` retain artifact/provenance/audit files.

Every saved feature is rechecked against direct predicates on its saved
barcode. Three real complexes receive an additional 2,100-bin audit using
full alpha simplices and independent SciPy connected components immediately
before each right boundary. Local tests additionally use random molecular
channels, an analytic obtuse-triangle coface example, exact endpoints and
hydrogen-driven cropping. Each model receives independent training-only
scaler, membership, restored prediction and metric checks. Smoke fits use
five trees on small subsets and are never reported as benchmark scores.

Read-only completion checks are active every 30 minutes in the originating
Codex thread. They inspect existing receipts and results, never restart or
change jobs, and stop after verified final reporting.
