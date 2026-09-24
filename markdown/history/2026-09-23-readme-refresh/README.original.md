# topokit

## Final topology ML selection — September 23, 2026

The final **FS-AN topology GBDT** averages predictions from seeds **0, 1 and 2**.
Each model uses 10,000 trees, learning_rate=0.005, depth7, min_samples_split=2,
subsample=0.4 and max_features=sqrt, with its own training-only StandardScaler.
All **12 models/scalers and six ensemble evaluations** are delivered and verified.
The default predictor is the **general-v2020R1 ensemble**; matching refined-set
ensembles are retained for the three historical benchmarks.

Feature selection used the **previous parameters**. Historical ablations retain
their original parameters, predictions and scores; the fixed FS-AN tensors and
memberships are unchanged. The new hyperparameters and three-seed consensus
produce the **final ML results**.

[Final parameters and model bundles](../datasets/protein_ligand_prediction/final/topology_gbdt/README.md) ·
[Final results](../datasets/protein_ligand_prediction/final/topology_gbdt/results/FINAL_REPORT.md) ·
[RMSE](../datasets/protein_ligand_prediction/final/topology_gbdt/results/RMSE.md) · [PCC](../datasets/protein_ligand_prediction/final/topology_gbdt/results/PCC.md).

## Final TopoFormer architecture — September 23, 2026

The user selected **A03** as the protocol's final DL architecture: width 768,
12 encoder layers, 12 attention heads, feed-forward width 3072 and 86,071,297
parameters. [Final profile, models and scalers](../datasets/protein_ligand_prediction/final/topoformer_a03/README.md)
provide the canonical entry point. The existing full predictors used
102/104/92 epochs; all three seed bundles and their prediction ensemble are retained.

[Unified DL ablation record](../datasets/protein_ligand_prediction/experiments/protocol_ablation/studies/ST-08/README.md)
organizes 20 candidate settings, 34 unique validation fits, six full fits and
24 CASF rows, with separate RMSE/PCC tables. Both original studies remain
archived in place. A03 wins the validation comparison and CASF-2007, while the
six-layer control retains better RMSE/PCC on CASF-2013/2016. Fewer epochs does
not imply less wall time; the explicit user selection fixes capacity, not a
claim of universal superiority.

[Active A03 tuning study](../datasets/protein_ligand_prediction/experiments/topoformer_a03_tuning_20260923/README.md)
started on both Cornell RTX3080 GPUs on September23 at17:58UTC. The fixed
16,648/1,850 split tests batch16/32/64 × actual LR3e-5/1e-4/2e-4, followed by
seed confirmation and weight-decay comparisons, up to25 new fits. Both maximum
epochs and cosine horizon are200, with10-epoch warmup and validation patience50.
The controller also runs a common-validation training-fraction diagnostic and
three full18498-row refits before CASF evaluation. All42 preflight tests and nine
GPU profiles passed. Optimizer selection is pending; the delivered A03 models,
feature strategy, GBDT and sequence selections retain their existing provenance.

## Final sequence configuration — September 23, 2026

The user selected **ESM-2 + CPZ, FS-AU NMI-inspired sqrt GBDT**, as the final
sequence ML configuration. [Final parameters, models, scalers and results](../datasets/protein_ligand_prediction/final/sequence_esm2_cpz/README.md)
provide the canonical saved profile: 10,000 trees, learning_rate=0.005,
max_depth=7, min_samples_split=2, subsample=0.4, max_features='sqrt',
random_state=None, with a training-only embedded StandardScaler.
All four model/scaler bundles, 19,066 × 1,792 embeddings and six evaluations are
preserved; the general-v2020R1 bundle is the primary reuse entry point.

[Previous-parameter FS-AW results and settings](../datasets/protein_ligand_prediction/experiments/sequence_esm2_cpz_previous_20260923/ARCHIVE.md) are archived in place,
with all existing artifacts retained. This explicit user selection supersedes
the historical comparison status below. Frozen scientific records and the
separate FS-AN topology selection remain unchanged. Monitoring is retired.

## Completed: previous GBDT settings on retained CPZ (FS-AW)

[FS-AW final report](../datasets/protein_ligand_prediction/experiments/sequence_esm2_cpz_previous_20260923/results/FINAL_REPORT.md) records four complete 10,000-tree models/scalers
and six evaluations, independently delivered locally at 15:33 UTC on September23.
The same 19,066 × 1,792 CPZ feature matrix was reused byte-for-byte. All resolved
parameters match the four original FS-AQ previous-setting model receipts.

The NMI-inspired settings give lower RMSE/MAE and higher PCC on all three
general-set tests and refined-2016. Previous settings improve all three metrics
on refined-2007. Refined-2013 is mixed: previous settings improve PCC, while
NMI-inspired settings improve RMSE and MAE. Overall NMI-inspired settings have
lower RMSE/MAE in five of six rows and higher PCC in four of six rows.

Separate [RMSE](../datasets/protein_ligand_prediction/experiments/sequence_esm2_cpz_previous_20260923/results/RMSE.md), [PCC](../datasets/protein_ligand_prediction/experiments/sequence_esm2_cpz_previous_20260923/results/PCC.md) and [MAE](../datasets/protein_ligand_prediction/experiments/sequence_esm2_cpz_previous_20260923/results/MAE.md) tables include all six
refined/general evaluations and signed differences.

This compares complete configurations, including learning_rate .005→.002,
min_samples_split 2→5, subsample .4→.8 and random_state None→0. Both use
10,000 trees, depth7, sqrt and identical features/memberships with training-only
scaling. One fit per training set does not establish significance or isolate
individual effects. RMSE/MAE use logKa/pK, not kcal/mol. FS-AU remains retained;
no automatic replacement or CASF selection. FS-AV stays archived with model
assets intentionally absent. Historical source and delivery receipts are preserved.

Delivery and documentation are complete; the existing monitor is retired.

## Current retention — September 23, 2026

The user retained **FS-AU, ESM-2 + CPZ**, for future sequence-model use: all
19,066 embeddings, four trained pipelines with embedded scalers and standalone
scaler arrays, results, encoder assets and required parent caches are preserved.
[CPZ reuse guide](../datasets/protein_ligand_prediction/experiments/sequence_esm2_cpz_sqrt_20260923/REUSE.md). **FS-AV, ESM-2 + ChEMBL27-only**, is archived with
settings, results and predictions retained; its four model bundles and standalone
scalers were removed from both Mac and Cornell (139,448,346 bytes per machine).
[Archive and pruning records](../datasets/protein_ligand_prediction/experiments/sequence_esm2_chembl27_sqrt_20260923/ARCHIVE.md). Earlier completion descriptions record
the successful delivery before pruning. Frozen scientific receipts remain unchanged;
ARCHIVE.json records current availability. Earlier FS-AQ/FS-AR and selected
FS-AN are preserved. FS-AW is also delivered and documented; monitoring is retired.


## Completed sequence revision: sqrt GBDT (September 23, 2026)

Both user-requested ESM-2 studies are complete and independently verified locally
as of 13:40 UTC: **FS-AU (CPZ)** and **FS-AV (ChEMBL27)**. Each delivered all
19,066 × 1,792 float32 feature rows (protein 1,280 + ligand 512), four full
10,000-tree model/scaler bundles and six CASF evaluations. All eight fits ran
concurrently on Cornell CPU; no embeddings or memberships were changed.

GBDT uses depth 7, learning rate 0.005, min_samples_split=2, subsample=0.4,
and max_features='sqrt' (resolved parameter 42). Remaining sklearn 1.9.0 defaults
include random_state=None; actual pre-fit RNG states are saved. Scaling uses
training data only. No early stopping, holdout, ensemble, tuning or target scaling.

[CPZ final report](../datasets/protein_ligand_prediction/experiments/sequence_esm2_cpz_sqrt_20260923/results/FINAL_REPORT.md) · [ChEMBL27 final report](../datasets/protein_ligand_prediction/experiments/sequence_esm2_chembl27_sqrt_20260923/results/FINAL_REPORT.md)

General-v2020R1 model results:

**RMSE (logKa/pK; lower is better)**

| Training → test | ESM-2 + CPZ (FS-AU) | ESM-2 + ChEMBL27 (FS-AV) |
|---|---:|---:|
| general_v2020R1 → casf_2007 | 1.401586 | 1.442955 |
| general_v2020R1 → casf_2013 | 1.389762 | 1.435526 |
| general_v2020R1 → casf_2016 | 1.198154 | 1.252878 |

**PCC (higher is better)**

| Training → test | ESM-2 + CPZ (FS-AU) | ESM-2 + ChEMBL27 (FS-AV) |
|---|---:|---:|
| general_v2020R1 → casf_2007 | 0.826199 | 0.813440 |
| general_v2020R1 → casf_2013 | 0.805119 | 0.789548 |
| general_v2020R1 → casf_2016 | 0.851202 | 0.837198 |

CPZ has lower RMSE and higher PCC in all six matched evaluations, including
the three refined models in the full reports. These are single stochastic
fits, so this does not establish significance or isolate every source of variation.
Both reports contain separate RMSE/PCC comparisons with FS-AQ, FS-AR and FS-AN;
FS-AU also labels the old FS-AS refined results explicitly as partial.

All model/feature hashes, pooling/cache provenance, memberships, labels,
training-only scaler arrays and metrics passed local verification without
unpickling incompatible remote models locally. Old max_features=None FS-AS
is preserved as partial; FS-AT was canceled while queued and must not restart.
Completed FS-AR/FS-AQ and selected FS-AN remain unchanged. The historical NMI
max_features attribution is not independently established. Monitoring has
completed its delivery scope; historical receipts remain intact.

TopoKit has six modules: **readers, builders, core, postprocessing, visualization,
and workflows**. The workflow module includes one selected protein–ligand feature
encoder and a guarded downstream GBDT prediction interface:
`topokit.workflows.protein_ligand_prediction`.

The representative **FS-AN** recipe fixes a 15 Å protein crop, 50 alpha radii
0.1–5.0 Å, 55 null/element channels, ten spectral summaries and **27,500 features**.
The default companion predictor is the **general-v2020R1 three-seed GBDT
ensemble**, with a fitted StandardScaler retained inside every member pipeline. Model weights remain separate data assets; no
training or dataset is loaded when importing the package.
[Feature workflow and API](workflows/protein_ligand_prediction/README.md),
[model profile](workflows/protein_ligand_prediction/MODEL_SELECTION.json),
and [trained companion model](../datasets/protein_ligand_prediction/pretrained/representative_gbdt/).

Historical feature/model ablations are collected outside the package in the
[protocol study catalogue](../datasets/protein_ligand_prediction/experiments/protocol_ablation/README.md):
42 complete strategy descriptions, stable FS letter IDs, scoped legacy aliases,
123 evaluation rows, separate RMSE/PCC tables, and preserved source/result receipts.
The [optional supervised TopoFormer module](workflows/protein_ligand_prediction/dl/README.md)
uses the same fixed features, with portable model/scaler bundles and lazy PyTorch.
The general-v2020R1 experiment is complete:23 fits, three trained models with
fitted scalers, nine per-seed CASF evaluations and three prediction ensembles.
Six layers was selected by validation RMSE. Bundles and results are audited and
delivered locally. [Final report](../datasets/protein_ligand_prediction/experiments/topoformer_general_v2020R1_20260922/reports/FINAL_REPORT.md).
The [article-informed follow-up](../datasets/protein_ligand_prediction/experiments/topoformer_article_followup_20260922/reports/FINAL_REPORT.md)
completed all 17 new fits and local delivery on September 23. Validation selected
A03 (width 768, 12 layers, 12 heads, feed-forward width 3072), with mean
RMSE/PCC 1.171782/0.772468. Its three full general-v2020R1 models trained for
102/104/92 epochs. Ensemble RMSE/PCC are 1.351085/0.831628 on CASF-2007,
1.450729/0.771897 on CASF-2013 and 1.206813/0.840541 on CASF-2016.
A03 improves CASF-2007; the preserved six-layer control remains better on
CASF-2013/2016 for both metrics. All 17 bundles, scalers, predictions and audits
are local. The installed implementation and defaults are unchanged.

A separate [sequence modality](workflows/protein_ligand_prediction/sequence/README.md)
adds frozen ESM-2 protein embeddings (1,280) and ChEMBL27 ligand embeddings
(512), followed by the same GBDT pipeline: **1,792 features**, four fixed
training memberships and six CASF evaluations. Optional dependencies remain
lazy (`topokit[sequence]`); pretrained weights and experiment artifacts are
external data assets. [FS-AQ final report](../datasets/protein_ligand_prediction/experiments/sequence_esm2_chembl27_20260922/results/FINAL_REPORT.md)
records 19,066 verified feature rows, four complete models with fitted scalers
and six evaluations, all delivered locally. ESM ran on the local Mac and
GBDT fitting on Cornell. FS-AN remains the selected topology representative;
it has lower RMSE and higher PCC in all six matching comparisons.

The separate [FS-AR experiment](../datasets/protein_ligand_prediction/experiments/sequence_esm1b_chembl27_nmi_20260922/README.md)
regenerates protein features with first-generation ESM-1b and uses the five
published NMI GBDT settings, leaving other estimator parameters at their
installed defaults. It retains ChEMBL27 ligand embeddings and uses one fit
per training set. All 19,066 feature rows, four 10,000-tree model/scaler bundles
and six evaluations were verified locally on September 23 at 10:34 UTC.
The general-v2020R1 model gives CASF-2016 RMSE 1.251483 (logKa/pK) and
PCC 0.836777; see the [final report](../datasets/protein_ligand_prediction/experiments/sequence_esm1b_chembl27_nmi_20260922/results/FINAL_REPORT.md).
Both encoder and GBDT changed relative to FS-AQ, so this single-fit comparison
does not isolate either effect. FS-AQ and selected FS-AN remain preserved.


## Historical ESM-2 + CPZ all-feature run (FS-AS; superseded)

The user-confirmed FS-AS study uses `esm2_t33_650M_UR50D` protein embeddings
and the supplied `chembl27_pubchem_zinc_512` ligand checkpoint. All 8,830 ESM-2
windows were verified and reused; 15,364 new CPZ BOS ligand vectors generated
on the Mac. The complete 19,066 × 1,792 float32 feature matrix passed its audit
on September 23 at 02:12 UTC. At the user's sqrt revision, this max_features=None
run was stopped at 13:04 UTC: three refined models/evaluations were completed
and archived with checked hashes; the general fit stopped at 7,401 trees and
is not complete. FS-AU reuses these audited CPZ features and is fully delivered.
Historical frozen sources, receipts and partial results remain preserved.

[Study strategy and receipts](../datasets/protein_ligand_prediction/experiments/sequence_esm2_cpz_nmi_20260923/README.md). All checkpoint assets, study adapters,
models and results live outside the installed package.

The [FS-AT follow-up](../datasets/protein_ligand_prediction/experiments/sequence_esm2_chembl27_nmi_20260923/README.md) was canceled while queued when the user
requested sqrt. It produced no production fits. FS-AV reuses its verified
ESM-2/ChEMBL27 feature design and is now fully delivered.

Version **0.3.0** — a modular Python toolkit for simplicial, sequence-hyperdigraph,
and two-factor interaction topology. Mathematical objects remain distinct;
shared numerical records make the layers easy to compose. Reusable object
views and dimensional building blocks support scientific diagrams. Descriptive
local names and independently checked algorithms support maintainable research
code. Dependency-free readers cover native JSON point clouds plus common
coordinate, molecular, docking, and crystallographic files; postprocessing
includes explicit spectral moments, variance, centered Laplacian spectral
energy, and opt-in zero encoding for complete empty Laplacian operators.

Ordinary hyperdigraph **L0 matrix construction now supports incremental
filtration sweeps**. `core.hyperdigraph.L0Sweep` inserts each appearing edge
once and reuses a dense matrix buffer. `matrix_at(scale)` returns a separate
matrix copy without computing eigenvalues; `laplacian(scale)` uses the existing
eigensolver. The generic `workflows.laplacian_series` selects this path for
eligible ordinary L0 filtrations. L1 and higher, missing-singleton-face cases,
and genuine two-scale persistent Laplacians retain their general algorithms.
[API, restrictions and validation](markdown/INCREMENTAL_L0.md) describe the
guarded optimization and the construction-only benchmark.

## Six responsibilities

| Layer | Responsibility | Boundary |
| --- | --- | --- |
| `readers` | Read formats and preserve coordinates, IDs, weights, scientific metadata | Never choose topology |
| `builders` | Construct explicit objects or filtrations from data | Never compute ML features |
| `core` | Homology, persistence, Laplacians, eigenvalues/eigenvectors of supplied objects | No parsing, geometric builders, plots, or ML |
| `postprocessing` | Numerical bars/spectra → bins, statistics, entropy, vectors | Never recompute topology |
| `visualization` | Display inputs, objects, results, and features | Numerical records remain authoritative |
| `workflows` | Compose layers; optional ML over feature matrices | Never reimplement their algorithms |

```text
topokit/
├── src/topokit/
│   ├── readers/          # format functions + extensible reader registry
│   ├── builders/         # geometry, conversions, explicit construction recipes
│   ├── core/             # analysis APIs + private native mathematical kernels
│   ├── postprocessing/   # barcode features + customizable spectral summaries
│   ├── visualization/    # independent plotting functions
│   ├── workflows/        # composition, selected protein–ligand API, optional ML
│   └── …                 # shared data/results, serialization, validation, CLI
├── examples/
│   ├── data/             # ALL supplied demo/test datasets and their provenance
│   ├── output/           # generated demo results and figures (ignored by Git)
│   └── *.py              # demonstrations, fixture loader, ML handoff
├── tests/                # tests; tiny mathematical fixtures may be inline
├── workflows/            # task-specific, non-installed application receipts
├── markdown/             # notation, architecture, change trace, notices, history
├── README.md
├── NOTICE.md             # attribution and licensing, not algorithm documentation
└── AGENTS.md             # maintenance rule: update this README and change trace
```

No dataset, demo runner, or example-specific loader is installed inside the
library. The original three research packages are retained as migration
references, not runtime dependencies. General TDA platforms are not required.
The selected encoder and inference API are installed under
`src/topokit/workflows/protein_ligand_prediction/`. Its recipe documentation and
thin examples live under `workflows/protein_ligand_prediction/`; historical source,
slurm/AWS/Cornell launchers and ablations live in the external dataset study catalogue.
No archive is a runtime dependency. The separate
[atom-deletion](workflows/protein_ligand_prediction_version2/README.md) and
[H0 barcode](workflows/protein_ligand_persistent_homology/README.md) research projects
retain their existing repository locations and are not the representative package
application. See the [feature definition](workflows/protein_ligand_prediction/README.md)
for geometry, summaries, complete benchmark counts and provenance.

## Install and run

Python 3.10+; NumPy and SciPy are the required numerical dependencies.

```bash
python -m pip install -e '.[plot,test]'
python -m pytest -q
python examples/point_cloud.py --plots
python examples/static_objects.py
python examples/features_to_ml.py
python examples/configured_pipeline.py
python examples/visualization_gallery.py
python -m topokit info
```

For the reproducible two-dimensional point-cloud tutorial, create the supplied
Conda environment from this project root and open the registered notebook
kernel:

```bash
conda env create -f environment.yml
conda activate topokit
python -m ipykernel install --user --name topokit --display-name "Python (topokit)"
jupyter lab examples/point_cloud_topology_workflow.ipynb
```

The notebook uses a fixed seed to sample two intersecting circles of different
radii, with controlled angular, radial, and Cartesian jitter. The resulting 34
points are introduced first, then shown as the Graph Representation, Simplicial
Complex Representation, Hyperdigraph Representation, and Interaction Complex
Representation. The graph has its own point-and-edge plot with no background
radius circles. The simplicial and directed constructions use the same
stationary cutoff; the interaction factors are the 14-point smaller-circle
sample and the full cloud with explicit overlap. Hyperdigraph figures collapse
repeated oriented segments separately for directed 1- and 2-hyperedges, limiting
each point pair to two arrows per displayed dimension and four arrows total.
When a directed 1-hyperedge and a displayed directed 2-hyperedge segment share
an orientation, the directed-2 segment is a wider translucent orange arrow below
the narrow blue directed-1 arrow on the same arc.

Each stationary representation plots GF(2) homology and complete real
Laplacian spectra with minimum, maximum, mean, numerical zero count, and
spectral energy. The graph is intentionally restricted to H0/L0, so its H1/L1
panels remain empty. A following persistent-analysis section repeats the same
four-representation order over 25 alpha values. It plots persistence intervals,
Betti curves, and ordinary L0/L1 snapshot curves for minimum, maximum, mean,
and energy. Four 13-frame GIFs animate connectivity as the cutoff grows. PNG,
editable-text SVG, GIF, and numerical CSV outputs are written under
`examples/output/point_cloud_topology_workflow/`. Each persistent topology is
constructed directly inside its corresponding analysis section, and H0 barcode
lines start at alpha zero without an open-circle birth marker. Demonstration
figures use consistent Nature-style panel typography, redundant geometry or
line styles, and a clean grid-free layout. Blue 1-simplices, green 2-simplices,
and cividis direction weights preserve the earlier topology colors while shape,
direction, and line width keep color from being the only mathematical cue. PNGs
are exported at 300 dpi and SVG text remains editable. The Simplicial Complex,
Hyperdigraph, and Interaction Complex GIFs
also show a pale circle of the current alpha radius around every point, so the
geometric origin of newly appearing cells remains visible as the filtration
grows; the independent graph GIF keeps its point-and-edge-only convention.

For a smaller tutorial in which every fixed object is specified directly by a
Python dictionary, open:

```bash
jupyter lab examples/fixed_topological_objects_homology_laplacian.ipynb
```

All vertices use one regular-hexagon layout. The four independent sections
demonstrate a six-cycle graph, one fixed closed octahedral simplicial shell
with `beta_2=1`, a fixed Hyperdigraph with ambient vertex `0` absent from its
directed singleton hyperedges, and one fixed interaction complex whose two
factors share vertex labels `0` and `3`. The six coordinates are written
directly in the editable `points` dictionary. Each object is defined in one
dimension-keyed dictionary named `graph_cells`, `simplicial_cells`,
`hyperdigraph_cells`, or `interaction_cells`, so readers can change its 0-,
1-, and 2-dimensional elements in one place. There is no filtration or
cross-object comparison in this fixed-case tutorial. Each section computes
GF(2) homology and complete ordinary real Laplacian matrices and spectra
through degree two. The executed notebook exports four 300-dpi PNGs, four
editable-text SVGs, and 16 numerical CSVs under
`examples/output/fixed_topological_objects_homology_laplacian/`.

The separate `examples/point_cloud.py` command-line example uses a 24-point
coordinate fixture, generic point IDs, and uniform assigned weights. It
exercises all three routes through H2/L2. It does not infer molecular
properties or claim predictive performance. Results go under
`examples/output/` by default; `--output` accepts another explicit directory.

The public, one-format-at-a-time reader tutorial is:

```bash
jupyter lab examples/different_input_formats_workflow.ipynb
```

After an input summary, the notebook gives PDB, MOL2, SDF, MOL, PDBQT,
AQUCOG CIF, and native JSON their own independent read, inspection, analysis,
and plot section. Each section constructs a sequence Hyperdigraph and computes
H0/H1 persistence plus complete ordinary L0/L1 snapshots before the next file
is introduced. It displays only barcodes, Betti curves, and curves of spectral
minimum, maximum, mean, and centered Laplacian energy. Every file is parsed in
full; explicit 24-site tutorial subsets keep
the 3,646-atom PDBQT receptor and 162-site AQUCOG MOF calculations small. The
JSON section uses all 29 coordinates, stable site IDs, element labels, and
explicit weights derived from `data_material.cif`. AQUCOG source, licence, and
derivative details are recorded in [the example-data inventory](examples/data/README.md).
Betti curves use a dense, inexpensive evaluation grid independent of the six
Laplacian snapshot scales. Infinite barcode deaths mean survival to the bounded
filtration cutoff (right-censoring), not proven survival beyond that cutoff.
The tutorial uses `plot_barcodes(..., mark_initial_stage=False)` because its H0
vertices are known to be born exactly at the initial scale. The plotting
default remains `True`, which adds an open-circle cue for bars merely known to
be present when a positively truncated filtration begins.

The closing spectral reference introduces the ordinary Hodge Laplacian,
explains what zero, small-positive, and large eigenvalues can indicate, and
keeps physical language such as “soft” or “stiff” explicitly analogical unless
the operator's weights and units encode that physics. A built-in-versus-custom
descriptor table is followed by an executable custom-statistic mapping and a
separate discussion of AUC, total variation, and other descriptors computed
across a sampled filtration curve. The six plotted L0/L1 records are ordinary
single-scale snapshots. They are not the two-scale operator returned by
`core.persistent_laplacian(start=s, end=t)` or by
`workflows.laplacian_series(mode="persistent", scale_pairs=[(s, t)])`.

## Compose the layers

```python
from topokit import readers, builders, core, postprocessing, visualization
from topokit.serialization import save_result

# 1. The file path is supplied by the application, never selected by the core.
cloud = readers.read("examples/data/point_cloud_24.csv")
# 2. Select construction explicitly. Point weights do not affect alpha geometry.
obj = builders.from_points(cloud, kind="simplicial", max_dimension=2)
# 3. Analyze the defined object; q+1 boundaries are built when needed for Hq/Lq.
bars = core.persistence(obj, max_dimension=2)
spectrum = core.laplacian(obj, dimension=2, scale=3.0)
# 4. Postprocessing consumes numerical outputs, never rebuilds topology.
vector = postprocessing.histogram_features(bars, birth_edges=[0, 1, 3, 6],
                                          death_edges=[0, 1, 3, 6])
summary = postprocessing.summarize_spectrum(spectrum, statistics=("min", "max", "mean"))
# 5. Optional display and independent authoritative numerical export.
axis = visualization.plot_barcodes(bars)
save_result({"bars": bars, "features": vector, "spectrum_summary": summary},
            "examples/output/custom.json")
```

For a compact fixed-scale demonstration, the reusable stationary workflow and
view functions replace notebook-local analysis and drawing helpers:

```python
from topokit import workflows, visualization as viz

result = workflows.analyze_stationary(obj, scale=0.55, dimensions=(0, 1))
axis = viz.plot_stationary_simplicial(obj, scale=0.55)
figure, axes = viz.plot_stationary_diagnostics(result)
```

Use `plot_stationary_graph`, `plot_stationary_hyperdigraph`, or
`plot_stationary_interaction` for the other representations, or
`plot_stationary(..., representation=...)` as a dispatcher. These functions
select and display supplied cells only. Animation timing and GIF writing remain
application/example concerns; a frame callback may reuse the same static
renderer at successive scales.

Use `topokit.builders.simplicial`, `.hyperdigraph`, and `.interaction` for
explicit route-specific constructors, and `topokit.core.simplicial`,
`.hyperdigraph`, and `.interaction` for analysis and mathematical object types.
The parent namespaces expose these modules lazily, so both explicit child-module
imports and attribute access after importing the parent are deterministic:

```python
from topokit import builders, core

obj = builders.simplicial.from_graph([(0, 1)])
spectrum = core.simplicial.laplacian(obj, dimension=0)
```

Importing a parent does not eagerly load all three routes; the requested child
is loaded on first access. Convenience aliases such as `topokit.from_points`
still delegate to these layers.

## Extension points

* Readers: CSV, native JSON point clouds, single-frame XYZ, PDB, MOL2,
  single-record SDF/MOL, PDBQT, and single-atom-loop CIF are implemented.
  `readers.read(path)` dispatches by extension; the named `read_json`,
  `read_pdb`, `read_mol2`, `read_sdf`, `read_mol`, `read_pdbqt`, and `read_cif`
  functions are also public. They return
  `PointCloud`, retain format-specific atom/site fields and declared bonds in
  metadata, and never pass those bonds to a builder implicitly. CIF fractional
  sites are converted through the declared cell without symmetry expansion.
  Molecular clouds distinguish source and current selections: `source_*`
  counts/bonds/connectivity remain provenance, while `atom_count`,
  `selected_atom_count`, `bonds`, and connectivity describe the current view.
  `PointCloud.subset(...)` induces those current records on the selected stable
  IDs and keeps cell/tags/properties as source-level metadata.
  Native JSON uses a root object with required nonempty rectangular
  `coordinates` and optional same-length `ids`, string `labels`, finite `weights`, explicit
  `coordinate_units`, and a `metadata` object. Optional identity fields are
  `"schema": "topokit.point_cloud"` and `"schema_version": 1`.
  `points` is accepted as an alias, but cannot appear with `coordinates`.
  Metadata/provenance-only `.json` sidecars are rejected because they contain
  neither coordinate key; `examples/data/point_cloud_24.json` documents its paired
  CSV fixture and is intentionally not a point-cloud JSON record.
  `examples/data/data_material_from_cif.json` is the canonical native-JSON
  example: it contains 29 Cartesian `coordinates` rows derived from
  `data_material.cif`, together with stable CIF site IDs, canonical element
  labels, and explicit Pauling-electronegativity weights.
  Register another callable with
  `readers.register_reader(name, reader, extensions=(... ,))`.
  Documented per-point metadata stays aligned when selecting/reordering point IDs.
  `readers.assign_element_weights(cloud)` explicitly applies the offline
  Pauling table to canonical element labels retained by XYZ and molecular
  readers; reading alone leaves uniform weights. Missing values require
  explicit overrides. [Reference values and source](markdown/REFERENCE_DATA.md).
* Builders: register a custom construction recipe using
  `builders.register_builder(name, callable)`, or compose the existing builders.
  It returns a `Topology` accepted by its mathematical core. Registration does
  not automatically implement the algebra of a new topology family.
  Alpha remains the default. Graph/flag and digraph/hyperdigraph modes accept
  stable-ID `bonds`, inclusive distance `cutoff`, and `filtration_range`.
  `builders.bonds_from_adjacency(matrix, ids=cloud.ids)` is the explicit adapter.
  Alpha rejects bonds/distance cutoffs rather than changing its definition.
* Postprocessing: select built-in statistics or pass an ordered mapping of names
  to scalar functions to `summarize_spectrum`. `summarize_spectra` handles
  sequences of ordinary or persistent spectra. `spectral_variance`,
  `spectral_moment`, and `spectral_moments` expose population variance and raw
  moments; summary names `variance` and `moment_1` through `moment_4` are
  predefined. `spectral_energy` is `sum(abs(lambda))`, whereas
  `laplacian_energy` is the centered full-spectrum functional
  `sum(abs(lambda - mean(lambda)))` and always requires the complete spectrum.
  It equals classical graph Laplacian energy for a combinatorial graph L0; for
  Hyperdigraph L0/L1 and other operators it is an explicit generalization, not
  a claimed standard graph invariant. `graph_entropy` means spectral von
  Neumann entropy of
  a supplied complete graph L0, not degree entropy.
  Empty undefined statistics remain NaN by default; partial spectra require
  explicit opt-in. A complete empty spectrum represents a zero-dimensional
  operator/source chain group, with no eigenvalues and nullity zero. Passing
  `empty_operator_policy="zero"` explicitly encodes every requested predefined
  summary slot as zero for a core-confirmed structural case, so requested
  degree blocks remain finite and fixed-width for downstream feature matrices.
  Authorization requires the empty basis, zero nullity, optional `0 x 0`
  matrix/eigenvectors, and the standardized core receipt to agree. It does not
  insert a fake zero eigenvalue, alter the raw result, fill a nonempty all-zero
  operator, or conceal a partial/failed calculation. Core spectra record
  `operator_dimension`, `source_chain_dimension`, and `structural_absence`;
  summaries record the policy and the names of any zero-filled statistics.
  Defaults are positive-eigenvalue mean/max/min/std plus a separate full zero
  count. Each custom callable receives its own copy of the eigenvalues selected
  by the call-wide `positive_only` setting and must return one real scalar;
  custom mappings always run and therefore define their own empty-input value.
  Encode parameters in stable names such as `positive_q25_linear` or
  `heat_trace_t1`; callback internals are not inferred, and TopoKit performs no
  automatic imputation. `PersistenceVectorizer.fit` learns one dataset-wide finite bin range;
  `transform` returns 1D sample vectors. Fit on training data only for ML;
  `fit_scope="descriptive"` records fitting on a full descriptive collection.
  Shared/per-degree explicit edge arrays override min/max/step settings.
* Visualization: filled simplex faces, directional hyperedge ribbons, reusable
  dimensional blocks, and fixed-scale graph/simplicial/Hyperdigraph/interaction
  views return composable axes/figures. `StationaryStyle` supplies the shared
  grid-free palette, construction circles are opt-in, and stationary
  Hyperdigraph views deduplicate displayed directed segments without changing
  the object. `plot_betti_curves` displays interval counts queried from a
  supplied persistence result, while `plot_spectral_summary_series` displays
  aligned, already-computed spectral summaries (minimum, maximum, and mean by
  default, with centered Laplacian energy on a secondary axis). Neither helper
  recomputes topology, persistence, spectra, or summary statistics. Native
  object views support inclusive scale selection, stable-ID labels, and
  3D/projected views. SVG/PDF/PNG export is explicit. See the
  [visualization guide](markdown/VISUALIZATION.md).
* Workflows/ML: `workflows.analyze` composes persistence and ordinary snapshots.
  `workflows.analyze_stationary` returns fixed-scale GF(2) homology, requested
  complete ordinary Laplacian spectra, and named scalar summaries in one
  `StationaryResult` for tables and figures.
  `workflows.laplacian_series` returns scale-by-degree spectra; observation scales
  are separate from construction ranges. Interaction supports one shared or two
  monotone paired schedules; optional higher overlap simplices are validation
  annotations only. See [contracts](markdown/CONTRACTS_0_3.md) and
  [configured example](examples/configured_pipeline.py).
  `workflows.ml` accepts precomputed feature matrices. Its optional default is
  gradient-boosted trees (regression or classification); custom estimators are
  accepted. Install `.[ml]` only when needed. No automatic training or splitting.
  The fixed protein–ligand application produces C-order `(10,50,55)` float32
  tensors using native alpha births and public incremental Hyperdigraph L0.
  Its GBDT uses 10,000 trees, learning rate .002, depth seven, sqrt features,
  minimum split five, subsample .8 and seed zero. Training-only scaling and
  unscaled targets apply to all four models, with no early stopping or tuning.
  The v2020R1 model is fitted once and evaluated separately on all three CASF
  editions. ML libraries remain lazy application dependencies.

## Object views and building blocks

```python
from topokit import visualization as viz

# obj is an existing simplicial or hyperdigraph Topology with coordinates.
axis = viz.plot_topology(obj, view="projected", labels=True)
figure, axes = viz.plot_building_blocks("hyperdigraph", dimensions=(0, 1, 2, 3))
viz.save_figure(figure, "examples/output/hyperdigraph_blocks.svg")
```

Use `plot_simplex_block(q)` or `plot_hyperdigraph_block(q)` independently in
later figures. A q-hyperedge displays q consecutive arrows; no directed clique
or missing hyperedge is inferred. Object plots preserve the supplied data and
topology. Blocks above dimension three are explicitly labelled schematics.
Existing low-level `plot_simplices` and `plot_hyperedges` remain available with
improved visual defaults. [Options and compatibility](markdown/VISUALIZATION.md).

## Scientific scope and maintenance

Simplicial/interaction alpha uses squared-radius scales; hyperdigraph uses
distance scales. Equal assigned weights preserve both edge orientations.
Interaction has exactly two independently constructed factors with explicit
overlap and matching declared coordinate units. Static objects are analyzed as supplied. Ordinary Laplacian snapshots
are the default; two-scale persistent Laplacians, eigenvectors, and matrices are
opt-in. GF(2) homology and real spectral nullity are not universally equivalent.

Alpha geometry remains floating-point; high dimensions remain expensive.
Weighted alpha, path topology, periodic/domain recipes, distributed execution,
symmetry-expanded or periodic CIF construction, and protocol-paper case studies
are outside this implementation milestone. Molecular bond records are preserved
as reader metadata but are not silently selected as construction support.

Every software change must update this README and append a dated notice to
[markdown/CHANGELOG.md](markdown/CHANGELOG.md), including impact, validation,
and limitations. Keep detailed notes in [markdown/](markdown/README.md), not
inside mathematical source folders. See [architecture](markdown/ARCHITECTURE.md),
[notation](markdown/NOTATION.md), [contracts](markdown/CONTRACTS.md),
[migration notes](markdown/MIGRATION_0_2.md), and [roadmap](markdown/ROADMAP.md).
MIT code license; source/data attribution is in [NOTICE.md](NOTICE.md).

Version 0.3 adds explicit bonds/cutoffs and weight assignment, independently
scheduled interaction factors, fitted dataset-wide barcode bins, native JSON,
molecular and crystallographic readers, and scoped spectral statistics. See
[updated contracts](markdown/CONTRACTS_0_3.md).

Current final-workflow regression reports **1,114 passed and one skipped**,
including the installed mathematical suite, independent alpha-incidence L0
comparisons, fixed shape/statistics, null-channel semantics, manifest guards,
artifact tampering and model-setting contracts. Two existing construction
truncation warnings remain explicit. No installed mathematical source changed;
higher-dimensional and persistent-operator tests continue to pass. The new
engine reproduces an audited molecular r4 tensor bit-for-bit. AWS preflight,
production, distribution and complete artifact checks are tracked separately
in [validation](markdown/VALIDATION.md) and the dataset run status; this is not
a claim that the new multi-year production run has already completed.

Earlier detailed release checks, production metrics and source hashes remain
in the [archived original README](../datasets/protein_ligand_prediction/experiments/protocol_ablation/source_snapshots/legacy_workflows_20260916/original_package_documents/README.md)
and unchanged historical validation/change records.

An [algorithm and efficiency audit (2026-09-05)](markdown/PERFORMANCE_AUDIT_2026_09_05.md)
identifies exact-result improvements with reproducible example-only benchmarks:
reuse of embedded boundaries and spectral workspaces, requested-degree
interaction reduction, diagonal spectra, and batched exact alpha queries.
These are evaluated prototypes; installed algorithms and scientific defaults
remain unchanged. The existing 621-test suite was rechecked during the audit.

The [readability review](markdown/CODE_READABILITY_2026_09_05.md) improves local
variable names without changing APIs or algorithms. The
[pressure-test suite](examples/pressure_test/README.md) covers alpha/Rips,
Delaunay/complete hyperdigraph support, equal/distinct weights, all three
families, and ordinary/two-scale persistent spectra through degree two. It
records wall/CPU time, peak RSS, stage timings, cell counts, numerical outputs,
and a **600-second wall limit per case**. Degree 0, degree 1, and degree 2
have separate run matrices; the existing compact hyperdigraph H0/H1 workflow
has an additional, separately identified scaling suite. Upload to the configured
AWS workspace and execution are complete: **358 cases, 282 successes, and 76
explicit resource-guard rejections**, with all 282 numerical artifacts verified.
No measured case timed out or reported a numerical/application failure. See the
[AWS findings](examples/pressure_test/results/aws_2026-09-05/FINDINGS.md),
[complete results](examples/pressure_test/results/aws_2026-09-05/OVERVIEW.md), and
[combined CSV](examples/pressure_test/results/aws_2026-09-05/combined.csv). Current
machine-readable state: [AWS_STATUS.json](examples/pressure_test/AWS_STATUS.json).
The [local smoke report](examples/pressure_test/results/local_smoke_2026-09-05/RESULTS.md)
contains all 72 cases and verified numerical exports; the
[planned pressure matrix](examples/pressure_test/matrices/pressure.csv) records the canonical test plan.
