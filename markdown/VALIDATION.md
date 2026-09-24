# Current validation and open issues

## 2026-09-24 — Initial GitHub release preparation

The public source is attached to `git@github.com:ChenDdon/TopoKit.git`, preserving
the repository's initial commit and removing its placeholder `tempfile`.
The publication file set excludes caches, builds, generated example results,
pressure-test transfer/history outputs and machine-specific AWS status.
A targeted credential-pattern review found no credential material in the
publishable source, notebooks and documentation. Data provenance notices remain
part of the distribution.

Release automation requires all six Linux/macOS/Windows × Python 3.10/3.12
test jobs to pass before publication, then rebuilds and verifies the released
wheel/source archive and writes SHA256SUMS. The GitHub Actions run attached to
each tag is the authoritative platform/release result; the local validation
records below describe the earlier preparation checks. No new scientific or
numerical implementation change was made for the initial upload.

## 2026-09-23 — Public guides, sample data and generic feature export

The `topokit_dev` working copy was made before edits; `topokit` is the public
release candidate. The fixed FS-AN scientific recipe is unchanged. Current
validation for the new user workflows supersedes no historical model results.

- Full relevant suite: **1,279 passed, 23 skipped, 1 deselected**, two expected
  truncation warnings, 20.22 seconds. Python 3.12.2, NumPy 1.26.4, SciPy 1.13.1,
  pytest 7.4.4 on macOS arm64; numerical threads limited to one. Supervised
  TopoFormer tests were excluded and the old external-weight sequence inference
  test was explicitly deselected. The remaining sequence and architecture tests
  ran. Skips concern optional GUDHI and Linux-specific resource measurement.
- Ten diverse byte-preserved protein PDB / ligand MOL2 pairs produce finite
  float32 `(10,50,55)` tensors. The generic exporter completed all ten with
  no warnings, a `(10,27500)` matrix, ordered IDs, exact canonical schema and
  per-sample input/output hashes. An independent agent following only the skill
  and linked guides reproduced and validated the full contract in 7.41 seconds.
- Five new exporter tests check API equivalence, schema byte identity, existing
  prediction-loader compatibility, refusal to overwrite, manifest-relative paths,
  explicit failed rows, resource-limit failures and PDB warning visibility.
- Fourteen new sequence-profile/runner tests pass without external weights.
  Historical default/profile behavior and GBDT helpers remain compatible.
  All seven actual retained ESM-2/CPZ asset-file hashes match the published
  recipe; model weights were not loaded for this check. Full inference on the
  added CPZ route has not been rerun in this change.
- Both repository skill validators pass. All local entry-guide links resolve.
  All three tutorial files validate as nbformat4; only the fixed-object
  notebook's stale rerun filename changed, and no notebook was reexecuted.
- Unix-only pressure tests now skip before Unix imports on unsupported systems.
  Their macOS regression run passed 113 tests with one Linux-specific skip;
  simulated Windows collection gates were checked, but actual Windows and
  Linux CI runs have not occurred locally.

The ten example structures are for the requested local smoke tests. Their
SOURCE.json and NOTICE record unresolved upstream redistribution provenance;
local inclusion and hash checks do not establish public data permissions.
No source, datasets or model assets were published. See the current workflow
guides for user input and output contracts.

Distribution validation passed on staged artifacts: source archive 4,993,646
bytes and wheel 252,310 bytes. The archive checks verify all three notebooks,
environment.yml, molecular inputs, ten-pair hashes, recipes and skill files;
large generated pressure outputs are excluded. Wheels exclude structures,
notebooks and pretrained weights. Focused tests from the unpacked source
archive passed **89 tests with one optional skip**, using only that archive's
source tree. A fresh base-dependency environment installed the built wheel and
passed the isolated three-route core smoke plus all ten real molecular pairs,
with optional ML/DL/reference-package imports blocked. CI now performs these
installed-wheel checks outside the checkout's Python path. Local artifact
receipts are retained in the workspace release_audit folder.

Final synchronization check: public and development copies have identical
source, tests, workflows, recipes, notebooks and example bytes; only README and
root AGENTS role labels differ. The public-copy exporter was also invoked with
`python -I` using the newly installed wheel, completing all ten samples under
NumPy 2.5.3 / SciPy 1.18.1. The sequence CLI's actual `--validate-only` invocation
verified the seven local model assets and a prepared illustrative input with
21 ligand tokens, no omissions and no UNK symbols, without loading a model.


## 2026-09-23 — Final topology GBDT consensus

The final FS-AN GBDT uses the new hyperparameters and equal prediction averaging
of seeds 0/1/2. All 12 full 10,000-tree models, their fitted scalers and six final
ensemble evaluations were delivered and independently verified locally at 20:15 UTC.
The historical feature-ablation protocol and numerical results remain unchanged.

The 45 focused inference/workflow/architecture/CLI tests passed. They cover
arithmetic prediction averaging with nontrivial embedded scalers, preservation
of the legacy single-model route, wrong/missing/duplicate ensemble members,
changed scaler/model hashes, feature identity/layout, no silent imputation and
lazy dependency boundaries. Profile assertions distinguish final hyperparameters
from the original feature-selection parameters. Remote fixed-seed replay and
four full-data smoke fits passed before the twelve production fits.

The audit verifies all 19,066 original tensor/record pairs, train/test memberships,
raw matrix/target hashes and all twelve independently recomputed scaler arrays.
Remote model reloads reproduce per-seed predictions. The exact public API module
reproduces all 1,350 consensus predictions bitwise across the six evaluations.
The recorded runtime is Python3.12.14, NumPy2.5.2, SciPy1.18.1, sklearn1.9.0 and
joblib1.5.3; local delivery checks hashes/arrays without incompatible unpickling.

[Final results and saved models](../../datasets/protein_ligand_prediction/final/topology_gbdt/README.md)
include LOCAL_DELIVERY.json, INFERENCE_AUDIT.json, installed-wheel validation and
bounded retirement receipts. General-set consensus scores improve the original
settings on all three tests, but are slightly below the earlier transferred
single fit. Three-seed metric SD describes variability, not a confidence interval.

## 2026-09-23 — Launch fixed-A03 optimizer tuning,200-epoch cap

New isolated execution `topoformer_a03_tuning_20260923` preserves the completed
DL ablations and final A03 architecture. The new trainer supports actual batches
16/32/64 and separate primary/diagnostic normalization; model/library code is
copied byte-for-byte from the authenticated completed snapshot. No public API,
feature generation, GBDT or sequence behavior changed.

All42 package/architecture/production-loop tests passed on the GPU host,
including exact checkpoint-resume at all three batches and the diagnostic
training scope. Tests cover split leakage,200-epoch scheduler preservation,
baseline retention, conditional WD confirmation and23/25-fit controller budgets.
All nine actual batch/LR profiles passed on RTX3080 GPUs, including backward
updates, deterministic evaluation and coexisting saved-model reload memory.
All19066 tensor/record hashes were rechecked. Primary and diagnostic scaler
means/variances were checked independently; all18498 full-training IDs and
parent array identities were authenticated. The only prelaunch correction was
a missing `time` import in the profiling helper; its failed attempt is preserved.

Controller started17:58:06UTC, PID1748052/start114644368, on GPU0/1. Production
results and final/local-delivery audits are pending. Maximum epochs and cosine
horizon both200; no claim of improvement is made from the launch checks.


## 2026-09-23 — Final A03 protocol profile and consolidated DL ablations

The user selected A03 as the final protocol DL architecture. Its explicit JSON
profile matches the delivered width-768/12-layer/12-head/FF-3072 bundles exactly;
the workflow and dataset selection receipts are identical. Generic constructor
defaults remain the historical baseline, while the protocol documentation loads
the explicit A03 profile. No numerical model, feature, GBDT or sequence code changed.

ST-08 now collects both completed DL executions: 20 candidate configurations,
34 unique validation fits, six full fits and 24 CASF rows. The shared control
is counted once. Original delivered files and 410 frozen source/input files
were hash-checked before organization. All 131,166 aggregate validation/CASF
prediction rows were independently rescored; maximum metric difference was
9.99e-16. The canonical final folder uses relative links to the original three
A03 predictors and scalers, preserving original paths and provenance.

Nine proposed LR/batch configs passed fixed-architecture/feature/preprocessing
and exact changed-field checks. The proposed 16,648/1,850 split is disjoint,
excludes every CASF ID and preserves a paired 14,799-training/1,850-validation/
1,849-reserved diagnostic. Config/split hashes, 25-fit maximum and 46 local
documentation links passed checks. No training or GPU profiling was launched;
batch-16/64 resource checks and all new performance results remain future work.
No improvement over existing scores is claimed. See
[organization verification](../../datasets/protein_ligand_prediction/experiments/protocol_ablation/studies/ST-08/VERIFICATION.json)
and [tuning proposal](../../datasets/protein_ligand_prediction/experiments/protocol_ablation/studies/ST-08/tuning_a03_20260923/README.md).

## 2026-09-23 — Final CPZ sequence selection and previous-setting archive

User selected FS-AU NMI-inspired sqrt as the final ESM-2 + CPZ configuration.
The [canonical saved profile](../../datasets/protein_ligand_prediction/final/sequence_esm2_cpz/README.md)
contains exact resolved parameters and links to all four trained model/scaler
bundles, 19,066 embeddings, six evaluations and predictions. Independent SHA256
checks verified all 37 final audited files for each of FS-AU and FS-AW and both
complete frozen source/input locks. Parameters match all four retained FS-AU
model receipts. FS-AW is archived in place with no deletions; FS-AV's intentional
model pruning remains in force. No model was loaded in incompatible local
sklearn, no fit was repeated and no numerical API changed. See
[selection audit](../../datasets/protein_ligand_prediction/final/sequence_esm2_cpz/SELECTION_AUDIT.json)
for final link, parameter, manifest and documentation checks. The saved final
sequence profile supersedes historical selection status, while FS-AN remains
the separate topology choice.

## 2026-09-23 — Article-informed TopoFormer results delivered

All 17 new fits completed: eight screens, six confirmations and three full
general-v2020R1 refits. The remote final audit passed at 05:30:18 UTC; independent
local delivery passed at 05:59:09 UTC. Verified 188 delivered files (4.19 GB),
all 17 weight/scaler bundles, 206 frozen follow-up files, 204 frozen parent files
and 67 pinned parent artifacts. Locally recomputed 51,786 new and 11,097 reused
control validation predictions, plus 2,700 CASF prediction rows per study.
Memberships, labels, both validation rankings, seed means/sample SD, arithmetic
prediction ensembles and fixed full-refit epochs/schedules passed. All scalers
exactly match authenticated training-only references. The prior audit of 19,066
raw tensors and independently recomputed scalers was authenticated and reused;
normalized caches were verified remotely and not downloaded. Weight hashes were
verified locally without Torch deserialization. No numerical code changed, so
the existing 36 preflight tests and eight GPU profiles were not repeated.

A03 (width 768, 12 layers/heads) led three-seed validation RMSE and PCC. Full
refits used 102/104/92 epochs. Ensemble RMSE/PCC: CASF-2007 1.351085/0.831628;
CASF-2013 1.450729/0.771897; CASF-2016 1.206813/0.840541. A03 improves CASF-2007;
the preserved six-layer control remains better on CASF-2013/2016. These are
reused benchmarks, and no significance or published-pretraining reproduction
is claimed. No default or original model was replaced. See the
[final report](../../datasets/protein_ligand_prediction/experiments/topoformer_article_followup_20260922/reports/FINAL_REPORT.md)
and [local audit](../../datasets/protein_ligand_prediction/experiments/topoformer_article_followup_20260922/receipts/LOCAL_DELIVERY.json).
Historical pending-status entries below describe the state at their recorded dates.

## 2026-09-22 — Launch the article-informed TopoFormer follow-up

A separate study, `topoformer_article_followup_20260922`, was authorized and
launched on two RTX3080 GPUs at14:37:56 UTC. All36 GPU-host tests passed, including
exact interrupted/resumed production fitting, model/bundle/tokenization and
architecture checks, registered config hashes, and unchanged numerical functions.
All8 batch32 float32 GPU profiles passed; the largest152,772,609-parameter model
reserved3.73 GiB during the profile. Both initial production fits completed their
first epochs. No completed new fit or winning configuration is claimed yet.

The new lock pins206 files; all204 parent frozen files and67 referenced control
artifacts passed local hash verification. All19066 remote features and reused
cache/scaler identities were authenticated. A profiler logging timestamp collision
was corrected before any production fit; the original attempt is archived.
Numerical trainer, fixed input contract, candidate hashes and package code were
unchanged. The existing hourly heartbeat now follows the separate execution.
See [execution receipts](../../datasets/protein_ligand_prediction/experiments/topoformer_article_followup_20260922/receipts/LOCAL_LAUNCH_VERIFICATION.json).
Full fits, final comparisons and output delivery remain pending.

## 2026-09-22 — Article-informed TopoFormer extension settings

Eight additional JSON configurations passed dimension/head/dropout and exact
field-difference checks against the selected six-layer control. Fixed features,
pooling, positions, batch32, dropout0.1 and the selection split are preserved.
SupplementS2/TablesS1-S2 were inspected as text and rendered pages; pinned source
hashes match. The printed4096 attention heads at width1024 is incompatible with
standard multihead attention; the proposed1024-wide candidate uses16 heads as an
explicit assumption. Paper learning rates and source-scaled rates are distinguished.
All204 frozen completed-study hashes still match. No new model was trained;
parameter counts are analytic and GPU memory checks remain a launch prerequisite.

## 2026-09-22 — Completed TopoFormer study and local delivery

All23 fits finished:12 initial screens, eight confirmation fits and three full
general-v2020R1 refits. The registered rule admitted no joint configurations.
Six layers/width256/eight heads led both separate validation RMSE/PCC rankings;
selection used mean RMSE. All nine per-seed CASF rows and three ensembles passed
the remote final audit. Local delivery verified232 artifact hashes, all23 model
bundles,204 frozen source/input hashes,19,066 feature record/tensor pairs,
73,980 validation prediction rows and2,700 CASF prediction rows. Membership,
labels, metrics, sample SD and arithmetic prediction ensembles passed checks.
Selection/full scalers recomputed from14,799/18,498 raw local tensors with zero
mean/scale differences. Full-refit epochs335/186/265, training IDs and the original
500-epoch scheduler horizon were verified. No numerical code or dependencies changed.

Ensemble RMSE/PCC: CASF2007 1.405379/0.816332; CASF2013 1.441888/0.775508;
CASF2016 1.190535/0.849187. These are three-model ensemble scores; individual
seeds and sample means/SD are separate in the
[final report](../../datasets/protein_ligand_prediction/experiments/topoformer_general_v2020R1_20260922/reports/FINAL_REPORT.md).
The completed GBDT baseline and frozen study snapshot are preserved. Earlier
pending-status entries below describe historical launch/preparation checks.

## 2026-09-22 — Compact supervised TopoFormer

GPU environment tests: **32 passed** (new model/bundle tests, architecture and
production-loop checks). A simulated interruption after epoch2 resumes to
bitwise-identical selected weights and identical validation predictions versus
an uninterrupted four-epoch run. Tiny synthetic data overfits; raw-feature
prediction matches explicit training-only scaling exactly. Token order and
width-first positions are independently checked. Baseline has3,366,913 trainable
parameters; blocks initialize independently. Bad schema and corrupted weights
are rejected. Local dependency-light checks: **26 passed,3 optional torch skips**.

Miniconda and a separate topokit environment on lambda-1 provide PyTorch2.7.1
with CUDA11.8, compatible with its NVIDIA535 driver. Both RTX3080 devices passed
a CUDA matrix calculation. All19,066 feature record/tensor pairs passed authentication. All12 candidates
and the largest candidate on the second GPU passed forward/backward smoke checks.
Another60 higher-dimensional Laplacian tests passed. Wheel contents and isolated
installed imports passed with torch/sklearn/GUDHI blocked. Production launched
at2026-09-22T04:23:29Z, with baseline and width128 observed training concurrently.
Frozen source SHA256: bee65a4efbb22c5a7bf5a461e3c279142fa22fcea0e0f108d8081e7a3051afd7
(SOURCE_LOCK.json). Current documentation may evolve; the launched source
snapshot remains unchanged. Detailed receipts are in the revision3 study.
No completed DL benchmark scores are claimed by these tests.


## 2026-09-21 — Unified protocol catalogue and installed selected application

The reorganized package passes **1,243 tests, 23 optional skips**, including
architecture and higher-dimensional checks. Run with `PYTHONPATH=<package>/src`
so cold subprocess checks see the checkout. No mathematical core, builder,
reader or higher-dimensional implementation changed. The new installed FS-AN
workflow reproduces the frozen 15 Å encoder and the stored `10gs` float32 tensor
bitwise; channel presence/counts and reference schemas match exactly. The guarded
prediction/load/read functions are AST-identical to the verified delivery helper;
only its streaming hash helper changed for Python 3.10 compatibility.

Clean temporary wheel/sdist builds pass distribution checks. An isolated wheel
import and two-point radius-boundary calculation pass while GUDHI, sklearn,
joblib and torch imports are blocked. The wheel contains six modules, the fixed
feature encoder, inference API and two small JSON profiles; no datasets, weights,
research launcher or historical ablation source is installed.

The external protocol catalogue verifies 42 feature records, ten study groups,
123 evaluation rows and 369 independently recomputed metrics over 50,685 saved
prediction rows. Maximum metric differences are below 5e-13 (tolerance 1e-12).
All 322 referenced input-file hashes, 215 relocated source hashes and 40 model
bundle artifact hashes match. Duplicate controls and early protocols remain
separate; no scores were averaged or reassigned. Current catalogue/package links
were checked. See `datasets/protein_ligand_prediction/experiments/protocol_ablation/`
for ORGANIZATION_AUDIT, PACKAGE_FEATURE_CONFORMANCE, PACKAGING_AUDIT and COMPLETION.

Only the selected feature/prediction API is working; the TopoFormer namespace
remains planned and untrained. No remote jobs or numerical study reruns occurred.
Weights are a separate trusted companion bundle, not embedded in the wheel.

## 2026-09-21 — Representative selection and previous-strategy preservation

The representative schema exactly matches the completed Cornell 15 Å recipe,
`hpcc-alpha15-cb4760366e92bd2e`; the selection receipt is identical in the
workflow and dataset folders. Three active aliases resolve to the audited
Cornell feature/model/result stores. The unchanged 20 Å schema remains the
ablation reference. Verified all 26 archive-copy hashes, all 26 original model
artifacts and all 326 frozen numerical source files. A fresh call through
`representative_strategy.py` reproduces the saved 10gs float32 tensor, channel
presence and atom counts exactly. Focused final/alpha15/controller checks:
**26 passed, 1 optional GUDHI skip**. Proof is saved in
`datasets/protein_ligand_prediction/experiments/cornell_alpha15_20260921/REPRESENTATIVE_SELECTION_VALIDATION.json`.
No new feature generation/training run or numerical-core change was needed.
The 20 Å comparison retains higher PCC on CASF-2013/2016; the user selected
15 Å for its smaller protein field and similar overall performance. No
hardware-independent speedup is inferred from different-platform runtimes.

## 2026-09-21 — Cornell alpha15 study complete; local delivery verified

Remote final audit passed at 19:10:00 UTC for all 19,066 features, four full
models and six evaluations. All scaler mean/variance/scale values matched
independent fits using training rows only; reloaded model predictions and
10,000-tree settings matched. Every production output was copied locally.
Independent local verification passed at 19:15:44 UTC for 19,066 tensor hashes,
19,066 record hashes, all tensor contracts, 26 model artifacts, nine manifests,
326 frozen source files, sample order, labels and all 1,350 prediction rows.
RMSE/PCC/MAE were recomputed and matched within 1e-12. Models were not unpickled
under the different local sklearn runtime; their bytes match the remotely
verified artifacts. Zero feature failures or dropped samples. See
`datasets/protein_ligand_prediction/experiments/cornell_alpha15_20260921/LOCAL_VERIFICATION.json`
and `results/RESULTS.md` in the same experiment folder. The new local verification
script only audits delivered outputs; the frozen scientific source is unchanged.
The 15 Å / shifted-grid comparison improves CASF-2007 but does not improve PCC
on CASF-2013/2016. Earlier pending-state entries below are historical.

## 2026-09-21 — Cornell alpha15 production features passed

**September 21, 18:09 UTC update:** All **19,066/19,066 features** are
complete, with zero failures. The complete feature audit passed at **17:55:54
UTC**, verifying tensor/record hashes, shape, finiteness, channel counts and
input identities. Training began automatically after that audit. The
refined-2007 and refined-2013 fits are complete and their saved artifact hashes
were checked; refined-2016 and updated official general v2020R1 are training
in parallel. Final model/scaler/prediction audits, the six-evaluation comparison
and local feature/model delivery remain pending. The copied audit receipt is
checksum-verified; this does not yet mean tensors or models were copied locally.

## 2026-09-21 — Cornell alpha15 controller started

The subsequent explicit user request authorized execution. At 15:41:36 UTC,
the frozen 326-file source and both launcher hashes were rechecked, no study
controller was running, and approximately 30 GiB RAM was available. Detached
controller PID 11442 and its preflight child were observed alive. This begins
full input hashing, molecular smoke checks and ML smoke checks; it does not
claim these checks or production complete. The controller automatically gates
16-worker feature generation, the feature audit, four fits, six evaluations
and final audit. See `experiments/cornell_alpha15_20260921/CORNELL_LAUNCH.json`
under the protein-ligand dataset. The preparation-only state below is historical.
Preflight **passed at 15:43:07 UTC**: all 38,132 structure-file hashes agree,
all 74 molecular smoke samples succeeded, four five-tree ML smoke fits passed,
and the 49 shared radii of the molecular control match bitwise. At **15:45:17
UTC**, 290/19,066 production feature/record pairs were complete (18,776 remain),
with zero failures and **16 live workers across four shards**. These are a
timestamped snapshot, not a completion claim. The existing continuation monitor
now follows Cornell only; the remote controller starts ML automatically after
the complete feature audit, independently of the monitor.


## 2026-09-21 — Cornell alpha15 preparation only

All 264 MSU job elements were confirmed canceled and their monitor paused.
The Cornell controller passed eight local checks covering read-only status
and planning, all 256 shards/four fits, CPU limits, failure gating and launcher
integrity. All 326 copied scientific source hashes match the original snapshot;
its original 1,162-test result is historical validation, not a new run.
Cornell preparation verified source identity, the presence and sizes of all
38,132 structures, all nine label/provenance hashes, exact runtime versions
and the read-only launcher. No molecule was featurized and no model was fitted.
Full structure-content hashing and scientific preflight are deferred. Current
proof is in `experiments/cornell_alpha15_20260921/CORNELL_PREPARATION.json`
under the protein-ligand dataset.

## 2026-09-20 — H0 protein–ligand smoke passed; production launched

The user chose H0 only, bars covering entire bins, and a maximum filtration
parameter of 5. The frozen application declares alpha radius in Å, squared
builder cutoff 25 Å², no separate pair-distance cap, and 100 bins ending at 5.
The new separate workflow saves 55 H0 barcodes and a `(55,100)` feature tensor
per sample. All 4,414 input complexes and the three complete matched-year
manifest pairs are available on Cornell WSL.

Local focused tests: **50 passed**. Remote activated `topokit` environment:
**274 passed** across new barcode/application tests, architecture, existing
spectral/ML tests and hyperdigraph persistence/homology/Laplacian regressions.
New application checks include random all-channel comparisons to full alpha
and independent graph components, an analytic obtuse-triangle alpha birth,
H-anchored inclusive cropping before exact deduplication, mixed versus explicit
Null edge policies, cutoff deaths and unbounded survivors. Model tests verify
training-only scaling and checksummed resume/reload. The native alpha and
higher-dimensional code are unchanged. A local 1a30 pilot completed in 0.476 s
with 9,243 bars; this is one timing, not a dataset-wide speed claim.

The end-to-end remote smoke passed at 18:49:27 UTC: 48 real complexes,
2,100 independent bin checks, and six short model/scaler/reload audits, with
zero failures. All 84 source hashes agree locally, on Cornell and in the
frozen receipt. Full production launched at 18:50:43 UTC (PID 9243), using
16 feature workers. All 4,414 feature/barcode pairs passed input/hash and
direct barcode-predicate audits, with zero failures; the 2,100 independent
real-bin checks also passed. All six complete model fits started at 19:00:14
UTC and were verified active. Metrics await full completion and model audit. Resume requires the frozen source and manifests.
Study identity: `edc0be35bd423140a9d0c6ed4e29d512eebbd05dea4f2f20d635603118623a7f`.

## 2026-09-20 — Explicit barcode bin counts; Cornell recipe pending

The new `postprocessing.barcode_bin_counts` passes **215 focused tests**
together with existing barcode, spectral, hyperdigraph, ML-schema and
architecture regressions. Independent bin-by-bin predicates check randomized
H0/H1/H2 intervals, nonuniform grids, partial-bin overlap, exact endpoints,
zero-length bars, infinite deaths and interval clipping. Real hyperdigraph
H0/H1 barcodes match independent fixed-scale homology on a triangle that fills
and an isolated vertex. Serialization preserves the vector/schema; ML stacking
rejects mismatched bin definitions. Bounds, malformed intervals and allocation
budgets fail explicitly. No core, geometry or spectral algorithms changed.

Command: `/tmp/topokit-incremental-l0-venv-20260915/bin/python -m pytest -q`
with `tests/test_barcode_curves.py`, `tests/test_barcode_extensions.py`,
`tests/test_architecture.py`, `tests/test_postprocessing_spectral_features.py`,
`tests/test_hyperdigraph.py`, `tests/test_ml_audit.py`, and legacy hyperdigraph
`test_persistence.py` / `test_homology.py` (215 passed in 3.65 s).

The Cornell input preflight found all structures for 4,414 unique complexes
across the existing refined/CASF 2007, 2013 and 2016 manifests. This is input
availability, not generated features. Homology degree, the bar-counting rule,
and the alpha cutoff coordinate remain unresolved; no production features or
models for the new 5,500-feature strategy have started. The final user rule
allows within-component alpha edges in explicit Null-side channels only.
See the separate `workflows/protein_ligand_persistent_homology/README.md`.


## 2026-09-20 — HPCC 15 Å alpha experiment prepared

The new repository-level hpcc_15a receipt changes only crop radius and alpha
sampling window. Native geometry, higher-dimensional code, source structures,
labels and ML settings are unchanged. Local regression reports 1,162 passed and
24 optional skips, including the scheduler and source-integrity gates. Tests cover inclusive 15 Å selection including H crop anchors,
radius² endpoints, an independent obtuse-triangle incidence oracle, exact
coordinate uniqueness, 49 shared scales and frozen baseline definitions.
A four-complex real molecular pilot (10gs, 1tps, 3tiw, 5etx) gives bitwise
agreement at all 49 common scales on identical selected clouds. Scheduler
checks verify dependency gates, group handling, resource limits and duplicate
submission prevention. The actual HPCC environment/smoke/full run has not
executed while a campus-wide power outage keeps jobs pending. No prediction
scores or runtime speedup are claimed. See the hpcc_15a workflow README and
experiments/hpcc_alpha15_20260920 receipts for current execution status.

## 2026-09-18 — Cornell VR atom-deletion study prepared

The remote `topokit` environment passes **1,217 tests and 240 subtests**, with
**23 optional-dependency skips**. The new optional core deletion API matches
independent induced-incidence matrices, eigenspectra and reconstructed
reference objects across random filtrations, reciprocal edges, mixed vertex
IDs, late singletons, isolates and empty operators. The feature suite compares
complete small tensors to independent D−A eigensolves and checks crop bounds,
exact keep-first deduplication, missing elements, independent deletion degrees,
population moments, raw extrema, aggregation order and cache reuse.
Local focused core/architecture/workflow checks pass **133 tests**.

Model tests check separate train-only scaling against a deliberately shifted
test distribution, complete model reload, immutable feature identities and
explicit undefined PCC for constant predictions. Historical workflows remain
unchanged apart from navigation documentation. Full higher-dimensional,
persistent, alpha and architecture regressions pass; GUDHI remains optional.
The new workflow is in the default pytest collection.

A real 1a30 pilot took **156.57 s** with repeated native-object reconstruction
and **81.81 s** with the new exact matrix-deletion reuse (one numerical thread,
about 118 MiB peak RSS, largest channel 399 atoms). This is one pilot rather
than a dataset-wide speed claim. The end-to-end smoke completed all 39 real
complexes, 45 independent molecular D−A slice checks and four short model
fits with successful independent scaler/reload audits (22:16:39 UTC).
Production launched at **22:17:33 UTC**, controller PID 4358, with 16 confirmed
active workers, each using one numerical thread. All 4,324 complexes passed
the input-file/hash preflight; all 83 frozen Python source hashes agree
between local and remote. Full feature/model completion and scores are
pending; authoritative status and independent real-feature/model
audits reside under remote
`datasets/protein_ligand_prediction_version2/experiments/study/`.

The running production study covers 4,324 unique complexes and four complete
fits (1,105/195 and 3,772/285 train/test; 10,000 and 30,000 trees). Frozen input,
source, schema and environment hashes guard every resume. No full-run result
is implied by passing the regression or smoke tests. Results stay remote.


## 2026-09-17 — ML feature comparison complete

All 12 new full-manifest fits and 18 evaluations are complete. Four existing
train-only standardized baseline models contribute six reused evaluations
(24 comparison rows). Full AWS model reload, independently fitted scaler,
separate matrix transform, sample membership/label/exclusion, prediction and
metric checks pass. Full local independent matrix/membership/metric audit
passes without incompatible sklearn model unpickling. All 78 model artifacts
and nine result/provenance files match fresh AWS checksums; all 388 frozen
Python sources match on both hosts. The full AWS audit passed at 18:58:25 UTC.
General-v2020R1 raw-extrema CASF-2016 RMSE is 1.2623574096529802 and PCC
0.8556211592220472; refined-2016 still favors baseline on both metrics.

Separate RMSE/PCC tables, full-precision heatmap matrices and completion
receipts are in experiments/ml_feature_comparison_20260917. All outputs are
local and retained on AWS. No geometry, stored features, labels, memberships,
GBDT settings, higher-dimensional source or fixed strategy changed. Earlier
partial/launch entries below are historical. This finalization changes only
reports and current documentation; earlier 21 local passes/one optional skip
and 22 AWS passes remain the relevant source regression checks.


## 2026-09-17 — Feature-only ML comparisons prepared

New repository-level comparison and independent audit scripts reuse the fixed
feature store. Local focused regression passes **21 tests, one optional-GUDHI
test skipped**. Checks cover exact H channel positions/flattening, positive
raw-extrema reconstruction to stored precision, empty spectra, unchanged
statistics, input immutability, combined views and independent batch transforms.
Existing fixed-workflow tests pass. No installed reader, builder, core or
feature generator changed. AWS focused tests pass22/22; all19,066 source tensors and all12 ML smoke fits
pass. All388 uploaded Python files match local sources. Production launched
at17:40:09 UTC with12 workers; production scores are not available yet.
See workflows/protein_ligand_prediction/ML_FEATURE_COMPARISON.md.

## 2026-09-17 — Full native benchmark complete and archived cleanly

All 19,066 features, four full models and six evaluations are complete,
copied locally and audited. General v2020R1 used all 18,498 rows and finished
at 16:44:18 UTC; final AWS model-reload/scaler/prediction audit passed at
16:44:59 UTC. CASF-2007/2013/2016 general RMSE is
1.5162041081237227 / 1.4785988416448097 / 1.27297726094721 and PCC is
0.804612239880753 / 0.7889076160720576 / 0.8522775980674216.
All 42 model/result artifact hashes match local copies. Local general-model
membership, label, exclusion and recomputed metric checks pass; no local
model unpickling across sklearn versions was performed. The complete AWS
feature/input inventories match the already-passed local feature audit.

The three refined predictions/metrics reproduce their historical results.
All 4,057 r4 tensors match exactly. Scientific geometry, features and ML
settings were not changed during finalization. Superseded production folders
were moved under dataset archive/2026-09-17_native_finalization with verified
before/after file hashes, original metadata intact and zero deleted files.
One final store remains under each active features/models/results directory.
COMPLETION.json, FINAL_TRANSFER_AUDIT.json and aws_audit.json in the current
experiment record full completion. All earlier progress entries below are
historical snapshots, superseded by this complete result.


## 2026-09-17 15:46 UTC — Three native refined models verified

All three refined fits completed with the full 1,105/2,764/3,772 training
rows and 195/195/285 CASF tests. AWS independent model reload, training scaler
and prediction checks pass. All 27 model/result artifact hashes match local
copies; local memberships, labels, exclusions and recomputed metrics pass.
Prediction CSVs and metrics exactly reproduce historical refined outputs.
The 18,498-row v2020R1 general model remains running, so the full benchmark
is incomplete. Evidence is in current experiment REFINED_COMPLETION.json,
partial_models_2007_2013_aws_audit.json, partial_models_2016_aws_audit.json,
local_refined_models_audit.json and local_refined_transfer_audit.json.

## 2026-09-17 15:35 UTC — Complete native feature store verified locally

All **19,066** required unique-atom feature tensors are complete, copied
locally and independently audited. AWS and local tensor/input inventories
agree, covering **38,132** input files and **4,057** bitwise-identical r4
controls. All **19,068** record/schema/compatibility file hashes match AWS.
The current store preserves 19,063 native-1.1.0 outputs and original records;
three cases (1tps, 3tiw, 5etx) were repaired under frozen native 1.1.1 and
independently match GUDHI tensors and all sampled alpha-filtration supports.
The source-compatibility inventory pins both actual recipe identities.
No source changes were made for this completion/transfer update.

The complete four-fit production stage is running with four processes.
Model completion, six evaluations, model reload/scaler/prediction audits
and final result transfers are still pending. Feature-only audits do not
claim model completion. Evidence is under
`datasets/protein_ligand_prediction/experiments/final_alpha_l0_unique_repair/`
in `feature_audit.json`, `local_feature_audit.json`, `local_transfer_audit.json`
and `FEATURE_COMPLETION.json`. Earlier launch snapshots below are historical.


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


## 2026-09-17 — Exact unique coordinates before alpha construction

The user explicitly chose to keep one atom per exact coordinate. Both alpha
backends now default to keep-first (`duplicates="merge"`); strict rejection
remains available. `PointCloud.unique_coordinates()` preserves first-row IDs,
weights and documented aligned reader metadata, with an original-to-retained
mapping. Molecular workflow 1.2.0 / schema 3 uses native engine 1.1.0 and recipe
`final-alpha-l0-65fa5d3acbf56ae0`. It deduplicates the combined cropped protein
then ligand atom list before element channels and writes removal receipts.
Near-coincident coordinates are not merged. Original files and manifests stay
unchanged. Duplicate atoms are removed from the operator, not lifted back.

Focused tests: **92 passed**. Full local suite: **1,150 passed, one skipped**,
with two existing truncation warnings. New tests cover first-row attributes,
conflicting element/component labels, signed zero, empty/all-duplicate input,
near coordinates, resource limits after deduplication, and exact L0/L1/L2 and
persistent-Laplacian matrix equality against manually unique input. Actual
1awf, 1exw and 1i8j input checks confirm recorded cross-component removals.
AWS full regression: **1,151 passed**, 240 subtests, three warnings. The
independent complete input scan passes **19,066 complexes**, confirming exactly
**208 affected complexes and 399 removed atom rows**. All other selected atoms
are unchanged. A no-GUDHI local runtime executes the unique-atom feature path.
AWS preflight passes all 297 samples and four smoke fits; all 208 former
failures are generated, copied locally and audited. Their 5287 channels and
11761438 edges match the optional GUDHI oracle at all 50 scales (maximum
squared-birth difference 1.954e-14). Full production launched with 64 workers
at 2026-09-17 12:06:28 UTC. Evidence and new
production use `experiments/final_alpha_l0_unique/`.
Historical GUDHI outputs and the earlier native migration audit remain intact.
The full new benchmark is not yet complete.



## 2026-09-17 — Native alpha, no GUDHI production requirement

The GUDHI-free native alpha implementation and workflow migration are complete.
Local regression: **1,134 passed, one skip**, two existing truncation warnings.
AWS regression: **1,135 passed**, 240 subtests, three warnings. The installed
wheel works with GUDHI imports blocked and matches all 76 current Python source
files. Real molecular feature generation and checksum-verified resume pass in
a local runtime where GUDHI is not installed.

A 64-worker AWS comparison covers **all 4,057 r4 complexes**, **106,086 selected
channels** and **228,691,848 edges**. Every edge identity and inclusion at all
50 observation scales matches GUDHI 3.13.0 exact geometry. Maximum squared-birth
roundoff is 1.883e-13 Å². All **92 final full-tensor checks are bitwise identical**
to the frozen AWS features, including the previous 89 preflight cases and three
additional large near-cospherical cases. Local cavity repair was used in 383
channels across the complete geometry comparison.

Summed native construction time is 13,279.72 worker-seconds versus 26,538.41 for
GUDHI exact geometry (**1.998×**); paired audit wall time is 645.99 seconds.
This measures geometry under parallel workload, not end-to-end feature/ML speed.
The eigensolver, summaries and all mathematical cores remain unchanged. Tests
cover full 2D/3D/4D geometry, thin full-dimensional clouds, non-Gabriel coface
births, cospherical/local repair, resource/duplicate policies, and direct
L0/L1/L2 matrix equality against the exact backend.

Local cross-platform checking gives 81/89 byte-identical saved AWS tensors;
the remaining eight differ by at most 3.553e-16 in summaries. All eight are
bitwise identical when recomputed with native and historical GUDHI engines on
the same local machine. Original strict cross-platform failures remain archived;
a separate reconciliation records the explicit 1e-12 absolute tolerance.
No feature values were changed to obtain agreement.

Workflow 1.1.0 / schema 2 has recipe `final-alpha-l0-8406e9c81e1f2f84`.
Both historical AWS molecular audits match that migration schema and package
source SHA256 `c8212fe51ade88079fc9f2033c17caf470518ce291e48392f939b78478e50e1c`.
Existing GUDHI results retain their original provenance. The native implementation
is floating-point geometry with rational repair, not a universal exact-predicate
guarantee. At that migration stage duplicate policy and the 208 general-only
cases were unchanged; the later user-authorized policy is documented above.

[Full validation report and receipts](../../datasets/protein_ligand_prediction/experiments/native_alpha_validation_20260917/REPORT.md)
and [algorithm/precision contracts](NATIVE_ALPHA.md).

Version 0.3.0; latest local verification 2026-09-16. This is an implementation
milestone, not a complete publication-ready molecular/materials product.

## 2026-09-16 final PCC strategy and consolidated workflow

The user fixed m2+m3+m5+r4 and requested complete refined 2007/2013/2016
and updated official v2020R1 general benchmarks. The active application is
`workflows/protein_ligand_prediction/`; old code, tests, labels, results and
model artifacts are archived with relocation receipts. The active dataset
has seven hash-pinned manifests and 19,066 unique selected structure pairs.
Training counts are 1,105/2,764/3,772 and 18,498 (v2020R1), with 195/195/285
CASF tests and the declared zero-overlap checks. No membership omissions.

Local regression: **1,114 passed, one skipped**, two existing truncation
warnings, 22.00 seconds. This includes the installed higher-dimensional and
persistent routes, architecture checks and seven new application tests.
Independent GUDHI/incidence spectra, null-channel/scale boundaries, summary
arithmetic, source/manifest/tensor contracts and no-early-stopping settings
pass. A freshly calculated 1a30 tensor is bitwise equal to the archived r4
float32 tensor. No installed mathematical source was edited.

AWS regression, molecular recovery preflight, four five-tree smoke fits,
full feature generation and four fixed 10,000-tree fits have separate status
receipts. A full feature audit requires all 19,066 tensors and equality of
all 4,057 historical r4 controls. Full-study completion requires all four
models and six evaluations; audited refined-only completion is recorded below.
See the [current run](../../datasets/protein_ligand_prediction/experiments/final_alpha_l0/RUN_STATUS.md).

AWS follow-through: 1,115 tests passed overall; the isolated no-GUDHI
subprocess check initially lacked checkout PYTHONPATH and passed after that
invocation was corrected. Molecular preflight passed all 89 samples,
including 66 bitwise matches to frozen r4; four five-tree smoke fits and
reload checks passed. Full generation launched with 64 active processes.
Fresh wheel/sdist inventory checks pass, with the active application only.

A separate exhaustive coordinate scan found 208 cross-component duplicate
coordinate cases, all in v2020R1 training, and none in refined or CASF samples.
Strict alpha rejects them. A proposed atom-preserving coincident-site rule
awaits the user's choice; no production policy or input has been changed.
Full general-set training remains blocked until all 18,498 rows are valid.
The strict generation pass ended at 16:31 UTC with **18,858 successful or
verified-skipped features** and those exact 208 failures. A separate scoped
audit verifies all available tensors and input hashes, all fixed manifests,
source identity and **4,057 bitwise-equal historical r4 tensors**. It explicitly
does not declare full-run completion. The immutable 347-file source inventory
matches. Three complete refined production fits launched in parallel at
16:48 UTC; general fitting remains blocked. All 18,858 tensors and records
were copied locally; the 16:51 UTC local audit passes with identical tensor
and input hash inventories, sources and 4,057 reproduced r4 controls.
All three refined models completed by **17:03 UTC**, with no failed model
jobs. Independent AWS model reload/scaler/prediction audits pass, and all
artifacts are copied locally and audited without cross-version unpickling.
Fresh transfer verification passes 47 files, including 29 model/result files.
CASF 2007/2013/2016 RMSE is **1.486219 / 1.492785 / 1.291531** and PCC is
**0.809891 / 0.788311 / 0.842641**. All 285 new refined-2016 predictions exactly
match a reload of the archived r4 model; saved RMSE/PCC/MAE are exactly equal.
The historical CSV agrees at its stored 12-significant-digit precision.
Local metric recomputation differs by at most 3.33e-16 in PCC (tolerance
1e-12); RMSE/MAE match exactly. The separate experiment auditor records this
roundoff, artifact identities and the explicit `full_run_complete: false`.
[Results and audit receipts](../../datasets/protein_ligand_prediction/results/final_alpha_l0/RESULTS.md).
The 208 general-only cases still await a user decision; no new geometry rule
or sample omission was introduced. All historical scientific results and
models are preserved; old generated arrays were pruned with checksum ledgers
(24.26 GB local, 24.24 GB AWS), retaining all original r4 controls for audit.

## 2026-09-16 completed alpha follow-up: crop, grid and internal edges

The AWS run and final audit completed at **05:46:42 UTC**: **4,057 successful
complexes, 28,399 tensors, 14 production models and zero failed samples**.
Feature generation used 64 workers; each training stage used seven parallel
fits. The fixed 2016 membership, 27,500 features and v2 10,000-tree estimator
settings remained unchanged throughout production.

The independent local audit passes all **28,399 tensors**, **8,114 structure
input hashes**, seven recipe/source identities and **14 model artifact sets**.
All **77 model/report files** match fresh AWS SHA-256 hashes. Every one of the
4,057 reference tensors is byte-identical to the completed r4 study, and its
validation/CASF metrics reproduce exactly. Matched feature checks pass for
overlapping r7 scales, r8 null channels, r6 ligand-only channels and r5 changes
to extrema alone. Reports and prediction CSV metrics recompute identically.
The AWS audit independently verifies 8,114 crop reads, reloads all models,
checks training-only scalers and reproduces saved predictions; local checks
do not unpickle models across different sklearn versions.

None of the six additions improves CASF RMSE over the unchanged r4 control:
**1.291531**, PCC **0.842641**. The best new CASF variant is **r7**, RMSE
**1.300913**. Training-only validation selects **r6+r7**, validation RMSE
**1.272556**, CASF RMSE **1.305975**. These are exploratory single-seed CASF
comparisons, not significance or protein-family generalization claims.
No numerical source changed after the AWS snapshot was frozen.
[Comparison and audit receipts](../../datasets/protein_ligand_prediction/archive/2026-09-16_pre_final/experiments/v2-alpha-followup-2016/COMPARISON_REPORT.md).

### Historical 2026-09-15 preparation and launch

Added an isolated application for six user-specified combinations on
m2+m3+m5+r4, plus a freshly generated r4 control. Local full regression:
**1,619 passed, 10 skipped**, two existing truncation warnings, **23.59 s**.
The 23 new tests cover independent alpha/incidence spectra, exact 15 Å
boundary selection including ligand H anchors, pre-grid r7 edges, r8 internal
edges, unchanged feature dimensions, cache/order/control identity, artifact
resume safety, resource failures and the fixed 2016 training partition.
The installed core and higher-dimensional algorithms are unchanged.

New AWS source/output locations isolate this run from completed studies.
The planned 19-complex preflight and 12-complex five-tree smoke trial precede
64-worker feature generation and seven parallel fits per stage. Results were
pending at preparation time. [Protocol and checks](../archive/protein_ligand_prediction_20260916/workflows/protein_ligand_prediction_v2_alpha_followup/README.md).

AWS regression passed **530 tests** (joblib warnings concern NumPy 2.5
array-shape deprecation). The 19-complex preflight passed **133 tensors**;
all controls reproduce and both crop selections verify. The smoke passed
**84 tensors and 14 five-tree fits**, including its full audit; downloaded
tensor and model artifact hashes verify locally. Production launched at
**22:51:19 UTC**, PID **19889**, from a new 239-file immutable snapshot.
At 22:54:45 UTC all 64 feature workers were active at 99.92% host CPU capacity,
with 23 complexes complete and no failure events. Final production results
and the full audit were pending at that launch snapshot.

## 2026-09-15 incremental ordinary L0 construction

The package adds a guarded `core.hyperdigraph.L0Sweep`, using one matrix
buffer and four accumulated updates for each newly born directed edge.
The generic spectral workflow uses it for eligible ordinary hyperdigraph L0;
higher-degree and genuine two-scale persistent calculations retain their
existing paths. The alpha feature engine 1.2.0 calls the public sweep, with
new recipe/source identities and preserved geometry/summary definitions.

The full package and all molecular workflow suites pass **1,596 tests**,
with **10 skips** and two existing intentional truncation warnings, in
**25.06 seconds**. Independent incidence products match every new L0 matrix;
tests cover tied/reciprocal edges, missing/late vertices, mixed labels,
inclusive boundaries, isolated zeros, partial/eigenvector queries, mutable
state isolation, resource guards, and no eigensolving in matrix-only calls.
L0-L3 ordinary and genuine persistent spectra/matrices match the reference;
the existing independent full-boundary audits cover higher non-face-closed
objects. Cold-import architecture isolation also exercises the new API.

The construction-only benchmark passes all **200 scale-specific matrices**
with bitwise equality. Over five repeats, median local construction speedup
is **30.5-39.6x** for 64/256-vertex alpha/distance graphs across 50 scales.
Geometry and eigensolving are excluded; both methods return dense copies.
This is not an end-to-end molecular or AWS throughput claim.
[API, benchmark and limitations](INCREMENTAL_L0.md).

All **28 complete molecular tensors** (seven strategies on 1a30, 4ej8, 1hps
and the large 4fys case) match the frozen 1.1.0 engine: float64 maximum
absolute difference is zero and float32 tensors are bitwise equal. The built
wheel passes an isolated installed-package smoke check outside the repository,
including the new API, all three routes, higher dimensions and persistent
queries, with legacy packages and optional geometry engines blocked.
[Validation receipt](../examples/output/incremental_l0_20260915/validation.json).
The full completed AWS study remains historical; it was not regenerated.

## 2026-09-15 alpha-complex and raw-extrema application

The exact production run completed at **20:33:39 UTC** with **4,057 successful
complexes, 28,399 tensors, 14 models and zero failed samples**. The final AWS
audit passed. All 10,000-tree models reproduce their predictions and metrics;
training-only scalers, split memberships and both previous controls verify.
All 77 downloaded model/report files match fresh AWS SHA-256 hashes.
Full local tensor retrieval and independent verification passed at
**20:49:29 UTC**: all **28,399 tensors**, **8,114 structure inputs**, seven
recipe/source identities and **14 model artifact sets**. All **4,057 tensors
for each of the two controls are byte-identical** to their prior study copies;
both validation and CASF scores also match exactly. The other eight summary
rows remain identical in all three r5 matched pairs. The local auditor
recomputes reports and CSV metrics without loading the AWS sklearn 1.9 models
into local sklearn 1.6; the AWS auditor independently reloads all models.
The final report includes all five requested variants, both controls and
six matched comparisons. No numerical source or estimator change was made
after the exact source freeze.

Lowest observed CASF RMSE: **r1+r4+r5, 1.270773**, 2.18% below the original
m2+m3+m5 reference. Validation selects **r1+r5**, with validation RMSE
**1.243183** and CASF RMSE **1.271012**. The triple's 0.000239 CASF advantage
does not establish significance. Highest CASF PCC: **r4, 0.842641**.
[Results and audit receipts](../../datasets/protein_ligand_prediction/archive/2026-09-16_pre_final/experiments/v2-alpha-2016/COMPARISON_REPORT.md).

### Historical preparation and execution checks

The native production attempt exited at 18:14:47 UTC with 4,041 complete
complexes and 16 singular-simplex errors; no model fitting was allowed.
The exact repair adds an explicit optional public alpha backend delegating
geometry to GUDHI 3.13.0 exact precision, without projection or perturbation.
Default native geometry and every core remain unchanged. All 485 local tests
pass in 13.11 seconds, including independent constrained-ball optimization
for both alpha backends, closed-form coface/degenerate geometry, optional
dependency isolation and the full application regressions. The revised
application pins the exact backend for every alpha channel. The immutable snapshot is
alpha2016_20260915_exact and new outputs are production-exact/smoke-exact.
Original production outputs remain intact. The exact source has 223 verified
files (manifest ba5a348054647d0a3be556f0b2df711dc8922950b397926aed0463e4bd216cfb).
All 485 AWS tests pass in 12.64 seconds. Recovery passes all 16 failed cases
and four large controls: 140 tensors, both prior controls exact, input hashes
and raw-extrema row invariance verified. The exact 12-complex pilot passes
all 84 tensors and 14 five-tree fits, including its full audit. Production
launched at 18:49:45 UTC, PID 15975, with 64 workers. At 18:51:18 UTC all 64
workers are active, CPU is 99.94%, feature RSS is 7.86 GiB, and no feature
failures are recorded. Historical preparation follows.

An isolated application requests r4, r5, r1+r4, r1+r5 and r1+r4+r5,
plus reference and r1 controls. r4 calls the existing public alpha builder
and preserves its full coface propagation before extracting edges.
The grid uses alpha radius=s/2; r1 keeps complete distance-filtered cross/
ligand-only pairs, so r1+r4 changes protein-only geometry. r5 replaces only
relative minimum/maximum. No installed builder or Laplacian implementation
changes. Local regression: 472 passed in 6.79 seconds, one optional GUDHI
test skipped because GUDHI is absent locally. Suites cover the new application,
both prior ablations, v2, architecture, simplicial/connection builders and the
independent alpha geometry oracle. AWS has GUDHI for exact-alpha checks.
AWS regression passes all 473 tests in 11.32 seconds. GUDHI 3.13.0 exact
precision independently agrees on all edges and alpha births for 16 molecular
channels (four complexes, up to 2,343 selected atoms), with largest absolute
squared-radius difference 1.106e-10. The 12-complex pilot passes all 84 tensors,
14 five-tree trial models, both exact prior controls, three raw-extrema pair
checks, scalers, predictions, metrics and reports. The immutable source has
220 verified files (manifest e4668bb8f42816e8c2b88067b94f04ec74506c2ed7fab181fd46c1a6de32c72f).
Production launched at 16:58:13 UTC with 56 feature workers and seven model
workers; expected output is 28,399 tensors and 14 full-length fits. Evidence
is under experiments/v2-alpha-2016; production performance is pending.

At the user's resource-utilization request, the fixed 56-worker pool was
stopped and resumed with 64 workers at 17:32:37 UTC, PID 14409. Source and
scientific configuration hashes are unchanged. All 1,779 previously complete
complexes passed integrity checks and resume verification; all 24,906 NPY and
record hashes were preserved. All 64 workers are active with one numerical
thread each; measured CPU use rose from 87.5% to 99.9%, with 7.66 GiB feature
RSS. The local helper syntax and production dry run pass. No numerical code
or ML settings changed. Eight deterministic native-alpha singular-simplex
errors remain under investigation; the independent exact-GUDHI diagnostic
succeeds on the original first three affected H-only clouds. No failures are
counted as completed, and no samples/categories have been dropped.

## 2026-09-15 m2+m3+m5 follow-up revisions

Added an isolated application workflow with seven predefined revision
subsets plus reference, sharing per-complex spectra across summary/category
variants. Local validation: 380 tests passed in 9.46 seconds across the new
receipt, original ablation, v2, and architecture suites. Tests cover complete
cross/ligand-only versus Delaunay protein-only edges, inclusive filtration,
positive summary scope and population variance, exact original-engine
equivalence, H-only channel removal with preserved crop inputs, missing
category sentinels, resource failures, and unchanged training/holdout settings.
After the additional undefined-correlation audit regression test, **381 tests
pass locally (4.92 s) and on AWS (9.87 s)**. The first AWS pilot exposed
an audit-only handling bug for constant predictions; the corrected source
and missing historical v1 test fixture were frozen in a new 207-file snapshot.
The smoke-verified run passed all 96 tensors, 16 five-tree fits, exact prior
reference reproduction, scaler/prediction/metric checks and report verification.
Production launched at 14:37:54 UTC with 56 feature workers and eight model
workers. Production completed at **15:41:57 UTC**. The final AWS audit and
independent local-copy audit both pass: **32,456 tensors**, **8,114 structure
input hashes**, **16 model artifact sets**, all exact split memberships,
training-only scaler statistics, 10,000 trees without early stopping,
persisted predictions/metrics, and reports. All **4,057 reference tensors
are byte-identical** to the completed m2+m3+m5 features, and its CASF scores
reproduce exactly. Downloaded artifacts match a fresh AWS checksum inventory.

r1+r3 has the lowest observed CASF RMSE **1.277164** (reference **1.299098**,
1.69% lower); r1 has highest PCC **0.840047**. The training-validation
choice is r1+r2+r3. Reports and audit receipts are under
datasets/protein_ligand_prediction/experiments/v2-revisions-2016.
No installed layer or original completed study artifact changes; repeated
single-seed CASF rankings are exploratory.

## 2026-09-15 completed refined-2016 ablation

The full production study completed at 08:27:03 UTC with **17 strategies**,
**68,969 feature tensors**, and **34 model artifacts**. All results and
features were retrieved locally. Final audits verify all tensor SHA-256
values, shapes, float32 dtype, finite values, schema identities, exact
manifest membership, and 8,114 structure-input hashes. All 34 local model
metadata hashes match AWS; both reports and combination selection recompute
identically after path relocation.

On the original AWS sklearn environment, all saved pipelines independently
reproduce their predictions and RMSE/MAE/PCC. All 10,000 fitted trees and v2
parameters match; scaler means/variances were recomputed from training rows.
All validation fits finished before any final core fit started. The top-four
single selection was m2, m5, m3, m6, yielding six pairs and four triples.
No sample or feature recipe failed or was omitted.

The validation-selected strategy is **m2_m5**, with CASF RMSE 1.297291 and
highest observed PCC 0.838336. **m2** has the lowest observed CASF RMSE,
1.293307, versus baseline 1.326745 (2.52% reduction). These repeated CASF
comparisons are exploratory, using one ungrouped holdout and one estimator
seed; no significance claim is made. The structure representation is the
existing v2-compatible files with 2016 membership, not an original-2016-byte
claim. Baseline comparison: 4,056 tensors byte-identical; one value in `4cu8`
differs by 2.384185791015625e-7 (one float32 step), with dimension/nullity rows
identical. No feature values were imputed.

Final evidence is under the parent dataset's
`experiments/v2-ablation-2016/`: `COMPARISON_REPORT.md`,
`comparison_results.json`, `completion_receipt.json`,
`final_local_artifact_audit.json`, `final_remote_model_audit.json`, and
`baseline_comparison_detail.json`. The read-only verification scripts are
saved there for reproducibility. Earlier sections below are dated historical
preparation and progress evidence; their pending-status statements are
superseded by this completed audit.

## 2026-09-14 AWS capacity upgrade and resume

The user supplied `ubuntu@3.131.153.182` with the existing `dongchen.pem` key.
Inspection confirmed 64 vCPUs, approximately 124 GiB RAM, preserved dataset,
and all 194 frozen source hashes unchanged. The old controller had exited and
the study lock was available. All seven recipes retained 1,792 NPY files each.
Production resumed at 19:27:20 UTC, PID 2860, with 56 feature workers and up to
12 model workers. The unchanged controller verifies cached tensor/input hashes
before reuse; worker counts are operational settings outside its scientific
configuration signature. The remote source remains immutable.

Only local AWS helper defaults and documentation changed. SSH and rsync now
both use the new address and an explicit configurable identity file. Shell
syntax and production-launch/result-sync dry runs pass. The follow-up monitor
uses the new address and treats verified `skipped` samples as completed, not
failures. No feature semantics, label membership, splits, or ML settings changed.
Live progress and memory verification are recorded in the parent dataset's
`experiments/v2-ablation-2016/aws_upgrade_20260914.json` and `RUN_STATUS.md`.

## 2026-09-14 corrected refined-2016 ablation preparation

The application receipt under
`workflows/protein_ligand_prediction_v2_ablation/` implements six independent
modifications and a staged selection of combinations. The corrected m4/m5
definitions each require Delaunay cross support; m5 null channels also require
selected-component internal Delaunay support. m1 is redundant whenever m4,
m5, or m6 is enabled. There are 30 unique canonical baseline/single/pair/triple
recipes before adaptive top-four selection. An earlier pilot used superseded
m4/m5 semantics and is excluded from scientific validation of this protocol.

Current evidence:

- The combined corrected ablation, v2-generator, and architecture suite
  reports **296 passes in 4.46 seconds**. The ablation tests are included in
  `pyproject.toml` default test discovery. Coverage includes selected-channel
  geometry, mask canonicalization, summary definitions, feature integrity and
  resume, training-only selection, aggregate resume verification, and explicit
  smoke/production separation.
- All **4,057** selected baseline NPY hashes and both structure-input hashes
  match their frozen success records. Complete training/core counts are
  **3,772/285**, with zero overlap and zero missing input/tensor pairs.
- The geometry-only scan parsed every complex in **135.95 seconds**, found
  **zero exact duplicate coordinate groups and zero parse errors**, and
  performed no eigensolves. Its per-complex receipt is
  `datasets/protein_ligand_prediction/experiments/v2-ablation-2016/preflight_geometry.json`
  in the parent repository. The largest channel union is `4fys`: 3,645 atoms
  and 13,286,025 dense entries. Maximum complete pairs within 9.9 angstroms
  are 282,377; maximum protein-internal pairs are 267,338 (`3ahn`). The
  declared 25M dense-entry/1M hyperedge limits exceed these observed counts.
- Corrected local calibration on **`1a30` and `4ej8`** succeeds for all seven
  baseline/single recipes. For both complexes, the generated baseline NPY
  SHA-256 is byte-identical to the frozen v2 tensor. The adjacent
  `corrected_calibration_audit.json` stores the evidence. `study_plan.json`
  stores exact scientific settings, source hashes, shapes, and execution
  status.
- The corrected local complete-flow smoke with eight training/four core rows
  and five trees completed **14 strategies and 28 model fits**, both reports,
  and features for all 12 complexes without errors. A fresh controller run and
  actual resume pass: each model fit command occurs once in its log, both
  reports recompute identically, and all 12 baseline NPY files are byte-identical
  to the frozen v2 tensors. `local_smoke_audit.json` records the evidence and
  `local-smoke-verified/` contains the completed output. The validation-report
  resume regression now excludes already-existing core runs from that report.
  Default smoke limits remain 40/10; production rejects reductions and uses
  the fixed 3,017/755 validation partition before full 3,772/285 evaluation.
  Smoke scores are not production performance evidence.

The user subsequently explicitly authorized the previously blocked upload
and AWS execution. Transfer verified **194 source files**, the exact two
2016 manifests, and all **4,057 structure pairs**. The AWS smoke completed
14 strategies/28 verified model artifacts with zero feature failures; its
report recomputed identically and all 12 baseline NPY files were byte-identical
to the frozen v2 files. Evidence is in the parent dataset's
`experiments/v2-ablation-2016/aws_smoke_audit.json`.

Production launched at **2026-09-14 03:01:18 UTC**, initial PID **72929**, on
the existing `aws-cpu` (`ubuntu@18.191.199.86`) host, using six feature/model
workers and one BLAS thread each. No new instance was provisioned. An hourly
task heartbeat (`finish-casf-2016-ablation-study`) will check progress and
retrieve/audit the completed results. The twelve-complex smoke mean was
100.56 seconds per complex for the seven baseline/single recipes, giving a
rough 18.9-hour first-pass extrapolation; this excludes model fitting and
combination extraction and is not a completion-time guarantee. Production
performance is pending. The remote snapshot remains immutable; its source
manifest SHA-256 is
`23c93af0b73fdd9f7d6549c8d7af9e194d19a6589d330371b00b7cf166dd5e95`.

The shell helpers explicitly allowlist research source, tests, documentation,
metadata, and the three required workflow directories; they omit raw
structures, examples, unrelated workflows, and other release label tables.
Launch is separate, durable, and source-verified. Both sync directions retain
unmatched artifacts; pulls back up changed local files and exclude transient
staging, temporary files, caches, and locks. `MANIFEST.in` includes workflow
shell scripts in source archives. All five helpers pass `bash -n`; help and
source-sync, production-launch, 8/4-smoke-launch, and result-sync dry runs pass
without SSH or writes. Production rejects smoke limits with exit status two.
Setuptools source-manifest processing confirms all five shell helpers are
included. No production command was executed.

Open limitations: the duplicate scan does not prove all Delaunay/Qhull calls
succeed; corrected size-distributed calibration must establish actual runtime
and memory. This comparison uses 2016 membership with the existing
v2-compatible structure representation. Historical 3,759-row model metrics are
not the new 3,772-row baseline. Smoke/calibration scores establish functionality,
not a best strategy, and ranking many CASF scores after observation remains
an exploratory comparison.

## 2026-09-13 completed historical refined-structure and feature recovery

The 28 unique IDs omitted across the frozen v2007/v2013/v2016 refined/core
feature intersections were located in the preserved NMI/TopoTransformer
working tree
`/Volumes/my-harddisk/HPC_guowei2_04262025/TopTransformer/Datasets_all/over_structures_set`.
Every requested directory exists and contains exactly one
`<id>_protein.pdb` and one `<id>_ligand.mol2`, with no pocket PDB, ligand SDF,
or unrelated file. The 56 recovered files total 15,447,020 bytes. They were
copied into the canonical store only after confirming each destination was
absent, then source/destination bytes and SHA-256 values were checked. The
store now contains 19,066 protein PDBs and 19,066 ligand MOL2s and still
contains zero pocket PDBs and zero ligand SDFs.

All 28 pairs pass the same current `_read_selected_atoms(..., 20.0)` input
path used by compact-six generation. The preserved source's original
`all_code/run_hpcc_job.py` explicitly constructs its feature inputs as
`over_structures_set/<id>/<id>_protein.pdb` and
`over_structures_set/<id>/<id>_ligand.mol2` and requests a 20-angstrom field.
Its label CSVs are byte-identical to the corresponding hash-pinned members of
the local `Benchmarks_labels.zip`, which independently connects the working
tree with this NMI/TopoTransformer protocol.

Cross-source validation is release-specific:

- v2007: all eight recovered pairs are byte-identical to the pinned PDBbind
  v2007 release files;
- v2016: all 13 recovered pairs are equivalent in the element labels and
  coordinates consumed by the feature receipt; all proteins and eight ligands
  are also byte-identical, while five ligands differ only in unconsumed file
  formatting;
- v2013: the files are validated as the historical TopTransformer working
  snapshot, but no immutable v2013 structure archive was available for an
  independent byte comparison.

The 28-row metadata manifest and detailed per-file record are
`labels/nmi_topotransformer_recovered_structures_20260913.csv` and
`labels/nmi_topotransformer_recovered_structures_20260913.provenance.json`.
The final sidecar records both structure and feature recovery as complete and
has SHA-256
`062cd5caaa8059feee40db6f69b1b822a26b180585ad1b33088ddc96f873ac56`.
The append-only eight-worker AWS run generated all 28 `(6, 100, 40)` tensors
under the unchanged compact-six recipe with 28 successes, zero failures, zero
skips, and zero unattempted samples. Its run record
`runs/20260914T004122.032819Z-pid68115-5e14372e.json` has SHA-256
`0a4b6407bde5da720aa1a0827c771a247b599b2988219d33bb4bf8102a8361a5`;
the copied recovery log has SHA-256
`f9fe9b20ff71a6ef0dc342042a531ea15e250e4f4114075e823cb4b422b412cd`.
All 28 tensors are finite, nonnegative, little-endian float32, C-contiguous,
and shape `(6, 100, 40)`; their global minimum is zero and maximum is 2001.
After copy-back, the local recipe contains 19,066 tensor/record pairs and its
stable sorted feature-ID SHA-256 is
`55c48f80d4e24bb72748274ba18fce7ef77adae659e2573026958df8583e5e13`.
Inventory is 19,066 NPYs (1,832,776,448 bytes), 19,066 success records
(73,335,534 bytes), five run records (8,293,442 bytes), one schema, zero
failures, and 38,138 regular files totaling 1,914,422,174 bytes.

Preparation version 1.3.0 adds a repeatable, non-overriding
`--supplemental-metadata-manifest` option; its SHA-256 is
`5ffefbbf9f0f78b36ad0e4be6f0b978505fc0306792a3330c1501bec648042a5`.
It published a distinct complete
manifest group under `labels/refined_nmi_complete_toptransformer_20260913/`.
The v2007/v2013/v2016 counts are 1,105/2,764/3,772 with zero omissions, all
matching core features available, and zero matching train/core overlap. A
repeat invocation reports every output unchanged. Preparation-provenance and
three manifest SHA-256 values are
`6fe3e9774701987ea7872e364d10da09cba93f51e0349acd9a9622dec42a3927`,
`108609f87dc1112fd821384faa31ead66e577218b473fbd08fccb06b255f8701`,
`f25161f8a511002f1f75828b1758c9a3de163ef1e98ae044ac729454d443f1c0`,
and `32edf19f6dc03d1e6809f640d99613f95f9de8e74505679393482786f6edeef9`.

Final local validation reports **1,275 passes, ten skips, and two expected
construction-truncation warnings** for the repository, **100 passes** for the
v2 workflow suite, and **19 passes** for the focused supplemental-preparer
slice. A no-network `pip wheel . --no-deps --no-build-isolation` build passes;
the wheel SHA-256 is
`a0b86aa3e81c1c863728d01ecbce7ea36380a9b75738495e5404926ed491cd8b`,
and inventory confirms that repository workflows, datasets, and examples are
not bundled. `python -m build` could not start because the active interpreter
lacks the PyPA `build` frontend and the repository `build/` directory shadows
that name from project root; therefore this recovery makes no new sdist
validation claim. The prior 1,097/2,749/3,759-row manifests, models, metrics,
and audit remain frozen historical evidence. No model was retrained on the
newly complete manifests.

## 2026-09-13 original v2020R1 compact-six feature production and GBDT receipt

The independent v2 workflow prepared the updated v2020R1 general P-L archive
and completed `(6, 100, 40)` singleton-element feature generation without
changing installed TopoKit mathematical source. The split contains 18,498
training IDs after excluding 539 archive-backed IDs in the union of the three
CASF cores; CASF-only `1xd1` brings the feature universe to 19,038 unique
complexes. The separate test manifests contain 195, 195, and 285 rows, with
zero train/test overlap. All active targets come from the NMI tables, including
`2cer=9.22`; updated-index measurements remain provenance rather than target
overrides. This 18,498-row available-archive subset and the Hyperdigraph
compact-six schema must not be described as exact reproduction of the paper's
18,904-row training membership, complete feature stack, or ensemble pipeline.

Preparation byte-verified all 76,148 canonical archive files and left existing
canonical bytes intact. Its sidecar hashes the four canonical `1xd1` files and
the pocket-repair provenance. Preparation version 1.1 preserves the canonical
12,422-byte `6djc_ligand.mol2` (SHA-256
`9234be921e87a6e4c6f0099e32a933efcebf00785819f9783d0df4ee7ce0a1d4`)
and creates a 12,408-byte normalized derivative by removing only the second of
two adjacent `@<TRIPOS>ATOM` headers. The derivative and deterministic
provenance-sidecar SHA-256 values are
`fd5b7ba7c3dea2e9a668ad71df3cf18a21cbd6bd655d7d72a837d84ca043a65a`
and `a018a8018d6c5c1420d446cfda0bf74bbc7ba892178f78da9c410674503f299a`.
Exactly one manifest row (`6djc`) selects the derivative. Generator and
preparation SHA-256 values are
`42075b820d2c49b9cc86873bc1845cc00d22f6fcae90b48c6f8bed59d09dbec4`
and `abe0907df6fd9ffca8955e22b1b72561d5b26c2bc4c5c5cddb36533788c09822`.
The migrated training-manifest and local completion-provenance SHA-256 values
are `df96d5b1c6f0a123019250bbe5e292f9c34b8171fb8eca6f38f7ce43a04ef8b0`
and `a9cd7266472b6e55b9227b8344017501b40eafa2ef9408c194fd34d127cd21f4`.
An idempotent repeat verified all canonical and repair files, added nothing,
and migrated nothing; the normalized ligand parses as 118 atoms.
The initial AWS preparation added 41,930 missing canonical files while
verifying all 76,148 and exited zero in 8:40.51 with 177,592 KiB maximum RSS
and no swap. The normalization/migration pass added its two repair files and
exited zero in 3:46.32 with 214,748 KiB maximum RSS and no swap.

The 64-complex AWS calibration succeeded in 34.86 seconds. The first full
eight-worker pass then generated 18,973 tensors, validated those 64 existing
results, and transparently reported the single `6djc` repeated-section input
failure. It ran for 2:42:37 at 797% CPU, reached 193,168 KiB maximum RSS, used
no swap, and exited 1 because the receipt treats any failed sample as a failed
run. After the deterministic input normalization, a targeted `6djc` run
succeeded in 7.53 seconds with 120,032 KiB maximum RSS. A clean full resume
then validated all 19,038 tensor/record pairs as existing results, with zero
failures and zero unattempted samples; it exited zero in 34.17 seconds at 341%
CPU with 148,676 KiB maximum RSS and no swap. Four immutable run records retain
the complete history: three `complete` records and the initial
`completed_with_failures` record. The final failure directory is empty.

The completed pre-recovery recipe snapshot is
`nmi-geometry-singleton40-compact6-hyperdigraph-l0-4a27520c26f43dba`.
It contained exactly 19,038 NPYs, 19,038 JSON records, one schema, and four run
records: 38,081 regular files totaling 1,911,614,102 bytes. The NPYs total
1,830,084,864 bytes, including 1,827,648,000 bytes of numeric float32 payload.
All tensors are finite, nonnegative, little-endian C-order float32 with shape
`(6, 100, 40)`. Stable feature-identity and aggregate path/content SHA-256
values are
`1ced88443ff408638e582c3b4e999a7fbf404c628d72b3ad6069836f25ee320f`
and `d6f733f27a502a435fa57256e39afb20c48560db6ef3af89434213832e0fd5ec`.
The schema SHA-256 is
`3fd66f389307a7b3e5670b745307377aebc502871d77358f191999320d3570b4`
and pins TopoKit source-tree SHA-256
`39419d85e3578b165fa14975954699f9671909721ac8fb616bf1c4f552df06e6`.

The exhaustive audit rehashed 38,076 selected protein/ligand inputs totaling
13,986,614,119 bytes; their stable identity SHA-256 is
`1c94eb94882436ec8bda423ec794e40c2cb8f8ac5389847a7997d8f64727785f`.
It verified exact manifest, NPY, record, and input identities; shape/dtype/order;
finite and nonnegative values; 20,021,662 positive-spectrum trace checks; and a
maximum trace/contact residual of `3.96728516e-4`. The first five feature rows
match v1 bit for bit for all 5,376 shared IDs. The AWS audit passed with zero
errors in 1:02.98 at 199% CPU, 233,504 KiB maximum RSS, zero swap, and exit
status zero. The post-copy local audit passed with zero errors in 17.95 seconds.
Their reports are byte-identical with SHA-256
`0579db43c7800c80b98f0a8cf0f319cfdeb4b777d736109c0cda75b6cb11036e`.
The checksum-only dry `rsync` comparison emitted no differences and exited
zero. Logs and both audit reports are retained outside the recipe in its
adjacent `.aws-logs/` directory, so they do not alter recipe identity.
The adjacent 5,346,712-byte `.aws.log` duplicates the historical first full
pass log and has SHA-256
`f814fd03328a4bba6c6a64c116608be6144bd2cd0cf93a1cc44094f403b33ac3`.

The complete v2 suite reports **69 passes in 1.75 seconds**, including **14
passes in 1.02 seconds** for the new standardized GBDT receipt. The complete
local suite reports **1,244 passes, ten skips, and two expected
construction-truncation warnings** in 20.20 seconds. A separate randomized
oracle matched direct `D - A` spectra for 240 bipartite channels and 2,160
scale snapshots. No ML training or CASF
prediction was part of the completed v2 feature-generation run, and all
preceding feature stores and model artifacts remain unchanged.

The downstream
`workflows/protein_ligand_prediction_v2/train_sklearn_gbdt.py` receipt now
defines the explicit model contract. It C-order flattens each `(6, 100, 40)`
tensor to 24,000 inputs and fits one
`Pipeline(StandardScaler, GradientBoostingRegressor)` using exactly the 18,498
training rows. The scaler is fitted only inside that training call; raw CASF
rows reuse it through `Pipeline.predict`, and the affinity target remains
unscaled. Requested estimator overrides are learning rate 0.01, 10,000 stages,
maximum depth 7, native per-split `max_features="sqrt"` resolving to 154,
random state zero, and verbosity zero; all other settings remain installed
scikit-learn defaults. The 195/195/285-row CASF manifests are evaluated
separately with RMSE, MAE, and PCC. No validation subset, tuning,
cross-validation, or early stopping is performed.

A full real-data dry run resolved all intended rows and feature identities
without creating output. An end-to-end smoke run used 50 training rows, ten
rows per CASF edition, and five estimators; it completed the pipeline,
predictions, metrics, metadata, and joblib reload check, with reloaded
predictions bit-identical to in-memory values. Trainer and focused-test source
SHA-256 values are
`a2179cc1a827a87ad23273c71b990b544f28f2f0fdd9d0924ad7eb908cc78d0d`
and `c8ebfd493e41fe1a30705739aa48f18ddf13a5777eda5fc3e09eb15d1508806f`.
The full AWS run used Python 3.12.14, NumPy 2.5.2, and scikit-learn 1.9.0. Its
initial approximately three-minute invocation was intentionally terminated and
preserved as an aborted preflight after the CLI-help wording correction; the
partial directory is not a result. Clean production then completed in the
newly created
`models/sklearn-gbdt-standardized-compact6-general-nmi-20260913-v1/`.
The run completed all 10,000 stages, wrote `metadata.json`, and exited zero.
Primary results are:

| Benchmark | N | RMSE | MAE | PCC |
| --- | ---: | ---: | ---: | ---: |
| CASF-2007 | 195 | 1.462757149449226 | 1.1323666580024467 | 0.8127462077634457 |
| CASF-2013 | 195 | 1.4534406512952716 | 1.157615444869793 | 0.7934419607156724 |
| CASF-2016 | 285 | 1.2374441451915585 | 0.9731991747296493 | 0.8550191429301052 |

Fitting took 6,698.197481 seconds and total receipt runtime was 6,732.784056
seconds. GNU `time` measured 1:52:31 wall time, 99% CPU, 5,780,004 KiB maximum
RSS, zero swaps, and exit status zero. The fitted scaler records exactly 18,498
samples and 24,000 inputs; it reports 2,245 zero-variance coordinates. Targets
remain unscaled.

The copied local model directory contains seven files totaling 51,828,928
bytes. `pipeline.joblib` SHA-256 is
`5e8c8f127038c1a6cec8db1c177b389ebf81bec6601478eab1930fdc0bc3ee78`;
`metrics.json` SHA-256 is
`8eb2d6f9e897c61938630c152a22c2c28332991933724b8648eef107bcb9c882`.
The independent aggregate path/content SHA-256 is
`8ba6f5f9d8cb2090d854d743014310004d15a6046b9778a01b4db51a5bc0f483`,
computed as relative path, NUL, file bytes, NUL for every file in sorted
relative-path order. The adjacent copied AWS log is 8,099 bytes and has SHA-256
`44e2eff8f16eb6e4679f89eb305f0f35e1e98c5eea366451e2aff382a5ba7f38`.

The independent AWS audit loaded the persisted pipeline, rehashed all 540
unique CASF tensors, reproduced every saved prediction and the exact
full-precision metrics above, confirmed identical predictions for all 119 IDs
shared between CASF editions, and reconfirmed zero train/test overlap. The
persisted-pipeline round trip also matches the in-memory predictions exactly.
No tuning, cross-validation, validation-set selection, target scaling, or early
stopping occurred. Earlier feature stores and completed model artifacts remain
unchanged.

## 2026-09-13 completed version-matched refined/core GBDT production

The independent refined/core preparation route was corrected to separate
membership authority from target authority. Hash-pinned PDBbind 2007, 2013,
and 2016 release sources define each refined membership. Preparation version
1.2.0 removes only the matching CASF core and verifies that the corresponding
training member inside the preserved Weilab
`labels/reference_indices/topoformer/Benchmarks_labels.zip` has exactly that
membership. The bundle then supplies the NMI training targets and training-row
order. Core labels and IDs are audited against it by PDB ID while the local
test-manifest order is retained; all 675 core target mappings pass. Its
local SHA-256 is
`e44d611e36589b5c01397449eee7083cb146af967f4f30d2b03cc20ae6e31b8b`.
The original URL referenced by TopoFormer was unavailable when checked, and
neither an embedded origin URL nor an upstream byte checksum is present, so this validation
claims local byte identity only.

The release-defined and feature-available counts are:

| Pair | Full refined-minus-core | Validated-feature intersection | Complete core | Matching overlap |
| --- | ---: | ---: | ---: | ---: |
| PDBbind v2007 / CASF-2007 | 1,105 | 1,097 | 195 | 0 |
| PDBbind v2013 / CASF-2013 | 2,764 | 2,749 | 195 | 0 |
| PDBbind v2016 / CASF-2016 | 3,772 | 3,759 | 285 | 0 |

The CASF-2016 row deliberately uses the 285-complex benchmark core, not the
distinct 290-complex PDBbind-v2016 core. The three current manifests have
SHA-256 values
`d0425d67e15e0e3c522cc3e5eee9d2ea6b463e92d252d979774a2a91111ca6e5`,
`e6fbcd4c5b0be0f796a5658792ae5fe3012b3ed6481f5bb6a5eaa055e4a4d4e5`,
and `fdcb46e764c324423e4b2e3166f86b36918406ccb2ecf8334ee9c27b3e2cefc9`;
their completion-provenance SHA-256 is
`7348702ba6b2f0b35a532f4391c37280d03da8057f9f018d715618ef5f69cd73`.
All selected target relations are exact (`=`), including `2cer=9.22`, and
production selects `--label-policy error`. Release measurements and any
historical inequality relations remain provenance only.

Trainer version 1.0.1 rejects mismatched train/core benchmark years before
loading model data. Each invocation fits one
`Pipeline(StandardScaler, GradientBoostingRegressor)` on one training
intersection and predicts only its matching raw core matrix. The scaler is fit
on training inputs only, and the affinity target is not scaled. The fixed
settings are learning rate 0.002, 10,000 estimators, maximum depth 7,
`max_features="sqrt"`, `min_samples_split=5`, `subsample=0.8`, and random state
0, with installed scikit-learn defaults otherwise. The frozen version-1.2
preparer and current trainer SHA-256 values are
`f5b319d603560f9ff487e9d8293f1e9e2aac1f9b9b236f8e891b30356a7e8495`
and `bf92e2f663bd4bbb33386053debf1acb1ba940f4ae87dce9d8d88728cd71369b`.

That frozen-model checkpoint reported **1,271 local passes, ten skips, and two
expected construction-truncation warnings**; the focused local
preparer/trainer slice reports **27 passes**. The AWS-focused refined trainer
suite reports **23 passes** and 18 warnings from external joblib/NumPy
deprecation paths. Fresh wheel and source-distribution checks also pass: the
wheel contains the six installed layers without datasets, examples, or
repository receipts, while the sdist contains the required receipts and
documentation and excludes generated outputs.

The three production fits completed under
`datasets/protein_ligand_prediction/models/sklearn-gbdt-standardized-compact6-refined-pairs-nmi-20260913-v2/`:

| Pair | Train N | Test N | RMSE | MAE | PCC |
| --- | ---: | ---: | ---: | ---: | ---: |
| PDBbind v2007 / CASF-2007 | 1,097 | 195 | 1.4767356443471618 | 1.153354580527072 | 0.8062946512163276 |
| PDBbind v2013 / CASF-2013 | 2,749 | 195 | 1.5179520255834025 | 1.2686667834755287 | 0.7745888010556891 |
| PDBbind v2016 / CASF-2016 | 3,759 | 285 | 1.3234052487088577 | 1.0583577486640605 | 0.828697281199449 |

The independent audit passed, reproduced the saved predictions and metrics,
and verified all 675 by-ID core targets while preserving each local test order.
Its JSON has SHA-256
`3146ecda37732f776b90baa10cbf47ac36fcb82389b6eebb318417e21f2e8aa4`.
These are single compact-six TopoKit models, not the NMI ensemble or an exact
reproduction of the paper's complete feature stack.

## 2026-09-12 protein-ligand v1 feature receipt and AWS production

The separate repository-level v1 receipt defines `(5, 100, 40)` features
without changing the installed TopoKit package, Hyperdigraph mathematical core,
or frozen 143-channel receipt/store. Its focused suite reports **33 passed**.
The current complete repository suite reports **1,175 passed, ten skipped, and
two expected construction-truncation warnings** in 20.98 seconds. A fresh source
archive includes the v1 script, README, design note, and focused test; the wheel
continues to exclude repository-level workflows.

The production command used the AWS Python 3.12.14 environment with NumPy
2.5.2, SciPy 1.18.1, eight worker processes, and one BLAS thread per worker:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
python workflows/protein_ligand_prediction_v1/generate_features_protein_ligand_v1.py dataset \
  --dataset-dir ../datasets/protein_ligand_prediction \
  --output-dir ../datasets/protein_ligand_prediction/features \
  --protein-column protein_file \
  --ligand-column ligand_mol2_file \
  --profile topoformer-large-2024 \
  --storage-dtype float32 \
  --spectral-zero-tolerance 1e-10 \
  --max-dense-entries 13000000 \
  --workers 8
```

A 32-complex calibration completed first. The resumable full run validated
those 32 results and generated 5,344 more, leaving all **5,376 unique
complexes complete with zero failures**. GNU `time` measured 42:51.27 wall
time, 796% CPU, 176,448 KiB maximum RSS, zero swaps, and exit status zero. The
recipe ID is
`nmi-geometry-singleton40-compact5-hyperdigraph-l0-f33790fac29d5722`.

AWS and post-copy local audits verified exact manifest/NPY/record ID equality,
all 10,752 source inputs by bytes and SHA-256, and every tensor's output hash,
shape `(5, 100, 40)`, little-endian `float32` dtype, C contiguity, and finite
values. The store contains 5,376 NPYs, 5,376 records, one schema, and two run
records: 10,755 regular files totaling 452,662,724 bytes. Its aggregate
path-and-content SHA-256 is
`e0b25c4ffd51685a9842b29bcbaef28b31afc8dfb50dd95b2f7330c9a90613bd`.
The copied 1,509,308-byte AWS log has SHA-256
`201ab2dbe41147d72390e18d615bf01b1ea4bfd625b43fa905919d51cdef1e10`.
The schema pins generator SHA-256
`4fff3aea0be8a9a49fa9a45b527234355a5df4185b5903c93b37b26b66fe0d5b`
and AWS TopoKit source-tree SHA-256
`ba22b5145743f23aa32ce2fd39d837c82941697713417a35e396736d569532ee`.
The local and AWS `1a1e` tensors match bit for bit.

All spectral invariants pass. Trace-derived contact counts agree with integers
within `3.55e-4`, well below the `1e-3` audit threshold. The only scientific
warning is `5f2u`: full-protein chain E residues 640--643 duplicate peptide-
ligand coordinates, so the declared `distance <= scale` rule correctly retains
27 scale-0 contacts (16 C--C, 4 N--N, 7 O--O). This is a source-selection
property, not corruption or a Laplacian failure; filtering it would require a
new receipt and recipe ID. No predictive result is claimed because this receipt
performs feature generation only.

## 2026-09-13 standardized singleton-40 scikit-learn GBDT production

The independent v1 model run consumed feature recipe
`nmi-geometry-singleton40-compact5-hyperdigraph-l0-f33790fac29d5722` and wrote
`models/sklearn-gbdt-standardized-singleton40-20260912-v1/` plus its adjacent
execution log. The persisted estimator is one scikit-learn pipeline:

```text
raw C-order (5, 100, 40) tensor -> 20,000 values -> StandardScaler -> GBDT
```

Only the 4,836 training rows entered `Pipeline.fit`; therefore the scaler was
fitted on exactly the 4,836-by-20,000 training matrix. It reports 3,039
zero-variance coordinates. The 540 unique CASF tensors were passed raw to
`Pipeline.predict`, which reused that fitted scaler, and the affinity target
was never standardized. Full-manifest validation again found zero train/test
ID overlap. The numeric label policy retains the requested `2cer=9.22` value
and provides a separate exact-only CASF-2007 sensitivity result.

With scikit-learn 1.9.0, the regressor completed the same overrides as the
earlier baseline: learning rate `0.01`, 10,000 stages, maximum depth 7,
`max_features="sqrt"` resolved to 141 candidates per split, random state 0,
and verbosity 0. Every other estimator setting remained at its installed
scikit-learn default. Primary results are:

| Benchmark | N | RMSE | MAE | PCC |
| --- | ---: | ---: | ---: | ---: |
| CASF-2007 | 195 | 1.607853941401926 | 1.2243135403887215 | 0.7688500041456876 |
| CASF-2013 | 195 | 1.5761652358408038 | 1.265168640352196 | 0.747099200175994 |
| CASF-2016 | 285 | 1.346295816342027 | 1.0504023703157173 | 0.8261117846707097 |

The 194-row CASF-2007 exact-only sensitivity result is RMSE
`1.6047232650544907`, MAE `1.219645710592681`, and PCC
`0.7684326083768271`.

Model fitting took 1,579.374834 seconds and total receipt runtime was
1,590.717836 seconds. GNU `time` measured 26:33.32 wall time, 1,361,176 KiB
maximum RSS, zero swaps, and exit status zero. The 43,285,710-byte
`pipeline.joblib` has SHA-256
`3b1f58f86fab4745167eac032d3ce0e52e80a49f54b66d9cfa576ebbb5335367`.
The seven regular files in the model directory total 43,420,496 bytes and have
aggregate path-and-content SHA-256
`257b599bd228f0280bdad120fe3b13897e5a99856799516daa21dc2a62f6b577`.
The 6,862-byte log has SHA-256
`e1c93972b4b448b70d9ea77d439c27b162e61e054e72573755cd6e7fe23636d4`,
and `metrics.json` has SHA-256
`a6f0eae8c34350cc13b85ef219bb4d2f73fbcd33000ad5194671835576d1925e`.
The executed receipt SHA-256 is
`50195a9faacf35e8c365ca83fb84ddffc2ffda57a027a5b41049ea6d0bbbc9cb`.
The copied artifacts are checksum-identical to AWS, and reloading the pipeline
reproduced all in-memory predictions exactly.

The final local suite reports **1,175 passed, ten skipped, and two expected
construction-truncation warnings** in 9.46 seconds. The v1 standardized GBDT
subset contributes **12 passing tests**, including direct proof that scaler
statistics come only from training rows and are reused for raw test inputs. A
fresh source archive contains the new receipt and focused test, while the fresh
wheel retains only the six installed layers; both pass the distribution audit.
`python -m pip check` reports no broken requirements.

## 2026-09-12 scikit-learn protein-ligand gradient boosting receipt

The repository-level `train_sklearn_gbdt.py` receipt adds a second conventional
tree baseline without changing the installed TopoKit package, mathematical
cores, feature generator, stored tensors, manifests, or existing XGBoost and
transformer artifacts. It reuses the task-local prediction contract for
full-manifest leakage checks, record/tensor SHA-256 validation, C-order
flattening, numeric/exact-only/error label policies, and separate CASF
RMSE/MAE/PCC reports.

The requested configuration is scikit-learn
`GradientBoostingRegressor(learning_rate=0.01, n_estimators=10000,
max_features="sqrt", max_depth=7)`. Native square-root feature sampling occurs
at every split. Random state zero is the sole reproducibility exception; all
other resolved estimator settings and the installed scikit-learn version are
recorded as run provenance. The classic estimator has no `n_jobs`, so a full
fit is single-process even on a multi-CPU AWS/HPC host.

Focused receipt checks cover requested defaults, retention and recording of
installed sklearn defaults, leakage and feature-contract reuse, deterministic
square-root sampling through seed zero, exclusive output reservation, metric
and prediction publication, model reload, and completion metadata written
last. They report **11 passed** in 1.04 seconds with scikit-learn 1.6.1. The
source-distribution check also requires the receipt and its focused test; wheels
continue to exclude repository applications. The complete repository suite
reports **1,130 passed, ten skipped, and two expected construction-truncation
warnings** in 20.33 seconds. The base-environment protein-ligand workflow suite
reports **64 passed and nine skipped** in 1.30 seconds; those skips are eight
optional PyTorch checks and the opt-in source comparison.

The full checksum-verified AWS production fit completed with Python 3.12.14,
NumPy 2.5.2, and scikit-learn 1.9.0. It used all 4,836 training rows, fitted all
10,000 stages, read 85,800 C-order features per row, and resolved native
`max_features="sqrt"` to 292 candidates per split. The three manifests contain
675 memberships over 540 unique test complexes and have zero training-ID
overlap. Independent same-environment loading confirmed the fitted counts,
rehashed all 540 test tensors, and exactly reproduced every 12-significant-digit
prediction CSV value. Metrics recomputed from those CSVs agree with the saved
full-precision metrics within `1e-10`.

Primary `(RMSE, MAE, PCC)` results are CASF-2007
`(1.5589951446757668, 1.1873557349969865, 0.7864024956048317)`, CASF-2013
`(1.5190792447089312, 1.1945678655160856, 0.7600982154664409)`, and CASF-2016
`(1.3029786085893282, 1.0155615997046952, 0.8257984374519883)`. The 194-row
CASF-2007 exact-only sensitivity result is
`(1.5573403560792864, 1.1839283478067546, 0.7854937722202958)`.

Fitting took 5,177.92085221599 seconds and receipt runtime was
5,192.595437048003 seconds. GNU `time` measured 1 h 26 min 35 s wall time,
1,889,964 KiB peak RSS (1.80 GiB), zero swaps, and exit status zero. The
47,049,346-byte joblib model has SHA-256
`146255f972a31268c27848a55e5397bed6ffbe1732946fa94b9646c939e15609`.
All seven model-directory artifacts plus the execution log are
checksum-identical between AWS and the local dataset folder. This trusted-only
pickle artifact should be loaded with the recorded environment; it is not a
cross-version model format.

## 2026-09-12 direct supervised protein-ligand transformer

The new task-local transformer path and extracted shared prediction contract
were validated without adding installed-package ML dependencies. The final
complete repository run reports **1,119 passed, ten skipped, and two expected
construction-truncation warnings** in 20.34 seconds. The base Python has no
PyTorch, so eight skips are optional transformer tests; the other two are the
existing Linux-only RSS and opt-in source-reference checks. Architecture and
existing XGBoost tests pass. The complete task workflow suite reports **61
passed and one opt-in skip** in the PyTorch environment. Its focused
transformer subset reports **32 passed** with PyTorch, or **24 passed and eight
skipped** without it.

PyTorch verification used the existing
`/opt/anaconda3/envs/mof_project/bin/python`: Python 3.12.9, NumPy 2.2.6,
PyTorch 2.7.1 on CPU. This environment lacks pytest, so the test runner imported
its NumPy/PyTorch/SciPy first and appended the base environment's pure-Python
pytest path for that process only. No package installation or environment
modification was performed. The unrelated `ITT_env` was not used after its
PyTorch import failed with duplicate OpenMP runtimes. CUDA and MPS have not been
validated.

Focused checks cover nonnegative, bounded-memory log1p normalization and
constant coordinates; exact statistic/scale/channel token order; nonidentical
encoder initial weights; gradient flow; seeded explicit holdouts; full manifest
train/test exclusion before smoke limits; a torch-free dry run; fixed training;
validation selection; full refitting; correct fitting-only feature and target
normalizers; metadata completion; and saved-model label-free inference.
Changing every CASF label and feature while holding training fixed produces
bit-identical trained weights and preprocessing. A selected-epoch full refit
also matches a fresh fixed-epoch run on all training rows exactly, including
with nonzero dropout. A manifest changed during fitting prevents completion
metadata from being written. These invariants are now persistent tests.

The complete real-data dry run resolves 4,836 training complexes and 540 unique
test IDs across 195/195/285 CASF memberships. An explicit 10% validation
fraction produces 4,352 fitting rows and 484 validation rows with split seed 0.
A separate read-only payload audit loaded all **5,376** canonical tensors and
verified their `(6,100,143)` shape, C-contiguous little-endian float32 storage,
finiteness, and nonnegative values required by the new log1p transform.
No dry run creates an output directory or claims to hash feature payloads;
record identities are checked during planning and full tensor SHA-256 checks
run when fitting or predicting.

A real-feature CPU smoke run used 32 training complexes, 24/8 internal
fit/validation rows, four rows from each CASF edition (eight unique test IDs),
two selection epochs, and a fresh two-epoch refit on all 32 training rows. It
used the default `(6,100,143)` input and 523,521-parameter model and completed
in 1.66 seconds. Artifacts are under
`examples/output/protein_ligand_transformer_smoke_2026-09-12/`. Reloaded
predictions for four real complexes match the evaluation CSV to within
`2.08e-7` target units despite the changed inference batch size. These small
prefix metrics are software checks, not benchmark performance or timing
estimates for the complete experiment.

Remaining work for a scientific transformer comparison is full-dataset training
with frozen settings, repeated model seeds, and hardware/runtime measurements.
The current encoder is TopoFormer-inspired and trained from scratch; it is not
the published pretrained/ensemble pipeline. The updated refined-membership
dataset and labels also differ from the paper. No advantage over XGBoost,
cross-family generalization, ranking/docking/screening performance, or
cross-device bitwise reproducibility is established by these checks.

## 2026-09-12 protein-ligand XGBoost receipt and AWS production

The repository-level `train_xgboost.py` receipt was checked independently of
the installed package. Its focused suite reports **12 passed** and covers the
requested defaults and square-root conversion, manifest split and benchmark
identity, train/test leakage rejection with intentional CASF overlap, censored
label policies including an all-non-exact benchmark, RMSE/MAE/PCC calculations,
C-order `float32` feature loading and hash rejection, XGBoost exact-tree
constraints, exclusive output-directory reservation, and a complete synthetic
train/evaluate/save path with a deterministic stand-in estimator.

A read-only dry run against the completed local feature recipe resolves 4,836
training rows, 195 CASF-2007 rows, 195 CASF-2013 rows, and 285 CASF-2016 rows.
The 675 test-manifest rows represent 540 unique IDs, with 119 IDs present in
multiple CASF editions and zero IDs shared with training. One training relation
and one CASF-2007 relation are non-exact; the default numeric compatibility
policy emits an explicit warning and the evaluation schema provides exact-only
sensitivity where applicable. The dry run verified all selected feature-record
identities and produced a stable aggregate feature fingerprint without fitting
a model or writing an output directory.

For the frozen `(6, 100, 143)` schema, `p=85,800` and the default receipt
shorthand `colsample_bytree=sqrt` resolves to
`0.0034139437099945944`, targeting about 292.916 features per tree before
integer handling. The raw training matrix estimate is 1,659,715,200 bytes and
the 540-row unique-test matrix estimate is 185,328,000 bytes; neither number is
a peak-memory bound because XGBoost needs additional quantized, histogram,
model, and allocator storage.

The production receipt subsequently completed on AWS with XGBoost 3.4.1 and
`n_jobs=8` on all eight logical CPUs. It used `gbtree` with
`reg:squarederror`, learning rate 0.01, depth 5, minimum child weight 2, row
subsampling 0.6, `colsample_bytree=0.0034139437099945944`,
`colsample_bynode=1`, 10,000 CPU `hist` estimators, and random seed zero. The
eight-thread setting changes execution parallelism from the receipt's
one-thread default; the requested scientific parameterization is otherwise
unchanged. All 4,836 training rows and all 540 unique test complexes were used.
The resolved scientific-configuration SHA-256 is
`07a5932f67e6cb8254dab14388e72228c77f635259dfe5618601781f9cc92d03`.

Fit time was 14,890.228262688004 seconds and total receipt runtime was
14,899.276525333 seconds, approximately 4 h 08 m 19 s. Peak observed RSS was
approximately 20.46 GiB; because this was an observation rather than an
instrumented allocation bound, the true instantaneous peak may differ. The
26,058,516-byte UBJSON model reports 10,000 boosted rounds/trees and 85,800
features. The copied local artifact directory is checksum-identical to the AWS
directory and has a 25M filesystem summary.

| Evaluation report | N | RMSE | MAE | PCC |
|---|---:|---:|---:|---:|
| CASF-2007, numeric policy | 195 | 1.5381142107194636 | 1.175577779574272 | 0.7916629735588632 |
| CASF-2007, exact-only sensitivity | 194 | 1.5365720385997774 | 1.172293924597121 | 0.7907548257830246 |
| CASF-2013, numeric policy | 195 | 1.518889291497816 | 1.1922867497175167 | 0.7559748795688793 |
| CASF-2016, numeric policy | 285 | 1.2976941358918523 | 1.0036906847702831 | 0.825679634347487 |

CASF editions are evaluated separately; no pooled CASF metric was computed.
The primary scores use the declared numeric compatibility treatment for bounded
labels, while the CASF-2007 exact-only sensitivity excludes its one bounded
test relation. The workflow is an updated refined-set training protocol, not an
exact replication of the Nature Machine Intelligence paper's machine-learning
experiment. It performs no implicit tuning, validation split, cross-validation,
early stopping, or test-guided model selection. The existing topology, feature
tensors, and installed `topokit.workflows.ml` behavior were not changed. After
integration, the complete repository suite reports **1,095 passed, two skipped,
and two expected construction-truncation warnings** in 19.56 seconds; the
combined task workflow suite reports **29 passed and one opt-in skip**.

## 2026-09-12 structurally empty Laplacian feature blocks

All three public cores were checked for ordinary and genuine two-scale
persistent Laplacians. An absent source degree-q chain group consistently
produces a complete `0 x 0` operator, `eigenvalues=[]`, `basis=()`, and
`nullity=0`. Shared spectrum metadata now records the full
`operator_dimension`, `source_chain_dimension`, and `structural_absence`; a
partial eigensolve retains the full operator size rather than the number of
returned modes.

Postprocessing now accepts the explicit
`empty_operator_policy="preserve"|"zero"`. The default retains existing
undefined NaNs. The zero policy fills every requested predefined scalar feature
only for a core-confirmed complete empty operator, records the policy and
filled statistic names, and does not alter the raw spectrum. Custom callbacks
continue to run and retain their explicit empty-input behavior. Regression
coverage rejects missing or contradictory receipts, bases, nullities,
matrices, and eigenvector shapes, and distinguishes structural absence
from a nonempty all-zero operator, an empty positive selection, a partial
spectrum, and an invalid computation. Same-policy empty/nonempty records stack
through `feature_matrix`; mixed policies are rejected as different schemas.
End-to-end tests cover simplicial, Hyperdigraph, and interaction routes in both
ordinary and persistent modes, including a persistent degree 1 that is absent
at the source scale but present at the target.

Focused postprocessing/core/ML validation reports **240 passed** with the two
expected explicit-skeleton warnings. The notebook-note regression now follows
the existing `examples/different_input_formats_workflow.ipynb` filename rather
than the removed earlier name. That notebook was executed after enabling the
explicit structural-zero policy and now asserts finite L0/L1 summary blocks at
every sampled scale. The complete local suite reports **1,095 passed, two
skipped, and two expected warnings** in 19.29 seconds. `python -m pip check`
reports no broken requirements. A fresh wheel passes the six-layer distribution
boundary and an isolated installed-package smoke run covering all three routes,
the existing 47/110/99 interval receipts, and the new persistent empty-operator
zero encoding. No example or dataset is present in that wheel.

## 2026-09-12 protein-ligand production validation

Relocating the task receipts under the TopoKit project exposed and repaired the
checkout-source path, renamed-entrypoint imports, slow-reference root, command
examples, default pytest discovery, and source-distribution inclusion. The
provenance-locked `reconstruct_pocket.py` was restored byte-for-byte at SHA-256
`c09ef8ac21c383d6c7c1e86c5f5def5e87e20fbe821d4f4517197d06fdf9fe71`.
It remains a dataset-repair utility outside the installed package.

The PDB reader now accepts an explicit neutral formal-charge token `0` and
preserves it as integer zero. Blank, signed, and invalid-token behavior is
unchanged. The new regression passes, the complete molecular-reader module
reports 37 passes, and both real `6dyn` protein and pocket inputs now parse.
No licensed structure was modified.

The application receipt now removes only isolated zero blocks from each
ordinary q=0 spectrum and memoizes duplicate selected-atom channel blocks. It
still builds the authoritative filtration through TopoKit, derives active
ordered edges from that object, and calls the unchanged native hyperdigraph
Laplacian core. Synthetic auto/reference tests and the pinned `1a1e` tensor
comparison pass exactly. On the AWS host, the optimized and original
full-protein `1a1e` `.npy` files are byte-identical; runtime fell from 613.91 to
21.73 seconds. The generic TopoKit hyperdigraph definition and implementation
were not changed.

All 5,376 full-protein/MOL2 pairs passed reader and atom-selection preflight.
The maximum full channel has 3,591 vertices, the maximum incident block solved
has 1,030 vertices, and the largest builder object has 13,938 singleton/edge
records. The conservative full-channel guard therefore uses an audited
13,000,000-entry override for this corpus, although no actual dense solve
exceeds 1,060,900 entries. A checksum-only `rsync` comparison found the staged
AWS dataset byte-identical to local, excluding `.DS_Store` and feature output.

Local validation excluding the independently absent notebook-note fixture is
**1,053 passed, two skipped, and two expected warnings**. The focused receipt
suite is **17 passed and one opt-in skip**; enabling the source check passes.
AWS focused receipt plus molecular-reader validation is **54 passed and one
opt-in skip**. The rebuilt wheel/source-archive policy check also passes: the
repository-level receipts are present in the source archive and absent from the
installed wheel.

The eight-worker AWS production job completed in **23,897.41 seconds (6 h 38 m
17.41 s)** with one BLAS thread per process and resumable atomic per-sample
storage. Its final run generated 5,375 tensors and validated the existing
byte-identical `1a1e` smoke tensor, covering all **5,376 unique complexes**.
Remote and independent local audits each found exactly 5,376 `(6, 100, 143)`
little-endian `float32` finite tensors, 5,376 matching success records, and zero
failure, missing, unexpected, or symlink entries. Every feature and all 10,752
protein/ligand inputs matched their recorded sizes and SHA-256 digests. The
copied recipe contains 10,755 regular files totalling 1,883,747,551 bytes; its
tensor payload is 1,845,731,328 bytes. A checksum-mode dry-run of `rsync`
reported no AWS/local difference. AWS records round source mtimes to whole
seconds whereas the original local inputs retain fractional nanoseconds; all
timestamps agree to the epoch second and content hashes agree exactly.

## 2026-09-11 public route-namespace repair

Fresh-process regression tests now verify that the `simplicial`,
`hyperdigraph`, and `interaction` child modules are lazily available through
both the public `builders` and `core` parents. Each requested module is loaded
on first attribute access, later routes remain unloaded until requested, and an
unknown attribute still raises `AttributeError`. The protein-ligand feature
receipt also imports its simplicial builder explicitly, retaining compatibility
with older installations that require an explicit child-module import.

The architecture/public-layer slice reports **28 passed**. The isolated
protein-ligand receipt suite reports **9 passed and one opt-in slow reference
test skipped**; no dataset feature generation was run. A clean aggregate run
excluding the independently missing notebook-note fixture reports **1,035
passed, one skipped, and two expected explicit-skeleton warnings** in 18.46
seconds. The unfiltered aggregate run has the same 1,035 passes and one skip,
plus one unrelated `FileNotFoundError` because
`examples/molecular_formats_hyperdigraph_workflow.ipynb` is absent from the
working tree.

A newly built temporary wheel passes the installed-package smoke workflow for
all three routes (47/110/99 persistence intervals and 42 features per route),
including the six lazy route attributes and the graph/digraph/interaction
constructor entrypoints. Its distribution check confirms all six layers and no
bundled datasets, examples, stale modules, or bytecode caches. `python -m pip
check` reports no broken requirements. Existing versioned distributions,
frozen paper vendor code, and pressure-test baselines were not overwritten or
presented as a new release.

## 2026-09-11 notebook spectral-reference notes

Focused source validation confirms that the molecular-format tutorial now ends
with an ordinary-Hodge spectrum primer, a zero/small-positive/large-mode
interpretation table, and an explicit warning that physical words such as
“soft,” “stiff,” and “interaction strength” are analogies for these unweighted
examples. The reference separates the four plotted summaries from a broader
built-in-versus-custom descriptor table and includes a public-API custom
mapping example whose callbacks return scalar quantiles, inverse sums, a log
pseudo-determinant, a higher raw moment, and a separately scoped full-spectrum
heat trace.

The same audit checks that callback text states the one-real-scalar return
contract, copy isolation, the call-wide `positive_only` selection, stable
parameter-bearing feature names, explicit non-finite values, and no automatic
imputation. Across-filtration AUC, mean/spread, range, total variation,
extremum-location, and largest-sampled-slope descriptors are presented
separately from per-spectrum callbacks, with grid/interpolation dependence
noted. The six plotted L0/L1
series are identified as ordinary single-scale snapshots and not as the
two-scale `persistent_laplacian(start=s, end=t)` operator.

The focused notebook/spectral/visualization slice reports **23 passed**. The
full local suite reports **1,034 passed, one skipped, and two expected
explicit-skeleton warnings** in 20.34 seconds. Direct execution of the
notebook's custom-summary block against its JSON L0 spectrum returns finite
quantile, inverse-sum, log-pseudo-determinant, sixth-moment, and heat-trace
values with the expected positive/full-spectrum scopes. `nbformat` validation
and HTML rendering pass; all five reference headings, four tables, and both
DOI links appear in the rendered document. The existing 17 execution counts,
21 output records, and seven inline PNG figures remain intact. `python -m pip
check` reports no broken requirements.

## 2026-09-11 result-only curves and public molecular-format tutorial

The public visualization surface now includes `plot_betti_curves` for querying
one supplied `PersistenceResult` over a caller-defined grid and
`plot_spectral_summary_series` for displaying an aligned sequence of supplied
spectral-summary records. Focused contract checks cover axis reuse, dimension
selection, scale validation, aligned schemas, metadata-derived scales, NaN
gaps, and the default minimum/maximum/mean plus centered-Laplacian-energy view.
They also verify that the plotting layer does not invoke builders, core
analysis, or postprocessing.

Barcode checks cover both initial-stage modes. The default
`mark_initial_stage=True` retains the open-circle cue for a potentially
left-truncated start; `False` hides that cue without changing interval
coordinates. The molecular-format tutorial uses `False` because every H0 vertex
in its bounded filtration is known to be born exactly at zero.

Source inspection confirms that
`examples/molecular_formats_hyperdigraph_workflow.ipynb` is a clean,
standalone public tutorial. It introduces the seven inputs in a summary and
then reads and analyzes PDB, MOL2, SDF, MOL, PDBQT, AQUCOG CIF, and native JSON
one at a time. Each section uses the public barcode, Betti-curve, and spectral-
summary-series helpers; the spectral panels show minimum, maximum, mean, and
centered Laplacian energy for complete L0/L1 spectra. The final aggregate
regression run reports **1,033 passed, one skipped, and two expected warnings**
in 19.62 seconds. Clean notebook execution completes all 17 code cells, creates
seven inline PNG figures, and produces no cell errors or stderr. The wheel
check and `python -m pip check` also pass.

## 2026-09-11 molecular/JSON readers, spectral information, and per-format notebook

The default reader registry now covers CSV, native JSON point clouds, XYZ, PDB,
MOL2, SDF, MOL, PDBQT, CIF, and the `.mmcif` suffix for CIF/mmCIF data-name syntax. Dependency-free
fixture checks read the supplied examples as 36 PDB atoms, 36 MOL2 atoms,
36 V3000 SDF atoms, 24 V2000 MOL atoms, 3,646 PDBQT atoms, and 29 fractional
CIF sites. The public AQUCOG CoRE-MOF fixture is independently read as 162
Cartesian atom sites (Ni18 O54 C72 H18), and the native JSON fixture is read as
29 Cartesian points with its IDs, element labels, and explicit weights intact.
The PDB and MOL2 ligand IDs and coordinates agree. Tests also cover
PDB model selection with model-induced current connectivity, MOL2 charges/bonds,
SDF data fields, PDBQT atom types and partial charges, V2000 charges, V3000
closure, Cartesian CIF, dotted mmCIF
names, quoted control-like values/internal apostrophes, quoted numeric coercion
policy, unit-cell validity and conversion, malformed/empty/multi-record
rejection, registry dispatch, uniform raw weights, and row-aligned metadata
after repeated subsetting. Subset tests
also verify selected/source counts plus induced versus retained-source bonds and
PDB connectivity.

Native JSON checks cover canonical `coordinates`, the exclusive `points`
alias, optional unique IDs, row-aligned string labels and finite weights,
explicit units, copied non-reserved metadata, extension dispatch, subset label
alignment, and optional schema identity/version. Adversarial cases reject both
coordinate keys together, unknown top-level keys, duplicate object keys,
non-finite or Boolean numeric values, ragged/empty coordinates, misaligned
arrays, metadata collisions, and caller/embedded unit conflicts. The existing
`examples/data/point_cloud_24.json` is verified to fail with a targeted
provenance-sidecar message; its coordinates remain in the paired CSV.

CIF fractional coordinates are converted to Cartesian coordinates through the
six declared cell parameters. The source fractional strings, transformation
matrix, tags, other loops, and no-symmetry-expansion policy remain in metadata.
The readers do not expand crystallographic symmetry, infer periodic images,
assign elemental weights, or select parsed bonds for topology. PDB-family
multi-model input requires a model selection; SDF/MOL/MOL2 and CIF currently
return exactly one structure/atom-site loop per call.

Postprocessing checks independently verify population spectral variance, raw
moments of caller-selected orders, the predefined first four moment names, and
the distinction between uncentered spectral energy and centered Laplacian
spectral energy. The latter equals classical graph Laplacian energy for a
combinatorial graph L0 and is a declared generalization elsewhere. It consumes
the complete full spectrum including tolerance-resolved zero modes even when
other summaries are positive-only. An
opted-in partial spectrum reports Laplacian energy as unavailable/NaN, and
feature-matrix schema checks reject same-named summaries with different scopes.
Large-offset regressions cover adjacent floating-point eigenvalues, rescaling
without spurious intermediate overflow, signed moment overflow/underflow, and
the opposite-sign fallback for genuinely unrepresentable centered statistics.

`examples/molecular_formats_hyperdigraph_workflow.ipynb` has 28 clean source
cells (17 code cells) and no embedded output or generated images. It begins
with one seven-file summary, then presents independent PDB, MOL2, SDF, MOL,
PDBQT, AQUCOG CIF, and native-JSON sections; no combined reader loop or input
dictionary is used. An independent `nbconvert` execution with the local Python
kernel produced no cell errors or stderr and seven inline PNG figures. Every
source file is read in full; the mathematical examples use all PDB/MOL2/SDF/MOL
and JSON points plus explicit first-24-site PDBQT/AQUCOG selections. Each route
constructs one bounded sequence Hyperdigraph, computes GF(2) H0/H1 persistence
and complete ordinary real L0/L1 spectra at six scales, then shows only a
barcode, 100-point independently sampled Betti curves, and
minimum/maximum/mean/centered-energy curves. Infinite deaths are explicitly labelled
as right-censored at the finite construction cutoff. These are API
demonstrations across unlike inputs, not comparable molecular/material
descriptors or a validated physical protocol.

`python -m pytest -q` reports **1,025 passed, one skipped, and two expected
explicit-skeleton warnings** in 19.34 seconds. `python -m pip check` reports no
broken requirements. A no-isolation wheel build contains the new reader and
postprocessing modules, contains no example data/notebook, and passes isolated
reader/spectral smoke checks, including the 29-row JSON example supplied from
outside the wheel. No remote cross-platform run or external
chemistry-toolkit equivalence claim is made.

## 2026-09-08 dictionary-defined fixed-object notebook

`examples/fixed_topological_objects_homology_laplacian.ipynb` was executed in
place with the registered `Python (topokit)` kernel from
`/opt/anaconda3/envs/topokit`. Its 18 cells include 11 code cells with
sequential execution counts 1 through 11, no missing counts, no error outputs,
no stderr, and no runtime warnings. Independent checks covered the Betti
numbers, matrix shapes and symmetry, complete spectra, nullity, the two shared
interaction vertex labels, and every exported file.

The code uses the six supplied coordinate rows directly in `points`. The four
editable inputs are `graph_cells`, `simplicial_cells`,
`hyperdigraph_cells`, and `interaction_cells`; each groups cells by dimension.
All four independent fixed objects are analyzed through the same public
`workflows.analyze_stationary` entry point with complete matrices requested.
The tutorial has no time parameter, filtration comparison, cross-object
comparison table, article-specific provenance, or special-case analysis path.

The graph six-cycle has Betti numbers `(1,1,0)` and Laplacian nullities
`(1,1,0)`. The fixed closed octahedral simplicial shell has `(1,0,1)` and
nullities `(1,0,1)`. The Hyperdigraph, with ambient vertex `0` omitted from the
directed singleton hyperedges, has `(1,1,0)`, nullities `(1,1,0)`, and `L2`
spectrum `[2,4]`. The fixed interaction factors are cycles containing edge
`(0,3)` and share vertex pairs `((0,0),(3,3))`; the resulting interaction
complex has `(0,0,1)` and nullities `(0,0,1)`. Its `H0` value belongs to
TopoKit's interaction chain complex and is not a connected-component count for
either factor.

Execution regenerated 24 artifacts: four PNGs, four SVGs, four
homology/Laplacian summary CSVs, and 12 basis-labelled Laplacian matrix CSVs.
The PNGs are RGBA at approximately 300 dpi. All SVGs parse, contain live text,
and contain no raster image nodes. Visual review covered the graph cycle, the
single closed simplicial shell, the hollow missing Hyperdigraph singleton,
directed 1/2-hyperedges, and the single two-factor interaction view. The fixed
six-point coordinates are display-only and do not infer any mathematical cell.

`pip check` in the prepared `topokit` Conda environment reports no broken
requirements. That environment does not include the optional test runner, so
the focused source regression was run with `/opt/anaconda3/bin/python`:

```bash
python -m pytest -q tests/test_architecture.py tests/test_simplicial.py \
  tests/test_hyperdigraph.py tests/test_interaction.py \
  tests/test_visualization_export.py
```

Result: **160 passed** with the two expected warnings from tests that
deliberately construct truncated point-cloud skeletons. No production source,
API, contract, notation, homology, or Laplacian implementation changed.

## 2026-09-08 stationary package API and notebook refactor

The notebook's reusable fixed-scale work now runs through public TopoKit APIs:
`workflows.analyze_stationary`, `StationaryResult`, `StationaryStyle`, the four
`plot_stationary_*` representation functions, the dispatcher, and
`plot_stationary_diagnostics`. Package import testing confirms that importing
`topokit.visualization` remains lazy with respect to Matplotlib. Architecture
checks confirm that the visualization layer imports neither builders nor core;
the workflow is the explicit composition point for core and postprocessing.

The full local suite passes **939 tests** with one Linux-only RSS check skipped
on macOS and the two expected explicit-skeleton truncation warnings. A focused
stationary, architecture, spectral, object, export, simplicial, Hyperdigraph,
and interaction run passes **271 tests**. `pip check` in the prepared `topokit`
Conda environment reports no broken requirements. A no-isolation wheel build
contains `visualization/stationary.py` and `workflows/stationary.py`; the wheel
imports version 0.3.0 and all public stationary symbols in a clean temporary
environment while preserving lazy Matplotlib import.

The refactored `examples/point_cloud_topology_workflow.ipynb` retains 31 cells,
including 17 executed code cells, and contains no error outputs. It regenerates
13 PNGs, 13 editable-text SVGs, four 13-frame GIFs, and eight CSVs. SHA-256
hashes of all eight CSVs match the immediately preceding notebook exactly.
This independently verifies that extracting plotting and one-scale composition
did not change coordinates, selected cells, homology, Laplacians, eigenvalue
statistics, or persistent curves.

Visual review covered the four stationary representations, graph and
Hyperdigraph diagnostic figures, and middle frames from the simplicial,
Hyperdigraph, and interaction GIFs. Graph rendering remains free of background
circles. The other three representations retain exact-radius pale circles,
blue edges, green triangles, clear overlap rings, and grid-free axes. The
Hyperdigraph retains cividis point weights and readable same-curvature orange
directed-2 underlays below blue directed-1 arrows. Animation scheduling and GIF
writing remain notebook code by design.

## 2026-09-08 grid-free topology figures and filtration circles

The executed `examples/point_cloud_topology_workflow.ipynb` retains 31 cells,
including 17 code cells, and has no error outputs. All 13 PNGs, 13 SVGs, four
13-frame GIFs, and eight CSVs were regenerated. The plots now suppress grids;
SVG inspection found no gridline elements. The earlier blue-edge and green-face
Simplicial Complex palette is restored, and continuous direction weights use
cividis.

The Simplicial Complex, Hyperdigraph, and Interaction Complex GIFs now draw a
pale circle with the current alpha radius around every point in every frame.
The radii therefore grow from 0 to 0.80 in step with the filtration. The graph
GIF remains a points-and-edges view. Directed 2-hyperedge segments are rendered
first as wider translucent orange arrows; directed 1-hyperedges are rendered
above them as narrow blue arrows with the same curvature. Unique oriented
segments remain deduplicated independently within each dimension, so this
layering changes display prominence without introducing or removing a directed
hyperedge.

Color and grayscale contact sheets of all static figures were inspected, as
were the first, middle, and final frames of all four GIFs and the full-resolution
static Hyperdigraph view. Points, blue edges, green triangles, growing circles,
directional arrowheads, orange underlays, reciprocal directions, overlap rings,
barcodes, and spectral curves remain legible. PNG metadata reports 300 dpi;
every SVG retains editable text.

Accuracy checks are independent of the visual review. SHA-256 hashes of all
eight numerical CSV files match the pre-change baseline exactly. The notebook's
deterministic topology, spectrum, display-limit, and output assertions pass.
The focused architecture, simplicial, hyperdigraph, interaction, and
visualization suite passes **160 tests**, with the two expected explicit-
skeleton warnings. `pip check` in the `topokit` Conda environment reports no
broken requirements. No package source, public API, coordinates, filtration,
homology, Laplacian, or eigenvalue statistic changed.

## 2026-09-08 Nature-style demonstration figures

This subsection records the preceding style pass; the current palette, grid,
and animation-circle rules are documented in the section above.

The executed `examples/point_cloud_topology_workflow.ipynb` retains 31 cells,
including 17 code cells, with no error outputs. Its 13 PNGs, 13 editable-text
SVGs, four 13-frame GIFs, and eight CSVs were regenerated after presentation-
only changes. PNGs use 300 dpi. GIFs use fixed axes and consistent frame
annotations. SVG inspection confirms that all text remains editable.

The visual system uses charcoal points, navy graph/H0 marks, cyan 2-simplex
faces, vermillion directed-2/H1 marks, teal mean-spectrum curves, neutral gray
annotation, and cividis for continuous point weights. Shapes, ordered arrows,
filled regions, curve placement, labels, and line styles provide redundant
encodings. Typography, ticks, panel letters, legends, construction-circle
opacity, white space, heatmap colors, and line widths are consistent across
stationary, persistent, and animated views.

Visual inspection covered a color contact sheet of all 13 PNGs, its grayscale
conversion, full-resolution Simplicial Complex, Hyperdigraph, Interaction
Complex, and persistent-analysis figures, and the first, middle, and final
frames of every GIF. The views retain visible points, edges, triangles,
directions, reciprocal arrows, overlap rings, bar endpoints, Betti changes, and
Laplacian curves without clipping. The hyperdigraph's four tied-weight labels
use separated white-backed annotations and leader lines.

Accuracy was checked independently of appearance. SHA-256 hashes for all eight
stationary and persistent numerical CSVs match the pre-style files exactly.
The notebook's fixed-seed coordinate, topology counts, homology, spectra,
deduplicated-arrow limits, table layout, and GIF assertions pass. The focused
architecture, simplicial, hyperdigraph, interaction, and visualization suite
passes **160 tests** with the two expected explicit-skeleton warnings. `pip
check` in the `topokit` Conda environment reports no broken requirements. No
package source, public API, mathematical object, or numerical result changed.

## 2026-09-08 hyperdigraph display and notebook organization

The executed `examples/point_cloud_topology_workflow.ipynb` now has 31 cells,
including 17 code cells, with no error outputs. Each persistent topology is
constructed directly under its own Graph, Simplicial Complex, Hyperdigraph, or
Interaction Complex persistent-analysis heading. The removed shared constructor
cell contained no numerical operation beyond creating those same objects.

The numerical stationary Hyperdigraph Representation remains 34/65/75 directed
degree-0/1/2 hyperedges with `(H0,H1)=(1,3)`, L0/L1 zero counts `(1,3)`, and
energies `(130,272)`. Its display now deduplicates consecutive ordered segments
within each dimension. The 65 directed 1-hyperedges produce 65 unique blue
arrows. Fourteen selected directed 2-hyperedges contain up to 28 segment
occurrences and produce 23 unique orange arrows after repeated orientations are
collapsed. Runtime checks confirm at most one copy of an orientation in each
layer, at most two arrows over an unordered point pair per layer, and at most
four across both layers. The same display helper is used for every hyperdigraph
GIF frame. No numerical cell, direction, weight, filtration event, or boundary
operator is removed.

H0 persistence intervals start at alpha zero as closed line endpoints without
the earlier hollow-circle marker. The interval values and Betti curves are
unchanged. Visual inspection covered the revised static hyperdigraph, first,
middle, and final hyperdigraph GIF frames, and Graph and Hyperdigraph persistent
barcode panels. The views are legible and unclipped, and reciprocal arrows and
the blue/orange dimension layers remain distinguishable.

The output directory still contains 13 PNGs, 13 editable-text SVGs, four
13-frame GIFs, and eight CSVs. Notebook assertions covering deterministic
geometry, stationary and persistent topology, spectra, deduplicated display
limits, tables, and GIF creation all pass. The focused architecture,
simplicial, hyperdigraph, interaction, and visualization suite passes **160
tests** with the two expected explicit-skeleton warnings. `pip check` in the
`topokit` Conda environment reports no broken requirements. No package source
or public API changed.

## 2026-09-07 overlapping-circles notebook and Conda environment

`examples/point_cloud_topology_workflow.ipynb` was executed in place with the
registered `Python (topokit)` kernel from `/opt/anaconda3/envs/topokit`. The
environment was created from root `environment.yml` and contains an editable
TopoKit 0.3.0 install with Python 3.12.14, NumPy 2.5.3, SciPy 1.18.0,
Matplotlib 3.11.1, pandas 3.0.5, nbformat 5.11.1, nbconvert 7.17.1,
ipykernel 7.3.0, and Pillow 12.3.0. All 18 code cells execute with no error
outputs. The notebook passes exact fixed-seed coordinate reproduction,
intersecting-circle geometry, graph scope, simplex and directed-hyperedge
construction, reciprocal tied-direction, complete-spectrum,
positive-semidefinite tolerance, persistent-table, and GIF assertions.

Seed `20260907` samples 20 points around a radius-1.00 circle and 14 around an
overlapping radius-0.70 circle. It applies fixed Gaussian angular, radial, and
Cartesian movement. At the stationary cutoff `epsilon=0.55`, the independent
graph view has 34 vertices and 63 edges, `H0=1`, one L0 zero eigenvalue, and L0
energy 126. H1 and L1 are intentionally absent. The Simplicial Complex
Representation has 34/63/39 degree-0/1/2 simplices, `(H0,H1)=(1,3)`, L0/L1
zero counts `(1,3)`, and energies `(126,243)`. The Hyperdigraph Representation
has 34/65/75 directed degree-0/1/2 hyperedges, `(H0,H1)=(1,3)`, zero counts
`(1,3)`, and energies `(130,272)`. Its two fixed equal-weight supports both
retain reciprocal directed 1-hyperedges. The Interaction Complex Representation
has 14/90/194 degree-0/1/2 cells, `(H0,H1)=(0,0)`, zero counts `(0,0)`, and
energies `(90,512)`. The interaction H0 value is not interpreted as a connected
component count for either factor.

Persistent analysis uses 25 alpha values from 0 to 0.80 in Graph, Simplicial
Complex, Hyperdigraph, and Interaction Complex order. Graph distance cutoff is
`epsilon=2*alpha`; its H1/L1 columns and panels remain empty. The other routes
use their default alpha constructors. Every route exports persistence intervals,
Betti curves, and ordinary L0/L1 filtration-snapshot curves for minimum,
maximum, all-eigenvalue mean, and spectral energy. CSV records also contain the
numerical zero count at every alpha. These snapshot curves do not claim the
two-time persistent Laplacian `Lq(s,t)`.

Notebook execution generated 13 PNG files, 13 SVG files, four 13-frame GIFs,
and eight numerical CSVs under
`examples/output/point_cloud_topology_workflow/`. Every SVG contains editable
text elements. All static figures and representative first, middle, and final
GIF frames were inspected for the two-circle geometry, graph-only point/edge
view, filled triangles, directed arrows and reciprocal edges, explicit factor
overlap, increasing connectivity, readable diagnostics, and absence of
clipping. The full numerical hyperdigraph uses all 75 directed 2-hyperedges;
the static object view explicitly displays 14 for readability.

Focused regression command:

```bash
python -m pytest -q tests/test_architecture.py tests/test_simplicial.py \
  tests/test_hyperdigraph.py tests/test_interaction.py \
  tests/test_visualization_export.py
```

Result: **160 passed**, with the two expected warnings for tests that
deliberately construct truncated point-cloud skeletons. No production source or
algorithm changed. The example is synthetic and demonstrates one deterministic
point cloud and filtration; it does not establish statistical or empirical
performance.

## 2026-09-05 completed AWS pressure tests and separate low dimensions

The user authorized uploading the reviewed source to the configured AWS host.
All **358 cases completed: 282 successes and 76 explicit resource guards**.
This comprises 76 successful small smoke cases and 282 pressure/supplemental
cases (206 successes, 76 guards). Canonical q0/q1/q2 suites each contain 72
cases; their success counts are 45/51/53 and guard counts 27/21/19. The compact
H0/H1 pressure suite passes all 12 cases, and the supplemental suite has 45
successes and nine guards. All 282 successful NPZ files pass SHA-256 checks,
with one stable package-source hash and no duplicate or missing case records.

No measured case timed out or reported a memory, application, controller, or
numerical failure. The longest case took 166.503 seconds; maximum observed
process peak RSS was 2,138.7 MiB. The 600-second wall, 16-GiB sampled RSS,
22-GiB address-space, and declared native allocation budgets remained active.
Guards rejected 66 cases during analysis and ten during construction. These
conditional observations do not establish unguarded capacity limits.

Final local validation: **926 passed, one Linux-only skip, two expected warnings**
in 8.69 seconds. AWS regression gate: **913 passed, three skips, two expected
warnings, and 240 passing subtests** in 21.45 seconds, using Python 3.14.4,
NumPy 2.5.2, SciPy 1.18.1, Matplotlib 3.11.1, pytest 9.1.1. The AWS skips cover
two optional scikit-learn tests and a local-only historical-smoke fixture.
The Linux live-RSS check runs on AWS. The local count additionally includes
11 aggregation tests added after that gate. Both local and AWS before/after
runs have **190 exact comparisons, 268 bitwise-identical arrays**, and 110
passing internal checks per revision. Production algorithms remain unchanged.

The compact route is explicitly distinct from canonical construction. Its
combined construction/analysis timing, process-only guards, implicit-path count
semantics, actual backend, and bounded eight-point canonical probe are recorded.
Every compact probe matched exact endpoints; none certifies all large-case
intervals. Full and requested-partial spectra remain separate experiments.

The initial upload omitted CHANGELOG; the initial controller dispatch also had
a local-variable scope error. Both were fixed before final measurements. Two
integration tests now launch actual canonical and compact workers, and original
failed-attempt logs remain under the run's `validation/` folder. No failed
attempt is counted as a successful pressure case. Source/runtime manifests and
the downloaded archive checksum preserve the executed revision.

Evidence: [AWS findings](../examples/pressure_test/results/aws_2026-09-05/FINDINGS.md),
[complete overview](../examples/pressure_test/results/aws_2026-09-05/OVERVIEW.md),
[raw table](../examples/pressure_test/results/aws_2026-09-05/combined.csv),
[protocol](../examples/pressure_test/README.md), and
[current machine-readable state](../examples/pressure_test/AWS_STATUS.json).

## 2026-09-05 readability review and pressure-suite preparation (historical stage)

All 70 package Python files were reviewed. Local names changed in 45 files;
1,590 identifier references differ. The structural audit allows only consistent
local renames and comments/docstrings, protecting function arguments,
attributes, constants, keys, calls, numerical expressions, and statement order.
It passes on the final source snapshot. A retained pre-change source archive,
hash manifest, and readable diff are under `examples/pressure_test/readability/`.

Ten isolated before/after scenarios cover all three families, alpha/Rips,
Delaunay/complete hyperedge support, equal/distinct weights, and full/half
interaction overlap. **190 scientific comparisons match exactly; all 268
numerical arrays are bitwise identical.** Each revision also passes 110
internal consistency checks, including equal-stage persistent versus ordinary
operators. Full payloads and per-check evidence are in
`examples/pressure_test/readability/local_verification.json` and adjacent
compressed files. These small-case results do not prove correctness universally.

The full local test suite passes **873 tests**, with **one Linux-only live-RSS
test skipped on macOS** and the same two expected explicit-skeleton warnings
(7.90 seconds). This includes 41 worker, 38 passing runner, 20 readability
checker, and 11 report-integrity tests. Environment: Python 3.12.2, NumPy 1.26.4, SciPy 1.13.1,
Matplotlib 3.9.2, macOS ARM64; one BLAS/OpenMP thread, Agg backend.
The new runner tests use tiny synthetic subprocesses to check actual hard
timeouts, CPU/RSS reporting, locks, stale checkpoints, interruption cleanup,
durable errors, CSV fields, and resume integrity.

All **72 local six-point smoke cases pass** using the production public APIs:
eight profiles × three operations × degrees 0/1/2. Numerical NPZ exports,
per-stage checkpoints, CSV/JSONL, source/input hashes, and environment records
are in `examples/pressure_test/results/local_smoke_2026-09-05/`. These are
local calibration results, not AWS pressure measurements.
The generated `RESULTS.md` verifies all 72 NPZ SHA-256 hashes, finds no duplicate
case IDs or mixed worker-source hashes, and identifies the local macOS host.

AWS host inspection confirmed 8 logical CPUs, an Intel Xeon Platinum 8259CL,
31,717 MiB RAM, and no swap. An isolated Python 3.14.4 environment was prepared
with NumPy 2.5.2, SciPy 1.18.1, Matplotlib 3.11.1, pytest 9.1.1, and
threadpoolctl 3.6.0. **No source transfer or AWS test execution occurred.**
Automatic approval review rejected the upload pending explicit authorization
for the private source/test payload and destination. The 216-case main matrix
and supplemental cases are prepared with a 600-second wall limit per case,
16-GiB RSS ceiling, and 22-GiB address-space bound. See
`examples/pressure_test/AWS_STATUS.json` for the current state; do not interpret
the prepared matrix or local timings as completed AWS capacity measurements.

Details: [readability review](CODE_READABILITY_2026_09_05.md) and
[pressure protocol](../examples/pressure_test/README.md). Historical validation
records below retain their original counts and scope.

## 2026-09-05 object visualization and building blocks

The full suite passes **763 tests** with the same two expected explicit-skeleton
truncation warnings (14.92 seconds locally). There are 142 new visualization
checks: 49 explicit-object, 38 native-object adapter, 35 building-block, and
20 export tests. Architecture tests also pass. The command was:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/topokit-mpl PYTHONPATH=src python -m pytest -q
```

Environment: Python 3.12.2, NumPy 1.26.4, SciPy 1.13.1, and Matplotlib 3.9.2
on macOS ARM64. Tests verify ordered consecutive arrows, reciprocal separation,
no clique filling, shared-face deduplication, inclusive scale selection,
stable-ID maps, unchanged scientific inputs, bounded iterables, lazy imports,
caller-axis reuse, and 2D/3D/projected rendering. Export tests inspect actual
PNG/SVG/PDF files, editable SVG text, embedded PDF fonts, transparency, local
rcParams restoration, and validation before output-directory creation.

`python examples/visualization_gallery.py` exports three illustrative figures
in all three formats under `examples/output/visualization_gallery/`. Rendered
PNGs were inspected for arrow direction, tetrahedron visibility, label
placement, aligned blocks, and clipping. All passed this local visual review.
The examples use explicitly illustrative coordinates, not experimental results.
The native 3D view remains a transparent schematic; projected views are useful
for fixed-layout vector figures. High-dimensional blocks are labelled diagrams
and dense overlapping sequences can require explicit selection or separate panels.

`python examples/point_cloud.py --plots --output examples/output/visualization_regression`
also passes all three H2/L2 routes with updated object rendering. Interval counts
remain 47/110/99 and feature counts remain 474 per route. A SHA-256 comparison
with the pre-change audit manifest confirms all **63 existing Python source
files outside visualization are unchanged**. The source changes comprise two
existing visualization files and five new visualization modules; mathematical
definitions, numerical algorithms, and dependencies are unchanged.

No new installed-wheel or remote cross-platform run is claimed for this
visualization extension. Earlier milestone records below retain their original
scope. Usage, semantics, resource limits, and compatibility are documented in
[VISUALIZATION.md](VISUALIZATION.md).

## 2026-09-05 audit recheck

The [algorithm and efficiency audit](PERFORMANCE_AUDIT_2026_09_05.md) adds only
documentation and example-level prototypes/results; installed source is unchanged.
The full suite passed **621 tests** with the same two expected warnings. The
initial run took 15.02 seconds; the final run with `OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1` took 12.41 seconds. These are test-run observations, not
controlled performance comparisons.

Audit benchmarks used Python 3.12.2, NumPy 1.26.4 and SciPy 1.13.1 on macOS
ARM64, with final numerical runs restricted to one BLAS thread. Demonstration
analysis retains interval counts 47/110/99. Exact-output checks and source hashes
are recorded in `examples/performance_audit/`; equivalent matrix-vector formulas
were checked separately with floating-point tolerances. No new wheel or remote
CI run was needed or claimed for this documentation/prototype audit. The
following milestone validation records remain historical evidence for 0.3.0.

## Checks

`python -m pytest -q`: **621 passed**, two expected explicit-skeleton truncation
warnings, in 4.15 seconds on the local Python 3.12 environment. This includes
the prior 476-test baseline, new configurable-builder and barcode-schema tests,
paired interaction endpoint/censor/empty-factor cases, reader reference-data
checks, spectral statistics, ragged-grid plotting/serialization, architecture,
and full configured examples. Timing is an observation, not a benchmark.
The warnings describe intentionally capped simplicial and interaction factors;
they are not numerical failures or implicit construction truncation.

An installed 0.3.0 wheel was tested from outside the checkout with Python `-I`,
using a fresh temporary virtual environment that reuses local scientific
dependencies. All three H2/L2 routes pass with the fixture supplied by an explicit
external path. Imports from the original research packages and general TDA
platforms are blocked by the smoke test. No dataset is present in the installed
package. New 0.3 installed-wheel checks also cover reference weights, adjacency
conversion, graph/digraph construction, fitted bins, spectral series, and the
genuine paired interaction operator. Static-object, feature-handoff, configured
pipeline, and CLI-info examples also pass. The fresh temporary virtual environment
uses `--system-site-packages` for existing local numerical dependencies; this
does not claim a clean download or testing every supported dependency version.

Wheel/source archive contents are checked with `tests/check_distribution.py`:
six library layer packages in the wheel, examples/data only in the source
archive, and no demo outputs, stale modules, or bytecode caches in either archive.
The source archive includes the Markdown documentation and change trace.

The example-only 24-point workflow completes all three routes through H2/L2,
including raw bars, numerical vectors, spectral summaries, and PNG views.
Raw interval counts remain 47 (simplicial), 110 (hyperdigraph), and 99
(interaction); the default barcode feature vector has 474 values per route.
The definitions/scale units differ across these routes; matching vector length
does not establish feature equivalence.
The smaller `configured_pipeline.py` example additionally covers all three
routes through H2/L2, explicit graph/digraph inputs, separate factor schedules,
and a two-sample training feature matrix of shape (2, 42).

## Open scope and limitations

* CSV, single-frame XYZ, PDB, MOL2, single-record SDF/MOL, PDBQT, and
  single-atom-loop CIF/mmCIF are implemented readers. Multi-structure collection
  APIs, crystallographic symmetry expansion, periodic-image construction, and
  external chemistry-toolkit parity remain future work.
* Six modules are separated physically and dependency-tested. A public reader
  registry and custom builder recipes are available; a genuinely new topology
  family still requires mathematical core and dispatch implementation.
* ML is optional and consumes explicit feature matrices. Its default trees are
  untrained until requested. No domain-specific split, fitted scientific model,
  imputation rule, predictive evaluation, batch scheduler, or HPC system is inferred.
* Default native alpha uses floating-point SciPy/Qhull geometry; no exact-predicate guarantee,
  arbitrary degeneracy guarantee, or GUDHI performance-parity claim is made.
  The explicit optional gudhi_exact backend uses GUDHI exact geometry; returned
  birth values are converted to float, and its simplex cap is checked after
  external construction rather than bounding peak allocation.
  Standard alpha rejects custom bonds/edge-distance cutoffs. Restricted or
  weighted alpha is not implemented. Separate paired interaction schedules give
  a sampled one-parameter progression, not general multiparameter persistence.
* High-order calculations remain expensive. Resource guards and partial spectra
  do not guarantee every input fits memory. Partial summaries require opt-in;
  min/max/mean/std over an empty positive-mode selection remain undefined and
  need explicit handling before ML. Only a complete `0 x 0` operator can use
  the explicit structural zero encoding; it does not generalize to other NaNs.
  Default zero-count features require full spectra; opted-in partial spectra
  record observed zeros but leave full zero count and graph-theoretic Laplacian
  energy undefined. Default feature guards reject oversized bin grids and
  spectral-observation collections.
* Cross-platform CI is configured but has not been executed remotely. Local
  tests do not establish behavior on every dependency/platform combination.
* Source data provenance is retained in examples. Dataset redistribution terms
  should be checked before public release; the MIT code license does not replace
  third-party source data terms. No public upload or manuscript submission occurred.

See the [change trace](CHANGELOG.md), [contracts](CONTRACTS.md), and
[roadmap](ROADMAP.md) for remaining development and protocol-paper milestones.

## Validated deployment and continuation

Local full suite: **1153 passed, one skipped**, two existing warnings. AWS:
**1154 passed**, 240 subtests, three warnings. The repaired 1tps feature was
generated on AWS in 26.25 seconds while the original 64-worker pass remained
active. It is bit-for-bit identical to the historical GUDHI tensor and copied
locally with its input/output/source hashes verified. Full simplex geometry
for the regression fixture also matches the independent exact backend.

New package Python SHA-256:
`095e2070e2d432516a4351913c0c89c5eb24c2cabe890ad084675c739765a97c`.
Recipe: `final-alpha-l0-48051949e7c7c0f2`.
Frozen source: `/home/ubuntu/topo_toolkit/protein_ligand_prediction_unique_repair_20260917/topokit`.

Continuation PID 46185 began at 2026-09-17 13:26:32 UTC. It waits for the
complete immutable native-1.1.0 feature pass, then verifies and copies only
its successful tensors AND JSON records without rewriting them. The pinned
AST/source equivalence guard and per-record compatibility inventory preserve
actual generation provenance. No GUDHI tensor is reused as a native output.
The already repaired 1tps tensor remains in the new store. The controller
then verifies/skips valid records, retries other failures if any, and performs
all feature audits, four fits and six evaluations. It has not completed the
full benchmark yet. Current status is in
`experiments/final_alpha_l0_unique_repair/transition_status.json`.

Source and verification receipts: `source_manifest.json`,
`repair_validation.json`, `local_transfer_audit.json`, `transition_launch.json`.
The original known 1tps failure record remains preserved under the old run;
it is resolved in the new store, not deleted from the historical record.

## 2026-09-21 — DL preparation and trained GBDT delivery

Scope: folder/configuration preparation and existing-model deployment only.
The reserved `workflows.topoformer` namespace has no model API yet. Pretraining
and all DL fits remain unstarted. The fixed alpha15 recipe and original
completion/selection receipts are unchanged.

- `pytest -q workflows/protein_ligand_prediction/ml/tests/test_predict.py tests/test_architecture.py`: **31 passed**. Guards cover raw C-order layout,
  sample/recipe/status identity, altered hashes, tensor shape/dtype/finiteness,
  Fortran order, missing samples and path traversal; layer boundaries pass.
- Original feature/model rsync with content checksums: zero scientific files
  required transfer; empty copied runtime locks were removed and excluded.
- Independent local full audit: **19,066 tensors and records**, 26 original
  model artifacts, 326 frozen source files, nine manifests, four models, six
  metric rows and 1,350 predictions pass. Historical audit receipt preserved.
- Compatible Cornell runtime: exported original model weights unchanged,
  checked each embedded scaler and separately exported NPZ arrays, and
  reproduced all **1,350 predictions bitwise**. The new inference helper also
  reproduces all six CASF prediction files with batch size 128.
- Local bundle/plan audit: **49 bundle files**, four 27,500-coordinate scalers,
  unchanged training counts (1,105/2,764/3,772/18,498), 18 single-factor configs
  and 2,844/711 split pass. All three CASF test sets are absent from the
  hyperparameter-selection pool. The 217 extra exclusions affect selection
  only, not the original full-training manifests. No local model unpickling.

Evidence is under
`../datasets/protein_ligand_prediction/deliveries/cornell_alpha15_20260921/`: 
`FEATURE_REVERIFICATION.json`, `REMOTE_PREDICTION_VERIFICATION.json`,
`DELIVERY.json` and reusable local audit scripts. See the dataset ablation
plan for explicitly untested candidate hyperparameters and pending DL work.

## 2026-09-21 — validation of supervised plan revision 2

Configuration/documentation change only. Checked all **12** active configuration
files and hashes, exact one-factor differences, head divisibility, width
divisibility by four, all fixed user settings and the baseline/template match.
Verified analytic baseline count **3,366,913** (actual module not implemented).
The frozen upstream NumPy positional helper on 50×1 grids agrees bitwise with
an independent width-first sine/cosine formula for widths **128/256/512**,
including the zero CLS row. This verifies the source convention, not a new
trained model or a completed positional implementation in TopoKit.

The 2,844/711 split is byte-identical, disjoint and excludes every CASF test ID.
The prior plan archive matches its checksums and the original delivery's plan
hash. All **49** existing GBDT bundle files retain their recorded hashes.
Pretraining and all training remain unstarted. No additional numerical-core
tests were needed for this plan-only change. Evidence:
`datasets/protein_ligand_prediction/experiments/topoformer_alpha15_supervised_20260921/PLAN_REVISION_02.json`
and repository `workflows/protein_ligand_prediction/dl/TOPOFORMER_REFERENCE.json`.

## 2026-09-22: FS-AQ frozen sequence modality

- 38 targeted tests passed: sequence token integrity, explicit truncation/UNK, complete protein windows and residue weighting, feature order/finite guards, lazy dependencies, training-only scaling, existing topology API and six-layer architecture.
- Supplied ChEMBL checkpoint: BOS embeddings match its saved smoke outputs (atol/rtol 3e-5); supported token IDs exactly match the original tokenizer for all 19,005 applicable complexes.
- Official ESM-2 snapshot SHA256 verified. CPU smoke passed (PyTorch 2.8.0): 1280-dimensional outputs, batch/single equality, direct exclusion of BOS/EOS from residue pooling.
- All 19,066 manifest IDs retained; split counts and overlap constraints checked. 8,830 unique protein windows and 15,364 unique SMILES. Explicit limitations: 589 truncated ligand inputs; 61 with ligand UNK; 598 protein inputs containing X; 4 declared source-graph valence exceptions; 4as6 uses observed ATOM sequence.
- Production GPU inference and the four full GBDT fits/six evaluations are staged separately; no production metrics are certified by these software/preparation checks. Runtime completion is recorded in the external study audits.

The built wheel was also inspected: the sequence implementation is included; no checkpoint, feature matrix, molecular dataset or study runner is bundled (`WHEEL_AUDIT.json` in the external FS-AQ study).

Production update 2026-09-22 15:56 UTC: all 15,364 unique ligand embeddings generated successfully on lambda-1 CPU; original CPU provenance preserved. Remote ESM CPU smoke passed with PyTorch 2.7.1+cu118. ESM production is queued behind the existing TopoFormer study; Cornell fitting and local delivery controllers are active and waiting on audited prerequisites.

## 2026-09-22: Local Mac ESM execution update

The user requested local Mac resources instead of the remote GPU queue. The original frozen ESM code was checked at 128/384/1022 residues in CPU FP32 and MPS FP32. Maximum absolute differences were 9.54e-7, 9.54e-7 and 1.43e-6; batch/single difference was 7.15e-7. All finite/dimension checks passed. The two verified remote wait processes were stopped with no protein cache entries yet generated; all 15,364 ligand embeddings are preserved. The separate Mac controller invokes the original runner and preserves the numerical source lock. No model, pooling, input, manifest or GBDT hyperparameter changed. Production completion remains governed by the original feature/model/local-delivery audits.

## 2026-09-22 — FS-AQ production complete and locally verified

All 19,066 × 1,792 float32 features, four full 10,000-tree model/scaler
bundles and six CASF evaluations passed final audits. ESM used local Mac
FP32 MPS, ChEMBL27 used lambda-1 CPU, and Cornell performed the four fits
with scikit-learn 1.9.0. Remote completion was 18:14 UTC; verified local
delivery was 18:15 UTC. This supersedes the earlier queued/running snapshots.

`LOCAL_DELIVERY.json` verifies all 27 delivered artifact hashes, independently
reconstructs every complex feature from the hashed embedding caches, and
checks memberships, labels, training-only scaler means/variances and RMSE/PCC.
Remote model pickles were not loaded in incompatible local scikit-learn.
The final documentation check reverified all 109 frozen source/input hashes,
the 27 final artifact hashes, completion-receipt links and convenience paths.
Source-lock SHA256 remains
`5b5095cab88743dc67ed174511f925e8827c5514fe43152504af3861cfce4f3e`.

The selected FS-AN topology baseline has lower RMSE and higher PCC for all
six identical train/test comparisons. General-v2020R1 → CASF-2016 sequence
RMSE/PCC is 1.310083/0.825491, compared with topology 1.273101/0.849913.
These are descriptive fixed-setting results; CASF was not used for tuning
or selecting a replacement. Input representation limitations remain explicit.
No numerical implementation changed during final delivery/documentation.

Evidence and separate RMSE/PCC tables:
`datasets/protein_ligand_prediction/experiments/sequence_esm2_chembl27_20260922/`,
especially `FINAL_AUDIT.json`, `LOCAL_DELIVERY.json`, `COMPLETION.json` and
`results/FINAL_REPORT.md`. Earlier execution receipts are preserved.

## 2026-09-22 — FS-AR ESM-1b / NMI GBDT launch validation

The user requested first-generation ESM protein regeneration and a single
GBDT fit per training set using the NMI paper's five parameters; all other
estimator settings use installed defaults. The isolated FS-AR experiment
uses pinned official ESM-1b t33/650M, reuses every ChEMBL27 ligand array and
retains all 19,066 IDs and the four training/six evaluation memberships.
The installed workflow and prior scientific snapshots are unchanged.

Three regression tests passed: only five GBDT overrides and a nonstopping
progress monitor; rejection of ESM-2 protein cache reuse and changed ligand
provenance; syntax/PID-start-identity checks for the remote launcher. Real
ESM-1b CPU/MPS checks at 128/384/1022 residues passed with maximum absolute
difference 1.91e-6. MPS padded-batch/single difference was 4.92e-7. Direct
residue pooling and a training-only scaler smoke fit passed.

The checkpoint SHA256 and upstream revision are pinned in ESM_COPY.json.
All 15,364 reused ligand arrays and records are checksum-pinned. The 19,153-file
source/input lock (including the copied original ligand inputs) is
`e93418925b037c6402845a5c04fd78f4525b4cfd55179474db34f70654a88afc`.
The Mac controller started at 18:55:09 UTC (PID 98959), and generation was
observed advancing with no failure. Cornell preflight, four full fits, final
audits and delivery are subsequent gates; this launch does not certify results.

Evidence: `datasets/protein_ligand_prediction/experiments/sequence_esm1b_chembl27_nmi_20260922/`.
Default parameters and actual random-state provenance are saved per model.
The NMI historical ESM checkpoint remains unidentified; this run uses the
stated ESM-1b choice, ChEMBL-only ligand features and no consensus.

## 2026-09-23 — FS-AS ESM-2/CPZ preparation and feature audit

Verified byte-for-byte reuse of all 8,830 FS-AQ ESM-2 windows and retained all
19,066 IDs, labels and manifests. The supplied CPZ archive was copied without
modifying its source. Strict checkpoint loading established eight layers and
heads, width 512, FFN 1,024 and context 256. Real CPU/MPS checks at token lengths
4, 59, 254 and 510 (before truncation), plus an unknown-token case, passed:
maximum absolute difference 4.18e-7; padding difference 9.54e-7; direct BOS and
MPS repeatability bitwise. The active CPZ audit records 589 truncated and 61
unknown-token complex ligand inputs. No samples were dropped or imputed.

Four regression tests passed: pinned protein reuse and rejection of old
ChEMBL-only ligand reuse; CPZ tokenization integrity/read-only protein cache;
five GBDT overrides, nonstopping progress and training-only scaling; remote
launcher syntax and PID/start identity. The source/input lock covers 19,155
files: `fd601ba7bdb13c189e80dc648c811365153b16b385a9b3213367ba94b67bba44`.
Mac launch was at 02:11 UTC. All 15,364 new ligand vectors and the full
19,066 × 1,792 matrix passed feature audit at 02:12 UTC. Four full Cornell fits,
six evaluations and local delivery require separate production receipts.
Evidence: `datasets/protein_ligand_prediction/experiments/sequence_esm2_cpz_nmi_20260923/`.
Installed numerical code and earlier frozen studies are unchanged.

FS-AS production-start check at 02:16 UTC: Cornell preflight passed all four
five-tree pipeline fits, exact default-parameter checks and train/test memberships.
Remote controller PID3326669/start ticks38336047 launched four production fits.
Independent feature comparison confirmed all 19,066 protein subvectors are
bitwise equal to FS-AQ, while every ligand subvector differs with the CPZ encoder.
`PRE_ML_VALIDATION.json` records this check. Final models/metrics remain pending.

## 2026-09-23 — FS-AT ESM-2/ChEMBL27 NMI GBDT queue validation

The follow-up reuses the completed FS-AQ 19,066 × 1,792 float32 feature matrix,
all 8,830 ESM-2 windows and 15,364 ChEMBL27-only ligand vectors. Every array
and provenance record was checked against pinned reuse manifests, and every
complex row was independently reconstructed with float64 residue-weighted
pooling and compared bitwise. The feature audit passed at 03:44 UTC; matrix
SHA256 is `37935fcba3f59aeb450e273e1ded0d7b4ddc340b1fd1185577158d08dba2b400`.
No encoder inference, sample dropping or feature imputation occurred.

Five regression tests passed: both cache provenance checks; dependency release
requiring verified local delivery and matching hashes; failure blocking; five
GBDT overrides with a nonstopping monitor and training-only scaler; remote
launch syntax and PID/start identity. All experiment scripts compile. The
19,159-file source/input lock is
`b04e6f7f02900c03462f314101a1e4aba238245fd3ef41166189d68b1aea3577`.

A single detached local controller started at 03:45:14 UTC (PID 12694).
It waits for FS-AS's passing final/local-delivery audits and copied_locally
completion before any remote upload, preflight or full fits. Cornell's new
study directory did not exist at launch; the two existing study controllers
were verified and preserved. Four five-tree Cornell preflight checks, four
full fits, six evaluations and independent local delivery are subsequent
gates, not completed results. The existing monitor now includes all three
unfinished sequence experiments. Evidence:
`datasets/protein_ligand_prediction/experiments/sequence_esm2_chembl27_nmi_20260923/`.

## 2026-09-23 — FS-AR completed production and verified local delivery

Cornell completed four 10,000-tree single fits and six CASF evaluations.
The final audit passed at 10:34:12 UTC and independent local delivery at
10:34:29 UTC. All 19,066 × 1,792 float32 rows, memberships, labels, training-only
scaler means/variances and recomputed RMSE/PCC passed. All 36 output hashes
and the frozen source/input lock were checked by the delivery verifier.
A subsequent artifact review reconfirmed those output hashes, all 15,364
reused ChEMBL27 array/record pairs, all four scaler scale arrays (zero maximum
error), and three artifact links. Remote sklearn models were not unpickled locally.

Evidence: `datasets/protein_ligand_prediction/experiments/sequence_esm1b_chembl27_nmi_20260922/`
contains `FINAL_AUDIT.json`, `LOCAL_DELIVERY.json`, `DELIVERY_REVIEW.json`,
`COMPLETION.json` and `results/FINAL_REPORT.md`. Final audit SHA256:
`7846f9b0e776a9ab1d26749f20ded10c424b25fb9ea40916ed4a627cc79de36a`;
local delivery SHA256:
`870e99c19f33afdaa1a289b18c747d275036767d19b3d2c6ba6c3778318889f4`.
General-v2020R1 → CASF-2016: RMSE 1.251483 logKa/pK, PCC 0.836777.
Separate RMSE/PCC reports preserve all six comparisons. Both encoder and
GBDT changed from FS-AQ; single stochastic fits do not isolate either effect.
FS-AS remains training and FS-AT awaits FS-AS verified delivery. The monitor
remains active, and prior studies and frozen scientific scripts are preserved.

## 2026-09-23 — User-requested sqrt revision for both ESM-2 studies

Prepared isolated FS-AU (ESM-2/CPZ) and FS-AV (ESM-2/ChEMBL27) recipes with
`max_features='sqrt'`; the remaining five explicit GBDT settings and all other
installed defaults are unchanged. Each sample retains 1,280 protein plus 512
ligand features. Both controllers launched at 13:15:52 UTC with no cross-study
completion dependency. Cornell preflight and production status are recorded
in each experiment's current receipts; this entry does not claim completed fits.

Four regression tests per study passed after the final code changes, covering
six estimator overrides, 42 resolved candidate features, training-only scaling,
cache tamper rejection, independent launch and PID/start-tick identity. Both
full feature audits passed at 13:15:04 UTC: all 19,066 rows are bitwise equal to
independently reconstructed pooled vectors and all 8,830 protein plus 15,364
ligand cache arrays and provenance records match their pinned hashes. No encoder
inference, input changes, dropped samples or local remote-model unpickling.

FS-AS's three completed refined results and partial general progress are archived
with verified model/scaler/prediction hashes; its 7,401-tree general fit is not a
completed result. FS-AT's unstarted queue is superseded. Historical locks and
receipts, completed FS-AR/FS-AQ and selected FS-AN are preserved. The existing
monitor now follows FS-AU/FS-AV through eight full fits, twelve evaluations and
independent local delivery. The historical NMI max_features attribution was not
independently established; sqrt implements the user's explicit request.

Launch verification at 13:18 UTC: both Cornell preflights passed under sklearn
1.9.0, each with four five-tree smoke fits and max_features_ equal to 42. Both
remote controller PID/start-tick identities and all eight single-thread production
workers were verified, with nonzero tree progress in every training set and no
failure receipt. Per-study LAUNCH_REVIEW.json preserves the evidence.

## 2026-09-23 — FS-AU/FS-AV sqrt studies delivered

Completed the requested ESM-2/CPZ and ESM-2/ChEMBL27 sqrt revision: 19,066 ×
1,792 verified features per study, eight 10,000-tree models with fitted scalers,
and twelve CASF evaluations, delivered locally at 13:40 UTC. General-v2020R1
CPZ RMSE is 1.401586/1.389762/1.198154 and PCC 0.826199/0.805119/0.851202
on CASF-2007/2013/2016; ChEMBL27 RMSE is 1.442955/1.435526/1.252878 and PCC
0.813440/0.789548/0.837198. CPZ improves both metrics in all six matched rows,
including refined fits; single stochastic fits do not establish significance.
RMSE is logKa/pK. No CASF selection or change to selected FS-AN.

Authenticated both frozen source locks (19,157/19,156 files), 37 final output
hashes per study, passing full local pooling/cache provenance, membership,
label and scaler audits, and copied_locally completion. Recomputed all twelve
RMSE/PCC values; refreshed both reports using their frozen scripts and checked
all baseline values, eight full model/scaler hashes and six artifact aliases.
Historical FINAL_AUDIT/LOCAL_DELIVERY/COMPLETION receipts are preserved; new
DELIVERY_REVIEW records report/artifact checks without local model unpickling.
The report refresh needed the original PYTHONPATH=code environment; no frozen
code was modified. Updated package/workflow/sequence/dataset/study READMEs and
catalogue status. No installed API, numerical implementation or dependencies
changed; documentation-only completion needed no new training or tests.
FS-AS partial results and canceled FS-AT queue, FS-AQ/FS-AR/FS-AN and unrelated
studies remain intact. Retire the existing heartbeat after documentation checks.

## 2026-09-23 — Retain CPZ for reuse; archive ChEMBL27-only results

Applied the user's retention request to FS-AU/FS-AV. FS-AU retains all four
10,000-tree GBDT pipelines, embedded training-only scalers, standalone scaler
arrays, embeddings and results. RETENTION.json pins the retained model/feature
hashes, 8,830 protein-window and 15,364 ligand array/record pairs, and nine encoder
asset files. A matching Cornell sklearn 1.9.0 inference check loaded the general
pipeline and exactly reproduced three saved CASF-2016 predictions. REUSE.md
documents feature order, runtime and reuse without refitting or double scaling.

FS-AV settings, source/input locks, manifests, predictions, metrics, RNG/model
receipts and historical delivery receipts are archived in place. Only the four
pipeline.joblib and four scaler.npz files were removed from each machine after
hash verification, reclaiming 139,448,346 bytes on the Mac and the same on Cornell
(278,896,692 bytes combined). The FS-AV pretrained alias was removed and an
archive alias added. ARCHIVE.json and local/remote pruning receipts explicitly
supersede historical delivery receipts for current model availability. No model
retraining, embedding changes, installed API/default changes or global topology
selection changes occurred. Source and historical receipts remain unchanged.
Current package/workflow/dataset/study/catalogue READMEs now distinguish retained
CPZ from archived ChEMBL27-only results. No new software tests were needed;
validation checked actual retained/pruned artifacts and reusable inference.

## 2026-09-23 — Launch FS-AW previous-parameter CPZ comparison

At the user's request, prepared isolated FS-AW using byte-identical retained
ESM-2/CPZ features and the full original FS-AQ parameter dictionary: 10,000
trees, lr=.002, depth7, max_features=sqrt, min_samples_split5, subsample.8,
random_state0, n_iter_no_change=None and remaining installed defaults. Exact
previous parameters were confirmed against all four original model receipts.
FS-AU's retained NMI-inspired configuration stays unchanged. Compare all six
refined/general evaluations; differences include random-state policy and
multiple parameter changes, so no isolated effects or significance claims.

Five regression tests passed: cache provenance/tamper checks, original-factory
parameters and seeded prediction equivalence, full-tree/training-only scaler
behavior, no cross-study gate, remote PID/start-tick launch checks, and matched
comparison keys/counts with signed RMSE/PCC/MAE differences. All 19,066 pooled
rows and 8,830/15,364 protein/ligand array-record pairs passed reuse verification.
Frozen lock 80c54146591586aee2adcae8543b9b1cae6a826db5ec8c11650465cab0a8258a
covers19,166 files. Cornell sklearn1.9.0 preflight passed four five-tree smoke
fits and exact resolved-parameter equality at14:49UTC. Four production workers
were verified at14:50UTC with one BLAS/OMP thread each and actual tree progress.

Changed only isolated experiment adapters, reporting and documentation outside
the installed package; no installed API/default/shared-package changes. Existing
monitor reactivated for this study through verified delivery and documentation.
Preserve retained FS-AU, archived/pruned FS-AV, completed FS-AQ/FS-AR and selected
FS-AN. Full metrics are pending; no future results or ETA claimed.

## 2026-09-23 — FS-AW previous-parameter CPZ comparison delivered

Completed four full 10,000-tree models/scalers and six evaluations using the
unchanged 19,066 × 1,792 CPZ features and original memberships. Remote completion
15:32UTC, verified local delivery15:33UTC. Previous-setting general-v2020R1
RMSE 1.482571/1.453735/1.254792, PCC .808693/.787990/.840157 and MAE
1.156930/1.186632/.996120 on CASF2007/2013/2016. NMI-inspired FS-AU improves
all three metrics on all three general tests and refined2016. Previous settings
improve all metrics on refined2007 and PCC on refined2013; NMI-inspired settings
improve refined2013 RMSE/MAE. Full comparison changes four parameters including
random-state policy; single fits establish neither significance nor isolated effects.

Authenticated19,166 frozen source/input files and37 final output hashes, original
passing full pooling/cache/scaler delivery audit and copied_locally completion.
Four parameter dictionaries exactly match original FS-AQ receipts, four portable
scalers are bitwise identical to matching FS-AU scalers, all model progress
receipts have10,000trees, all six RMSE/PCC/MAE rows recompute from predictions,
and all signed comparison differences match actual FS-AU receipts. Three artifact
aliases resolve to the delivered features/models/results. No local unpickling.
The general fit plus artifact/evaluation work took2567.10seconds (observed).

Updated current package/workflow/sequence/dataset/study/catalogue READMEs and
completion documentation. No installed API, dependency, scientific source or
settings changed; no retraining or additional tests for documentation changes.
FS-AU retention/model/scaler hashes remain intact; FS-AV pruned assets remain
absent. Older FS-AQ/FS-AR and selected FS-AN are preserved; no automatic selection.
Historical source locks and FINAL_AUDIT/LOCAL_DELIVERY/COMPLETION remain unchanged.
DELIVERY_REVIEW and DOCUMENTATION_COMPLETION supplement the original receipts.
Retire the existing heartbeat once documentation verification passes.
