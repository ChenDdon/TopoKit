# Six-layer architecture

## 2026-09-24 — GitHub distribution

The standalone public repository is `ChenDdon/TopoKit`; cloning it does not
require the maintainer's sibling `topokit_dev` or dataset workspace. Releases
are tagged from the package version and publish a wheel, source archive and
checksums after the platform test matrix and installed-wheel smoke checks pass.
Temporary caches, generated outputs, transfer archives and operational status
files are excluded from the repository. The local development copy and source
research records remain outside this publication workflow. Scientific algorithms
and feature recipes are unchanged by release preparation.

## 2026-09-23 — Public examples and explicit sequence profile

The repository now supplies a generic FS-AN manifest exporter under
`workflows/protein_ligand_prediction/extract_features.py`. It composes the
installed reader/selection/compute API, writes canonical schema bytes, hashes,
per-sample outcomes and a complete matrix, and never changes the fixed topology
recipe. Data, CLI orchestration and agent skills remain outside `src/topokit`.
Ten byte-preserved example complexes and all tutorial notebooks stay in
`examples/` and are included in source distributions, not wheels.

The installed sequence encoder adds explicit `ligand_profile="cpz"` with pinned
checkpoint/vocabulary identities and a distinct feature recipe ID. Its existing
default remains `chembl27`; historical GBDT helpers keep their existing recipe.
Both profiles reuse the same protein pooling, ligand BOS inference and optional
lazy dependencies. There is no mathematical-core, topology-recipe or training
change. The repository sequence runner accepts explicit chains/SMILES and local
assets; no dataset lookup or model download is inserted into the package.

`topokit_dev` is the development copy; the sibling `topokit` is a synchronized
public release candidate. Both retain the same Python import/distribution name
and should be installed in separate environments. Neither folder is published
by this local preparation step. Older dated records below retain their context.

The normal numerical path is:

`readers → builders → core → postprocessing → optional ML workflow`

Visualization consumes data/results independently. Workflows call the layers
to organize experiments. Shared `data.py`, `results.py`, serialization,
exceptions, and validation are infrastructure, not a seventh scientific layer.

Installed `src/topokit/workflows` contains reusable compositions and the selected
protein–ligand application. Following the September 21 organization request,
`workflows.protein_ligand_prediction` is an installed package submodule: one
fixed FS-AN feature encoder, a model profile, and lazy optional GBDT inference.
It composes public readers, alpha builders and Hyperdigraph L0; no geometric
or Laplacian algorithm is copied into the workflow. The six-layer dependency
direction and optional-ML boundary are unchanged.

Dataset memberships, research launchers, ablation switches, trained weights and
historical source now live outside the package under
`datasets/protein_ligand_prediction/experiments/protocol_ablation` or linked
artifact stores. The final companion predictor is the general-v2020R1
three-seed GBDT ensemble. Each member retains its embedded scaler; the workflow
averages predicted pK values after applying each pipeline once. The prediction
API also accepts historical single-model bundles. Model weights are separate from the wheel;
only its small profile and the selected reference feature schema are packaged.
Recipe reference hashes preserve original generation provenance; the installed
implementation has its own independently recorded source hash.

Descriptions below are dated historical records where they describe the older
repository-only application boundary. They do not override this current layout.
The unrelated atom-deletion and H0-barcode projects retain their existing locations.

The protein-ligand v2 receipt follows this boundary: general/core-set
selection, the fixed six-statistic tensor encoding, parallel execution, and
artifact provenance remain under repository-level `workflows/`. It continues
to call the public Hyperdigraph builder and degree-zero Laplacian unchanged;
neither the dataset split nor the relative largest-eigenvalue summary is an
installed core definition. Source-specific label selection, CASF manifest
versioning, and safe archive extraction likewise remain in the application
preparation receipt and never enter a reader, builder, core, or installed
workflow API.

The 2026-09-13 recovery of 28 historical NMI/TopoTransformer working
structures follows the same boundary. The external source choice, additive
full-schema metadata manifest, source/release comparison, append-only feature
job, and publication of a distinct complete refined-manifest directory are
repository-application concerns. They do not extend the molecular readers,
change the 20-angstrom geometry, or modify Hyperdigraph construction or `L0`.
Preparation version 1.3.0 accepts repeatable supplemental metadata files only
as non-overriding additions and records each source in provenance; built-in
metadata remains authoritative for every existing ID.

The 2016 v2 ablation receipt also remains under repository-level `workflows/`.
Its masks, selected-element channel policies, fixed train-only selection
holdout, GBDT fitting, aggregate resume checks, and AWS snapshot/launch helpers
are application concerns. Channel geometry calls the public connection and
Hyperdigraph builders; full spectra call the native degree-zero core. The
receipt never constructs a substitute graph Laplacian or triangulates the
entire complex before inducing element channels. It changes no installed
reader, builder, core, or generic workflow API. Related source helpers ship
with the source distribution, while datasets and application receipts remain
excluded from wheels.

## Current protein–ligand application (2026-09-16)

The active repository receipt is now `workflows/protein_ligand_prediction/`.
Earlier application descriptions above and below are historical architecture
records; their code is under `archive/protein_ligand_prediction_20260916/`.
The final m2+m3+m5+r4 engine calls public exact-alpha construction and public
incremental ordinary Hyperdigraph L0. It exposes no ablation switches and
imports no archived workflow. Structure selection, ten spectral summaries,
55-channel encoding, manifest validation, AWS orchestration and GBDT remain
application concerns. No installed reader, builder, core or generic workflow
source changes. Higher-dimensional and two-scale persistent routes retain
their algorithms. Source distributions contain only the active application;
wheels contain none of these dataset/demo receipts.

## Boundaries

The **2026-09-20 Cornell H0 barcode experiment** is separate at
`workflows/protein_ligand_persistent_homology/`. Its 15 Å crop, 55 channels,
mixed-versus-Null edge policy, radius conversion, benchmark membership and
parallel ML remain application concerns. Each selected cloud calls the public
native alpha builder, then the public hyperdigraph H0 persistence core once.
The reusable `barcode_bin_counts(..., mode="cover")` postprocessor creates
100 full-bin coverage counts. The application saves original intervals and
feature tensors with checksums; independent audits use direct interval
predicates and full alpha simplices with SciPy graph components. No existing
geometry, homology or higher-dimensional algorithm changes.

The **2026-09-18 Cornell atom-deletion comparison** adds a separate repository
receipt at `workflows/protein_ligand_prediction_version2/`, leaving the fixed
alpha baseline intact. It selects a 12 Å field, supplies explicit unit-edge
VR graphs, and aggregates complete deletion spectra into 12,960 features.
The induced-L0 deletion algebra lives in `core/_hyperdigraph/l0_sweep.py` and
is exposed by `core.hyperdigraph.L0Sweep.vertex_deleted_laplacian`. Applications
do not copy the degree-update kernel. Exact component/twin reuse, missing-side
encoding, benchmark manifests, conda activation, parallel scheduling, model
fitting and audit receipts remain application concerns. Tests include direct
induced-incidence and D−A oracles; higher-dimensional algorithms are unchanged.
Both current application directories are included in source distributions;
neither is installed in wheels.

| Package | May use | Must not use |
| --- | --- | --- |
| readers | shared PointCloud, format parser dependencies | builders, core analysis, plots, ML |
| builders | shared data, mathematical object models, geometry helpers | homology/spectral analysis, postprocessing, plots, ML |
| core | defined object models, chain algebra, numerical primitives | readers, builders, postprocessing, visualization, workflows |
| postprocessing | numerical result records, numerical transformations | builders or topology/spectral recomputation |
| visualization | shared data/results, plotting primitives | builders or topology recomputation; result mutation |
| workflows | any public layer, optional estimator library | copies of mathematical/building/postprocessing algorithms |

Object-model operations such as inserting an explicitly supplied simplex,
validating a boundary, or forming a chain group are intrinsic representation
and algebra. They are not scientific-data-to-topology construction. Alpha,
Rips, Delaunay support, direction assignment, ID overlap mapping, and construction
recipes live in `builders/`. The core may analyze a prebuilt filtration at a
requested scale; selecting that subobject is not a new geometric construction.

Native kernels are private `core/_simplicial`, `core/_hyperdigraph`, and
`core/_interaction`. They preserve distinct algebra. Public `core/*.py` adapters
return shared result records and do not collapse the native objects into one
representation. Core import/use must work even when upper layers are unavailable.
The public `builders` and `core` parent namespaces resolve the `simplicial`,
`hyperdigraph`, and `interaction` route modules lazily on attribute access.
This namespace convenience does not change dependency direction: importing a
parent alone does not preload all route implementations.

The incremental ordinary-L0 accumulator lives in
`core/_hyperdigraph/l0_sweep.py`, with the public `core.hyperdigraph.L0Sweep`
adapter. It consumes an explicit filtered object, checks singleton-face birth
closure, and updates its native L0 matrix from supplied edge events. It never
constructs geometric support or changes an alpha birth. The generic spectral
workflow chooses this guarded L0 path, while higher degrees and genuine
persistent pairs retain their existing core algorithms. The alpha application
1.2.0 constructs an explicit edge filtration and calls the public sweep; no
matrix-update algebra is copied into the application. The completed 1.1.0
source and scientific outputs remain separate historical artifacts.

## Extending a reader

Implement a callable `read_format(path, **options) -> PointCloud`, preserving
units, source records, IDs, charges/labels/cell/other attributes in metadata.
Registration uses `readers.register_reader("format", read_format,
extensions=("ext",))`. An isolated `ReaderRegistry` can be used per application.
Unsupported formats fail explicitly. Built-ins cover CSV, native JSON point
clouds, single-frame XYZ, PDB, MOL2, single-record SDF/MOL (V2000 and V3000),
PDBQT, and one atom-site loop from CIF. Native JSON canonically requires a
`coordinates` array (`points` is an exclusive alias);
optional IDs, labels, weights, units, and metadata are validated without
guessing missing values or allowing duplicate object keys. JSON provenance
sidecars without either coordinate key fail explicitly. The 29-row
`examples/data/data_material_from_cif.json` fixture demonstrates the canonical
form with Cartesian coordinates, stable source-site IDs, element labels, and
explicit Pauling weights derived from `data_material.cif`. Molecular readers keep atom/site columns, source records,
charges, declared bonds, unit cells, and data fields where their format supplies
them. PDB model selection and single-structure constraints are explicit. CIF
fractional sites are transformed with the declared unit cell; symmetry and
periodic-image expansion are intentionally not performed by a reader.
Molecular-reader metadata separates immutable source provenance (`source_*`,
raw declarations, cells, tags, properties) from the current point selection.
`PointCloud.subset` updates current counts and induces declared bonds/PDB
connectivity on selected stable IDs while retaining the original records.
An explicit reader-layer attribute helper assigns offline Pauling weights to
element symbols and records the source. It does not select a builder or alter
geometry; all raw parsers still return uniform weights.

A parser does not infer bonds, choose alpha versus a graph, orient hyperedges,
or manufacture physical feature rules. Metadata preservation does not imply
that the downstream builder understands every attribute, such as periodicity.

## Extending builders and postprocessing

The explicit optional exact-alpha route delegates geometric construction to
GUDHI inside `builders/_alpha_gudhi.py`, then returns the same native explicit
simplicial object. GUDHI is imported lazily only when `backend="gudhi_exact"`
is selected; default native construction and every mathematical core remain
independent of it. The adapter performs no homology, spectra or feature work.
The historical protein-ligand alpha application selected that backend.
Since 2026-09-17 the active application selects the GUDHI-free native builder
in `builders/_alpha_native.py`: batched circumsphere calculations, empty-ball
checks, rational degeneracy repair and coface propagation all remain in the
builder layer. SciPy supplies candidate triangulation. No geometric algebra
is copied into the molecular workflow; it passes retained edges to the same
native L0 core. Optional GUDHI use is limited to explicit legacy/reference
calls. See [native alpha contracts](NATIVE_ALPHA.md).

New construction recipes return `Topology(kind, native, cloud, metadata)`,
where `native` is well-defined for the chosen core. Preserve raw/effective
filtration values, units, ID maps, bounds, and the scientific construction rule.
`register_builder` extends recipes for supported mathematical families; a new
family still needs a defined core and dispatch integration.

Postprocessing consumes `PersistenceResult` or `SpectrumResult`. Scalar spectral
callbacks receive copies, never live eigenvalue storage. Built-in barcode
histograms, variance, raw spectral moments, spectral/Laplacian energies,
entropy, and future token/image encoders are separate transformations.
`barcode_bin_counts` likewise consumes only recorded intervals: explicit
overlap, full-bin coverage and finite-death modes live in postprocessing,
with no builder or core imports. Sorted endpoint searches and range additions
avoid a bars-by-bins allocation. The result uses the existing `VectorResult`
serialization and schema validation; topology and homology algorithms are
unchanged. Molecular channel rules and any alpha-unit conversion belong to
the repository-level barcode experiment, not this numerical API.
Centered Laplacian spectral energy uses a complete full spectrum, including
zero modes; it is unavailable for a partial result. It equals classical graph
Laplacian energy only for a combinatorial graph L0 and is an explicitly
generalized functional for other operators or dimensions. Undefined values and
every statistic's selection scope remain explicit. The sole zero-fill route is
an explicit postprocessing policy for predefined summaries of a core-confirmed
complete zero-dimensional operator; custom callbacks still execute. The raw
spectrum remains empty, and partial or nonempty operators are never filled by
that policy. A feature matrix requires a consistent schema, including
units, binning, representation, fill policy, and relevant mathematical
definitions.

## Visualization and optional ML

Views are disposable projections of authoritative numerical records. Interaction
views show factors and overlap, not a claimed embedding of the full quotient
chain. General high-dimensional embeddings are not inferred automatically.

`visualization.objects` draws explicit cells using shared point records;
`visualization.topology` reads supplied native objects through their public
cell accessors without importing core or builders. Scale/dimension selection
only changes the view. `visualization.blocks` provides illustrative canonical
layouts for one supplied simplex or ordered hyperedge, not geometric scientific
construction. `visualization.stationary` composes those supplied-cell views into
fixed-scale graph, simplicial, Hyperdigraph, and two-factor interaction figures.
`visualization.plots` displays supplied numerical records:
`plot_barcodes` and `plot_betti_curves` consume one `PersistenceResult`, and
`plot_spectral_summary_series` consumes an aligned sequence of already-computed
spectral-summary records. Betti values are interval queries on that persistence
record; spectral values are read directly from the summary records. These
functions do not import or invoke builders, core analysis, or postprocessing.
The barcode initial-stage marker is display metadata only and never rewrites a
birth coordinate.
The stationary plotting module may consume a `StationaryResult` for
diagnostics, but it never calls core or a builder.
`workflows.analyze_stationary` is the corresponding numerical
composition point: it calls public core and postprocessing functions once at an
explicit scale and returns the shared result record. GIF frame schedules and
writers remain in examples or applications. Style records are independent of
Matplotlib; plotting imports remain lazy. Explicit export uses local font settings. See
[the visualization guide](VISUALIZATION.md) for these public interfaces.

`workflows.ml.make_estimator` creates an unfitted gradient-boosted tree only on
request. `fit_features` sees only caller-supplied training features and targets.
There is no invented target, implicit train/test split, tuning loop, or model
fitting during topological analysis. Custom fit/predict estimators are accepted.
The optional defaults delegate to scikit-learn, not reimplemented tree algorithms:
[regression API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
and [classification API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html).

The repository-level
`workflows/protein_ligand_prediction_v1/generate_features_protein_ligand_v1.py`
is a second feature application receipt. It does not alter the original
protein-ligand generator or any installed API. The receipt selects four
singleton protein elements and ten singleton ligand elements, including
hydrogen, in protein-major Cartesian-product order. It asks the public
Hyperdigraph builder and core for each complete ordinary L0 spectrum, then
applies its declared five-value application summary. It is not an alternative
Hyperdigraph definition or Laplacian implementation.

The v1 numerical boundary is explicit: eigenvalues and summary arithmetic use
`float64`; numerical zero is `abs(lambda) <= 1e-10` with no five-decimal
pre-rounding; the stored tensor is C-order `float32`. Matrix size is repeated at
each scale so the artifact remains rectangular `(5, 100, 40)`. The two relative
statistics divide the minimum positive eigenvalue and positive-spectrum mean
absolute deviation, respectively, by the positive-spectrum mean. Empty
positive spectra use declared zero sentinels for all three positive-only
statistics while matrix size and nullity preserve the distinction. These are
application feature semantics, not new postprocessing defaults. The receipt's
NMI-inspired geometry does not make its changed channels and summaries a paper
replication.

The adjacent repository application `train_sklearn_gbdt.py` consumes this
completed schema without calling a reader, builder, core, or postprocessor. It
fits a two-step sklearn `Pipeline(StandardScaler, GradientBoostingRegressor)`
on explicit training-manifest rows only. Raw test matrices enter only
`Pipeline.predict`, so the already fitted training scaler is reused and test
data cannot contribute preprocessing statistics. The target is not scaled.
Both estimator and preprocessing dependencies remain optional and outside the
installed package API.

The v2 sibling
`workflows/protein_ligand_prediction_v2/train_sklearn_gbdt.py` uses the same
application boundary while binding the completed compact-six general-set
contract: C-order `(6, 100, 40)` tensors flatten to 24,000 coordinates, and
only the 18,498 explicit training-manifest rows enter `Pipeline.fit`. Raw CASF
rows enter `Pipeline.predict`, which reuses the training-fitted scaler; the
target is unscaled. For this width, native `max_features="sqrt"` resolves to
154 candidates independently at each split; the other explicit overrides are
learning rate 0.01, 10,000 depth-seven stages, random state zero, and verbosity
zero, with installed defaults otherwise. The three CASF manifests retain
their separate 195/195/285-row evaluations. There is no implicit validation
split, tuning, cross-validation, or early stopping.

The v2 trainer writes an immutable plan before fitting and refuses a
pre-existing output directory. It rechecks manifests, feature schema, receipt,
shared prediction helper, and selected feature identities before publication;
persists the fitted scaler and regressor together; verifies predictions after
joblib reload; and writes completion metadata last. These transactional and
provenance checks remain repository-application behavior, not installed
workflow or mathematical-core APIs.

The sibling `prepare_refined_intersections.py` and
`train_refined_sklearn_gbdt.py` receipts remain at the same repository-only
boundary. Preparation version 1.3.0 uses explicit, hash-pinned release sources
only to form the three version-matched refined-minus-core memberships. A
separately hash-pinned, preserved Weilab `Benchmarks_labels.zip` supplies the
NMI training point targets and training-row order after its training membership
is proved equal to the release-defined set. Core IDs and targets are audited by
PDB ID against the bundle while local test-manifest row order is retained; all
675 core target mappings pass. The original TopoFormer download URL is
unavailable and no upstream byte checksum is known, so the pinned local archive hash is the
workflow's byte-identity authority. The frozen version-1.2 preparation and
model run records 1,097/2,749/3,759 usable rows from full
1,105/2,764/3,772 memberships; it neither invents a split nor changes a tensor.
Version 1.3 may add metadata for feature-available IDs from one or more
separately audited supplemental manifests, but cannot override an existing
metadata row. The independently validated append of the 28 recovered structure
pairs closes all three historical feature gaps, and version 1.3 publishes
1,105/2,764/3,772-row manifests in a new output directory with zero omissions.
The old intersection manifests and models remain immutable; no retraining was
performed as part of data repair.
All 195/195/285 matching core rows are complete and each matching train/core
overlap is zero. The 285-row CASF-2016 core is deliberately distinct from the
290-row PDBbind-v2016 core.

Trainer version 1.0.1 consumes exactly one such training manifest and its
matching CASF manifest per invocation, rejecting mismatched benchmark years and
non-exact labels in the production protocol. It cannot pool benchmark cores or
infer additional test sets. C-order compact-six tensors flatten to 24,000
coordinates; `StandardScaler` is fitted only by the supplied training rows and
the target remains unscaled. Its explicit GBDT settings are learning rate
0.002, 10,000 stages, maximum depth 7, per-split `max_features="sqrt"`,
`min_samples_split=5`, `subsample=0.8`, and random state zero. These single
TopoKit-feature models are neither the NMI ensemble nor an exact reproduction
of the paper's full feature stack. Corrected AWS production and independent
artifact validation completed under
`datasets/protein_ligand_prediction/models/sklearn-gbdt-standardized-compact6-refined-pairs-nmi-20260913-v2/`.
CASF-2007/2013/2016 `(RMSE, MAE, PCC)` values are
`(1.4767356443471618, 1.153354580527072, 0.8062946512163276)`,
`(1.5179520255834025, 1.2686667834755287, 0.7745888010556891)`, and
`(1.3234052487088577, 1.0583577486640605, 0.828697281199449)`. The passing audit
JSON has SHA-256
`3146ecda37732f776b90baa10cbf47ac36fcb82389b6eebb318417e21f2e8aa4`.
These results remain operational application evidence rather than installed
workflow or mathematical-core behavior.

The completed AWS/local production artifact preserves this boundary: it is a
seven-file task output under
`datasets/protein_ligand_prediction/models/sklearn-gbdt-standardized-compact6-general-nmi-20260913-v1/`,
not package source or an installed default. Independent reload and evaluation
rehashed the 540 unique CASF inputs and reproduced every prediction and metric;
the operational evidence is recorded in [VALIDATION.md](VALIDATION.md), not in
the mathematical layers.

The repository-level
`workflows/protein_ligand_prediction/train_xgboost.py` is intentionally more
specific and is not an extension of that installed API. It consumes the frozen
protein-ligand tensor schema, a named PDBbind training manifest, and three named
CASF test manifests. It validates train/test ID separation, fits only after an
explicit command, and evaluates each CASF edition independently. It does not
call a reader, builder, core, or postprocessor and cannot change the feature
recipe. XGBoost is imported lazily inside the receipt, so it remains absent from
the required package dependency graph.

The application fixes estimator arguments but does not infer a validation set,
tune them, or inspect CASF results during fitting. Its `sqrt` spelling is a
receipt-level parser convention rather than an XGBoost value: for `p` flattened
features it resolves to the numeric fraction `1/sqrt(p)`. Per-tree
`colsample_bytree=sqrt` and per-node `colsample_bynode=sqrt` are different
experimental policies, and the latter with `colsample_bytree=1` is the closer
analogue of scikit-learn's per-split `max_features="sqrt"`.

The parallel repository application `train_sklearn_gbdt.py` uses
scikit-learn's classic `GradientBoostingRegressor` directly. It shares
`prediction_common.py` with the XGBoost and transformer receipts, so manifest
membership, feature-record hashes, C-order flattening, label policy, CASF
metrics, and transactional output rules are identical. It fixes only the
requested learning rate 0.01, 10,000 stages, depth 7 per tree, and native
`max_features="sqrt"`, plus random state zero for reproducibility. Remaining
estimator parameters come from the installed scikit-learn defaults; the full
resolved mapping and library version are recorded in each run.

Native scikit-learn square-root sampling is applied at every split, unlike the
XGBoost receipt's current per-tree shorthand. Classic
`GradientBoostingRegressor` exposes no `n_jobs`, so this receipt does not invent
parallel estimator semantics; one fit is single-process. Scikit-learn is loaded
only when the application is invoked through the existing optional `.[ml]`
extra. The receipt does not alter `workflows.ml`, the installed dependency
graph, any topological core, or the frozen feature store.

The companion `train_transformer.py` uses the same task-local
`prediction_common.py` data, hashing, and evaluation contracts. XGBoost's
original helper names remain importable from its script; estimator arguments
and behavior are preserved. The transformer and its NumPy preprocessing live
in `transformer_model.py`, which imports PyTorch only when a model is built.
There is no dependency change in the installed package.

This model reshapes existing tensors into filtration-scale tokens and trains
random weights with supervised labels. Validation is optional and requires an
explicit `--validation-fraction`; it draws only from the training manifest.
Preprocessing is fitted after that split. An explicit `--refit-full-train`
initializes a fresh model and preprocessing on all training rows for the
internally selected epoch count. CASF never selects checkpoints or supplies
normalization statistics. `predict_transformer.py` verifies the checkpoint,
recipe, schema, and feature records before applying saved preprocessing and
weights to unlabeled tensors. See the
[application protocol](../workflows/protein_ligand_prediction/TRANSFORMER_PROTOCOL.md).

## Data, documentation, and distribution

All repository datasets and demo-specific code are in `examples/`; generated
outputs default to `examples/output/`. Source distributions may contain examples
in that folder. Installed wheels contain library code only, without datasets,
demo loaders, examples, or Markdown scientific notes. Markdown records are
centralized here; root README/NOTICE/LICENSE remain standard project entry files.
The documented elemental-property dictionary is a runtime scientific reference
constant, not an example dataset; it is shipped as reader-layer Python code.

## Exact-coordinate selection (2026-09-17)

`PointCloud.unique_coordinates()` belongs to the domain-independent data layer;
it selects existing records without changing coordinates. Alpha builders use
the shared first-row coordinate-index routine before geometric construction.
Readers remain lossless; core operators receive the resulting explicit unique
vertices. The molecular application decides the component ordering and applies
the operation before element channels, recording every selection. No deduplication
policy is inserted into generic mathematical cores or unrelated graph builders.


## 2026-09-20 — HPCC application variant

`workflows/protein_ligand_prediction/hpcc_15a/` contains the explicit 15 Å crop,
alpha-radius grid, immutable transfer/recipe metadata, scheduler gates and
application audit. It reuses baseline parsing/statistics/training helpers and
composes public native-alpha and Hyperdigraph L0 APIs. No installed source
or higher-dimensional mathematical algorithm changes. Its data and outputs
remain in the HPCC project and separate local experiment tracking folder.

## Supervised TopoFormer preparation and GBDT delivery (2026-09-21)

The installed `workflows/topoformer` namespace is reserved with an empty public
API and no torch imports. Model, configuration and checkpoint implementations
are planned there; no DL algorithm is added at this stage. Protein–ligand
configuration files and the training/inference contract live under repository
`workflows/protein_ligand_prediction/dl`. Dataset-specific ablation plans,
selection IDs, output folders and all artifacts live outside the package in
`datasets/protein_ligand_prediction`. Pretraining is explicitly disabled.

Repository `ml/export_bundle.py` and `ml/predict.py` export/validate/use existing
scikit-learn pipelines. They do not retrain, construct topology, or change the
fixed feature recipe. A pipeline contains its fitted StandardScaler; separate
NPZ scaler arrays are redundant inspectable artifacts. The application enforces
recorded runtime, feature-schema and content hashes. Source distributions now
include the new DL JSON template, representative strategy receipt and ML
runtime files. Wheels exclude application data and models.

## 2026-09-21 — Package application and external study catalogue

The former `archive/protein_ligand_prediction_20260916` source tree and mixed
20/15 Å application were relocated outside TopoKit, with all 215 tracked source
file hashes preserved. The installed selected module is self-contained and has
no dependency on an archive, dataset root, sibling script or `sys.path` mutation.
Its feature schema is byte-identical to the model's reference schema. Reorganized
feature functions retain their numerical arithmetic; an implementation receipt
makes the current file hash distinct from the frozen original engine hashes.

`workflows/protein_ligand_prediction/` now provides detailed recipe documentation,
selection receipts, thin API/CLI wrappers and the future supervised-DL contract.
Model-fitting and hyperparameter experiment records belong to the external study.
The catalogue assigns feature IDs FS-A…FS-AP, study IDs ST-00…ST-09, and retains
original run IDs independently. Backend/duplicate-policy revisions are explicit;
repeated controls and altered model hyperparameters do not silently become new
feature strategies. Historical research code is excluded from current distributions.

## Optional compact TopoFormer (2026-09-22)

`workflows.topoformer` now provides model/configuration, scale tokenization,
fixed positions and verified prediction bundles. PyTorch imports occur only
when building/loading a model. Dataset selection, ablation orchestration, GPU
launching and weights live outside the package in the general-v2020R1 study.
No geometry/core/postprocessing algorithm or feature recipe changes.

## Sequence modality in workflows (2026-09-22)

`workflows.protein_ligand_prediction.sequence` composes frozen ESM-2 and ChEMBL27 encoders and optional GBDT fitting/inference. The attributed compact ChEMBL loader is private `_chembl.py`; it is loaded only after verification of the trusted supplied checkpoint. PyTorch, Transformers and scikit-learn imports remain lazy, with a new optional `sequence` extra. The six scientific layers are unchanged; no reader, builder, core, postprocessor or topology recipe is modified. Dataset-specific protein-chain and SMILES preparation, exceptional-valence receipts, fixed split manifests, schedulers, weights, caches and trained bundles live in the external FS-AQ study. The wheel contains reusable code, not dataset runners or model weights.
