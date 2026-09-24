# Construction and feature contracts — 0.3

These additions refine [CONTRACTS.md](CONTRACTS.md). The six-layer boundary
remains unchanged: only builders choose geometric objects; core computations
consume already defined objects; postprocessing never recomputes topology.

## 2016 v2 ablation application contract

The repository-only `workflows/protein_ligand_prediction_v2_ablation` receipt
uses the complete 3,772-row refined-2016 training and 285-row CASF-2016
manifests, fixed by their byte hashes. These are membership/label contracts;
the existing structure representation is the v2-compatible updated archive
plus recovered historical working structures, not a claim that every input
matches original 2016 release bytes. Earlier incomplete manifests, feature
stores, and trained model results remain immutable.

Baseline geometry retains the supported `C/N/O/S` protein crop and
`C/N/O/S/P/F/Cl/Br/I/H` ligand groups. Modification m1 restricts cross edges
to selected-channel Delaunay support; m2 adds positive-spectrum quartiles and
entropy; m3 uses 50 stop-exclusive scales `0.0,0.2,...,9.8`; m4 adds `all`
groups with Delaunay cross support; m5 adds `null` groups with Delaunay cross
support and selected-component internal Delaunay support for null channels;
m6 permits internal and cross edges on selected-channel Delaunay support.
m4, m5, and m6 each subsume m1, which is removed from canonical masks.
Their other effects remain separate. Baseline plus all singles/pairs/triples
has 30 unique canonical recipes; default adaptive selection evaluates only
the top four singleton modifications' unique pairs/triples.

`all` is a union of supported groups, not permission to add chemistry.
Null-null remains an explicit zero channel. Intentional null-side absence
differs from an absent expected element: valid single-component channels
retain vertices and numerical nullity, while structurally absent channels
carry a false presence mask. Each channel is triangulated on its own selected
points; inducing channels from global Delaunay support is not equivalent.
All distances and coordinates remain unchanged. Degeneracy and resource
errors never trigger jitter, deduplication, pruning, or imputation.

The six frozen complete-spectrum summaries remain first. Added quartiles
use linear interpolation over eigenvalues strictly greater than `1e-10`;
added entropy is `-sum(p*ln(p))` for normalized positive eigenvalues,
unnormalized and measured in nats. Empty positives yield zero derived
summaries, and singleton positives yield zero entropy. Full-spectrum size
and numerical nullity are retained separately. Computation is float64 and
canonical storage is finite little-endian float32, C order. Every feature
schema fixes groups, scales, summary definitions, source hashes, and masks.

An explicitly declared SHA-256 ID-ranked holdout with seed 2016 reserves
755 of the training rows, leaving 3,017 for validation fitting. The original
manifest order is retained within partitions. Baseline and six singles are
evaluated first; the top four singles by holdout RMSE determine all distinct
pair/triple candidates. CASF scores do not select combinations. Full core
fits then use all 3,772 training rows and all 285 evaluation rows with
train-only `StandardScaler`, unscaled targets, and the refined-v2 GBDT:
10,000 stages, learning rate 0.002, depth 7, `max_features="sqrt"`,
`min_samples_split=5`, `subsample=0.8`, and random state zero.

Feature and model successes require validated source/input/schema/output
identities. Resume also verifies completed selection/report aggregates
against their referenced model runs. Explicit smoke limits are rejected
outside smoke mode and never become performance evidence. AWS staging,
durable launch, status inspection, and result copy-back are separate commands;
neither importing the receipt nor copying source launches training. Sync
never deletes remote or unmatched local artifacts, and source helpers are
included in source distributions without entering installed wheels.

The [application README](../workflows/protein_ligand_prediction_v2_ablation/README.md)
contains exact manifests, paths, commands, layouts, and current execution
status. These policies do not change the installed mathematical contracts.

The three public route modules are available both by explicit child import and
as lazy attributes of their parent layers: `builders.simplicial`,
`builders.hyperdigraph`, `builders.interaction`, and the corresponding names in
`core`. Access is deterministic in a fresh process and does not depend on a
previous dispatch call or unrelated test import. Merely importing a parent
continues to avoid loading all route implementations.

The independent visualization extension is documented in
[VISUALIZATION.md](VISUALIZATION.md). Object views select only explicitly
supplied cells, preserve stable IDs and sequence order, and use inclusive
filtration thresholds inside the recorded domain. Exact display dimensions
do not change construction caps or remove q+1 information from the object.
No clique completion, inferred singleton hyperedges, coordinate perturbation,
topology recomputation, or silent sampling occurs. Above degree three, simplex
views are labelled 2-skeleton schematics. Plotting remains optional and lazy;
functions return axes/figures, and saving requires an explicit export call.

Result-curve plotting preserves the same boundary. `plot_betti_curves` accepts
one existing `PersistenceResult`, a finite strictly increasing scale grid inside
its recorded domain, and unique available dimensions. It queries the stored
intervals and does not rerun persistent homology. `plot_spectral_summary_series`
accepts a nonempty ordered sequence of aligned scalar summary records, with one
strictly increasing scale per record supplied explicitly or in metadata. Records
must share their ordered names and, when present, one nonnegative Laplacian
dimension; infinities are rejected and NaNs remain plot gaps. The default
primary curves are `min`, `max`, and `mean`; `laplacian_energy` is shown on a
secondary axis. The function does not compute spectra or summaries.

`plot_barcodes(..., mark_initial_stage=True)` uses open circles to distinguish
bars already present at the recorded filtration start, where the displayed
left endpoint may be a truncation boundary. Setting
`mark_initial_stage=False` removes only this visual cue and is appropriate when
births are independently known to be exact, including the tutorial's H0
vertices born at zero. Neither setting changes the supplied birth values or
the right-censoring convention for terminal infinite bars.

The fixed-scale composition API follows the same boundary. Calling
`workflows.analyze_stationary(topology, scale=s, dimensions=degrees)` computes
GF(2) homology through the largest requested degree and ordinary real
Laplacians only for the requested degrees. Its `StationaryResult` retains the
authoritative `HomologyResult` and complete `SpectrumResult` records alongside
derived scalar summaries. Missing display degrees may be represented by NaN
summary rows; they are not fabricated spectra.

`visualization.plot_stationary_*` functions consume an existing object, and
`plot_stationary_diagnostics` consumes a `StationaryResult`. Display-only
Hyperdigraph segment deduplication or cell subsetting never edits the topology
or changes numerical analysis. Proximity circles use an explicit caller-
supplied radius and do not reinterpret native filtration units. Interaction
views retain two factor panels and explicit overlap because no quotient-chain
embedding is inferred. Animation scheduling and GIF writing are outside this
library contract.

The 2026-09-05 readability refactor changes local identifiers only: public
parameters, attributes, result keys, numerical definitions, and resource
defaults are retained. Pressure-test budgets and thread limits are explicit
example-run settings, not new library defaults. See the
[review](CODE_READABILITY_2026_09_05.md) and
[benchmark protocol](../examples/pressure_test/README.md).

Pressure matrices separate requested q0, q1, and q2. The additional compact
hyperdigraph H0/H1 suite selects the existing public score-directed workflow
only for distinct weights, with complete or Delaunay edge support. Its packed
representation and implicit two-path handling are recorded separately from
native explicit-generator counts. It exposes no canonical allocation-budget
arguments; process wall/RSS/address-space limits remain explicit. A small
canonical probe reports exact endpoint equality separately from declared
floating-point tolerance and never certifies every large-case interval.
No production dispatch, spectrum accuracy, filtration definition, or default
guard changes as a consequence of selecting this benchmark route.

## Points, weights, and bonds

`PointCloud` carries coordinates, stable string/integer IDs, finite scalar
weights, and scientific metadata. Raw inputs default to equal weights.
`weights=None` on a builder preserves weights already on a PointCloud;
an explicit vector or complete ID-to-weight mapping overrides them.
The XYZ reader does not automatically infer weights. Call
`readers.assign_element_weights(cloud)` to use the sourced Pauling table;
missing/unknown entries fail unless explicitly resolved. See
[reference data](REFERENCE_DATA.md). Weights affect hyperedge orientation,
not alpha geometry, filtration radii, or Hodge inner products.

The native JSON reader accepts one root object with a canonical required
nonempty, rectangular, finite numeric `coordinates` array. `points` is accepted
as an exclusive alias; supplying both coordinate keys is an error. Optional `ids`, `labels`, and
`weights` arrays contain exactly one value per point; IDs are stable strings or
integers, labels are strings, and weights are finite JSON numbers. Optional
`coordinate_units` is a nonempty label and `metadata` is an object whose
non-reserved keys are copied to `PointCloud.metadata`. Optional identity fields
are `schema="topokit.point_cloud"` and integer `schema_version=1`. Unknown
top-level fields, duplicate JSON keys, booleans in numeric arrays, non-finite
constants, reserved metadata collisions, and conflicting embedded/caller units
fail explicitly. A JSON file without `coordinates` or its `points` alias is
metadata/provenance, not a point cloud; extension dispatch does not reinterpret it.
The canonical example `examples/data/data_material_from_cif.json` contains 29
Cartesian coordinate rows derived from `data_material.cif`, with the source
site IDs, canonical element labels, and explicit Pauling-electronegativity
weights aligned row for row.

PDB, MOL2, SDF, MOL, PDBQT, and CIF readers return the same `PointCloud`
contract and retain canonical element labels plus row-aligned format columns.
PDB-family model selection is explicit. Blank PDB formal-charge fields map to
`None`; standard signed charge tokens and the explicit neutral token `0` map to
integers, while malformed nonnumeric tokens fail. MOL/SDF accept V2000 or V3000 but one
structure record per call; MOL2 likewise accepts one molecule. CIF accepts one
atom-site coordinate loop, uses Cartesian sites directly or converts fractional
sites through all six declared cell parameters, and does not expand symmetry or
periodic images. Declared bond records are preserved as metadata only. Raw
reader weights remain one until an application calls an assignment helper or
supplies another explicit vector.

Initial molecular clouds report equal `atom_count`, `source_atom_count`, and
`selected_atom_count`. A stable-ID subset retains the original source count,
IDs, raw declarations, `source_bonds`, and source PDB connectivity; it updates
current counts and exposes only bonds/connectivity induced by the selected IDs.
Cells, CIF tags, SDF properties, and non-atom records remain attached as source
provenance rather than being misrepresented as row-aligned selected data.

Use stable-ID pairs such as `[("a", "b"), ("b", "c")]` for bonds. IDs are not
implicitly interpreted as array positions. With default integer IDs 0..n−1,
they coincide with indices. `bonds_from_adjacency(matrix, ids=cloud.ids)`
converts binary dense/sparse adjacency with a mandatory row/column ID order.
Undirected matrices must be symmetric; loops and malformed entries are rejected.
Weights/birth values are not smuggled into a binary adjacency matrix.

| Construction | Missing bonds | Supplied bonds | Filtration units |
| --- | --- | --- | --- |
| Alpha (default) | Delaunay alpha geometry | Rejected | Alpha squared radius |
| Rips | Complete distance graph | Rejected | Edge distance |
| Graph / flag | Delaunay edges | Replace inferred pairs | Edge distance |
| Digraph / sequence-hyperdigraph | Delaunay edges | Replace inferred pairs | Edge distance |

`bonds=[]` means no edges, not missing input. `cutoff=None` imposes no extra
distance pruning. A finite cutoff retains edges of length <= cutoff and applies
to supplied bonds too; removed pairs are recorded. Standard alpha rejects
distance cutoffs: use its native `max_scale`/`filtration_range` instead. A
restricted-alpha construction is not implemented. Flag means clique completion
of the chosen support, not standard alpha or automatically full Rips.

Digraph is the degree-one case. Hyperedges use ordered distinct-vertex sequences
with consecutive directed support; they are not required to be directed cliques.
Direction follows lower to higher assigned weight. Equal weights retain both
orientations without index tie-breaking or perturbation. A graph/digraph stays
degree one unless the corresponding explicit higher-order construction is chosen.

## Ranges, persistence, and Laplacian observations

`filtration_range=(start, end)` is an inclusive **construction domain** in the
selected builder's native units. It is not a two-point observation schedule.
Absent bounds mean start 0 and all finite constructed events, subject to explicit
resource guards. Nondefault legacy `filtration_start`/`max_scale` must agree with
a supplied range. Births below a positive start are clamped, not translated;
initial bars may be left-truncated, and terminal infinite bars in a bounded
domain are right-censored. They are not automatically essential in a larger object.

`core.persistence` uses the built object's actual events by default. Observation
scales do not change these barcodes. `core.homology(..., max_dimension=q)` and
`core.laplacians(..., max_dimension=q)` return degrees 0..q. Where construction
permits, builders include q+1 cells to allow deaths and the full up-Laplacian.
Explicit lower skeleton caps are meaningful truncations and are reported.

`workflows.laplacian_series(obj, max_dimension=q, scales=[...])` returns
`scale -> degree -> SpectrumResult`. Ordinary snapshots are the default.
`scales=None` chooses finite critical grades; the default guard allows at most
256 observations, and an explicit larger `max_snapshots` is required beyond it.
Eigenvalue vectors can have different lengths at different scales; no artificial
padding or truncation aligns them. Matrices and eigenvectors are opt-in.
A curve assembled from these single-scale ordinary snapshots is a multiscale
trajectory, not a persistent-Laplacian spectrum.

For genuine two-stage operators, use `core.persistent_laplacian(start=s, end=t)`
or `laplacian_series(mode="persistent", scale_pairs=[(s,t), ...])`. No quadratic
all-pairs grid is generated implicitly. Higher degrees use the general algebra
with existing resource guards; no claim of universal high-dimensional efficiency
or parity with other platforms is made.

## Two-factor interaction

Only k=2 is supported. One cloud means two factors with full vertex overlap.
Two clouds require a partial one-to-one map `overlap_vertices=[(id_a,id_b), ...]`
(the existing `overlap_pairs` name remains available). Input rows and IDs are
never reordered to manufacture overlap. A matched vertex may have different
local IDs; internal factor namespaces distinguish all unmatched vertices.

`overlap_simplices` contains paired unordered simplices, represented by sequences
of vertex IDs, for example `[(("a","b"), ("x","y"))]`. Such an annotation must
exist in both factors and agree with the declared vertex map. It validates and
documents a higher correspondence; it does not restrict the quotient chain,
insert missing simplices, or replace the native interaction definition.

Construction choices/ranges remain in `factor_options` (shared or two separate
dictionaries). A `filtration_a` numeric schedule and optional `filtration_b`
define paired observation thresholds. Missing b copies a, while both supplied
arrays must have equal lengths and be individually nondecreasing. They trace
one chosen monotone path, not a full independent two-parameter barcode.
Stage coordinates default to 0..n−1; explicit `progression` values must be
strictly increasing. Metadata retains both factor thresholds and their units.
Without schedules, exact compatible-unit scalar max-coupled persistence remains
the default. Different factor rules cannot be silently compared as one physical
scalar; use an explicit paired path where appropriate.

The interaction workflow also provides static pair-threshold homology/Laplacian
queries and persistent operators between two comparable factor-threshold pairs.
Each query must remain inside the respective constructed factor domain.
Independent original factor filtrations are retained in serializable metadata,
so another threshold pair can be evaluated without rerunning geometry. An
empty early factor yields a zero interaction chain and empty operators; it is
not the forbidden empty simplex and is never replaced with invented vertices.

## Barcode feature schemas

Default numerical features are fixed-length **1D vectors**, obtained by flattening
per-degree joint birth–death count grids in a documented order. The 2D grids
remain available for heatmaps; this is not a Betti-curve occupancy histogram.
Each interval contributes to its birth/death cell, not every bin it spans.

Explicit birth/death edge arrays take priority over min/max/step specifications.
Edges must be finite and strictly increasing. Grids may be shared or defined
per homology degree. Final upper edges are inclusive according to NumPy histogram
semantics; intermediate bins are left-closed/right-open. Unbounded bars and
out-of-range values have explicit separate feature channels/policies.

`PersistenceVectorizer.fit(results)` fits one dataset-wide finite maximum and
freezes the edges; `transform` never recalibrates them per sample. With supervised
ML, fit on the entire training split only (and independently within each CV fold).
Use the full collection for explicitly descriptive analysis. Infinite deaths do
not set the finite maximum. With `min_value=None`, the start is derived from
the known filtration start, bounded below by zero. Schema checks prevent mixing
units, coefficient fields, topology definitions, and construction settings.

## Spectral feature schemas

`summarize_spectrum` defaults to `(mean, max, min, std, zero_count)`. The first
four and user callbacks operate on positive eigenvalues; `positive_only=False`
explicitly selects all supplied eigenvalues. Zero count always uses the full
spectrum, independent of that selection. Other built-ins include median, sum,
positive_count, count, spectral_entropy, `variance`, `moment_1` through
`moment_4`, and `laplacian_energy`; named scalar callbacks are supported.
`spectral_variance`, `spectral_moment`, and `spectral_moments` expose the same
population-statistic definitions directly. `spectral_energy` remains the
uncentered `sum(abs(lambda))`. `laplacian_energy` is the centered full-spectrum
`sum(abs(lambda - mean(lambda)))`; the summary built-in always uses every mode
of a complete spectrum, regardless of `positive_only`, and returns NaN with an
explicit unavailable scope for an opted-in partial spectrum. It equals
classical graph Laplacian energy for combinatorial graph L0; its application to
Hyperdigraph L0/L1 or other operators is an explicit generalized functional.

A custom `statistics` mapping is ordered `nonempty_name -> callable(values)`.
Each callable receives its own copy of the one-dimensional eigenvalues selected
by the call-wide `positive_only` policy and must return one real scalar. With
the default `positive_only=True`, selection means eigenvalues strictly above
the resolved numerical-zero tolerance; `False` supplies all modes. Documented
built-ins such as `zero_count` and `laplacian_energy` retain their special
complete-spectrum contracts. Callback code and parameter values are not
automatically encoded in feature metadata, so parameterized descriptors need
stable names such as `positive_q25_linear` and `heat_trace_t1`. Undefined or
non-finite scalar results remain explicit by default, and neither
`summarize_spectrum` nor `summarize_spectra` performs automatic imputation.

Every requested statistic already contributes one named slot, including at
higher dimensions. When the source degree-q chain group is absent, every core
returns a complete zero-dimensional operator with `eigenvalues=[]`,
`basis=()`, and `nullity=0`; core metadata records `operator_dimension=0`,
`source_chain_dimension=0`, and `structural_absence=True`. It never fabricates
the spectrum `[0]`, which would incorrectly report a one-dimensional nullspace.
With the default `empty_operator_policy="preserve"`, each summary retains its
ordinary empty-input definition: extrema, means, variance, and moments are NaN,
whereas counts, sums, energies, and the entropy convention are zero.

`empty_operator_policy="zero"` is an explicit structural feature encoding for
predefined summaries. It sets every requested built-in slot to zero if and
only if the full spectrum is complete and empty; the basis is empty; nullity is
zero; any stored matrix and eigenvector array are `0 x 0`; and the standardized
core metadata jointly records zero operator/source dimensions and structural
absence. This makes the built-in block finite for `feature_matrix` while
leaving the authoritative spectrum unchanged. A complete empty hand-built or
legacy result without that receipt is preserved by the default policy but is
not authorized for automatic zero encoding.

Custom mappings remain authoritative under both policies: every callback still
receives a copy of its selected array and determines its own empty-input value.
The zero policy does not apply to a nonempty all-zero operator, a merely empty
positive-mode selection, an opted-in partial spectrum, or a failed computation.
Summary metadata records `empty_operator_policy`, confirmed
`structural_absence`, and `zero_filled_statistics`; the policy, but not
sample-specific absence, is part of feature-schema compatibility.

A callback summarizes one spectrum only. Across-filtration AUC, mean and spread,
sampled range, total variation, extremum locations, and largest sampled slopes
consume an already sampled curve and remain separate application-level
calculations with explicit grid and interpolation conventions.

Numerical zero is `abs(lambda) <= max(atol, rtol * max(abs(lambda)))`. Defaults
use `tol=1e-10` as atol and `rtol=0`, matching the core's absolute rule. Significant
negative values cause an error before positive-value selection. Input arrays
are not modified. Positive-only min/max/mean/std of an empty set are NaN, while
counts are zero. Structural zero filling requires the explicit policy above;
other undefined-value handling remains a separate application decision.

For self-adjoint PSD chain Laplacians, complete ordinary numerical nullity
estimates the real-field Betti number; persistent nullity estimates an image
rank. Neither is a general assertion about GF(2) homology. Partial spectra are
rejected by default. With explicit opt-in, full `zero_count` is NaN, and observed
zero count is separately recorded without claiming full nullity. Changing the
tolerance changes the numerical interpretation and is included in feature schema.

## Repository protein-ligand v1 feature receipt

`workflows/protein_ligand_prediction_v1/generate_features_protein_ligand_v1.py`
is a task-specific, non-installed receipt. It leaves the original
`workflows/protein_ligand_prediction/` generator, its `(6, 100, 143)` schema,
and its completed artifacts frozen. The v1 recipe retains only the earlier
large-profile geometry convention: a ligand-centered 20-angstrom protein crop
and the ordered scale grid `0.0, 0.1, ..., 9.9` angstroms. Its changed element
and statistic definitions mean it is NMI-inspired, not paper-exact.

The ordered protein element groups are the singletons `C`, `N`, `O`, and `S`.
The ordered ligand groups are the singletons `C`, `N`, `O`, `S`, `P`, `F`,
`Cl`, `Br`, `I`, and `H`. Protein-major, ligand-minor Cartesian-product order
therefore defines 40 channels, with channel index
`protein_index * 10 + ligand_index`. Ligand hydrogen participates in spectral
feature construction rather than serving only as a crop-selection atom. The
construction remains the receipt's cross-component, protein-to-ligand,
distance-thresholded sequence Hyperdigraph, and the authoritative operator is
TopoKit's complete ordinary degree-zero Hyperdigraph Laplacian.

For every scale and channel the five statistic rows, in order, are:

1. `n0`, the complete L0 matrix dimension;
2. `beta0`, the number of eigenvalues satisfying `abs(lambda) <= 1e-10`;
3. the arithmetic mean of eigenvalues strictly greater than `1e-10`;
4. the relative soft mode `min(lambda_positive) / mean(lambda_positive)`; and
5. the relative spectral heterogeneity
   `mean(abs(lambda_positive - mean(lambda_positive))) /
   mean(lambda_positive)`.

There is no five-decimal pre-rounding. Eigenvalue and summary arithmetic use
`float64`, and final C-order tensors use `float32` storage by default. With no
positive eigenvalue, statistics 3--5 are explicitly zero while `n0=beta0`
records the all-null spectrum. With exactly one positive eigenvalue, relative
soft mode is one and relative heterogeneity is zero. `n0` is repeated across
all 100 scales to preserve the rectangular `(5, 100, 40)` contract. An absent
element channel is a five-row zero block and remains distinguishable from a
present all-null channel, whose `n0` and `beta0` are nonzero.

Recipe identity covers the element order, scale/crop configuration,
construction semantics, tolerance, summary definitions and order, compute and
storage dtypes, generator source, and TopoKit source. Dataset execution uses
explicit manifests, validates resumable sample records and hashes, and writes
to its own recipe-addressed directory; it may not overwrite or claim
compatibility with the original 143-channel feature directory. It performs no
model training or test-set evaluation.

## Repository protein-ligand v2 feature receipt

`workflows/protein_ligand_prediction_v2/generate_features_protein_ligand_v2.py`
is an independent, non-installed evolution of the v1 receipt. It preserves
the four-by-ten singleton element order, the NMI-inspired large geometry, the
absolute `1e-10` zero rule, and TopoKit's ordinary sequence-embedded
Hyperdigraph `L0` construction. It does not modify or replace any builder or
mathematical-core implementation.

The v2 statistic axis appends one row after the five v1 rows:

6. `relative_largest_positive_eigenvalue`, defined as
   `max(lambda_positive) / mean(lambda_positive)`.

Here `lambda_positive` contains precisely the eigenvalues greater than the
declared absolute tolerance. The value is zero when the positive spectrum is
empty and one when it contains exactly one mode. Significant negative
eigenvalues remain errors. The existing relative soft mode and relative mean
absolute deviation retain indices 3 and 4, so no prior statistic is silently
reordered. The complete C-order tensor shape is `(6, 100, 40)`, with float64
working arithmetic and canonical little-endian float32 storage. An absent
element partner is a six-row zero block with an explicit JSON presence mask.

The updated local v2020R1 P-L index contains 19,037 IDs. Excluding the 539
indexed IDs in the union of the CASF-2007, CASF-2013, and CASF-2016 cores gives
18,498 training IDs; the CASF-only `1xd1` makes 19,038 unique structures across
training and evaluation. The preparation and generation receipts must use an
explicit new training manifest and preserve the older refined manifest, CASF
manifests, structures, feature stores, and model-provenance hashes. The three
CASF editions keep separate memberships while sharing per-ID tensors.

When the NMI tables are selected, preparation writes the new training manifest
and three `_nmi` CASF manifests as one source-consistent set. `label_logka` and
its active equality relation come from those NMI tables. The existing
`label_logka_unrounded` field repeats that active NMI number because the NMI
source supplies no additional precision. The raw binding token, measurement
relation, and converted concentration separately preserve the updated PDBbind
index measurement, while cross-source discrepancies belong in the provenance
JSON. Each source table, output manifest, membership set, and archive is
hash-recorded; pre-existing structure bytes may be accepted when identical but
never silently replaced when different.

The schema namespace and recipe prefix contain `compact6`, and the generator
hash is distinct from v1. Resume may validate only artifacts with this exact
recipe identity, shape, dtype, finite values, and input/output hashes. No v1
tensor can be accepted or overwritten as a v2 result.

## Repository protein-ligand v1 standardized GBDT receipt

`workflows/protein_ligand_prediction_v1/train_sklearn_gbdt.py` is an explicit,
non-installed consumer of a caller-named completed feature recipe. It validates
the three-axis, C-order, canonical-float32 schema and sample record/hash before
flattening. Training and CASF memberships come only from the four explicit
manifests, and any train/test ID overlap is rejected.

The preprocessing/model contract is one persisted sklearn pipeline with steps
`scaler` then `gbdt`. `Pipeline.fit` receives only training rows;
`StandardScaler(with_mean=True, with_std=True)` therefore learns no CASF
statistics. Raw CASF rows are supplied to `Pipeline.predict`, which applies the
same fitted scaler before regression. The affinity target remains on its
original dimensionless log-affinity scale. Fitted scaler row/feature counts,
zero-variance count, and hashes of `mean_`, `var_`, and `scale_` are recorded.

The estimator defaults are learning rate `0.01`, 10,000 stages,
`max_features="sqrt"`, maximum depth 7, random state 0, and otherwise the
installed `GradientBoostingRegressor` defaults. The square-root rule is resolved
from the actual flattened feature count; it is 141 candidates per split for the
20,000-value v1 tensor. The output directory is exclusive, the pipeline is
round-trip prediction checked, and `metadata.json` is the completion marker.
No training is initiated by import or feature generation.

## Repository protein-ligand XGBoost receipt

`workflows/protein_ligand_prediction/train_xgboost.py` is a task-specific,
non-installed receipt rather than a `topokit` package API. It consumes one
completed feature recipe whose schema declares C-order
`(statistic, scale, element_channel)` tensors with canonical `float32` storage.
Each tensor is validated against its success record and flattened in C order;
the receipt does not regenerate topology, fill missing tensors, fit a scaler, or
change the 85,800-feature schema.

Training and evaluation membership come only from caller-visible CSV manifests.
The default files are the PDBbind-v2020R1 refined training manifest and the
CASF-2007, CASF-2013, and CASF-2016 test manifests. Duplicate IDs inside one
manifest are invalid, any training/test ID overlap is rejected, and intentional
overlap among CASF editions is retained. A shared complex is predicted once but
contributes independently to every CASF edition in which its manifest places it.
No test manifest supplies observations to model fitting.

The requested estimator defaults are XGBoost `gbtree` regression with
`objective="reg:squarederror"`, RMSE as the internal evaluation metric,
`learning_rate=0.01`, `max_depth=5`, `min_child_weight=2`, `subsample=0.6`,
10,000 estimators, histogram trees on CPU, one XGBoost worker thread, and random
seed zero. No validation split, early stopping, cross-validation, feature
selection, or hyperparameter search is implicit; changing any setting requires
an argument.

XGBoost column-sampling parameters accept numeric fractions, not the literal
string `sqrt`. The receipt resolves its `sqrt` shorthand to `1/sqrt(p)`, where
`p` is the flattened feature count. With `p=85,800`, the default
`colsample_bytree=sqrt` resolves to approximately `0.0034139437`, or about 293
candidate features selected once per tree before XGBoost's integer handling.
This is deliberately aggressive and is not scikit-learn GBDT
`max_features="sqrt"`, which samples candidates at each split. The closer
XGBoost analogue is explicit `colsample_bytree=1` with
`colsample_bynode=sqrt`. Column-sampling fractions are cumulative, so setting
both to the square-root fraction would leave only about one candidate per node.

The label relation remains part of the evaluation contract. `numeric` is the
default compatibility policy and treats a reported bound as its numeric value;
it is not a censor-aware loss. `exact-only` removes non-exact training labels
and excludes them from each primary benchmark metric. `error` rejects a run if
any selected manifest contains a non-exact relation. When numeric evaluation
contains a bound, the receipt also reports exact-only sensitivity metrics.

Each CASF edition receives a separate RMSE, mean absolute error (MAE), and
Pearson correlation coefficient (PCC), plus a per-row prediction CSV. The
output directory also contains an XGBoost UBJSON model, `metrics.json`, and
`metadata.json` with the resolved parameters, feature and manifest provenance,
sample IDs, software versions, and hashes. `plan.json` is written when the new
output directory is reserved, while `metadata.json` is written last and marks a
complete artifact set. An existing output path is always refused; the receipt
has no implicit resume or overwrite mode. These artifacts describe this
application run and do not certify that its defaults are optimal.

## Repository protein-ligand scikit-learn GBDT receipt

`workflows/protein_ligand_prediction/train_sklearn_gbdt.py` is a second
task-specific, non-installed tree receipt. It consumes the same completed
feature recipe and calls the same task-local `prediction_common.py` functions
for manifest parsing, train/test leakage rejection, record and tensor hashing,
C-order flattening, label policy, CASF RMSE/MAE/PCC, and prediction CSVs. It
never regenerates topology or changes the feature schema.

The frozen requested estimator choices are scikit-learn
`GradientBoostingRegressor` with `learning_rate=0.01`,
`n_estimators=10000`, `max_features="sqrt"`, and `max_depth=7` for every
regression tree. Random state zero is the sole additional scientific choice,
made to make the stochastic feature sampling reproducible. Verbosity may be
changed for progress reporting without changing predictions. Every other
estimator parameter is left at the default supplied by the installed
scikit-learn implementation; both the installed version and the complete
resolved `get_params(deep=False)` mapping are run provenance rather than
unstated assumptions.

For a feature count `p`, native `max_features="sqrt"` chooses approximately
`sqrt(p)` feature candidates at each split. With the frozen 85,800-feature
schema, integer conversion resolves that to 292 candidates per split. This
policy is not the XGBoost receipt's `colsample_bytree=sqrt` shorthand, which
resolves a fraction once per tree. `max_depth=7` constrains each individual
weak learner, not the number of boosting stages. Classic
`GradientBoostingRegressor` has no
`n_jobs`; the receipt does not claim that one model fit can use multiple CPU
workers. Scikit-learn may inspect more than the nominal feature count when it
must find at least one valid partition.

There is no implicit validation set, early stopping, cross-validation,
hyperparameter search, scaling, feature selection, or CASF-guided decision.
Training uses only the explicit PDBbind manifest. CASF editions are evaluated
separately with intentional cross-edition membership overlap and no train/test
ID overlap. The numeric/exact-only/error label semantics and exact-only
sensitivity report are identical to the XGBoost contract.

The run directory contains `plan.json`, a `model.joblib` estimator,
`metrics.json`, separate CASF prediction CSVs, and final `metadata.json` with
source, feature, manifest, parameter, software, output, and model hashes. Files
are published through temporary-file replacement; final metadata is withheld
if provenance changes during fitting. Existing directories are refused. The
joblib model uses pickle serialization, so it must be treated as trusted-only
and version-sensitive; the JSON/CSV records remain the portable audit surface.
These artifacts do not establish parameter optimality, publication-protocol
equivalence, or production performance until a full run is completed.

## Repository protein-ligand transformer receipt

The parallel `train_transformer.py` application consumes the same manifest,
feature-record, and CASF metric contracts, extracted into task-local
`prediction_common.py`. Existing XGBoost helper names remain re-exported for
compatibility. Its estimator and splitting behavior are unchanged; receipt
provenance now also hashes the shared helper source.

The transformer accepts raw `(batch, statistic, scale, element_channel)` arrays,
uses nonnegative `log1p` features standardized per statistic/channel across
fitting samples and scales, then transposes to scale-major order before
flattening statistic/channel coordinates. For the current recipe this is
100 tokens with 858 values each. Zero sentinels are retained; they do not define
padding masks. No topology is recomputed. Two independently initialized
128-wide encoder layers with four heads, fixed sinusoidal positions, a learned
CLS token, and a scalar head form the default supervised model. No pretrained
weights or reconstruction decoder are loaded.

No validation split is implicit. `--validation-fraction` explicitly selects
`ceil(fraction * training_rows)` validation rows using `--split-seed`, requiring
at least two fitting and two validation rows and preserving manifest order.
Feature and target normalizers use only fitting rows. Validation RMSE selects
the checkpoint; CASF remains evaluation-only. `--refit-full-train` requires
validation, reinitializes the model, fits new normalizers on all selected
training rows, and trains for the selected epoch count. Without validation,
the final fixed epoch is used. Without refitting, the final model retains the
smaller fitting membership. Actual IDs and smoke limits are recorded.

The checkpoint stores weights, architecture, both normalizers, recipe ID, and
schema SHA-256. Original target units are restored before all metrics and
prediction exports. `predict_transformer.py` requires completed metadata,
verifies model and feature hashes and exact recipe/schema compatibility, uses
`torch.load(..., weights_only=True)`, and needs no labels. New CSV outputs and
training directories refuse overwrites. Training metadata is the final
completion marker and is withheld if source, schema, or manifests change
during fitting. This workflow is scoring regression, not a reproduction of the
published TopoFormer training set, pretrained pipeline, or other CASF tasks.
