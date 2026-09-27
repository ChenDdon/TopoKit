# Scientific contracts

These contracts describe the maintained public APIs and fixed application recipes.
See [notation](NOTATION.md) for mathematical symbols and [workflow guides](../workflows/README.md) for commands.

## Default user feature exports

The repository Topo exporter accepts explicit `sample_id,protein_file,ligand_file`
CSV rows, resolving relative inputs against the manifest directory. It has no
fixed cohort-size or label requirement. It writes one success/failure record
per requested sample; complete float32 `(N,27500)` matrices and matching sample
IDs appear only when all rows succeed. Failure is never imputed as an all-zero
sample. The exact canonical schema bytes, input/output hashes and installed
implementation identity accompany each store. The prediction manifest's
`pdb_id` column contains user IDs and does not require a real PDB accession.

The Topo recipe fixes its `(10,50,55)` tensor shape, 15 Å crop, 50 alpha radii,
channel order, statistics and exact-coordinate selection policy. Exporter PDB
alternate-location/occupancy diagnostics are warnings, not atom-selection rules.
Prepare the intended conformer upstream. Complete ordinary L0 spectra are
required; explicit resource caps cannot silently substitute partial spectra.

`SequenceEncoder(ligand_profile="cpz")` selects the pinned CPZ checkpoint and
dictionary and reports `sequence-esm2-t33-cpz-bos-v1`. Omitting the profile keeps
the ChEMBL27 default and its distinct recipe. Both use ESM-2 residue-count-weighted
pooling followed by the ligand BOS vector, yielding float32[1792] in protein-then-
ligand order. The explicit profile changes ligand weights/vocabulary, not pooling
or feature width. It cannot make a CPZ vector compatible with a ChEMBL27 model.
The installed sequence GBDT helpers use the ChEMBL27 recipe; the repository CPZ
runner extracts embeddings only and does not train or perform ensemble inference.

## Public route namespaces

`topokit.builders` and `topokit.core` expose `simplicial`, `hyperdigraph`, and
`interaction` as lazy module attributes. Consequently, parent-first access such
as `from topokit import builders; builders.simplicial.from_graph(...)` has the
same route meaning as `from topokit.builders import simplicial`. Parent import
alone does not eagerly load every route, and an unknown attribute raises
`AttributeError`. Construction, filtration and analysis semantics are defined
by the selected route.

## Inputs and identity

`PointCloud(points, ids=None, weights=None)` is a domain-independent numerical
record. Coordinates are finite n-by-d arrays; IDs are unique strings/integers;
assigned weights are one finite scalar per point, defaulting to 1. Input arrays
are copied and made read-only. Format readers are outside mathematical cores.
CSV/XYZ, native JSON, and molecular/crystallographic readers return this same record. Native
JSON requires finite rectangular coordinates and validates optional IDs,
labels, weights, units, and metadata before constructing the cloud. Readers
may preserve declared elements, charges, bonds, cell parameters, and raw fields
as metadata, but reading never assigns chemical weights, expands symmetry,
chooses periodic images, or sends bonds into a builder. The mathematical core
therefore receives a defined numerical object, not an inferred chemistry or
physical model.

For molecular-reader outputs, source-level counts, bonds, and PDB connectivity
are retained under `source_*` keys. `atom_count`, `selected_atom_count`,
`bonds`, and `connectivity_records` describe the current `PointCloud` view.
Selecting by stable IDs induces the current bonds/connectivity and preserves raw
declarations, cells, CIF tags, SDF properties, and non-atom records as source
provenance.

## Distinct objects

Simplicial complexes, ordered sequence hyperdigraphs, and two-factor interaction
quotient chains retain different mathematical representations. Shared result
envelopes do not identify their homology or force them through a simplex tree.
An explicit graph is a one-dimensional simplicial object unless flag expansion
is explicitly requested. An explicit digraph contains the supplied directed
edges, not implicit higher hyperedges. Point-cloud construction is a separate,
explicit operation. No path-homology family is included.

## Filtration rules

* Simplicial default: unweighted alpha. Births are squared radii; point weights
  have no effect. Rips is an explicit alternative with diameter/edge-length
  births. Weighted alpha is not implemented. The default alpha backend is
  `native`, implemented in TopoKit with batched NumPy/SciPy geometry and
  rational repair of uncertain cases. It does not import GUDHI or jitter
  coordinates; it preserves exact affine rank in near-degenerate clouds.
  The full coface closure determines births before skeleton truncation.
  Small-cloud repair is bounded at 300,000 candidate maximal simplices;
  failure beyond that bound is explicit. Cospherical triangulation choices
  and floating-point births need not be identical to GUDHI. The native
  `geometry_tolerance` argument is validated but does not expand empty
  spheres. See [precision and resource limits](NATIVE_ALPHA.md).
  Explicit `backend="gudhi_exact"` uses the optional GUDHI dependency
  with exact precision on unmodified coordinates. There is no automatic
  fallback or coordinate projection on that route. Exact births are converted
  to float by GUDHI, and the backend/version are recorded in metadata. Full
  cofaces are computed before the requested radius/skeleton is returned.
  The full-complex simplex cap is checked after external construction, so it
  does not cap GUDHI's transient allocation. Geometry tolerance is validated
  but not applied by the exact backend. Both alpha backends default to
  `duplicates="merge"`, keeping the first input row per exact coordinate before
  geometry. Counts and original-to-retained stable-ID mappings are recorded;
  `duplicates="error"` remains an explicit strict mode. Near coordinates stay
  distinct. `PointCloud.unique_coordinates()` provides the same input operation
  with first-row attributes and documented aligned reader metadata.
* Hyperdigraph default: Delaunay adjacency oriented from smaller to larger
  assigned weight. Exactly equal weights retain both orientations. Vertices
  enter at zero; edges have Euclidean-distance births. A higher hyperedge is
  a distinct-vertex ordered sequence whose consecutive edges are allowed,
  and its birth is their maximum. This is not an ordered clique rule.
* Interaction: exactly two factors in the public API, constructed independently,
  unweighted alpha by default. Explicit one-to-one overlap pairs identify shared
  nodes. Unmatched local IDs remain distinct. One cloud creates two fully
  overlapping factor views. The paired example is alpha(subset) versus
  alpha(whole), sharing subset IDs; no inclusion between the factor complexes
  is assumed. Shared settings do not mean copied birth arrays. Interaction cell
  birth is max(factor births) on one common scalar axis, not two-parameter
  persistence. Factor simplex caps may be configured separately.

For geometric builders, `filtration_start=s0` is finite and nonnegative.
Effective births are max(s0, raw birth) for every retained cell. This starts
observation at s0 without shifting later events. Raw geometric births are
retained. Bars born at s0 are flagged `at_initial_stage`; they may predate the
observed filtration, or be genuinely born at its first stage. Snapshot sampling
does not change filtration births. Cutoff-surviving intervals have infinite
death in the supplied filtration; this is not proof of survival beyond a cutoff.

Alpha and hyperdigraph scales differ: squared coordinate units versus coordinate
units. They must not be silently concatenated as a single physically calibrated
axis. Two interaction factors need compatible units for the default common
scalar axis; an explicit paired progression records both axes and their units.

## Algebra, dimensions, and spectra

Defaults cover H0 through H2 and L0 through L2. Degree q analyses require the
appropriate q+1 boundary information. Higher degrees are accepted when the
object supplies those chain groups; a plain graph does not acquire H2 merely
because a higher degree was requested. Explicit low caps describe truncated
objects and can change persistence and Laplacians.

GF(2) persistence is the common denominator. The simplicial core also supports
other prime fields. Real Hodge spectra are not universally equivalent to GF(2)
Betti numbers. Ordinary snapshot Lq is the default spectral operation; genuine
two-scale persistent Lq is a separate opt-in API, acting on the source-time basis.

`core.hyperdigraph.L0Sweep` is a guarded ordinary-L0 optimization: every edge
endpoint must have a singleton birth no later than its edge. Reciprocal edges
are distinct incidence columns; births are inclusion coordinates, never
operator weights. Nondecreasing inclusive queries update one dense buffer in
fixed ambient-vertex order; returned matrices are independent copies, and
late/absent singleton vertices keep their original semantics. The full sweep
buffer and edge schedule obey explicit dense/sparse resource budgets.
Missing-face cases use general embedded-chain algebra. Generic
`workflows.laplacian_series` uses the sweep only for eligible ordinary
hyperdigraph L0; higher dimensions, explicit reference backends and genuine
two-scale persistent operators use their respective core algorithms. See
[incremental L0](INCREMENTAL_L0.md) for the API, memory/copy costs and validation.

Eigenvalues are returned by default. `return_eigenvectors=True` opts into
eigenvectors; `return_matrix=True` opts into explicit matrix export. Basis labels
are always retained; a hyperdigraph basis may be a linear combination of ordered
hyperedges rather than one hyperedge. Eigenvector signs and repeated-eigenspace
bases are non-unique. `k` requests a partial spectrum; partial nullity is not
reported as a complete Betti count. Excessive full spectra are refused before
dense materialization where supported, with no silent topology modification.

## Features and visualization

Features operate on numerical interval tables. Fixed birth/death histograms
separate infinite intervals and out-of-range counts. Display or feature
`min_persistence` thresholds never mutate raw intervals. ML bin ranges must be
fixed or learned once from the training collection. Version 0.3 supports fitted
dataset-wide ranges and separate per-degree grids; full descriptive collections
require explicit descriptive scope. Default vectors flatten the count grids.
Point, simplex, hyperedge, interaction-factor, barcode, spectrum, eigenvector-
coefficient, and feature views consume those same results. Factor views are not
claimed to embed the full interaction quotient tensor object geometrically.

Spectral variance and raw moments are population statistics over the explicitly
selected eigenvalues. `summarize_spectrum` uses positive modes by default and
records each statistic's scope. Centered Laplacian spectral energy is centered
at the full-spectrum mean and is therefore defined by the built-in only for a
complete spectrum; it is not the existing uncentered spectral energy. It equals
classical graph Laplacian energy for combinatorial graph L0 and is an explicit
generalization for other operators or dimensions.

A complete spectrum of length zero is the spectrum of a
zero-dimensional operator/source chain group, not a one-element zero spectrum.
The default summary policy preserves the statistic definitions, including NaN
for undefined extrema and moments. Applications that require finite,
fixed-width blocks may opt into `empty_operator_policy="zero"`; this encodes
all requested predefined summaries as zero only for a complete structural
absence confirmed by the core receipt and result envelope. Custom mappings
always execute and own their empty-input behavior. The raw `SpectrumResult`,
nonempty zero operators, and partial spectra are not changed or filled. The
selected policy is part of the feature schema.

## Optional general-set sequence predictions

The saved ESM-2 + CPZ general-set predictor averages seeds 0, 1 and 2 with
weight 1/3 each. Each member applies its own training-only StandardScaler to
unscaled 1,792-dimensional embeddings. Ensemble metrics are evaluated on the
mean prediction vector; individual metric means/SD remain separate summaries.
Ensemble inference requires external model/scaler bundles and a compatible
inference implementation, as described in the
[model card](../workflows/protein_ligand_prediction/sequence/MODEL_CARD.md).
The repository sequence runner extracts embeddings only; installed ChEMBL27
estimator helpers are not compatible with CPZ features.

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

## Stationary analysis and result views

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
