# Scientific contracts

The 0.3 parameter details are in [CONTRACTS_0_3.md](CONTRACTS_0_3.md).

## Default user feature exports (2026-09-23)

The repository FS-AN exporter accepts explicit `sample_id,protein_file,ligand_file`
CSV rows, resolving relative inputs against the manifest directory. It has no
historical cohort-size or label requirement. It writes one success/failure record
per requested sample; complete float32 `(N,27500)` matrices and matching sample
IDs appear only when all rows succeed. Failure is never imputed as an all-zero
sample. The exact canonical schema bytes, input/output hashes and installed
implementation identity accompany each store. The prediction manifest's
`pdb_id` column contains user IDs and does not require a real PDB accession.

The scientific FS-AN `(10,50,55)` tensor, 15 Å crop, 50 alpha radii, channel order,
statistics and exact-coordinate selection policy are unchanged. Exporter PDB
alternate-location/occupancy diagnostics are warnings, not atom-selection rules.
Prepare the intended conformer upstream. Complete ordinary L0 spectra are still
required; explicit resource caps cannot silently substitute partial spectra.

`SequenceEncoder(ligand_profile="cpz")` selects the pinned CPZ checkpoint and
dictionary and reports `sequence-esm2-t33-cpz-bos-v1`. Omitting the profile keeps
the historical ChEMBL27 default and recipe. Both use ESM-2 residue-count-weighted
pooling followed by the ligand BOS vector, yielding float32[1792] in protein-then-
ligand order. The explicit profile changes ligand weights/vocabulary, not pooling
or feature width. It cannot make a CPZ vector compatible with a ChEMBL27 model.
Historical sequence GBDT helpers retain their earlier recipe; the new CPZ runner
extracts embeddings only and does not train or claim FS-AU bundle inference.

## Public route namespaces

`topokit.builders` and `topokit.core` expose `simplicial`, `hyperdigraph`, and
`interaction` as lazy module attributes. Consequently, parent-first access such
as `from topokit import builders; builders.simplicial.from_graph(...)` has the
same route meaning as `from topokit.builders import simplicial`. Parent import
alone does not eagerly load every route, and an unknown attribute raises
`AttributeError`. This import contract changes no construction, filtration, or
analysis definition.

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

The application-level alpha follow-up records its observation grid separately
from native alpha units: r7 uses 50 grid values 2.0,...,11.8 Å, with radius
grid/2 and squared-radius birth inclusion. Starting observations at 2 does
not discard earlier edges. Its r6 crop is <=15 Å before channel geometry;
r8 retains all internal and cross alpha edges of each selected element union.
These are declared molecular recipes, not changes to any builder/core default.

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
  `geometry_tolerance` argument is validated but no longer expands empty
  spheres. See [precision and resource limits](NATIVE_ALPHA.md).
  Explicit `backend="gudhi_exact"` uses the optional GUDHI dependency
  with exact precision on unmodified coordinates. There is no automatic
  fallback or coordinate projection on that route. Exact births are converted
  to float by GUDHI, and the backend/version are recorded in metadata. Full
  cofaces are computed before the requested radius/skeleton is returned.
  The full-complex simplex cap is checked after external construction, so it
  does not cap GUDHI's transient allocation. Geometry tolerance is validated
  but not applied by the exact backend. Both alpha backends now default to
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
  overlapping factor views. The initial paired example is alpha(subset) versus
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
Missing-face cases continue through general embedded-chain algebra. Generic
`workflows.laplacian_series` uses the sweep only for eligible ordinary
hyperdigraph L0; higher dimensions, explicit reference backends and genuine
two-scale persistent operators retain their previous paths. See
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

## Native alpha 1.1.1 error-only recovery

When all older repair paths fail on a nearly planar hull sliver, a bounded
facet-neighbor cavity may replace its misleading enormous sphere neighborhood.
All new cells still face full-cloud empty-ball and existing boundary checks.
Previously successful branches and higher-dimensional coface propagation are
unchanged. Application-only compatible reuse preserves original JSON/source
identity, requires an inventoried exact predecessor and AST/source equivalence,
and freezes the compatibility-inventory checksum during generation and ML.
