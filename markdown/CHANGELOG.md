# Change trace

## 2026-09-21 — User selected 15 Å representative; 20 Å ablation preserved

Promoted the existing audited 15 Å / alpha-radius 0.1–5.0 configuration to the
representative protein–ligand application at the user's request, favoring a
smaller protein field and similar scores. Added the full selection receipt,
`representative_strategy.py` adapter and relative feature/model/result aliases.
Kept recipe `hpcc-alpha15-cb4760366e92bd2e` and all original numerical records.
Saved the preceding 20 Å schema, source, result files and pre-selection notes
in `archive/2026-09-21_20a_ablation_reference`, with hash receipts and links to
all old feature/model/audit artifacts. Updated current README and recipe notes.
No source geometry, L0/higher-dimensional code, data label, feature tensor,
model or prediction was changed or deleted. Validation: 26 workflow tests
passed, one optional GUDHI test skipped; the selected adapter reproduces the
saved 10gs tensor/presence/counts exactly; all 326 frozen source hashes, 26
archived copied files and 26 retained model artifacts agree. This is a user
selection of a representative, not a claim of best PCC on every benchmark.

## 2026-09-21 — Cornell alpha15 full results and verified local delivery

Completed the user-authorized 15 Å / alpha-radius 0.1–5.0 study: 19,066 features,
four full GBDT fits and six CASF evaluations. Remote audits verified training-only
scalers and reloaded predictions. Copied all production outputs into the separate
Cornell experiment folder and added a repeatable local audit script; its complete
run verified every tensor/record/model hash, sample/label membership and all six
metrics from 1,350 predictions. Added separate RMSE/PCC report tables and a
machine-readable baseline comparison, and updated workflow/package/dataset notes.
The 15 Å variant improves CASF-2007; the 20 Å baseline retains higher PCC on
CASF-2013/2016. No numerical package code, public API, baseline files or ML
parameters changed. All delivery and audit work is complete; MSU remains canceled.

## 2026-09-21 — Cornell alpha15 features complete; full ML underway

All 19,066 production features and their complete audit passed with zero
failures at 17:55:54 UTC. The unchanged controller automatically started ML.
At 18:09 UTC two refined fits (2007/2013) were complete with artifact hashes
verified, and the 2016/general-v2020R1 fits were active. Updated status notes
and copied the feature-audit receipt locally with a checksum check. No numerical
code or settings changed. Final audits, full comparison and local output
delivery remain pending.

## 2026-09-21 — Cornell alpha15 execution authorized and started

The user subsequently authorized running the prepared Cornell experiment and
its later ML stage. Started a detached controller at 15:41:36 UTC, after
rechecking all 326 scientific source hashes, the launcher hashes and available
resources. The fixed sequence is scientific preflight, 16 feature workers,
feature audit, four model fits (two simultaneously) and final audit. No
scientific source, feature definition or ML parameter changed. Updated current
workflow/package/dataset notes and saved the launch receipt. MSU remains
canceled. Preflight passed at 15:43:07 UTC: 38,132 input hashes, 74 smoke
features, four smoke fits and the molecular control's 49 shared radii passed.
Production was verified with 16 live workers and zero failures. The existing
continuation monitor was retargeted to Cornell only. Full production audit
results and scores are pending.

## 2026-09-21 — Cornell preparation for the 15 Å alpha-L0 study

At the user's request, canceled all 264 MSU Slurm jobs and paused the MSU
monitor. Prepared a separate Cornell WSL copy of the 326 pinned source files,
preserving the exact m2+m3+m5+r4 / 15 Å / alpha-radius 0.1–5.0 recipe.
Added the application-only `protein_ligand_prediction/cornell_15a` controller,
environment launcher, manual commands and eight passing controller tests.
The default action is read-only status; numerical work requires an explicit
action. Reuse Cornell's existing structures, labels and matching conda runtime.
No installed mathematical API, feature definition or higher-dimensional route
changed. No Cornell features, models, scheduled jobs or automatic launches were
started; full scientific preflight remains deferred to a later explicit run.

Append a dated notice for each software change; update the root README in the
same change. Historical entries describe the state when they were recorded.

## 2026-09-03 — 0.1.0 initial integration (historical)

Integrated the three reconstructed mathematical packages, shared numerical
records, point-cloud examples, barcode features, visualization, and exports.
Validated 354 tests and a standalone installed-wheel workflow. The original
research folders remained unchanged. See [initial report](history/INITIAL_INTEGRATION.md).
This early version bundled a coordinate fixture and mixed public construction
and analysis APIs; those decisions are superseded below.

## 2026-09-03 — 0.2.0 six-layer organization

Request: make the package follow six explicit responsibilities; remove datasets
and demo code from library source; centralize notes and maintain a clear trace.

Changes:

* Split each route into `builders/` and `core/`. Extracted actual alpha/Rips/
  Delaunay/conversion logic, not just facade names. Native object/algebra modules
  now live in `core/_<route>`. Construct-and-analyze helpers live in workflows.
* Moved coordinate data, provenance, fixture loader, and demo runner to
  `examples/`. Moved generated demo outputs to `examples/output/`. Wheels no
  longer bundle any of them. `python examples/point_cloud.py` replaces the
  package demo command; `python -m topokit info` remains available.
* Added extensible `readers/` and a reader registry. CSV/XYZ preserve scientific
  fields; subset operations keep documented per-point metadata aligned. No
  claim is made that MOL2/PDB/CIF adapters already exist.
* Added `postprocessing/` with barcode encoders, customizable spectral summaries,
  sequence summaries, and explicitly defined spectral graph entropy. Undefined
  statistics and partial-spectrum scope remain visible. Custom callbacks do
  not receive authoritative mutable result storage.
* Converted visualization and workflows into packages. Added an independent
  graph display and optional `workflows.ml`: gradient-boosted trees by default,
  or caller-supplied fit/predict estimators. No implicit training or splitting.
* Centralized notation, contracts, architecture, validation, migration,
  provenance, and history under `markdown/`. Added maintenance instructions in
  `AGENTS.md`; rewrote root README and updated NOTICE/packaging/examples/tests.
* Independent audits hardened schema checks, metadata preservation, malformed
  reader input validation, partial/empty spectra, and optional ML data handling.
  Core independence is tested in fresh processes with upper layers blocked.
* Promoted coordinate units/route identity into derived feature schemas for all
  three routes. Interaction rejects mismatched declared input units instead of
  combining physically incompatible filtration coordinates.

Scientific/API impact: construction definitions and tested homology/Laplacian
results are preserved; package import paths change as recorded in
[MIGRATION_0_2.md](MIGRATION_0_2.md). New spectral summaries are explicit derived
quantities, not topology recalculation. Root convenience aliases remain lazy
delegates. Portable result schema remains version 1.

Validation: **476 tests pass**, including architecture and independent extension
audits; the coordinate-only three-route H2/L2 demo and installed-wheel check pass
after the split. Full details and distribution checks are recorded
in [VALIDATION.md](VALIDATION.md); this entry does not imply cross-platform CI
or public release. Remaining limitations: floating-point geometry, high-degree
cost, domain-format adapters/recipes, batch/HPC orchestration, and paper studies.

## 2026-09-03 — 0.3.0 explicit construction and feature contracts

Request: implement the approved reader/builder/filtering/postprocessing design,
retain alpha by default, validate higher overlap annotations, use default 1D
barcode features with dataset-wide bins, and support paired factor schedules.

Changes across the six layers:

* Readers: added the offline PubChem Pauling reference mapping and explicit,
  nonmutating element-weight assignment; missing values require overrides.
  Source URL, retrieval date, and checksum are in REFERENCE_DATA.md.
* Builders: added stable-ID bond pairs and binary adjacency conversion; explicit
  graph/flag and digraph variants, inclusive distance cutoffs, weight overrides,
  and a validated construction range. Alpha remains unweighted and rejects
  custom bonds/distance cutoffs; Rips retains its full-distance definition.
* Interaction: added explicit overlap aliases/validation annotations and paired
  scalar schedules, independently bounded factors, and factor-pair snapshots.
  Sampled schedule results are labelled separately from exact event persistence.
* Core/workflows: added degree-indexed ordinary/persistent Laplacian collections
  and bounded spectral series. Eigenvalues remain the default; matrices/vectors
  and genuine source-to-target persistent operators are opt-in.
* Postprocessing: positive mean/max/min/std and complete zero count by default;
  invalid negative eigenvalues are rejected. Full nullity is not inferred from
  partial spectra. Barcode features use fitted shared or per-degree grids with
  explicit edge precedence, immutable fit schemas, and construction/unit checks.
* Visualization: per-degree barcode grids remain independently displayable.
  Documentation/examples and portable-result checks track the new APIs.

API impact: spectral summaries now default to positive eigenvalues; pass
`positive_only=False` to reproduce old all-eigenvalue statistics. Undefined
statistics remain NaN. Paired interaction schedules represent one chosen
monotone progression, not an unrestricted multiparameter persistence module.
Original research folders remain unchanged. No public release is performed.

Validation: **621 tests pass**, with two expected explicit-skeleton truncation
warnings. All three 24-point H2/L2 routes retain interval counts 47/110/99;
configured pipeline, static objects, feature handoff, and CLI checks pass.
The 0.3 wheel passes isolated outside-checkout checks including the new APIs,
and both archives pass content checks. Reference values were verified against
the official PubChem response. See VALIDATION.md for environment and scope.
Limitations remain floating-point alpha geometry, costly high dimensions,
explicit NaN handling before ML, one-path interaction schedules, future domain
adapters/recipes and HPC orchestration, and unexecuted remote cross-platform CI.

## Template for the next change

Use a new heading: `YYYY-MM-DD — version / short change title`.

Record: request; affected layers/files; API/scientific impact; README/notation
updates; commands/tests and outcomes; unresolved issues and intentional scope.

## 2026-09-05 — algorithm and efficiency audit

Request: review overall TopoKit algorithms and identify efficiency improvements
that preserve accuracy, using local documentation and external references.

Changes: added `markdown/PERFORMANCE_AUDIT_2026_09_05.md`, four standalone audit
scripts and recorded benchmark/profile outputs under `examples/performance_audit/`;
updated root README, documentation index and current validation. Reviewed all
three native mathematical families plus spectral, workflow and supporting layers.
No `src/topokit` implementation, scientific definition, dependency, API or
notation/contract changed. Original research packages were not edited.

Evidence: exact-output prototypes for overlap lookup reuse, requested-degree
interaction reduction, transformed hyperdigraph boundary reuse, alpha query
batching, trusted canonical simplex access, diagonal spectra, and workflow reuse.
The report distinguishes measured kernels from complete operations, conditional
H0/H1 specializations from general backends, and numerical equivalence from
bitwise identity. Runtime monkeypatches in standalone experiments are scoped
and restored; they are not installed optimization implementations.

Validation: full suite **621 passed**, two expected explicit-construction-cap
warnings; final single-thread-BLAS run 12.41 seconds. Three demonstration routes
retain 47/110/99 intervals. Additional differential checks include 132 alpha
cases, 168 cutoff-boundary comparisons, 60 arbitrary-hyperedge chain comparisons,
and 129 interaction backend/empty/randomized comparisons. Final benchmark
environment and measured scope are recorded in the report and JSON files.

Unresolved: implementation and production-workload acceptance remain future
work; timings are local observations. Floating-point rank/geometry, cache
memory budgets, invalidation, diagnostics and large-input behavior need the
documented safeguards before adopting new backends. No release, upload or
publication occurred.

## 2026-09-05 — object visualization and reusable dimensional blocks

Request: improve simplicial-complex and hyperdigraph views using the supplied
Python plotting examples, and provide independently reusable building blocks
across dimensions.

Changed layer/files: added `visualization/objects.py`, `topology.py`, `blocks.py`,
`style.py`, and `export.py`; updated `visualization/plots.py` and `__init__.py`.
Added four focused test modules and `examples/visualization_gallery.py` with
generated PNG/SVG/PDF figures under `examples/output/visualization_gallery/`.
Updated root README, documentation index, architecture, notation, contracts,
and current validation; added `markdown/VISUALIZATION.md` with executable usage.

API impact: new `PlotStyle`, `plot_simplicial_complex`, `plot_hyperdigraph`,
`plot_topology`, `plot_simplex_block`, `plot_hyperdigraph_block`,
`plot_building_blocks`, and `save_figure`. Existing low-level plotting import
paths are retained. Object views now default to dark nodes and hidden axes,
with filled simplex faces and directional hyperedge ribbons. Weight colors
and coordinate axes remain explicit options. Native-object views support
inclusive scale filters, exact degree selection, stable-ID mapping, explicit
coordinate overrides, and separate selection/inspection/primitive budgets.
Blocks support canonical point/segment/triangle/tetrahedron layouts and
labelled higher-dimensional schematics, caller axes, custom coordinates/order,
and 3D or orthographic projected views. Saving is explicit and preserves
editable SVG text and embedded PDF fonts using local settings.

Scientific impact: ordered q-hyperedges display exactly q consecutive arrows;
the directed-flag template is not used to infer all-pairs support or additional
hyperedges. Simplicial edges/faces are deduplicated without clique filling.
Display filtering and curved lanes change no scientific coordinates, births,
cell membership, q+1 construction, algebra, or numerical results. All 63
existing source files outside visualization match the pre-change audit hashes.
Optional Matplotlib loading and the six-layer architecture remain intact.

Validation: **763 tests passed**, two expected explicit-skeleton warnings,
including 142 new visualization checks and existing architecture/end-to-end
checks. The three-route 24-point H2/L2 example with plots passes and retains
47/110/99 intervals and 474 features per route. Inspected all three generated
gallery PNGs for clear arrows, tetrahedra, labels, alignment, and clipping.
Actual SVG/PDF tests verify editable text and font embedding. See
VALIDATION.md for environment, commands, and validation scope.

Limitations: above-degree-three simplex drawings are 2-skeleton schematics;
transparent 3D rendering is not exact volumetric occlusion. Dense overlapping
sequences may require explicit subsets or separate panels. No general layout
for absent/high-dimensional scientific coordinates is inferred. Installed-wheel
and cross-platform checks were not rerun for this extension. No release or
publication occurred.

## 2026-09-05 — code readability and reproducible pressure-test preparation

Request: review overall code naming, retain useful abbreviations, then stress
all three persistence/Laplacian families under alpha/VR and related settings
on AWS, recording measurements with a ten-minute wall limit.

Source changes: reviewed all 70 Python files; selectively renamed locals in
45 files across the six layers and shared infrastructure. Added concise
comments for ambiguous algebra and geometry steps. Existing argument names,
attributes, schemas, imports, numerical operations, and resource defaults
remain unchanged. No earlier audit optimization was adopted. Adapted the
source-extraction strings in the historical simplicial benchmark example;
stored benchmark results remain intact.

Example/test changes: added `examples/pressure_test/` with a generated-case
matrix, standalone worker, isolated runner, before/after checker, AWS entry
point, protocol, raw local records, numerical exports, and retained pre-change
source evidence. Added an integrity-checking Markdown report generator and
focused worker/runner/readability/report tests. Updated README,
documentation index, current validation, and contracts; added the readability
review report. Pressure cases retain q+1 cells and distinguish true alpha/Rips
from Delaunay/complete distance-supported ordered hyperdigraphs. Full spectra
and explicitly requested partial spectra have separate case records.

Validation: structural audit permits only 1,590 local-identifier changes across
45 files; ten small isolated scenarios give 190 exact before/after comparisons
and 268 bitwise-identical arrays, with 110 internal checks per revision.
The full local suite passes 873 tests, one Linux-only live-RSS test is skipped
on macOS, and the two expected explicit-skeleton warnings remain. All 72 local
smoke cases pass; actual barcodes/eigenvalues are retained as NPZ files.
All 72 numerical-file hashes verify in the generated local report.
Historical alpha prototype compatibility passes 30 exact checks; hyperdigraph
and spectral prototype smoke checks also pass without rewriting old results.

AWS state: the configured host was inspected (8 logical CPUs, 31,717 MiB RAM),
and a separate scientific Python environment was prepared. The main 216-case
matrix uses a 600-second wall limit per case, 16-GiB sampled RSS limit, and
22-GiB address-space bound, with CPU/peak-RSS measurement from wait4 and atomic
stage checkpoints. **Source upload and AWS execution remain pending** because
automatic approval review requires explicit authorization for exporting the
private source/test payload to the specific configured AWS destination.
No rejected transfer was bypassed. Local smoke data are not AWS measurements.

Limitations: renaming regression checks are bounded evidence, not a universal
correctness proof. Actual AWS capacity results and Linux live-RSS validation
are not yet available. Dense higher-order cases can be rejected by explicit
budgets; no timeout or resource refusal is counted as a completed result.
No package publication or new instance provisioning occurred.


## 2026-09-05 — Authorized AWS upload and independent low-dimensional tests

Request: upload the code to the configured AWS host and test low-dimensional
fast paths independently. Source upload is now explicitly authorized and the
first bundle checksum verified. Split canonical pressure cases into q0/q1/q2
matrices and give H0/H1 larger operation-specific point ladders. Added an
example-only compact hyperdigraph worker and explicit 12-case scaling plan for
its existing public H0/H1 algorithms; no installed scientific API or algorithm
changed. Record actual routes/backends, combined compact construction/analysis
time, process-only guards, exact numerical exports, and bounded canonical
equivalence evidence. Reports distinguish stored representations and dimensions.

Changed example harness, its tests, pressure protocol, root README, and current
validation. Local validation: 913 passed, one Linux-only skip, two expected
warnings. The first AWS before/after check passes 190 comparisons; initial
remote pytest found a missing CHANGELOG file in the upload payload, corrected
in the next bundle. Large AWS measurements remain in progress. The earlier
transfer rejection remains recorded above as history; no publication or new
instance provisioning was performed.


## 2026-09-05 — Completed AWS measurements and verified combined results

Completed the authorized AWS run with separate canonical q0/q1/q2 matrices,
compact H0/H1 scaling, and partial-spectrum/half-overlap supplements. All 358
cases have terminal records: 282 successes and 76 native resource guards. Of these,
76 are smoke trials; the 282 pressure/supplemental trials contain 206 successes and
76 guards. No measured timeout, memory stop, application/controller error, or
numerical-check failure occurred. All 282 successful numerical artifacts and the
6,654,695-byte download archive checksum verify. The package source hash is
stable across all suites. No production algorithm or scientific default changed.

Added a local-only combined report generator and 11 focused tests; corrected
worker dispatch scope and added two real worker-launch integration tests before
the final run. Initial failed-attempt logs remain preserved and excluded from
measured totals. Updated root README, pressure protocol, contracts, current
validation, machine-readable AWS status, and result findings/CSV/JSON/reports.
Local full suite: 926 passed, 1 Linux-only skip, 2 expected warnings. AWS gate:
913 passed, 3 optional/fixture skips, 240 passing subtests, 2 expected warnings.
Both environments retain 190 exact before/after comparisons and 268 bitwise-equal
arrays. Source/readability snapshots and experiment matrices remain reproducible.

Measured limits: largest runtime 166.503 s, peak RSS 2,138.7 MiB, explicit 600 s wall
per case. Native guards determine many observed stopping points, so these are
not estimates of unguarded host capacity. Compact probes validate an eight-point
prefix only; no universal correctness or equal-budget speedup claim follows.
Identified an exact conditional persistent-L0 identity-space shortcut as a
future optimization, without changing or benchmarking a replacement algorithm.

## 2026-09-07 — Executable overlapping-circles topology notebook

Request: add a Jupyter example that generates a circular two-dimensional point
cloud, displays graph, simplicial, directed-hypergraph, and interaction
representations at one cutoff, computes H0/H1 and L0/L1, summarizes each
spectrum, and provide a ready-to-run Conda environment. The point generator
was then refined to match a supplied two-overlapping-circles reference while
remaining exactly reproducible.

Added `examples/point_cloud_topology_workflow.ipynb`, `environment.yml`, and
the corresponding root README instructions. Seed `20260907` generates 20
samples around a radius-1.00 circle and 14 around an intersecting radius-0.68
circle. Their centers are 1.55 units apart. Seeded Gaussian movement has
standard deviations 0.024 radians angularly, 0.022 radially, and 0.008 in each
Cartesian coordinate. Repeating the generator with the same seed is asserted
to reproduce the coordinates exactly. The inclusive distance cutoff is 0.38
and the displayed proximity-ball radius is 0.19.

The graph and two-dimensional flag complex share a 43-edge 1-skeleton over 34
vertices. The graph has GF(2) `(H0,H1)=(1,10)`; filling its eight 3-cliques
gives `(1,2)`, retaining the two circle-scale classes. Random direction weights
include one forced adjacent tie on each source circle, producing verified
reciprocal directed edges. The full directed object contains 34/45/32
degree-0/1/2 hyperedges and has `(H0,H1)=(1,3)`. Interaction factors use the 14
smaller-circle samples and the full cloud with 14 explicitly paired overlap
IDs; their degree-0/1/2 simplex counts are 14/14/0 and 34/43/8, yielding
14/65/72 interaction cells.

Every stage exports a topology view and a four-panel diagnostic with GF(2)
Betti numbers, complete real spectra, all-eigenvalue minimum/maximum/mean,
numerical zero count, and spectral energy `sum(abs(eigenvalues))`. Eleven PNG
and eleven editable-text SVG figures plus a CSV comparison are generated under
`examples/output/point_cloud_topology_workflow/`; the executed notebook embeds
its results. The pale circles are explicitly documented as the ball
interpretation of the flag/Rips cutoff, not TopoKit's Delaunay-supported
squared-radius alpha constructor. Fourteen of 32 degree-2 hyperedges are
selected explicitly for the display only; every numerical calculation uses the
full directed object.

Scientific/API impact: this is example, environment, and documentation work;
no package source, public API, topology definition, or numerical algorithm
changed. The notebook was executed successfully with its registered `topokit`
kernel in the new `/opt/anaconda3/envs/topokit` environment (Python 3.12.14,
NumPy 2.5.3, SciPy 1.18.0, Matplotlib 3.11.1, pandas 3.0.5). All 13 code cells
completed without errors and their deterministic assertions passed. Eleven PNG
exports were visually inspected for legibility and clipping; all 11 SVGs retain
editable text. Focused architecture, simplicial, hyperdigraph, interaction, and
visualization checks pass: **160 tests**, with the two expected explicit
construction-truncation warnings. Limitations are the synthetic fixed-scale
demonstration and the intentionally selected directed display subset; it is not
a statistical study, persistence sweep, or strict alpha-complex comparison.

## 2026-09-07 — Representation-first persistent notebook revision

Request: make each introductory representation stand on its own, use the
package's exact representation names, remove the cross-representation
comparison, and add persistent homology, Laplacian curves, and cutoff animations
in Graph, Simplicial Complex, Hyperdigraph, and Interaction Complex order.

Reworked `examples/point_cloud_topology_workflow.ipynb` into 32 cells with 18
executable cells. The Graph Representation now uses the independent graph
plotter, labels its cutoff, and draws only points and edges. The Simplicial
Complex Representation explains 0-simplices as points, 1-simplices as edges,
and 2-simplices as filled triangles. The Hyperdigraph Representation explains
directed 0-, 1-, and 2-hyperedges in the prose and uses tied fixed-seed weights
to verify both directions on two supports. The Interaction Complex
Representation uses the 14-point smaller-circle sample as factor A, the full
cloud as factor B, and matching point IDs as the explicit overlap. The prior
cross-representation figure and table were removed.

Added persistent sections over 25 displayed alpha values from 0 to 0.80. The
graph uses cutoff `epsilon=2*alpha` and remains limited to H0/L0. The other
three routes use TopoKit's default alpha construction. Each section exports
persistence intervals, Betti curves, and ordinary filtration-snapshot
Laplacian curves for eigenvalue minimum, maximum, mean, and spectral energy;
numerical CSVs also retain zero-eigenvalue counts. Four 13-frame GIFs show how
connectivity changes with the cutoff. Added Pillow explicitly to
`environment.yml` for reproducible GIF writing.

At the stationary cutoff `epsilon=0.55`, the graph has 34 vertices and 63 edges
with `H0=1`; H1/L1 are intentionally undefined. The simplicial complex has
34/63/39 degree-0/1/2 simplices and `(H0,H1)=(1,3)`. The hyperdigraph has
34/65/75 directed degree-0/1/2 hyperedges and `(H0,H1)=(1,3)`. The interaction
complex has 14/90/194 degree-0/1/2 cells and `(H0,H1)=(0,0)`. The executed
notebook generated 13 PNGs, 13 editable-text SVGs, four GIFs, and eight CSVs.

Scientific/API impact: the package source, public API, topology definitions,
and numerical algorithms are unchanged. Laplacian persistence plots report
ordinary `Lq(alpha)` snapshot statistics along each filtration; they are not
the separate two-time persistent operator `Lq(s,t)`. All 18 code cells execute
without errors in the `topokit` Conda kernel and pass the notebook's deterministic,
algebraic, spectral, table-layout, and GIF assertions. Static figures and the
first, middle, and final animation frames were visually checked for labels,
connectivity progression, clipping, and legibility. Focused package validation
passes 160 tests with the two expected explicit-skeleton warnings.

## 2026-09-08 — Deduplicated hyperdigraph display and localized persistence setup

Request: explain and remove repeated same-direction arrows from the notebook's
Hyperdigraph Representation, place each persistent topology construction inside
its corresponding representation section, and remove the hollow birth marker
from H0 barcode intervals beginning at alpha zero.

Updated `examples/point_cloud_topology_workflow.ipynb` only at the example and
display-composition level. Distinct directed 2-hyperedges can share a
consecutive ordered segment; the previous object renderer preserved every such
occurrence, producing parallel same-direction arrows. The notebook now extracts
unique ordered segments independently for the directed 1- and directed 2-
hyperedge display layers. It draws at most one arrow per orientation in each
layer, so an unordered point pair has at most two blue directed-1 arrows, two
orange directed-2 segment arrows, and four arrows total. Separate curvatures
make the two dimensions and reciprocal directions visible. The full numerical
hyperdigraph remains unchanged: 34/65/75 directed degree-0/1/2 hyperedges are
used for homology and Laplacians. The static view still selects 14 of the 75
directed 2-hyperedges; their 28 possible occurrences collapse to 23 unique
orange segments.

Removed the shared persistent-topology construction cell. The graph,
simplicial-complex, hyperdigraph, and interaction-complex persistent objects are
now constructed directly below their respective headings; the interaction
factor objects used by its GIF are colocated there as well. Generic conversion,
plotting, and analysis helpers remain shared. H0 barcode lines now begin
directly at alpha zero; the open-circle convention remains available only for
other dimensions marked as initial-stage intervals.

Scientific/API impact: no package source, public API, mathematical object,
homology, persistence, or Laplacian result changed. Segment collapsing is an
explicit notebook visualization rule and does not merge or discard numerical
hyperedges. The notebook now contains 31 cells, including 17 executable cells;
all execute without errors in the `topokit` Conda kernel. Its assertions verify
one oriented segment per layer, at most two arrows per point-pair support in
each layer, and at most four across both layers. Thirteen PNGs, 13 editable-text
SVGs, four 13-frame GIFs, and eight CSVs were regenerated. Static and animated
hyperdigraph views and persistent barcodes were visually inspected. Focused
package validation passes 160 tests with the two expected explicit-skeleton
warnings.

## 2026-09-08 — Nature-style demonstration figure polish

Request: use the Nature figure workflow to improve the notebook's colors and
figure styles while preserving the accuracy of every demonstration.

Restyled all figures generated by
`examples/point_cloud_topology_workflow.ipynb`. The notebook now uses a
restrained Nature Publishing Group-inspired palette: charcoal points, navy
graph edges and H0, cyan simplex faces, vermillion directed-2 segments and H1,
and teal mean-spectrum curves. Continuous direction weights use cividis. Hue is
reinforced by point/edge/face geometry, arrow curvature, labels, dotted or
dashed line styles, and separate axes. Typography, tick marks, panel labels,
legends, line widths, white space, background circles, heatmaps, and GIF frame
annotations were standardized for compact double-column-style layouts. Tied
hyperdigraph weights use separated annotations with subtle leader lines.

All 13 PNGs were regenerated at 300 dpi; all 13 SVGs retain editable text. Four
13-frame GIFs use the same visual system and preserve stable limits between
frames. Color and grayscale contact sheets were inspected together with the
full-size Simplicial Complex, Hyperdigraph, Interaction Complex, and persistent
analysis panels. Titles, axes, legends, arrows, overlap rings, intervals, and
curves remain legible and unclipped.

Scientific/API impact: this is notebook presentation and documentation work.
No coordinate, point ID, weight, cutoff, simplex, directed hyperedge,
interaction cell, filtration value, homology interval, Betti number, Laplacian,
or eigenvalue statistic changed. SHA-256 hashes of all eight numerical CSVs
match the pre-style baseline exactly. All 17 code cells execute without errors
in the `topokit` Conda kernel and pass the deterministic topology, spectrum,
display-limit, table, and GIF assertions. No package source or public API
changed. Focused architecture, simplicial, hyperdigraph, interaction, and
visualization validation passes 160 tests with the two expected
explicit-skeleton warnings.

## 2026-09-08 — Grid-free plots, filtration circles, and directed underlays

Request: remove plot grids, restore the earlier Simplicial Complex colors, show
the current-radius construction circles in the Simplicial Complex,
Hyperdigraph, and Interaction Complex GIFs, and clarify higher-dimensional
Hyperdigraph segments without adding parallel arrow lanes.

Updated `examples/point_cloud_topology_workflow.ipynb`, its generated figures,
and the README/validation record. Every axis now suppresses grids. Simplicial
and interaction views use blue 1-simplices and green 2-simplices; continuous
point weights use viridis. The three higher-order GIFs draw pale circles of
radius alpha around every point, while the graph remains an independent
point-and-edge view without circles.

The Hyperdigraph display now places each directed 2-hyperedge segment on the
same curved path as the corresponding directed 1-hyperedge orientation. The
directed-2 layer is drawn first as a wider translucent orange arrow, and the
directed-1 layer is drawn above it as a narrow blue arrow. Oriented segments
remain deduplicated separately in each dimension. The static figure still
shows 14 selected directed 2-hyperedges for readability, while all 75 remain in
the numerical topology and all homology/Laplacian calculations.

Scientific/API impact: display composition and documentation changed; package
source, public APIs, mathematical objects, and numerical results did not. All
17 notebook code cells execute without errors. The 13 PNGs are 300 dpi, the 13
SVGs retain editable text and contain no gridline elements, and each of the four
GIFs has 13 frames. Color and grayscale contact sheets and representative GIF
frames were visually inspected. All eight numerical CSV SHA-256 hashes match
the pre-change baseline exactly. The focused regression suite passes **160
tests** with two expected explicit-skeleton warnings, and `pip check` reports no
broken requirements in the `topokit` Conda environment. The synthetic example
and selected static directed-2 display remain intentional demonstration scope.

## 2026-09-08 — Reusable stationary visualization and analysis API

Request: move the reusable stationary representation logic from the current
point-cloud notebook into TopoKit so future examples can import concise plotting
and fixed-scale analysis functions, while keeping GIF orchestration outside the
package.

Added `visualization/stationary.py` with `StationaryStyle`, clean local figure
helpers, representation-specific graph, simplicial, Hyperdigraph, and
interaction renderers, a dispatcher, and a four-panel stationary diagnostic
plot. The graph renderer always uses explicit 0/1 cells and omits construction
circles. Simplicial and interaction renderers show explicit cells and optional
exact-radius circles. The Hyperdigraph renderer deduplicates oriented segments
separately by displayed dimension, draws directed-2 segments as wider
translucent orange underlays below directed-1 arrows on the same curvature, and
supports deterministic or explicit display subsets without changing its source
topology. Interaction views use retained factor filtrations and explicit
overlap; no quotient embedding is inferred. Matplotlib remains lazy, plotting
does not change global `rcParams`, and plot calls do not save, show, or close.

Added `workflows.analyze_stationary` and the immutable `StationaryResult` record
to compose fixed-scale GF(2) homology, requested complete ordinary Laplacian
spectra, and named summaries. Added `spectral_energy` as the public built-in
`energy` statistic. Refactored the existing object-view selectors for reuse
inside visualization without introducing a visualization-to-builder/core
dependency. Exported the new APIs from their package namespaces and from the
shared result namespace.

Refactored `examples/point_cloud_topology_workflow.ipynb` to import these public
functions for all four stationary views, stationary diagnostics, and the static
drawing inside each animation frame. Only frame scheduling, annotations,
persistent-analysis curves, file publication, and GIF writing remain local to
the tutorial. The notebook preserves the user's current blue/green/orange and
cividis styling.

Scientific/API impact: this adds public convenience APIs without changing any
topology, filtration, homology, Laplacian, or persistence definition. All 17
notebook code cells execute without error and regenerate 13 PNGs, 13 editable-
text SVGs, four 13-frame GIFs, and eight CSVs. SHA-256 hashes for all eight CSVs
match the pre-refactor notebook exactly. The full suite passes **939 tests**
with one platform-specific skip and the two expected truncation warnings; a
focused 271-test stationary/architecture/object regression also passes. The
prepared `topokit` Conda environment reports no broken requirements. Current
stationary presentation helpers are intentionally two-dimensional, while the
existing general object API retains its 3D and projected views. A built wheel
contains both new modules and imports the public stationary symbols in an
isolated environment.

## 2026-09-08 — Dictionary-defined fixed-object homology and Laplacian notebook

Request: add a compact notebook under `examples/` that defines simple fixed
topological objects with dictionaries, places their vertices on a common
hexagon, and demonstrates homology and ordinary Laplacians in Graph,
Simplicial Complex, Hyperdigraph, and Interaction Complex order.

Added
`examples/fixed_topological_objects_homology_laplacian.ipynb`. The graph is
the six-cycle. The simplicial section first verifies that one filled
2-simplex is contractible, then adds the eighth face of a six-vertex
octahedral boundary so that `beta_2` changes from 0 to 1. The Hyperdigraph
section uses an ordered six-vertex example while preserving
ambient vertex `0` without a directed singleton. The interaction section
uses two complementary four-vertex paths that share exactly vertices `0`
and `3`; adding their common closing edge changes interaction `beta_2`
from 0 to 1. Coordinates affect only the illustrations.

Each final object reports GF(2) Betti numbers, the exact basis for every
ordinary real `L0`, `L1`, and `L2` matrix, complete eigenvalue spectra,
and real nullity. The notebook writes four 300-dpi PNGs, four editable-text
SVGs, four summary CSVs, 12 matrix CSVs, and one cross-representation CSV
under `examples/output/fixed_topological_objects_homology_laplacian/`.
The final simplicial face is highlighted directly, the missing Hyperdigraph
singleton is shown as a hollow ambient vertex, and interaction overlap uses
redundant ring and label encoding.

Scientific/API impact: this adds a synthetic educational example and
documentation only. It changes no package source, public API, topology
definition, notation, contract, or numerical algorithm. The 20-cell notebook,
including 12 code cells, executes sequentially in the registered
`Python (topokit)` kernel with no errors, stderr, or warnings. Its topology,
matrix, spectrum, nullity, overlap, and exports pass independent checks. Focused
architecture, simplicial, Hyperdigraph, interaction, and visualization
validation passes **160 tests** with the two expected explicit-skeleton
warnings; `pip check` in the prepared Conda environment reports no broken
requirements. All four PNGs and SVGs were visually inspected; PNG metadata is
approximately 300 dpi and the SVGs retain live text with no embedded raster
images. The planar octahedral drawing intentionally has crossing projected
faces, so the dictionary remains the authoritative combinatorial object.

## 2026-09-08 — Simplified fixed-object notebook notation

Request: present the four fixed-object examples as ordinary editable examples,
remove paper/publication framing, use the six supplied coordinates directly,
and simplify code and notation for readers who want to make small adjustments.

Updated `examples/fixed_topological_objects_homology_laplacian.ipynb` and its
README/validation documentation. The notebook now writes the six rows directly
in one `POINTS` dictionary; it no longer computes square roots or stores a
nested hexagon description. The mathematical inputs are short, visible
dictionaries named `GRAPH`, `TRIANGLES`, `HYPEREDGES`, `FACTOR_A`, and
`FACTOR_B`. Removed an unused face-closure helper, its combinations import,
the source-path search/injection code, nested object wrappers, redundant
fields, and provenance-bearing Hyperdigraph names and metadata.

All four final calculations now use the same public
`workflows.analyze_stationary(..., return_matrices=True)` call. A small shared
display helper keeps the returned bases, complete spectra, and exported
Laplacian matrices visible without repeating those operations in every
section. Prose and figure titles use `beta_2=1` rather than the imprecise
`H2=1`. The Hyperdigraph is described only by its ordered-hyperedge
dictionary and omitted singleton; the interaction explanation is shorter and
uses plain shared-label language.

Scientific/API impact: presentation and notebook composition changed; the
coordinates, graph edges, simplex births, ordered hyperedges, interaction
factors, Betti numbers, Laplacian matrices, spectra, and exported filenames
remain unchanged. No package source, public API, contract, or numerical
algorithm changed. All 12 code cells execute sequentially in the existing
`Python (topokit)` kernel without errors, stderr, or warnings. The long
hard-coded baseline assertion block was removed so dictionary edits flow into
the final comparison table without requiring a second edit. Short `Edit here`
comments mark the graph, simplex, ordered-hyperedge, and interaction inputs.
The highlighted simplex, omitted singleton, displayed simplex counts, and
earliest/latest filtration moments are derived from those inputs. Filtration
metadata and figure `beta_2` labels use the same computed values, and adding
all singleton hyperedges no longer makes the annotation cell fail. The
regenerated figures were visually checked for readable labels, missing/new
simplex annotations, the omitted Hyperdigraph singleton, two shared
interaction vertices, and clipping.

## 2026-09-08 — Fixed-only topology tutorial

Request: make the dictionary-defined notebook a set of independent fixed-case
examples instead of a comparison, retain only one simplicial-complex case and
one interaction-complex case, use consistent editable names, and remove
`Representation` from section titles.

Updated `examples/fixed_topological_objects_homology_laplacian.ipynb`, its
generated output directory, the root README, and current validation record.
Each section now starts from a dimension-keyed dictionary:
`graph_cells`, `simplicial_cells`, `hyperdigraph_cells`, or
`interaction_cells`. The supplied coordinates remain literal rows in `points`.
The simplicial section contains only the fixed closed shell with
`beta_2=1`; the interaction section contains only the fixed two-factor case
with shared labels `0` and `3`. The compact comparison cells, all time-based
construction, former comparison-oriented names, and stale comparison output
were removed.

Scientific/API impact: this simplifies an educational notebook and its
documentation. No package source, public API, topology definition, homology or
Laplacian algorithm changed. The 18-cell notebook, including 11 code cells,
executes sequentially in the registered `Python (topokit)` kernel with no
errors, stderr, or warnings. The graph, simplicial, Hyperdigraph, and
interaction Betti numbers are respectively `(1,1,0)`, `(1,0,1)`, `(1,1,0)`,
and `(0,0,1)`, matching the real Laplacian nullities. Execution regenerates 24
artifacts: four 300-dpi PNGs, four editable-text SVGs, four summary CSVs, and
12 basis-labelled matrix CSVs. All figures were visually inspected; every SVG
retains live text and contains no embedded raster image.

## 2026-09-11 — Molecular readers and predefined spectral information

Request: support PDB, MOL2, SDF, MOL, PDBQT, and CIF input with XYZ-like
`PointCloud` outputs; add predefined spectral variance, moments, and
graph-energy-like centered Laplacian spectral energy; and add a simple
per-format notebook from reading through H0/H1 Hyperdigraph persistence, L0/L1
spectra, postprocessing, barcodes, Betti curves, and Laplacian feature curves.

Reader layer changes add dependency-free public `read_pdb`, `read_mol2`,
`read_sdf`, `read_mol`, `read_pdbqt`, and `read_cif` functions in
`src/topokit/readers/molecular.py`, register their filename suffixes (including
`.mmcif` for CIF syntax), and export them from `topokit.readers`. They preserve
stable file IDs, canonical element labels, row-aligned raw atom/site columns,
charges/types, declared bonds, headers/properties, PDB model information, and
CIF cell/tags/loops as applicable. CIF fractional sites are transformed to
Cartesian coordinates without symmetry expansion. Raw weights remain uniform;
parsed bonds and attributes never choose a builder. PDB models and all
single-structure/atom-loop limitations fail explicitly rather than silently
discarding additional coordinate records.
Reader metadata now separates original `source_*` counts/bonds/connectivity
from current selected counts and induced records; repeated stable-ID subsets
remain internally consistent while preserving source declarations, cells,
tags, properties, and non-atom provenance. CIF lexical parsing retains quoted
token state, including control-like quoted values and internal apostrophes.
Quoted numeric site identifiers therefore remain strings, while quoted
coordinates, cell parameters, and charges are not silently coerced to numbers.
For a selected PDB model, current connectivity is induced onto that model's
atom IDs while all source `CONECT` records remain available as provenance.

Postprocessing adds public `spectral_variance`, `spectral_moment`,
`spectral_moments`, and `laplacian_energy`. Population variance and raw moments
are available through predefined `variance`, `moment_1` through `moment_4`, and
explicit-name aliases in `summarize_spectrum`. Existing `spectral_energy`
remains `sum(abs(lambda))`; the new centered Laplacian spectral energy is
`sum(abs(lambda - mean(lambda)))`. It equals classical graph Laplacian energy
for a combinatorial graph L0 and is explicitly a generalized functional for
Hyperdigraph L0/L1 or other operators. Its built-in always uses a complete full
spectrum including numerical zero modes, is unavailable for a partial spectrum,
and records per-statistic scope. ML feature schema checks now retain these
definitions and scopes. No topology, filtration, homology, Laplacian operator,
or persistence algorithm changed.
Centered-statistic evaluation translates and range-scales when possible, so
small representable gaps atop very large eigenvalue offsets survive; guarded
magnitude/log-rescaling fallbacks preserve signed underflow and true overflow
behavior.

Added `examples/molecular_formats_hyperdigraph_workflow.ipynb` and focused
reader/spectral tests. The 12-cell notebook reads every supplied source file in
full, uses explicit 24-site tutorial subsets for PDBQT/CIF analysis, constructs
bounded sequence Hyperdigraphs, and computes/displays only the requested H0/H1
and L0/L1 results. A clean temporary `nbconvert` execution produced six inline
figures with no cell error or stderr output; all panels were visually checked.
Betti curves use a separate dense evaluation grid, while infinite barcode
deaths are labelled as right-censored survival to the finite cutoff.
Added `examples/data/README.md` and extended `NOTICE.md` with fixture-level
provenance, exact-copy relationships, SHA-256 checksums, known attribution
gaps, and a conservative redistribution warning.

Validation: the full local suite reports **989 passed, one Linux-only RSS check
skipped on macOS, and the two expected explicit-skeleton warnings** in 17.07
seconds. Supplied fixture counts are PDB 36, MOL2 36, SDF 36, MOL 24, PDBQT
3,646, and CIF 29. `pip check`
reports no broken requirements. A no-isolation wheel contains all new modules,
excludes examples, and passes an isolated PDB-reader/spectral-feature smoke
test. Remaining concerns: these are coordinate parsers rather than complete
chemistry engines; SDF/MOL/MOL2 and CIF are single-structure/loop APIs; symmetry,
periodic topology, automatic bond use, physical featurization, remote
cross-platform validation, and protocol claims remain outside this change.

## 2026-09-11 — Native JSON PointCloud reader

Request: add direct `.json` coordinate input with required coordinates and
optional point IDs, labels, weights, units, and metadata, while resolving the
existing provenance-sidecar filename conflict explicitly.

Added public `readers.read_json` in `src/topokit/readers/formats.py`, exported
it from `topokit.readers`, and registered `.json` dispatch. The canonical
version-1 object uses a nonempty rectangular finite numeric `coordinates`
array; the exclusive `points` alias is accepted for ergonomic compatibility.
Optional same-length `ids`, `labels`, and `weights` arrays map directly to the
`PointCloud` contract. Optional `coordinate_units` and non-reserved `metadata`
are preserved, and optional identity fields are
`schema="topokit.point_cloud"` and integer `schema_version=1`.

The parser rejects duplicate keys, unknown top-level fields, both coordinate
keys together, non-finite/Boolean numeric values, ragged or empty arrays,
misaligned attributes, invalid or duplicate IDs, reserved metadata collisions,
and conflicting caller/embedded unit labels. Missing IDs and weights retain the
usual PointCloud defaults; missing units are labelled `unspecified`. The
existing `examples/data/point_cloud_24.json` remains a provenance sidecar, not
a coordinate record: extension dispatch now raises a targeted missing-
`coordinates` error and applications should read its paired CSV.

Added `examples/data/data_material_from_cif.json` as the canonical native-JSON
fixture. It contains 29 Cartesian coordinate rows derived from
`data_material.cif`, preserving the CIF site IDs and canonical element labels,
with explicit Pauling-electronegativity weights aligned row for row.

Changed reader/API files are `src/topokit/readers/formats.py`,
`src/topokit/readers/__init__.py`, and focused reader tests. Updated the root
README, reader architecture/contracts, current validation record, and example
data inventory. Scientific impact is limited to explicit point ingestion; the
reader selects no topology, performs no coordinate conversion, and adds no
inferred weights or labels. No new runtime dependency was introduced.

Validation: the JSON/fixture/registry/architecture/contract/readability slice
reports **149 passed**. The full local suite reports **1,025 passed, one
Linux-only RSS check skipped, and two expected explicit-skeleton warnings**.

## 2026-09-11 — One-format-at-a-time tutorial with AQUCOG and native JSON

Request: make the reader notebook clearly demonstrate each input type in its
own section instead of loading/analyzing all formats together; use the AQUCOG
MOF as the CIF example; add a coordinate JSON format with optional point
labels and weights; and derive its example from the existing material CIF.

Refactored `examples/molecular_formats_hyperdigraph_workflow.ipynb` into a
seven-file summary followed by independent PDB, MOL2, SDF, MOL, PDBQT,
AQUCOG-CIF, and JSON sections. Each section visibly performs its own read and
inspection, then its own call to the shared small Hyperdigraph analysis/plot
helpers before the next format begins. There is no combined input dictionary or
format-reading loop. Every section computes GF(2) H0/H1 persistence, complete
ordinary real L0/L1 spectra at six scales, spectral variance, first/second raw
moments, and centered Laplacian energy, and displays one four-panel barcode,
Betti-curve, and spectral-feature figure. The full PDBQT and CIF sources are
reported before deterministic first-24-site tutorial selections; the other
five examples use every parsed point.

Added `examples/data/AQUCOG_clean.cif`, byte-identical (SHA-256
`8afe10ec5002a0154fb96e5f067f5d96c7789ad42676bda8a94cf7712e2b725b`)
to `structure_10143/AQUCOG_clean.cif` in the CC BY 4.0 CoRE MOF 2019 v1.1.4
all-solvent-removed archive (DOI 10.5281/zenodo.7691378). It contains 162 P1
sites (Ni18 O54 C72 H18). Attribution in `NOTICE.md` and
`examples/data/README.md` distinguishes this computation-ready derivative from
the raw CSD AQUCOG deposition: the CoRE file omits the neon guest/solvent and
must not be described as a WebCSD export.

Added `examples/data/data_material_from_cif.json` with all 29 Cartesian
coordinates obtained from `data_material.cif`, the original site IDs,
row-aligned canonical element labels, and explicit Pauling-electronegativity
weights. The native reader/API and its strict version-1 contract are described
in the preceding JSON entry. An independent archive-member comparison and SHA
check verify AQUCOG byte identity; new integration tests verify its
counts/cell/Cartesian conversion and exact agreement
between the JSON coordinates/IDs/labels/weights and the source-CIF-derived
cloud. Updated the root README, data/reference notices, architecture,
contracts, migration/roadmap, and current validation record.

Validation: `python -m pytest -q` reports **1,025 passed, one macOS-skipped
Linux RSS check, and two expected explicit-skeleton warnings** in 19.34 seconds;
`python -m pip check` reports no broken requirements. A clean temporary
`nbconvert` run executes all 17 code cells sequentially with no cell errors or
stderr and creates exactly seven inline PNG figures; all seven panels were
visually inspected, while the 28-cell source notebook remains unexecuted. A
fresh wheel excludes every example/data file and passes an isolated zip-import
smoke test of the 29-row JSON record. Remaining scope: the first-24-site
selection and all-pair Hyperdigraph recipe are intentionally small API demos,
not a chemically bonded or periodic MOF model; CIF symmetry/periodic expansion,
multi-record JSON, and physical descriptor validation remain outside this
change.

## 2026-09-11 — Reusable result curves and neutral molecular-format tutorial

Request: make the multi-format notebook a standalone public tutorial and use
reusable plotting functions for the requested persistence and spectral curves.

Added public `visualization.plot_betti_curves`, which queries an existing
`PersistenceResult` on a caller-supplied scale grid, and
`visualization.plot_spectral_summary_series`, which displays an ordered aligned
sequence of already-computed scalar summaries. The latter defaults to minimum,
maximum, and mean on its primary axis and centered `laplacian_energy` on a
secondary axis. Undefined NaNs remain gaps. Both APIs consume supplied results;
the visualization layer does not recompute topology, persistence, Laplacians,
or spectral statistics.

Extended `plot_barcodes` with `mark_initial_stage=True`. The default retains an
open-circle cue for intervals already present at a recorded start and therefore
potentially left-truncated. Setting it to `False` hides the cue without changing
birth coordinates; the molecular tutorial uses this setting because its H0
vertices are known to be born exactly at zero.

Revised `examples/molecular_formats_hyperdigraph_workflow.ipynb` as a neutral,
one-format-at-a-time tutorial for PDB, MOL2, SDF, MOL, PDBQT, AQUCOG CIF, and
native JSON. Each independent section reads one file, analyzes H0/H1 and L0/L1,
and shows barcodes, Betti curves, and minimum/maximum/mean/centered-energy
curves through the public visualization API. Updated the README, visualization
guide, architecture boundary, 0.3 contracts, and validation record. Focused
coverage is recorded in `markdown/VALIDATION.md`. The full integration run
reports **1,033 passed, one skipped, and two expected warnings** in 19.62
seconds. Clean notebook execution completes 17/17 code cells with seven inline
PNGs and no errors or stderr; the wheel check and `python -m pip check` pass.

## 2026-09-11 — Spectral reference and custom-descriptor guidance

Request: make the molecular-format tutorial explain what its Laplacian spectra
represent, show the broader summary vocabulary, and demonstrate how users can
add descriptors without implying unsupported physics or persistent operators.

Extended the notebook's closing reference with the ordinary Hodge-Laplacian
formula and a cautious interpretation of zero, small-positive, and large
eigenvalues. “Soft,” “stiff,” “flexible,” and interaction-strength language is
identified as structural analogy unless weights and units encode the relevant
physics. A comprehensive table now separates built-in or settings-derived
summaries from custom quantiles, power sums, inverse-positive-spectrum sums,
log pseudo-determinants, entropy normalizations, and heat traces.

Added an executable ordered-mapping example for custom scalar summaries. The
documented contract is one real scalar from an isolated copy of the eigenvalues
selected by the call-wide `positive_only` setting. Parameter values belong in
stable feature names, undefined results remain explicit, and TopoKit does not
impute them. Across-filtration AUC, mean/spread, sampled range, total variation,
extremum locations, and largest sampled slopes are documented as a separate
second stage over sampled curves.

The reference explicitly identifies the plotted L0/L1 trajectories as ordinary
single-scale snapshots. A genuine two-scale operator requires
`core.persistent_laplacian(start=s, end=t)` or persistent-mode
`workflows.laplacian_series` with explicit scale pairs. Updated the README,
visualization guide, 0.3 contracts, and validation record. Focused validation
covered the reference structure, rendered tables, custom-summary execution, and
contracts. The focused notebook/spectral/visualization slice reports **23
passed**. The full integration run reports **1,034 passed, one skipped, and two
expected warnings** in 20.34 seconds; `python -m pip check` also passes. The
source notebook remains valid JSON/`nbformat`, and its existing 17 execution
counts, 21 output records, and seven inline PNG figures were preserved.

## 2026-09-11 — Lazy public route namespace repair

Request: determine whether the missing `builders.simplicial` attribute was
normal behavior or a software error, then repair the maintained package rather
than relying on test/import order.

Python does not automatically import arbitrary child modules when a parent is
imported, so the underlying behavior was normal; however, TopoKit's documented
parent-first usage and the protein-ligand receipt relied on those attributes.
That mismatch was a real repository-level API defect. Added lazy child-module
resolution and discoverability for all three public routes in both `builders`
and `core`, while retaining the existing function-only wildcard export and
avoiding eager route imports. The receipt now explicitly imports the simplicial
builder as an additional compatibility safeguard for older TopoKit installs.

Updated the maintained package source, public/architecture tests, installed-
wheel smoke test, README, architecture, scientific/0.3 contracts, migration
note, and current validation record. No topology, filtration, Laplacian,
feature statistic, output dtype, or data-selection rule changed. Existing
release archives, the frozen paper vendor tree, and historical AWS/pressure
baselines remain untouched; no package was published.

Validation: the architecture/public-layer slice reports **28 passed** and the
isolated protein-ligand receipt reports **9 passed, one skipped**. A clean run
excluding one unrelated absent-notebook test reports **1,035 passed, one
skipped, and two expected warnings**. The full run reaches the same passing
tests but retains one pre-existing `FileNotFoundError` for the missing
`examples/molecular_formats_hyperdigraph_workflow.ipynb`. A fresh temporary
wheel passes six-layer distribution checks and the installed three-route smoke
workflow; `python -m pip check` reports no broken requirements. The opt-in slow
protein-ligand reference calculation and dataset feature generation were not
run.

## 2026-09-12 — Relocated protein-ligand receipt and AWS production launch

Request: relocate the task-specific protein-ligand workflows into the TopoKit
project, repair the receipts for their new location, stage the package and
licensed dataset on the specified AWS host, and generate the paper-profile
features in parallel before copying them back locally.

Repaired the repository-level receipt path and renamed entry point, its focused
test import and slow-reference root, all command examples, default pytest test
discovery, root project inventory, and source-archive inclusion. Restored the
provenance-locked `reconstruct_pocket.py` exactly (SHA-256
`c09ef8ac21c383d6c7c1e86c5f5def5e87e20fbe821d4f4517197d06fdf9fe71`)
and corrected the dataset documentation link without changing the derived
`1xd1` pocket or its historical provenance record. Wheels continue to exclude
repository-level receipts; source distributions include them.

Extended the PDB reader to preserve the explicit neutral formal-charge token
`0` as integer zero. This resolves the only dataset-wide parser incompatibility,
141 neutral-water records in the full `6dyn` protein and 18 in its pocket,
without altering licensed input files. Blank fields remain `None`; signed
tokens and invalid-token rejection retain their existing behavior.

Optimized only the application receipt, not the generic TopoKit mathematical
core. It still constructs the complete filtered sequence hyperdigraph through
the public builder and derives scale membership from TopoKit's stored ordered
edges/births. At each changed scale it calls the unchanged native q=0
hyperdigraph Laplacian on the incident-vertex block and restores one exact zero
eigenvalue per isolated singleton before the existing five-decimal summaries.
Exact atom-index channel selections are memoized. Full-channel dense and
hyperedge guards are still enforced before this decomposition. The schema
records the optimization while retaining
`direct_graph_laplacian_substitution=false`; tensor shape, ordering, dtype,
statistics, and scientific parameters are unchanged.

Validation: the focused receipt suite reports 17 passes plus one opt-in skip,
and its pinned `1a1e` source-tensor check passes when enabled. The complete
local suite excluding one independently absent notebook-note fixture reports
1,053 passes, two skips, and two expected warnings. All 5,376 full-protein/MOL2
pairs passed exact reader/selection preflight. On AWS, the optimized and
unoptimized full-protein `1a1e` files are byte-identical (SHA-256
`131ed0a5dcd287bb22a07fb057c70294369d9998e50a85c0e8d2039f79023a6c`),
while runtime fell from 613.91 to 21.73 seconds. A matched eight-complex test
took 68.53 seconds with four workers and 47.57 seconds with eight. The staged
dataset passed a checksum-only comparison against local.

The eight-worker AWS production run used one BLAS thread per process,
`float32` atomic per-sample output, resume validation, and the audited
`--max-dense-entries 13000000` override. The largest actual incident matrix was
only 1,030 square. It completed all 5,376 unique complexes in 23,897.41 seconds
(6 h 38 m 17.41 s), with 5,375 newly generated tensors and one validated
existing `1a1e` tensor. Exhaustive AWS and local audits found 5,376 finite
`(6, 100, 143)` tensors, 5,376 matching success records, zero failures, and
matching SHA-256 digests for every tensor and 10,752 molecular inputs. The
checksum-identical local copy contains 10,755 files totalling 1,883,747,551
bytes under the dataset's `features/` directory. No feature artifact was
altered during reconciliation.

## 2026-09-12 — Explicit zero encoding for absent Laplacian chain groups

Request: keep higher-dimensional ordinary or persistent Laplacian feature
blocks finite and fixed-width when the corresponding source chain group is
absent.

Confirmed across simplicial, sequence-Hyperdigraph, and interaction cores that
structural absence already has the correct raw representation: a complete
`0 x 0` operator with an empty eigenvalue array, empty basis, and nullity zero.
No fake zero eigenvalue is inserted. Standardized shared spectrum metadata now
records the full `operator_dimension`, `source_chain_dimension`, and
`structural_absence`; partial eigensolves retain the full source-basis size.

Added `empty_operator_policy="preserve"|"zero"` to
`postprocessing.summarize_spectrum` and, through option forwarding,
`summarize_spectra`. The backward-compatible default preserves the existing
per-statistic empty behavior. The opt-in zero policy encodes every requested
predefined statistic as zero only for a core-confirmed complete empty operator
whose basis, nullity, optional matrices, and standardized metadata all agree.
Custom callbacks always execute and define their own empty-input value. The
authoritative spectrum remains unchanged. Missing or contradictory receipts,
nonempty all-zero operators, empty positive-mode selections, partial spectra,
and failed computations are not filled. Feature-matrix schema checks include
the selected policy, while sample-specific fill provenance is retained without
preventing empty and nonempty records from stacking.

Changed the three public core adapters, postprocessing spectral summaries, ML
schema validation, focused tests, root README, architecture, notation,
contracts, current validation record, and the multi-format notebook. Focused
postprocessing/core/ML tests report **240 passed** with two expected warnings.
The notebook was re-executed with finite L0/L1 summary assertions under the
explicit structural-zero policy. During integration, the
notebook-note regression and current README command were also aligned with the
existing `examples/different_input_formats_workflow.ipynb` filename; historical
filename records were not changed. The complete local
suite reports **1,095 passed, two skipped, and two expected warnings** in 19.29
seconds, and `python -m pip check` reports no broken requirements. A fresh wheel
passes the six-layer distribution check and an isolated installed-package smoke
run covering all three routes and the new zero-fill policy; examples and data
remain excluded from the wheel.

## 2026-09-12 — Explicit protein-ligand XGBoost training and CASF evaluation receipt

Request: add a reviewable, argument-driven XGBoost receipt that trains on the
completed protein-ligand features with 10,000 estimators and evaluates
CASF-2007, CASF-2013, and CASF-2016 using RMSE, MAE, and Pearson correlation.

Added the repository-level application
`workflows/protein_ligand_prediction/train_xgboost.py` and its focused tests.
The receipt reads only explicit label manifests and the completed feature
schema, rejects training/test PDB-ID overlap, preserves intentional overlap
among CASF editions, validates and flattens each `(6, 100, 143)` tensor in C
order, and imports XGBoost lazily. It neither regenerates topology nor changes
the TopoKit hyperdigraph core, feature-generation receipt, installed workflow
API, or required dependency set.

The requested defaults are `gbtree` squared-error regression with learning rate
0.01, depth 5, minimum child weight 2, row subsampling 0.6, 10,000 histogram
trees on CPU, one worker thread, and seed zero. Because XGBoost accepts a numeric
column fraction rather than a literal `sqrt`, the receipt maps its shorthand to
`1/sqrt(p)`. For the 85,800 flattened features, default per-tree sampling is
approximately 0.0034139437, or about 293 candidates per tree. This is more
aggressive than scikit-learn's per-split `max_features="sqrt"`; the documented
closer analogue is `colsample_bytree=1` with `colsample_bynode=sqrt`, and the two
fractions are explicitly reported as cumulative.

CASF editions receive separate metrics and prediction CSVs. The default numeric
label policy retains paper-compatibility by treating bounds as their recorded
values, emits a warning, and provides exact-only sensitivity when a benchmark
contains a bound; `exact-only` and `error` are explicit alternatives. No
validation split, tuning, cross-validation, early stopping, scaler, or
test-guided model choice is introduced. Each new output directory receives
`plan.json`, `model.ubj`, `metrics.json`, three prediction files, and a final
`metadata.json` completion marker with resolved parameters and provenance.

Documentation impact: updated `README.md`, `workflows/README.md`, the task
README, `markdown/ARCHITECTURE.md`, `markdown/CONTRACTS_0_3.md`, and
`markdown/VALIDATION.md`; this entry records the change. Scientific impact is
limited to a new downstream modeling/evaluation application over unchanged
stored features.

Validation: the focused XGBoost-receipt suite reports **12 passed**, the
combined task workflow suite reports **29 passed and one opt-in skip**, and the
complete repository suite reports **1,095 passed, two skipped, and two expected
warnings**. A real-data
dry run resolves 4,836 training rows and 675 CASF rows representing 540 unique
test IDs, including 119 cross-edition memberships and zero training/test
overlap. It verifies the selected feature-record identities and reports raw
matrix sizes of 1,659,715,200 training bytes and 185,328,000 unique-test bytes.

The production-fit concern in the original receipt review is resolved by the
completed AWS run recorded below. Cross-version numerical reproducibility has
not been established. The raw matrix estimates are not peak-memory guarantees,
and the numeric treatment of censored labels remains a declared compatibility
convention rather than a censor-aware loss.

## 2026-09-12 — Completed protein-ligand XGBoost AWS production

Completed the explicit full-matrix receipt with XGBoost 3.4.1 on AWS using all
eight logical CPUs (`n_jobs=8`). The model retained the requested `gbtree`
squared-error configuration: learning rate 0.01, depth 5, minimum child weight
2, row subsampling 0.6, 10,000 CPU `hist` estimators, seed zero,
`colsample_bynode=1`, and
`colsample_bytree=0.0034139437099945944` from the receipt's per-tree `sqrt`
mapping. The eight-thread setting changes execution parallelism from the
conservative default, not the scientific estimator configuration. Training used
all 4,836 rows and inference covered 540 unique test complexes. The resolved
scientific-configuration SHA-256 is
`07a5932f67e6cb8254dab14388e72228c77f635259dfe5618601781f9cc92d03`.

Fit time was 14,890.228262688004 seconds and total runtime was
14,899.276525333 seconds (about 4 h 08 m 19 s), with approximately 20.46 GiB
peak observed RSS. The completed 26,058,516-byte UBJSON artifact reports 10,000
boosted rounds/trees and 85,800 features. The copied local output directory is
checksum-identical to AWS and has a 25M filesystem summary.

Separate numeric-policy results were: CASF-2007, N=195, RMSE
1.5381142107194636, MAE 1.175577779574272, PCC 0.7916629735588632; CASF-2013,
N=195, RMSE 1.518889291497816, MAE 1.1922867497175167, PCC
0.7559748795688793; and CASF-2016, N=285, RMSE 1.2976941358918523, MAE
1.0036906847702831, PCC 0.825679634347487. CASF-2007 exact-only sensitivity was
N=194, RMSE 1.5365720385997774, MAE 1.172293924597121, PCC
0.7907548257830246. No pooled CASF result was computed. The numeric label policy
is a declared compatibility convention, and this updated refined-set training
protocol is not an exact replication of the Nature Machine Intelligence
paper's machine-learning experiment. There was no implicit split, tuning,
cross-validation, early stopping, or CASF-guided model selection.


## 2026-09-12 — direct supervised protein-ligand transformer

Request: provide a simple TopoFormer-inspired training/prediction application
using the completed TopoKit features, without pretraining, alongside the
existing traditional ML workflow.

Changes: added task-local `transformer_model.py`, `train_transformer.py`,
`predict_transformer.py`, optional `requirements-transformer.txt`, focused
tests, and `TRANSFORMER_PROTOCOL.md` under
`workflows/protein_ligand_prediction/`. Extracted common manifest, feature,
hashing, metric, and prediction-CSV code into `prediction_common.py`;
`train_xgboost.py` re-exports its original helpers and preserves its estimator
defaults and CLI behavior. Its receipt version is 1.0.1 and provenance also
hashes the shared helper source. No installed TopoKit source/dependency or
stored feature recipe was changed.

Scientific impact: the new randomly initialized encoder uses 100 filtration
scale tokens, each with 858 statistic/channel values, for a 523,521-parameter
affinity regressor. It fits log1p feature normalization and target scaling
using optimization rows only. Validation requires an explicit fraction of
training-manifest rows; optional full refitting initializes again on all
training rows for the internally selected epoch count. CASF is evaluated only
after the final fit. Model/preprocessing/checksum artifacts support standalone
label-free prediction. Existing output directories and prediction CSVs are
refused, and final metadata is withheld if manifests, schema, or training
source change during fitting.

Documentation: updated root and workflow READMEs, application protocol,
`markdown/ARCHITECTURE.md`, `markdown/CONTRACTS_0_3.md`, and
`markdown/VALIDATION.md`. Historical XGBoost production records are preserved.

Validation: the final complete repository run passed 1,119 tests with ten
expected skips and two expected warnings; the full task workflow suite passed
61 checks with one opt-in skip in the PyTorch environment. Its transformer
subset passed 32 checks with PyTorch 2.7.1 on CPU, including CASF isolation, exact fresh-refit
equivalence, fitting-only preprocessing, and checkpoint/inference restoration.
All 5,376 stored tensors pass shape/dtype/order/finiteness/nonnegativity checks.
A 32-training-complex smoke run used the full default feature/model dimensions,
completed two selection and two refit epochs, and reloaded predictions agree
within 2.08e-7 target units. See the current validation log for final combined
suite counts and environments.

Unresolved: full transformer training and repeated-seed comparisons have not
been run; the smoke scores are not benchmark results. GPU execution, complete
training memory/runtime, and cross-device reproducibility remain unmeasured.
This is a compact supervised model and an updated refined-membership protocol,
not a reproduction of TopoFormer's pretrained and ensemble pipeline.

## 2026-09-12 — scikit-learn protein-ligand gradient boosting receipt

Request: add a simple scikit-learn GBDT training and prediction path over the
completed protein-ligand features, using learning rate 0.01, 10,000 estimators,
square-root feature sampling, maximum tree depth 7, and otherwise library
defaults.

Changes: added the task-local, repository-only `train_sklearn_gbdt.py` receipt
and focused tests. It reuses `prediction_common.py` and therefore retains the
same explicit PDBbind/CASF manifests, full-manifest train/test leakage check,
C-order flattening, tensor and manifest hashes, label policies, separate CASF
RMSE/MAE/PCC reports, transactional artifact publication, and completion
marker as the existing models. The receipt is included in the source archive
but remains outside installed wheels. Scikit-learn stays lazy and is supplied
by the existing optional `.[ml]` extra.

Scientific and API impact: no TopoKit mathematical core, reader, builder,
postprocessor, installed workflow API, feature tensor, manifest, XGBoost model,
or transformer model is changed. The new estimator fixes
`learning_rate=0.01`, `n_estimators=10000`, native
`max_features="sqrt"` at every split, and depth 7 per tree. Random state zero is
the sole reproducibility exception to the requested library defaults. The
installed scikit-learn version and complete resolved parameter mapping are
recorded for every run. Classic `GradientBoostingRegressor` provides no
`n_jobs`, so the receipt does not imply multithreaded fitting.

Documentation: updated the root and workflow READMEs, the shared model
protocol, architecture boundary, 0.3 contracts, current validation record, and
source-distribution assertions. Historical XGBoost production results and
transformer validation remain intact.

Validation: the focused receipt reports 11 passes with scikit-learn 1.6.1. The
complete repository suite reports 1,130 passes, ten expected skips, and two
expected construction-truncation warnings; the base-environment protein-ligand
suite reports 64 passes and nine skips. The full AWS production fit subsequently
completed with scikit-learn 1.9.0: all 4,836 rows, 10,000 stages, 85,800 input
features, and 292 candidates per split were independently confirmed. Primary
`(RMSE, MAE, PCC)` results are CASF-2007
`(1.558995, 1.187356, 0.786402)`, CASF-2013
`(1.519079, 1.194568, 0.760098)`, and CASF-2016
`(1.302979, 1.015562, 0.825798)`; the CASF-2007 exact-only sensitivity result is
`(1.557340, 1.183928, 0.785494)`. Fitting took 5,177.92 seconds and total receipt
runtime was 5,192.60 seconds, with 1.80 GiB measured peak RSS. The 47,049,346-byte
joblib model and every output/log file were copied back with AWS-identical
SHA-256 values. Independent same-environment loading rehashed all 540 test
tensors and reproduced all saved predictions and metrics. Joblib output remains
pickle-based, trusted-only, and sensitive to the recorded Python/scikit-learn
environment.

## 2026-09-12 — singleton-element protein-ligand v1 feature receipt

Request: introduce an independent protein-ligand feature-generation strategy
that retains the earlier NMI-inspired large-profile geometry while using four
singleton protein elements, ten singleton ligand elements including hydrogen,
and five compact L0 spectral summaries. Keep the original receipt and feature
store unchanged, support explicit dataset/HPC execution, and publish results to
a separate recipe directory.

Changes: documented the repository-only
`workflows/protein_ligand_prediction_v1/generate_features_protein_ligand_v1.py`
receipt, its focused test location and design note, and added the v1 test path
to default pytest discovery. Source-distribution assertions now require the v1
script, README, design note, and focused test; the existing wheel boundary
continues to exclude all repository-level workflows. Root, workflow,
architecture, 0.3 contract, and current-validation documentation now distinguish
the v1 schema from the frozen original recipe.

Scientific and API impact: there is no installed API, dependency, reader,
builder, Hyperdigraph definition, or Laplacian-core change. The v1 schema uses
protein-major ordering over `C/N/O/S` by
`C/N/O/S/P/F/Cl/Br/I/H`, producing 40 singleton channels. At each of 100 scales
it records `n0`, `beta0`, positive-spectrum mean, relative minimum-positive
mode, and relative mean absolute deviation. Absolute tolerance is `1e-10` with
no five-decimal eigenvalue pre-rounding; arithmetic is `float64` and default
storage is C-order `float32`, giving shape `(5, 100, 40)`. Empty positive spectra
use explicit zero sentinels while size and nullity retain structural information.

Validation status: focused receipt tests, a fresh source archive check, and AWS
production generation remain pending at this documentation checkpoint. No v1
artifact count, checksum, runtime, or modeling result is claimed. The existing
143-channel production artifacts and validation history remain unchanged.

## 2026-09-12 — completed singleton-element v1 AWS generation and local copy

Request: execute the new 40-channel, five-summary protein-ligand receipt in
parallel on the supplied AWS host, copy the completed features into a new local
dataset feature directory, preserve the original store, and keep `2cer`'s
numeric modeling label at `9.22`.

Changes: uploaded the v1 repository receipt and reused the checksum-matched
5,376-complex dataset on AWS. A 32-complex calibration preceded the resumable
eight-worker production run, which validated those 32 artifacts and generated
the remaining 5,344. The independent recipe directory and execution log were
copied back beside the existing feature store. Root, workflow, dataset, and
current-validation documentation now record exact artifacts, hashes, resource
use, and the `5f2u` source-data caveat. The CASF-2007 manifest retains numeric
`label_logka=9.22` for `2cer`; its source inequality remains provenance only.

Scientific and API impact: no installed API, dependency, reader, builder,
Hyperdigraph definition, Laplacian core, original generator, original feature,
label value, or trained model changed. The production schema pins generator
SHA-256 `4fff3aea0be8a9a49fa9a45b527234355a5df4185b5903c93b37b26b66fe0d5b`
and AWS TopoKit source-tree SHA-256
`ba22b5145743f23aa32ce2fd39d837c82941697713417a35e396736d569532ee`.
The local tree has later metadata-only differences in its public Hyperdigraph
adapter and unrelated unused files; the readers, builder, solver, and private
Hyperdigraph algebra used for generation are byte-identical, and an independent
`1a1e` run matches the AWS tensor bit for bit.

Validation: focused v1 checks report 33 passes; the complete suite reports
1,163 passes, ten skips, and two expected warnings. Source/wheel distribution
checks pass. The AWS run completed all 5,376 complexes with zero failures in
42:51.27 wall time at 796% CPU and 176,448 KiB maximum RSS. Exhaustive AWS and
post-copy local audits rehashed all 5,376 tensors, 5,376 success records, and
10,752 source inputs. Every tensor is finite, C-contiguous, little-endian
`float32`, shape `(5, 100, 40)`. The 10,755-file, 452,662,724-byte recipe has
aggregate SHA-256
`e0b25c4ffd51685a9842b29bcbaef28b31afc8dfb50dd95b2f7330c9a90613bd`;
the copied 1,509,308-byte log has SHA-256
`201ab2dbe41147d72390e18d615bf01b1ea4bfd625b43fa905919d51cdef1e10`.

Unresolved concerns: `5f2u`'s selected full-protein PDB duplicates its peptide
ligand as chain E residues 640--643. The frozen inclusive distance rule therefore
encodes 27 zero-distance contacts at scale 0; no silent chain filtering was
introduced. The current resume path validates recipe and input/output hashes,
tensor shape/dtype/finiteness, but not every redundant field in an existing
JSON record. The exhaustive production audit checked those fields independently.
Any future resume-validator hardening or chain-filtering policy must change the
generator identity and create a new recipe rather than mutate this store.

## 2026-09-13 — completed standardized singleton-40 GBDT production run

Request: train the same scikit-learn GBDT configuration used by the earlier
baseline on the new singleton-40 compact-5 features, normalize inputs from
training data only, reuse that fitted scaler for all test sets, and copy the
completed artifacts back without replacing the earlier model.

Changes: added the repository-only
`workflows/protein_ligand_prediction_v1/train_sklearn_gbdt.py` receipt and its
focused tests, extended source-distribution assertions, and documented the
application-layer normalization boundary in the architecture and 0.3 contracts.
The completed independent model was then recorded under
`models/sklearn-gbdt-standardized-singleton40-20260912-v1/` with its adjacent
AWS log. Updated the v1 workflow README, workflow index, root README, current
validation record, and dataset README to distinguish this pipeline from the
historical unscaled 85,800-feature GBDT. No generated feature tensor, label
manifest, earlier training receipt, or earlier model artifact was modified.

Scientific and API impact: no installed TopoKit API, mathematical core,
reader, builder, postprocessor, feature recipe, or earlier model changed. The
new artifact consumes
`nmi-geometry-singleton40-compact5-hyperdigraph-l0-f33790fac29d5722`, flattens
each `(5, 100, 40)` tensor to 20,000 values, and persists one
`Pipeline(StandardScaler, GradientBoostingRegressor)`. Only all 4,836 training
rows fit the scaler; its 3,039 zero-variance coordinates remain finite with the
standard scaler convention. Raw CASF rows reuse that scaler during prediction,
and targets remain unscaled. With scikit-learn 1.9.0, the GBDT completed 10,000
stages at learning rate 0.01, depth 7, `max_features="sqrt"` resolved to 141,
random state 0, verbosity 0, and otherwise installed defaults.

Validation: the final suite reports 1,175 passes, ten skips, and two expected
warnings in 9.46 seconds; the v1 GBDT subset reports 12 passes. Fresh source
and wheel archives pass their distribution boundary checks, and `pip check`
reports no broken requirements. Pipeline reload reproduced the in-memory
predictions exactly. Primary
`(RMSE, MAE, PCC)` values are CASF-2007
`(1.607853941401926, 1.2243135403887215, 0.7688500041456876)`, CASF-2013
`(1.5761652358408038, 1.265168640352196, 0.747099200175994)`, and CASF-2016
`(1.346295816342027, 1.0504023703157173, 0.8261117846707097)`. The 194-row
CASF-2007 exact-only sensitivity result is
`(1.6047232650544907, 1.219645710592681, 0.7684326083768271)`. Fitting took
1,579.374834 seconds, total receipt runtime was 1,590.717836 seconds, and GNU
`time` measured 26:33.32 wall time, 1,361,176 KiB maximum RSS, zero swaps, and
exit status zero.

Artifact validation: the 43,285,710-byte pipeline has SHA-256
`3b1f58f86fab4745167eac032d3ce0e52e80a49f54b66d9cfa576ebbb5335367`.
The seven-file, 43,420,496-byte model directory has aggregate path-and-content
SHA-256
`257b599bd228f0280bdad120fe3b13897e5a99856799516daa21dc2a62f6b577`;
the 6,862-byte log has SHA-256
`e1c93972b4b448b70d9ea77d439c27b162e61e054e72573755cd6e7fe23636d4`,
and `metrics.json` has SHA-256
`a6f0eae8c34350cc13b85ef219bb4d2f73fbcd33000ad5194671835576d1925e`.
The executed receipt SHA-256 is
`50195a9faacf35e8c365ca83fb84ddffc2ffda57a027a5b41049ea6d0bbbc9cb`.

Unresolved concerns: this is one fixed-seed baseline without hyperparameter
selection, cross-validation, repeated-seed uncertainty, or censor-aware loss.
The numeric compatibility policy treats the one bounded CASF-2007 label,
including `2cer=9.22`, as a point label; the exact-only sensitivity result is
reported separately. Standardization is retained because it was explicitly
requested even though ideal decision-tree splits are invariant to positive
affine feature transforms. Joblib remains a trusted-only pickle artifact whose
portable reuse requires a compatible recorded Python/scikit-learn environment.

## 2026-09-13 — v2020R1 general-set compact-six protein-ligand receipt

Request: replace the refined-set scope of the next feature experiment with the
available updated PDBbind v2020R1 general P-L archive, exclude the union of the
CASF-2007/2013/2016 cores, retain the Nature Machine Intelligence label tables,
and append `max(lambda_positive) / mean(lambda_positive)` to the five existing
singleton-40 Hyperdigraph `L0` summaries. Prepare the complete structures and
run the independent feature store on the supplied AWS host without replacing
earlier stores.

Changes: added repository-only
`workflows/protein_ligand_prediction_v2/prepare_v2020r1_general.py` and
`generate_features_protein_ligand_v2.py`, their focused tests and README, and
updated the root/workflow/architecture/contracts documentation, pytest
discovery, and source-distribution assertions. Preparation atomically adds only
missing archive files, byte-validates existing ones, pins its own version/hash,
and writes a new 18,498-row training manifest plus three `_nmi` CASF manifests.
The archive-external `1xd1` files and pocket-repair sidecar are fingerprinted;
existing manifests, feature stores, models, and structure bytes are preserved.

Scientific and API impact: no installed reader, builder, Hyperdigraph
definition, Laplacian core, dependency, or ML API changed. The v2 tensor is
C-order `(6, 100, 40)` with float64 spectral arithmetic and canonical float32
storage. Its first five rows match v1 exactly; statistic index 5 is the new
relative largest positive eigenvalue. The active NMI labels and equality
relations remain distinct from updated-index measurement provenance. The local
updated archive supplies 19,037 IDs; excluding 539 archive-backed CASF IDs gives
18,498 training IDs, and retained `1xd1` makes 19,038 unique feature inputs.

Validation: the focused v2 suite reports 51 passes and the complete local suite
reports 1,226 passes, ten skips, and two expected construction-truncation
warnings. An independent oracle matched direct `D - A` spectra for 240 random
bipartite channels and 2,160 snapshots. All 76,148 archive structure files were
hash-verified locally, the new manifests have 18,498/195/195/285 rows, and a
local `10gs` smoke tensor is finite C-order float32 with shape `(6, 100, 40)`.
The production AWS run and checksum-identical local copy remain pending at this
checkpoint and must be recorded before any completed-store claim.

Unresolved concerns: the provider's updated v2020R1 archive is a 19,037-entry
v2020/v2024 intersection, so its core-excluded training set has 18,498 rather
than the 18,904 rows in the earlier paper label release; 406 labeled paper IDs
have no structure in this archive. Feature generation remains deliberately
label-independent, and no predictive result is claimed.

## 2026-09-13 — Reproducible `6djc` MOL2 input normalization

Request: repair the sole v2 production input failure without changing the
TopoKit mathematical implementation, frozen v2 generator, or archive-supplied
structure bytes, then make the corrected current general/NMI manifests safe to
resume on AWS.

Changes: advanced
`workflows/protein_ligand_prediction_v2/prepare_v2020r1_general.py` to
preparation version 1.1. It retains `6djc_ligand.mol2` byte-identically and
adds `6djc_ligand.normalized.mol2`, which removes exactly the second member of
the sole adjacent `@<TRIPOS>ATOM` header pair. A deterministic sidecar pins the
archive/member, source and derived sizes/hashes, and exact byte replacement.
Only the `6djc` row in the current general/NMI manifests selects this new path;
all other manifest bytes are unchanged. Dry-run reports both pending repair
files and any legacy-output migration without writing. Existing generated
outputs can be replaced only when they match the exact pinned preparation-v1.0
manifest/provenance hashes and the old `6djc` path semantics; arbitrary
differences remain hard failures. Updated the focused tests, root/workflow/data
READMEs, and current validation record.

Scientific and API impact: this is a task-specific input syntax normalization,
not a molecular-coordinate edit or mathematical change. TopoKit's installed
reader, builder, Hyperdigraph definition, Laplacian core, and package API are
unchanged. The v2 generator SHA-256 remains
`42075b820d2c49b9cc86873bc1845cc00d22f6fcae90b48c6f8bed59d09dbec4`;
the updated preparation SHA-256 is
`abe0907df6fd9ffca8955e22b1b72561d5b26c2bc4c5c5cddb36533788c09822`.

Validation: the canonical source remains 12,422 bytes / SHA-256
`9234be921e87a6e4c6f0099e32a933efcebf00785819f9783d0df4ee7ce0a1d4`;
the normalized derivative is 12,408 bytes / SHA-256
`fd5b7ba7c3dea2e9a668ad71df3cf18a21cbd6bd655d7d72a837d84ca043a65a`;
and its provenance sidecar SHA-256 is
`a018a8018d6c5c1420d446cfda0bf74bbc7ba892178f78da9c410674503f299a`.
The migrated 18,498-row training manifest has SHA-256
`df96d5b1c6f0a123019250bbe5e292f9c34b8171fb8eca6f38f7ce43a04ef8b0`;
replacing its one normalized path with the canonical path reconstructs the
exact pinned predecessor hash
`4518042d72f4f7b989cb4be01e99555ba66ede332cf3147e9e92fca28d5549ab`.
The local completion-provenance SHA-256 is
`a9cd7266472b6e55b9227b8344017501b40eafa2ef9408c194fd34d127cd21f4`.
An immediate repeat was idempotent and verified 76,148 canonical files plus
both repair files. The normalized ligand parses as 118 atoms; the preserved
source continues to raise the expected repeated-section error. The focused v2
suite reports 55 passes, and the complete local suite reports 1,230 passes,
ten skips, and two expected warnings in 18.89 seconds.

Unresolved concerns: the interrupted AWS run must be resumed for `6djc` with
the migrated manifest and normalized input, then the complete 19,038-sample
store must pass the independent artifact/input-hash audit and checksum-identical
copy-back verification before production completion is claimed.

## 2026-09-13 — Completed v2020R1 compact-six AWS production and copy-back

Request: finish the independent general-set feature generation on the supplied
AWS host, exhaustively audit the feature/input store, and copy the completed
recipe back without replacing prior features or models. This completion entry
supersedes the pending state recorded in the two historical checkpoints above;
those entries remain unchanged as an execution trace.

Execution: the 64-complex calibration completed with 64 successes in 34.86
seconds. The first eight-worker full pass generated 18,973 new results,
validated 64 calibration results, and transparently recorded one `6djc`
failure because the canonical MOL2 contains adjacent repeated
`@<TRIPOS>ATOM` sections. That pass took 2:42:37 wall time at 797% CPU,
reached 193,168 KiB maximum RSS, used no swap, and exited 1. Preparation 1.1
then retained the canonical bytes and selected the deterministic,
hash/provenance-pinned normalized derivative for only `6djc`. Its targeted run
succeeded in 7.53 seconds with 120,032 KiB maximum RSS. A final full resume
validated all 19,038 existing results, reported zero failures and zero
unattempted IDs, and exited zero in 34.17 seconds at 341% CPU with 148,676 KiB
maximum RSS and no swap. The four run records deliberately preserve the
calibration, initial failure, targeted repair, and clean complete-resume history.

Artifacts: recipe
`nmi-geometry-singleton40-compact6-hyperdigraph-l0-4a27520c26f43dba`
contains 19,038 NPY tensors, 19,038 success records, one schema, four run
records, and no failure file. Its 38,081 regular files total 1,911,614,102
bytes. NPY files total 1,830,084,864 bytes, including exactly 1,827,648,000
numeric float32 bytes. The feature-identity and aggregate path/content SHA-256
values are
`1ced88443ff408638e582c3b4e999a7fbf404c628d72b3ad6069836f25ee320f`
and `d6f733f27a502a435fa57256e39afb20c48560db6ef3af89434213832e0fd5ec`.
The schema SHA-256 is
`3fd66f389307a7b3e5670b745307377aebc502871d77358f191999320d3570b4`.
Copied logs and audit reports live outside the recipe in its adjacent
`.aws-logs/` directory; the main historical full-pass log is also retained as
the adjacent `.aws.log`.

Validation: the exhaustive audit rehashed all 38,076 selected protein/ligand
inputs (13,986,614,119 bytes; identity SHA-256
`1c94eb94882436ec8bda423ec794e40c2cb8f8ac5389847a7997d8f64727785f`),
verified exact manifest membership and zero train/test overlap, checked every
tensor/record/input hash and spectral invariant, and compared the first five
rows bit for bit against v1 for all 5,376 shared IDs with zero mismatches. The
AWS audit passed with zero errors in 1:02.98, 233,504 KiB maximum RSS, no swap,
and exit zero. The copied local store passed the same audit with zero errors in
17.95 seconds; its report is byte-identical to the AWS report, SHA-256
`0579db43c7800c80b98f0a8cf0f319cfdeb4b777d736109c0cda75b6cb11036e`.
A checksum-only dry `rsync` emitted no differences and exited zero. The final
full local suite reports 1,230 passes, ten skips, and two expected
construction-truncation warnings in 18.89 seconds.

Scientific/API impact: no installed TopoKit mathematical source, feature
definition, or generator changed during production repair. This completion is
feature generation only; no v2 model or CASF prediction was run. The selected
NMI labels and inspired geometry do not imply exact reproduction of the paper's
full feature or ensemble pipeline. Earlier feature stores, manifests, and
model artifacts remain unchanged.

## 2026-09-13 — v2 standardized general-set GBDT receipt and production

Request: add the same standardized scikit-learn gradient-boosting strategy to
the completed compact-six general-set features, using all available training
rows and keeping the three CASF editions separate.

Changes: added the repository-only
`workflows/protein_ligand_prediction_v2/train_sklearn_gbdt.py` receipt and
`tests/test_train_sklearn_gbdt.py`, then updated the root, workflow, v2,
architecture, validation, and dataset documentation. The receipt reuses the
task-local `prediction_common.py` contract, refuses a pre-existing output
directory, writes an immutable plan before fitting, revalidates manifests,
schema, source, and feature identities before publication, persists scaler and
regressor together, confirms joblib reload predictions bit for bit, and writes
completion metadata last. Trainer and focused-test SHA-256 values are
`a2179cc1a827a87ad23273c71b990b544f28f2f0fdd9d0924ad7eb908cc78d0d`
and `c8ebfd493e41fe1a30705739aa48f18ddf13a5777eda5fc3e09eb15d1508806f`.

Scientific contract: C-order `(6, 100, 40)` tensors flatten to 24,000
coordinates. Only the 18,498 training rows enter
`Pipeline(StandardScaler, GradientBoostingRegressor).fit`; raw CASF matrices
reuse that fitted scaler through `Pipeline.predict`, and affinity targets stay
unscaled. Requested GBDT overrides are learning rate 0.01, 10,000 stages,
maximum depth 7, per-split `max_features="sqrt"` resolving to 154 candidate
features, random state zero, and verbosity zero. Other estimator settings remain
at scikit-learn defaults. The 195-row CASF-2007, 195-row CASF-2013, and 285-row
CASF-2016 manifests receive separate RMSE, MAE, and PCC reports. There is no
hyperparameter search, cross-validation, validation-set selection, early
stopping, or CASF-dependent fitting.

Validation: the focused trainer suite reports 14 passes in 1.02 seconds, the
complete v2 suite reports 69 passes in 1.75 seconds, and the complete repository
suite reports 1,244 passes, ten skips, and two expected construction-truncation
warnings in 20.20 seconds. A real-data dry run resolved the complete
18,498/195/195/285 contract without writing output. A five-estimator smoke run
using 50 training rows and ten rows per CASF edition completed all artifacts;
its reloaded pipeline predictions are bit-identical to the in-memory values.

AWS production used Python 3.12.14, NumPy 2.5.2, and scikit-learn 1.9.0. An
initial approximately three-minute invocation was intentionally
terminated and preserved as an aborted preflight after correcting cosmetic
v1-to-v2 wording in CLI help. It is not a model result. The clean full run used
all 18,498 training rows and completed all 10,000 stages under
`models/sklearn-gbdt-standardized-compact6-general-nmi-20260913-v1/`.
It wrote completion metadata and exited zero. CASF-2007/2013/2016
`(RMSE, MAE, PCC)` results are
`(1.462757149449226, 1.1323666580024467, 0.8127462077634457)`,
`(1.4534406512952716, 1.157615444869793, 0.7934419607156724)`, and
`(1.2374441451915585, 0.9731991747296493, 0.8550191429301052)` for 195,
195, and 285 rows, respectively. Fitting took 6,698.197481 seconds; total
receipt runtime was 6,732.784056 seconds. GNU `time` measured 1:52:31 wall
time, 5,780,004 KiB maximum RSS, zero swaps, and exit status zero.

Artifact validation: the seven model files total 51,828,928 bytes. Pipeline
and metrics SHA-256 values are
`5e8c8f127038c1a6cec8db1c177b389ebf81bec6601478eab1930fdc0bc3ee78`
and `8eb2d6f9e897c61938630c152a22c2c28332991933724b8648eef107bcb9c882`.
Their independent aggregate sorted-relative-path/NUL/content/NUL SHA-256 is
`8ba6f5f9d8cb2090d854d743014310004d15a6046b9778a01b4db51a5bc0f483`;
the copied adjacent AWS log SHA-256 is
`44e2eff8f16eb6e4679f89eb305f0f35e1e98c5eea366451e2aff382a5ba7f38`.
An independent AWS audit loaded the saved pipeline, rehashed all 540 unique
CASF tensors, reproduced every prediction and full-precision metric, confirmed
identical predictions for all 119 shared CASF IDs, and found zero train/test
overlap. No installed API, TopoKit mathematical source, feature tensor,
manifest, prior feature store, or completed model artifact changed.

## 2026-09-13 — Completed corrected version-matched refined/core GBDT protocol

Request: train three separate scikit-learn GBDT models from the PDBbind 2007,
2013, and 2016 refined sets minus their respective CASF cores, using learning
rate 0.002, `min_samples_split=5`, `subsample=0.8`, and the previously selected
10,000-stage, depth-seven, square-root-feature configuration; evaluate each
model only on its matching core and report the final results after validation.

Changes: preparation receipt `prepare_refined_intersections.py` version 1.2.0
now keeps membership authority and target authority explicit. The three
hash-pinned PDBbind release sources define refined membership. The separately
hash-pinned, preserved Weilab
`labels/reference_indices/topoformer/Benchmarks_labels.zip` supplies exact NMI
training targets and training-row order only after the receipt proves that each
training member set equals the corresponding release-defined refined-minus-core
set. Core labels and IDs are audited by PDB ID, and the local CASF test order is
retained; all 675 core target mappings pass.
Its local SHA-256 is
`e44d611e36589b5c01397449eee7083cb146af967f4f30d2b03cc20ae6e31b8b`;
the original TopoFormer URL is unavailable and no upstream byte checksum is
known. Trainer `train_refined_sklearn_gbdt.py` version 1.0.1 rejects mismatched
train/core years and gives label-policy diagnostics only for relations actually
present. Production uses `--label-policy error`. Updated the root, architecture,
validation, workflow, v2 receipt, and dataset documentation; no generated
tensor was changed.

Scientific/API impact: full refined-minus-core memberships are
1,105/2,764/3,772, while the validated compact-six feature intersections are
1,097/2,749/3,759. Their matching feature-complete cores contain 195/195/285
rows, and all three matching overlaps are zero. CASF-2016 means its 285-complex
benchmark core, not the distinct 290-complex PDBbind-v2016 core. Every selected
target relation is exact, the saved NMI training-row order is retained, the
scaler is fit on training inputs only, and targets remain unscaled. Each
independent GBDT uses learning rate 0.002, 10,000 stages, maximum depth 7,
`max_features="sqrt"`, `min_samples_split=5`, `subsample=0.8`, and random state
zero. This repository application does not alter installed TopoKit APIs,
Hyperdigraph mathematics, or feature tensors and is neither the NMI ensemble
nor an exact reproduction of the paper's full feature stack.

Validation: the preparer and trainer SHA-256 values are
`f5b319d603560f9ff487e9d8293f1e9e2aac1f9b9b236f8e891b30356a7e8495`
and `bf92e2f663bd4bbb33386053debf1acb1ba940f4ae87dce9d8d88728cd71369b`.
The preparation-provenance SHA-256 is
`7348702ba6b2f0b35a532f4391c37280d03da8057f9f018d715618ef5f69cd73`.
The complete local suite reports **1,271 passes, ten skips, and two expected
construction-truncation warnings**; the focused local preparer/trainer slice
reports **27 passes**. The AWS-focused trainer suite reports **23 passes** plus
18 warnings from external joblib/NumPy deprecation paths.
Fresh wheel and sdist checks pass; the wheel contains the six installed layers
without datasets, examples, or repository receipts, while the sdist contains
the required receipts/documentation and excludes generated outputs.

Production completed and the copied artifact is
`datasets/protein_ligand_prediction/models/sklearn-gbdt-standardized-compact6-refined-pairs-nmi-20260913-v2/`.
CASF-2007/2013/2016 `(RMSE, MAE, PCC)` results are
`(1.4767356443471618, 1.153354580527072, 0.8062946512163276)`,
`(1.5179520255834025, 1.2686667834755287, 0.7745888010556891)`, and
`(1.3234052487088577, 1.0583577486640605, 0.828697281199449)` for
1,097/2,749/3,759 training rows and 195/195/285 matching test rows. The
independent audit passed, reproduced the artifacts/predictions/metrics, and has
SHA-256
`3146ecda37732f776b90baa10cbf47ac36fcb82389b6eebb318417e21f2e8aa4`.
The remaining provenance limitation is that the preserved label bundle cannot
be independently byte-matched to its unavailable original endpoint.

## 2026-09-13 — Recovered 28 historical refined-set structure/feature pairs

Request: search the external
`/Volumes/my-harddisk/HPC_guowei2_04262025/TopTransformer/Datasets_all/over_structures_set`
collection for the structures missing from the PDBbind 2007, 2013, and 2016
refined/core training intersections, repair the local dataset where provenance
is sufficient, and retain the storage-pruned full-protein PDB/ligand MOL2
layout.

Changes: copied exactly 28 previously absent full-protein PDB/ligand MOL2
pairs, 56 files and 15,447,020 bytes, into the canonical structure store
without overwriting any existing input. The store now has 19,066 pairs and
still has no pocket PDB or ligand SDF. Added the 28-row additive metadata
manifest
`datasets/protein_ligand_prediction/labels/nmi_topotransformer_recovered_structures_20260913.csv`
and its detailed `.provenance.json` sidecar with source evidence and all file
hashes. Preparation receipt `prepare_refined_intersections.py` version 1.3.0
adds repeatable, additive `--supplemental-metadata-manifest` inputs, rejects
attempts to override built-in metadata, and pins individual and aggregate
supplemental provenance. Updated the root, workflow, architecture, validation,
and dataset documentation. The historical incomplete manifests and completed
model artifacts were not overwritten.
The version-1.3.0 receipt SHA-256 is
`5ffefbbf9f0f78b36ad0e4be6f0b978505fc0306792a3330c1501bec648042a5`.

Scientific/API impact: the original TopTransformer `run_hpcc_job.py` explicitly
uses `over_structures_set/<id>/<id>_protein.pdb`,
`over_structures_set/<id>/<id>_ligand.mol2`, and a 20-angstrom field. All 28
pairs pass the current input selector. All eight v2007 pairs are byte-identical
to the pinned release. All v2016 pairs are equivalent in consumed elements and
coordinates; proteins and eight ligands are byte-identical, while five ligand
files differ only in unconsumed formatting. The v2013 files retain historical
NMI/TopoTransformer working-snapshot provenance because no immutable v2013
structure archive was available for an independent byte comparison. This is
an application-data repair only: the compact-six schema, Hyperdigraph
definition, Laplacian, installed APIs, and training method are unchanged.

Validation/status: source/destination byte comparison and all 56 recorded
SHA-256 checks pass. The active structure inventory reports 19,066 PDB/MOL2
pairs and zero optional pocket/SDF files. The append-only eight-worker AWS run
completed 28/28 compact-six tensors with zero failures, skips, or unattempted
samples. All 28 are finite, nonnegative, little-endian float32, C-contiguous
`(6, 100, 40)` arrays with global range 0--2001. Checksum-verified copy-back
leaves 19,066 local tensor/record pairs,
zero failures, and sorted feature-ID SHA-256
`55c48f80d4e24bb72748274ba18fce7ef77adae659e2573026958df8583e5e13`.
The recipe inventory is 19,066 NPYs (1,832,776,448 bytes), 19,066 success
records (73,335,534 bytes), five run records (8,293,442 bytes), one schema,
and 38,138 regular files totaling 1,914,422,174 bytes.
The run record and copied log hashes are
`0a4b6407bde5da720aa1a0827c771a247b599b2988219d33bb4bf8102a8361a5`
and `f9fe9b20ff71a6ef0dc342042a531ea15e250e4f4114075e823cb4b422b412cd`.
The final recovery-provenance SHA-256 is
`062cd5caaa8059feee40db6f69b1b822a26b180585ad1b33088ddc96f873ac56`.

Version 1.3.0 published complete 1,105/2,764/3,772-row manifests under
`labels/refined_nmi_complete_toptransformer_20260913/`, with zero omissions;
a repeat invocation reports all outputs unchanged. The preparation provenance
hash is `6fe3e9774701987ea7872e364d10da09cba93f51e0349acd9a9622dec42a3927`.
Final local validation reports 1,275 passes, ten skips, and two expected
warnings; the v2 suite reports 100 passes and the focused supplemental slice
reports 19. A no-network wheel build passes with SHA-256
`a0b86aa3e81c1c863728d01ecbce7ea36380a9b75738495e5404926ed491cd8b`
and excludes workflows, datasets, and examples. `python -m build` was not
available because the active interpreter lacks the PyPA frontend and the local
`build/` directory shadows that module name, so no new sdist result is claimed.
Existing 1,097/2,749/3,759-row models and metrics remain frozen; no retraining
was performed.

Unresolved concern: v2013 source authenticity is bounded by the preserved
working snapshot rather than an immutable release-byte comparison.

## 2026-09-14 — complete refined-2016 feature ablation preparation

Request: compare the v2 feature baseline with six declared modifications,
then select useful pairs/triples using complete PDBbind-2016 refined training
membership and the CASF-2016 core, retaining refined-v2 ML settings and
supporting parallel execution on the existing AWS host.

Changed application files: new
`workflows/protein_ligand_prediction_v2_ablation/{feature_strategies.py,generate_features.py,train_ablation.py,run_study.py}`,
focused tests, README, and explicit AWS source-sync, durable-launch, status,
and result-sync shell helpers. `pyproject.toml` now includes ablation tests in
default discovery, and `MANIFEST.in` includes workflow `*.sh` in source
distributions. Updated root/workflow README, architecture and affected 0.3
contracts, and current validation. No installed mathematical core, reader,
builder, or generic workflow API changes.

Scientific contract: m1 adds selected-channel Delaunay cross support; m2 adds
linear positive-spectrum quartiles and unnormalized Shannon spectral entropy;
m3 uses 50 scales from 0.0 through 9.8; corrected m4 adds `all` groups with
Delaunay cross support; corrected m5 adds `null` groups with Delaunay cross
support and selected-component internal Delaunay null channels; m6 admits
cross and internal edges on selected-channel Delaunay support. m4/m5/m6 each
subsume m1, yielding 30 distinct canonical baseline/single/pair/triple recipes.
`all` includes only the original supported elements; null-null remains an
explicit zero slot. Coordinates, raw structures, and earlier artifacts are
unchanged. A superseded pilot with the earlier m4/m5 interpretation is not
scientific validation of this corrected protocol.

Selection first evaluates the baseline and six singles on the fixed
3,017/755 training-only holdout, then evaluates all unique pairs/triples of
the four best singleton modifications. CASF does not select combinations.
Final fits use all 3,772 training rows and 285 core rows with the refined-v2
train-only scaler and 10,000-stage GBDT. Completed selection/report aggregates
are now verified against their referenced model artifacts during resume.
Explicit smoke train/core limits default to 40/10; the quicker 8/4 smoke uses
five trees, and non-smoke runs reject these reductions.

Validation: 296 corrected ablation/v2-generator/architecture tests pass in
4.46 seconds. All 4,057 frozen baseline/input identities pass exhaustive
hash verification. The full geometry-only scan reports no parse failures or
exact coordinate duplicates. Corrected `1a30`/`4ej8` calibration succeeds for
all seven recipes and reproduces both baseline NPY files byte-for-byte. The
corrected local 8/4 complete-flow smoke completed 14 strategies, 28 model
fits, both reports, and all 12 complexes without feature errors. A fresh
controller run and actual resume succeed after the regression that mixed
already-existing core runs into `validation_report` was fixed. Each model fit
command occurs once, both reports recompute identically, and all 12 baseline
NPY files match the frozen v2 bytes. The regression test enforces report-stage
separation; `local_smoke_audit.json` records the completed audit. No smoke
score is production evidence. Shell syntax/help/dry runs, production rejection
of smoke limits, and source-manifest inclusion of all five helpers pass.
Current evidence and limitations are tracked in `markdown/VALIDATION.md`.

AWS preparation uses an isolated source snapshot with a strict payload
allowlist and exactly two 2016 label manifests; no raw structures or unrelated
workflows are transferred. Sync performs no deletion, and copy-back preserves
changed local files in backup history while excluding transient staging and
lock files. Automatic approval review blocked the attempted SCP, and explicit
upload authorization is pending. No remote snapshot or production job was
created by that attempt. Production performance and a best strategy remain
unresolved until the authorized corrected study completes.

## 2026-09-14 — Authorized AWS ablation production launch

The user explicitly approved uploading the prepared code/two 2016 label
manifests to aws-cpu and running the ablation. Transfer verified all 194
snapshot files and all 4,057 input pairs. The AWS 8/4-row smoke completed
14 strategies and 28 verified model artifacts, zero feature failures, an
identical recomputed report, and 12 byte-identical frozen-v2 baseline tensors.
Production launched at 03:01:18 UTC, initial PID 72929, with six feature/model
workers. No scientific code, geometry, labels, split, or estimator settings
changed during launch. The remote source snapshot remains immutable.

README and current VALIDATION now reflect active production; dataset
study_plan.json and aws_smoke_audit.json contain execution and audit evidence.
An hourly heartbeat will retrieve and verify completed results and report the
comparison. The first feature-pass estimate is 18.9 hours from twelve smoke
complexes, excluding model fits and selected combinations. Full production
metrics and a winning strategy remain pending.


## 2026-09-14 — Resume ablation on the user-upgraded AWS instance

- Request: use the user-supplied `ubuntu@3.131.153.182` address and increase
  parallelism after the AWS CPU upgrade.
- Changes: local ablation `aws_common.sh` defaults use the explicit existing
  SSH key, 56 feature workers, and 12 model workers. Updated root/workflow
  README, validation notes, study runtime evidence, and the existing heartbeat.
- Scientific/API impact: no feature, core, label, split, or estimator changes;
  no edits to the immutable remote source. Reused the 1,792 preserved complexes.
- Validation: 194 remote source hashes match; old process absent and study lock
  free before resume; shell syntax and launch/sync dry runs pass. Live resume
  verification is in `aws_upgrade_20260914.json` under the dataset study folder.
- Open work: full feature generation, model fits, selected combinations, and
  verified performance reporting continue; no production winner claimed.


## 2026-09-15 — Complete and verify the CASF-2016 feature ablation study

- Completed all six individual modifications and the six pairs/four triples
  selected from the four best training-holdout singles: 17 feature strategies,
  68,969 tensors, 17 validation models, and 17 full-training CASF models.
- Retrieved the complete artifacts and independently verified structure and
  tensor hashes, source/manifest identities, all model hashes, exact splits,
  training-only scaler statistics, unchanged 10,000-tree settings, reproduced
  predictions/metrics, and both reports. Read-only audit scripts and receipts
  are saved with the dataset; no frozen scientific source was edited.
- Results: m2 has lowest observed CASF RMSE 1.293307 (baseline 1.326745);
  m2_m5 is the validation choice and highest CASF PCC at 0.838336.
- Updated root/workflow README, validation record, study status, and full
  comparison report. Functional smoke and superseded pilot are excluded.
- Limits: single seed and ungrouped holdout; repeated CASF rankings are
  exploratory. One float32 baseline value in 4cu8 differs by one representable
  step; all other 4,056 tensors are byte-identical to frozen v2.

## 2026-09-15 — Seven follow-up revisions of m2+m3+m5

- Request: test r1/r2/r3, all three pairs, and the triple on the 2016 set,
  using parallel AWS execution and frozen v2 learning settings.
- Added application-only protein_ligand_prediction_v2_revisions with feature
  engine, resumable generator/controller, transactional trainer, AWS helpers,
  independent completion audit, tests, and protocol README.
- Scientific impact: r1 removes Delaunay only for cross/ligand-only channels;
  r2 drops the ligand H category while preserving crop anchors; r3 uses eight
  raw positive-spectrum summaries plus nullity. Native cores are unchanged.
- Updated root README and current validation. Local contract/regression/
  architecture validation: 380 passing tests. Fresh AWS smoke, full sample
  generation, sixteen production fits, and final artifact verification remain
  pending; the completed original study is preserved.

### Follow-up launch validation

- Corrected the independent verifier's undefined PCC handling for constant
  smoke predictions and included the historical v1 generator test fixture
  in the isolated upload. Added its regression test.
- 381 local and AWS tests pass. The corrected smoke-verified run passes
  96 tensors, 16 five-tree fits, reference equality and full artifact audit.
- Production launched at 14:37:54 UTC on September 15, PID 10990, from
  revisions2016_20260915_verified: 56 feature workers and eight model workers.
  The existing task continuation now follows this new experiment hourly.
  Local study plan/status are recorded; performance results remain pending.

## 2026-09-15 — Complete and verify all m2+m3+m5 revisions

- All seven requested revision subsets plus reference finished on AWS at
  15:41:57 UTC: 32,456 tensors and 16 full-length model fits, with no missing
  complexes. Retrieved complete artifacts into the separate local namespace.
- AWS and independent local audits pass all feature/input hashes, schema and
  split identities, training-only scalers, unchanged 10,000-tree parameters,
  reproduced predictions/metrics, and report consistency. A fresh remote
  checksum inventory verifies downloaded model/report files. All 4,057 new
  reference tensors are byte-identical to prior m2+m3+m5; scores match exactly.
- Added local verification/report-generation receipts and a complete comparison
  report. r1+r3 has lowest observed CASF RMSE 1.277164 (1.69% below reference
  1.299098); r1 has highest PCC 0.840047; training-validation selects r1+r2+r3.
- Updated README, workflow protocol/status, and validation evidence. Original
  completed experiments and frozen AWS sources remain unchanged. No installed
  mathematical/API changes. Repeated CASF rankings remain exploratory.

## 2026-09-15 — Alpha and raw-extrema follow-up implementation

- Request: run r4, r5, r1+r4, r1+r5 and r1+r4+r5 on AWS using the same
  m2+m3+m5 reference, complete 2016 membership and frozen v2 GBDT settings.
- Added the application-only protein_ligand_prediction_v2_alpha_ablation
  engine, generator, trainer, controller, independent calibration/audit,
  AWS helpers, tests and protocol. Root README and validation record updated.
- r4 uses public alpha edge births with radius=s/2; r1 overrides cross and
  ligand-only geometry while retaining alpha protein-only channels.
  r5 replaces only normalized min/max with raw positive min/max.
- No installed mathematical changes. Tests cover non-Gabriel coface births,
  unchanged reference/r1 features, channel precedence, statistics and budgets.
  Calibration and production results remain pending in a new source/output
  namespace; previous experiments are preserved.

## 2026-09-15 — Verify alpha geometry and launch five AWS comparisons

- Local regression passed 472 tests with one optional GUDHI skip; AWS passed
  all 473 tests. Independent GUDHI 3.13.0 exact-alpha checks agreed on all
  edges and births across 16 complete molecular channels from four complexes.
- The 12-complex pilot passed 84 tensors and 14 five-tree model artifacts,
  exact reproduction of both prior controls, raw-extrema row invariance,
  scaler/prediction/metric checks and reports. No installed algorithm changes.
- Production launched at 16:58:13 UTC, PID 13589, using the immutable
  220-file alpha2016_20260915 source, 56 feature workers and seven model
  workers. Expected output: 28,399 tensors and 14 models on all 4,057 complexes.
- Saved study/preflight/launch receipts and a local artifact verifier in the
  separate v2-alpha-2016 namespace. Reactivated the existing hourly follow-up
  for final verification and reporting. Performance results remain pending.

## 2026-09-15 — Use all 64 AWS vCPUs for alpha feature generation

- User requested fuller use of the available computation resources. Measured
  56 active workers at 87.5% CPU utilization, with ample free RAM.
- Changed only the local alpha AWS helper's worker default from 56 to 64.
  Stopped the verified old process tree and resumed the same frozen source
  and configuration at 17:32:37 UTC, PID 14409. No estimator, feature or
  geometry definitions changed. Shell syntax and launch dry run pass.
- Verified preservation of all 1,779 completed complexes and 24,906 NPY/record
  hashes. The resumed generator independently verified all existing complexes
  before reuse. All 64 new workers are active with one numerical thread;
  CPU utilization is 99.9%, with 7.66 GiB feature RSS.
- Updated protocol/status and the existing follow-up. Eight deterministic
  native-alpha numerical failures remain; they are not considered complete
  and still require a validated construction repair before fitting.

## 2026-09-15 — Explicit exact-alpha construction for degenerate molecular clouds

- The approved study's native attempt ended with 4,041 successes and 16
  singular-simplex errors, correctly preventing fitting. Exact GUDHI geometry
  succeeds on the original diagnostic clouds without changing coordinates.
- Added a lazy, opt-in gudhi_exact adapter in builders/_alpha_gudhi.py and
  dispatch through the public simplicial builder; native remains the default.
  Added the optional alpha_exact dependency and backend/version provenance.
  Full cofaces are processed before truncation; the full simplex cap is
  checked after external construction. No point projection, jitter or fallback.
- The alpha application selects exact GUDHI 3.13.0 for every alpha channel,
  pins new recipe identities, adds a full recovery-probe verifier and maps
  helpers to the new alpha2016_20260915_exact source and exact output folders.
  Original source/output and both earlier completed studies are preserved.
- Updated README, architecture/contracts, dependency notice and validation.
  All 485 local tests pass, including an independent constrained-ball oracle,
  analytical coface/near-coplanar geometry, lazy import and native isolation.
  AWS recovery/pilot evidence and performance results remain pending.

## 2026-09-15 — Recover all singular cases and launch exact-alpha production

- All 485 regression tests pass on AWS. Every formerly failed complex passes
  the exact-backend recovery probe, together with four large controls: 140
  tensors, original inputs and both prior controls verified exactly.
- The exact pilot passes 84 tensors and 14 five-tree fits with its full audit.
  No categories, atoms, scales, summaries, labels or estimator settings changed.
- Froze 223 source files in alpha2016_20260915_exact and launched separate
  production-exact outputs at 18:49:45 UTC, PID 15975, with 64 workers. Initial
  verification shows 99.94% CPU use, 7.86 GiB feature RSS and zero failures.
- Archived the preceding mutable study plan, saved recovery/preflight/runtime
  receipts, adapted local artifact/report verification to the exact output,
  and updated documentation and the existing follow-up. Native-attempt
  artifacts and both earlier completed studies remain preserved. Final
  production verification and performance reporting are still pending.

## 2026-09-15 — Complete and verify the alpha/raw-extrema comparison

- The approved exact-alpha production completed at 20:33:39 UTC. All 4,057
  complexes succeeded across seven strategies: 28,399 tensors and 14 models.
  Feature generation used 64 workers; each training stage used seven parallel
  10,000-tree fits under the unchanged v2 protocol and fixed 2016 membership.
- The AWS audit passes all feature/source/input checks, saved-model reloads,
  training-only scalers, predictions, metrics, split membership and reports.
  The independent local audit passes all 28,399 tensors and 8,114 structure
  hashes. All 77 local model/report artifacts match fresh AWS SHA-256 hashes.
  Both controls reproduce all 4,057 tensors and validation/CASF scores exactly;
  all three r5 pairs preserve the other eight summary rows.
- Added the final dataset-local comparison report and machine-readable results,
  including all five requested variants, both controls and six matched effects.
  Lowest observed CASF RMSE: r1+r4+r5, 1.270773 (2.18% below reference).
  Validation selects r1+r5, CASF RMSE 1.271012; the triple's 0.000239 test
  advantage does not establish significance. r4 has highest PCC, 0.842641.
- Updated root/workflow README and current validation. Frozen numerical source,
  estimator settings, prior completed studies and native-attempt records are
  preserved. Results remain exploratory single-seed comparisons on a repeatedly
  examined CASF test set; no publication or new experiment is implied.

## 2026-09-15 — Incremental ordinary L0 matrices across a filtration

- Request: accelerate matrix construction across filtration scales while
  preserving correctness and higher-dimensional calculations.
- Added the private core L0 accumulator and public `core.hyperdigraph.L0Sweep`
  with a matrix-only API, guarded singleton-face birth closure, inclusive
  nondecreasing queries, stable IDs, reciprocal-edge multiplicity, independent
  returned matrices and explicit resource limits. Each edge is inserted once.
- The generic spectral workflow selects incremental ordinary hyperdigraph L0
  when eligible; general missing-face cases, higher dimensions, reference
  requests and genuine two-scale persistent operators keep their algorithms.
  The alpha application 1.2.0 calls this public API and records new source/recipe
  IDs; exact geometry, isolated-zero handling, summaries and ML remain unchanged.
- Added independent incidence, state/budget, matrix-only, mixed-degree and
  persistent-path tests, plus cold-import and installed-wheel API checks.
  Full package/application regression: 1,596 passed, 10 skipped, two existing
  truncation warnings. A 200-matrix synthetic benchmark is bitwise identical,
  with 30.5-39.6x median construction-only speedup; no end-to-end claim is made.
- All 28 molecular tensors across four representative complexes and seven
  strategies match the frozen 1.1.0 engine exactly in float64 and float32.
  The built wheel passes isolated installed-package checks for the new API,
  all three topology routes, higher dimensions and persistent queries.
- Added reproducible benchmark and molecular-equivalence scripts under examples,
  and updated README, architecture, contracts, validation and application notes.
  Existing completed studies and their frozen remote source remain preserved.
  The dense sweep buffer and exported snapshots retain their documented memory
  costs; no alpha geometry acceleration or eigenvalue approximation is included.

## 2026-09-15 — Alpha follow-up with r6/r7/r8

- Request: compare r6, r7, r6+r7, r7+r8, r6+r7+r8 and r6+r7+r8+r5 on
  m2+m3+m5+r4, using only 2016 and parallel AWS execution.
- Added protein_ligand_prediction_v2_alpha_followup with the six variants
  and an unchanged r4 control. It composes the existing exact-alpha workspace
  and public incremental L0 API; no core or higher-dimensional code changes.
  Each strategy retains shape (10,50,55), 27,500 features.
- r6 applies <=15 Å before channel geometry. r7 observes 2.0,...,11.8 Å
  with radius=grid/2, retaining pre-grid births. r8 includes internal edges
  from the selected-channel alpha complex; r5 changes only the raw extrema.
- Added independent scientific, artifact and fixed-protocol tests, AWS
  snapshot/launch/status/sync helpers, molecular preflight and an expanded
  final audit. Local regression: 1,619 passed, 10 skipped, two existing
  warnings. Prior completed studies/source remain preserved. AWS preflight,
  smoke and production results are pending in distinct output directories.
- AWS regression passed 530 tests. Preflight passed all 133 tensors over
  19 complexes; smoke passed 84 tensors and 14 five-tree fits with the full
  audit. Downloaded pilot tensors and model artifact hashes verify locally.
- Froze 239 source files in alpha_followup2016_20260915 and launched production
  at 22:51:19 UTC, PID 19889. All 64 workers are active at 99.92% host CPU
  capacity; the initial snapshot has 23 successful complexes and zero failures.
  Added dataset-local audit/comparison scripts, plan/status and preparation
  receipts. Reactivated the existing follow-up for this study. Final results
  remain pending; previous study outputs and remote sources are preserved.

## 2026-09-16 — Complete the alpha crop/grid/internal-edge follow-up

- Completed the six requested additions on m2+m3+m5+r4 plus its unchanged
  control: 4,057 complexes, 28,399 tensors, 14 production models and zero
  failures. AWS finished its final audit at 05:46:42 UTC. Production used
  64 feature workers and seven parallel fits per stage under the frozen v2
  estimator settings and 2016 split; no numerical source changed after freeze.
- Both AWS and independent local audits pass. Local checks verify all tensors,
  8,114 input structure hashes, seven recipe identities, 14 model artifact
  sets and reports. All 77 model/report hashes match a fresh AWS inventory.
  The AWS audit independently checks 8,114 crop reads, reloads all models
  and verifies scalers and predictions. All 4,057 control tensors and its
  validation/CASF scores reproduce exactly; matched feature slices verify.
- None of the six new variants improves CASF RMSE over the unchanged r4
  control (1.291531, PCC 0.842641). The best new CASF variant is r7 (1.300913).
  Training-only validation selects r6+r7 (validation RMSE 1.272556,
  CASF RMSE 1.305975). Results remain exploratory single-seed comparisons.
- Added the final dataset-local report with all scores and eight matched
  comparisons, machine-readable results and audit receipts. Updated root and
  workflow README, validation and run status. Prior studies, frozen AWS
  snapshots, installed mathematical APIs and higher-dimensional code remain
  unchanged; this completion changes reports and documentation only.
- Paused the existing hourly completion monitor after both audits, artifact
  retrieval, comparison reporting and the completion receipt were finished.


## 2026-09-16 — fix PCC-leading alpha strategy and consolidate prediction workflow

Request: fix m2+m3+m5+r4, archive previous protein–ligand studies, reclaim
regenerable feature storage, and run complete refined 2007/2013/2016 plus the
updated official v2020R1 general benchmark on AWS with local artifact copies.

Changed application: one active `workflows/protein_ligand_prediction/` with
fixed strategy, explicit manifest preparation, verified parallel generation,
four fixed GBDT fits/six evaluations, audits and AWS helpers. Previous seven
workflow directories and old dataset results/models/labels are archived;
source hashes and historical scientific records remain intact. README,
workflow index, dataset README, architecture, validation, test discovery and
source-distribution checks now identify only the final application. Archives
are excluded from the active distribution inventory.

Scientific/API impact: fixed 20 Å crop, 55 null/element channels, exact alpha
squared births <=(s/2)^2 at 50 scales, ten original+m2 summaries, ordinary
unweighted native L0, 27,500 float32 features. No installed mathematical API
or source changes; no higher-dimensional algorithm changed. v2020R1 means
19,037 official updated entries minus 539 CASF IDs = 18,498 training rows,
with the existing audited NMI point-label targets retained. Complete refined
training counts are 1,105/2,764/3,772. All four final fits use the refined-v2
10,000-tree/.002 configuration, training-only scaling and no early stopping.

Validation: 1,114 local tests pass, one skip, two existing explicit truncation
warnings. Independent alpha/incidence and encoding tests pass; one complete
molecular tensor reproduces frozen r4 bitwise. AWS preflight/full audits,
distribution checks and new production completion are tracked in VALIDATION.md
and dataset run receipts; no pending production result is claimed here.

Limitations: CASF-2016 contributed to prior exploratory feature selection;
new scores are not independent selection validation. Historical membership
uses updated/recovered structure bytes, not original historical media. Large
geometry/dense-spectrum resource failures remain explicit and block fitting.

AWS follow-through checkpoint: 89 molecular preflight features and four
smoke models passed, with 66 prior-r4 tensors bitwise equal. All 1,115 AWS
tests pass after correcting PYTHONPATH for one isolated subprocess invocation.
Fresh wheel/sdist inventories pass. Full generation is running with 64
workers. An exhaustive scan found 208 cross-component coincident-coordinate
cases solely in v2020R1 training; strict r4 records failures without omission.
The proposed atom-preserving extension is pending a user choice. Cleanup
reclaimed 24,263,138,336 local bytes and 24,242,889,472 AWS bytes from obsolete
NPY arrays, with per-file hashes, while retaining historical results and all
4,057 original r4 controls. Completion and new production scores remain pending.

## 2026-09-16 — strict generation audit and independent refined fits

The strict production pass ended with 18,858 successful/verified features and
exactly 208 known general-only coincident-coordinate failures. Added a
dataset-experiment audit receipt script, separate from the frozen workflow,
that validates every available tensor/input hash, fixed manifest, source
identity and failure membership while explicitly marking the full run
incomplete. All 4,057 historical r4 tensors reproduce bitwise. All 347 frozen
source files match their recorded hashes; no geometry, installed API, training
setting, sample membership or core algorithm changed.

At 16:48 UTC the three complete refined benchmarks started under a separate
flock with three parallel model workers. Their training counts remain
1,105/2,764/3,772; the 18,498-row v2020R1 fit awaits the user’s duplicate policy
choice. Root/workflow/dataset README, validation and run receipts distinguish
the completed strict feature pass, partial full-run status and active refined
fits. All 18,858 available features and records were synchronized locally;
the independent local audit passes with identical tensor/input inventories,
source identity and 4,057 historical r4 reproductions. Models remain running.

## 2026-09-16 — refined benchmark completion and reproduction audits

All three complete refined models finished by 17:03 UTC and were copied to
the canonical local model/result folders. Added a separate experiment audit
for the three completed fits; the immutable production source remains intact.
Independent AWS checks reload all models, refit exact training-only scalers,
verify 10,000-tree settings and reproduce every saved prediction. Local
checks verify all artifacts, memberships and scores without cross-version
model loading. The fresh AWS inventory matches all 47 local files, including
29 model/result files. Canonical metrics remain unchanged; PCC recomputation
across NumPy versions differs at most 3.33e-16, recorded under 1e-12 tolerance.

Refined-to-matching-CASF RMSE is 1.486219/1.492785/1.291531 and PCC is
0.809891/0.788311/0.842641 for 2007/2013/2016. Refined-2016 exactly reproduces
the archived r4 model's full-precision predictions and RMSE/PCC/MAE. Its
historical CSV matches at the original 12-significant-digit precision.
Added RESULTS.md and REFINED_COMPLETION.json with explicit partial-study
scope, updated root/workflow/dataset README and validation, and retained all
historical controls. The general v2020R1 model still requires all 18,498 rows;
208 duplicate-coordinate features await the user's choice. No scientific
recipe, membership, installed API or higher-dimensional algorithm changed.


## 2026-09-17 — GUDHI-free native alpha construction and workflow migration

Request: implement alpha construction inside TopoKit so the selected protein–
ligand strategy no longer requires GUDHI, retaining efficiency and higher-degree
correctness. Added builders/_alpha_native.py; the native dispatcher now uses
batched circumspheres, empty-ball checks and array-based coface propagation.
Rational rank/sphere/predicate checks, local near-cospherical cavity repair,
consistent symbolic ties, bounded global repair, and explicit resource/geometry
failures preserve coordinates and duplicate policy. No mathematical core,
L0/L1/L2 definition, molecular crop, channel, scale, statistic or ML setting changed.

The active molecular workflow is 1.1.0 / schema 2 with native alpha 1.0.0.
GUDHI remains an explicit optional legacy/oracle backend. New recipe/source
identities and native output/snapshot roots prevent silently mixing artifacts.
Added a read-only molecular migration auditor; existing features, models and
scores retain their original provenance. Updated root/workflow/dataset READMEs,
architecture, contracts, final-strategy notation and NATIVE_ALPHA.md.

Local full regression: 1,134 passed, one skip and two existing truncation
warnings. Installed-wheel checks block GUDHI imports; real feature CLI generation
and verified resume also pass in a runtime without GUDHI. Added thin-geometry,
large local-cavity, symbolic-tie, bounded-failure, independent geometry and direct
L1/L2 matrix tests. Final molecular audit coverage and timings are recorded in
VALIDATION.md and experiments/native_alpha_validation_20260917. Cross-platform
near-zero summary roundoff is retained and separately reconciled, not hidden by
changing stored features. The native implementation does not claim a universal
exact-predicate guarantee; the 208 general-only duplicate-coordinate cases remain
subject to the unchanged strict policy.

## 2026-09-17 — User-authorized exact-coordinate keep-first input policy

Request: count duplicate-coordinate atoms once and update TopoKit input handling.
Changed data layer (`PointCloud.unique_coordinates`), both alpha builders and
the fixed molecular application, plus tests, contracts, README and validation
notes. Alpha defaults to merge; explicit error mode preserves strict replay.
The first input row supplies ID and attributes. Molecular ordering is cropped
protein then ligand, with global deduplication before element categories.
No rounding, jitter, coordinate averaging, edge lifting or sample exclusion.
Native engine 1.1.0 / workflow 1.2.0 / schema 3 have a new source/recipe identity
and `final_alpha_l0_unique` output roots. Historical artifacts stay unchanged.
Focused 92 tests and full local 1150 tests pass (one optional skip in full suite).
Higher-dimensional ordinary/persistent L0/L1/L2 equality is verified.
AWS production completion and full v2020R1 training remain pending execution;
exact-predicate universal equivalence is not claimed.

AWS validation for the same update: 1151 tests pass (240 subtests); the
independent 19066-complex selection audit confirms 208 affected samples and
399 exact repeated atom rows removed. Native unique-atom production started
with a 64-worker first pass over the affected samples and regression controls.
See experiments/final_alpha_l0_unique/IMPLEMENTATION_REPORT.md for receipts.

Follow-through: all 208 previously blocked samples now have features, locally
transferred and audited. The 297-sample/four-smoke-fit preflight and all 208
optional exact-alpha comparisons passed. Complete uniform native production
launched with 64 feature workers; full benchmark completion remains pending.

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

## 2026-09-17 — Native feature production and local transfer complete

All 19,066 fixed-strategy unique-atom features finished on AWS and were copied
locally. Independent local and AWS audits agree on all tensors and 38,132
input hashes; all 4,057 r4 controls match exactly. A separate inventory verifies
19,068 record/schema/compatibility files. Updated current README paths/status
and validation notes; no geometry, feature, model or frozen-source code changed.
Actual native-1.1.0/1.1.1 generation identities remain intact. Four production
fits are now running; six evaluations and final model/result audits remain.

## 2026-09-17 — Full native benchmark, transfer and final archive complete

Completed the 18,498-row updated v2020R1 model and all three CASF evaluations;
four full fits and six evaluations now pass independent AWS and local audits.
General CASF-2016 PCC is 0.8522775980674216 and RMSE 1.27297726094721.
All 19,066 features and 42 model/result artifacts are copied locally and
verified. Updated current package/workflow/dataset READMEs, validation and
native-alpha notes, complete results and machine-readable receipts. Moved
superseded production folders to a checksum-verified archive without deleting
files; one final feature/model/result store remains active. Frozen source,
scientific inputs, feature arithmetic, higher-dimensional code and ML settings
are unchanged. Earlier partial-completion receipts remain historical.

## 2026-09-17 — ML-only H-channel and raw-extrema comparisons

Added repository-level ml_feature_comparison.py, audit_ml_feature_comparison.py
and scientific view tests. Existing standardized baseline is reused; variants
drop five ligand-H channels (25,000 dimensions), reconstruct raw positive
min/max from stored ratios and means (27,500), or combine them (25,000).
Multiplication is float64 before float32 storage; exact pre-rounding spectra
are not claimed. MAD/mean, retained geometry, manifests, labels and fixed
GBDT settings remain unchanged. Twelve new AWS fits and18 evaluations are
planned, with pinned provenance, independent matrix/scaler/reload audits and
separate experiment outputs. Local focused regression:21 passed,1 skipped.
No installed mathematical or geometry code changes; no feature regeneration.

## 2026-09-17 — ML feature comparison, audits and transfer complete

Finished all 12 new AWS fits and 18 evaluations using existing feature tensors;
reused the already-standardized baseline for six more comparison rows. All
independent AWS scaler/model/prediction audits and local matrix/membership/
metric audits pass. All 78 model artifacts plus nine aggregate/provenance
files match checksums; all 388 frozen Python files remain unchanged. Added
separate RMSE/PCC result tables and full-precision heatmap matrices, final
transfer/completion receipts, and updated package/workflow/dataset READMEs,
protocol and validation. Raw min/max gives general-v2020R1 CASF-2016 RMSE
1.2623574096529802 and PCC 0.8556211592220472; refined-2016 favors baseline.
No single view wins every pair. Fixed strategy, source tensors and all
scientific/ML code remain unchanged; stored-float32 reconstruction and
single-seed/CASF-selection limitations are documented. No requested fits or
evaluations remain.

## 2026-09-18 — Cornell VR atom-deletion feature and model experiment

Request: run the agreed 12 Å, 40-channel, nine-cutoff, nine-statistic atom
leave-one-out recipe (12,960 dimensions) on refined/CASF 2007 and 2016 using
Cornell WSL, the activated topokit environment and separate workflow/data
folders, then compare the prior 10,000-tree protocol with 30,000 trees.

Added `workflows/protein_ligand_prediction_version2/` with exact feature
generation, bounded component/twin reuse, immutable provenance/manifests,
parallel resumable orchestration, explicit four-fit ML protocol, independent
molecular/model audits, launcher and documentation. Added optional public
`L0Sweep.vertex_deleted_laplacian` and its private degree-correction kernel;
applications do not duplicate induced-L0 algebra. Existing alpha, higher
degree and genuine persistent algorithms are unchanged. Updated the root and
baseline README, architecture, incremental-L0 and validation notes, and
default pytest collection. Existing structures, labels, features and results
are preserved; the new dataset references existing structures through a
read-only-by-contract symlink. New results remain remote as requested.

Validation: 1,217 remote tests and 240 subtests pass (23 optional skips);
133 local focused core/architecture/workflow tests pass. Tests use independent
induced-incidence and D−A oracles, missing-side/dedup/crop checks and independent
scaler/model validation. The 1a30 pilot improved from 156.57 s to 81.81 s with
exact deletion reuse; it is not a full-dataset speed estimate. The 39-complex
end-to-end smoke is active; production results are pending. Full completion
requires all 4,324 tensors, four full models and successful remote final audit.

Same-day execution update: the 39-complex smoke passed at22:16:39UTC, including
45 independent molecular slice checks and all four short fit/reload/scaler
audits. All4,324 input complexes passed preflight and all83 frozen Python
sources agree locally/remotely. Production launched at22:17:33UTC (PID4358),
with16 confirmed active feature workers and automatic subsequent four-model
training/audit. The successful smoke tensors are reused only after identity
verification. Full results remain pending; no Python source changed after
preparation. New features/models/results stay on Cornell as requested.

## 2026-09-20 — MSU HPCC 15 Å alpha sampling experiment

Request: reduce the finalized protein crop from 20 to 15 Å and sample alpha
radii 0.1–5.0 Å by 0.1, using MSU HPCC for feature generation and ML. Added
hpcc_15a application strategy, Slurm pipeline/submission helpers, tests and
notes; registered its tests in pyproject and updated current READMEs and
validation. Public native-alpha and L0 APIs remain unchanged. The separate
recipe retains 55 channels, normalized extrema and 27,500 features. It reuses
all audited manifests and the unchanged four-model baseline trainer. Local
regression and four molecular pilot checks pass. HPCC power issues currently
block compute execution; installation, full features and ML are not complete.
The baseline/AWS and Cornell experiments are preserved independently.
The follow-up parallelism request uses 256 feature-array tasks with four
workers each (up to 1,024 CPUs), within the verified MSU limits of 1,040 CPUs,
520 running jobs and 1,000 queued plus running jobs per user. An audited Slurm
throttle update changes scheduling without modifying the frozen source.

## 2026-09-20 — Barcode bin postprocessing for the proposed Cornell study

Request: generate actual protein–ligand persistence barcodes before a
0–5 / 0.05 bin representation, preserving isolated atoms and allowing
within-component alpha edges in explicit Null-side channels. Added public
`postprocessing.barcode_bin_counts` with explicit overlap, full-bin coverage
and finite-death modes. It preserves original/infinite endpoints, excludes
zero-length intervals, validates observation bounds, and returns the existing
serializable VectorResult with a complete bin schema. Endpoint search and
range additions avoid a bars-by-bins matrix. No topology or geometry kernel
was copied into application code; core and higher-dimensional algorithms
are unchanged.

Updated exports, architecture enforcement, root README, notation, architecture
and current validation; added independent endpoint/property/integration tests
and the draft `workflows/protein_ligand_persistent_homology/README.md`.
Focused validation: **215 passed**. The Cornell preflight verifies all inputs
for 4,414 unique complexes. Scientific choices (homology degree, counting mode,
alpha-radius versus distance-cutoff interpretation) await the user, so this
entry records preparation only, with no new production features, fits or
performance claims. Existing experiment artifacts and remote code are intact.

## 2026-09-20 — Fixed alpha H0 full-bin molecular workflow on Cornell

The user resolved the recipe: H0 only, full-bin barcode coverage and filtration
maximum 5. Added the separate `protein_ligand_persistent_homology` application:
15 Å H-inclusive ligand-centered crop, exact keep-first deduplication, 55
mixed/Null channels, native alpha radius 0–5 Å and 100 full-bin counts per
channel. The squared builder bound is 25 Å²; no additional distance cap or
factor-of-two conversion is used. Original infinite-death barcodes are saved
beside the 5,500-value tensors. The application calls public geometry, H0 and
postprocessing APIs, with no new mathematical core implementation.

Added source/manifests/input/artifact identity checks, parallel resumable
orchestration, 10,000/30,000-tree complete refined 2007/2013/2016 fits, independent
barcode/real-graph/model audits, launcher and tests. Registered the tests and
updated README, architecture, validation and the fixed workflow recipe.
Local focused checks: 50 passed; remote checks: 274 passed. Cornell contains
all 4,414 unique inputs. A local 1a30 pilot completed in 0.476 s. The remote
smoke is active; full production and performance remain pending. Existing
baseline, atom-deletion and HPCC artifacts remain separate.

Same-day execution update: the 48-complex Cornell smoke passed at 18:49:27 UTC,
including 2,100 independent full-alpha graph/bin checks and all six short
model/scaler/reload audits. All 84 Python source hashes match locally and
remotely. Production launched at 18:50:43 UTC (PID 9243), with 16 feature
workers and six automatic subsequent full-manifest fits. Scientific Python
source is frozen; existing verified smoke features are checksum-validated
before reuse. Full scores remain pending.

Execution update at 19:00 UTC: all 4,414 feature/barcode pairs are complete,
with no failed samples. All input/artifact hashes and saved-barcode feature
predicates pass, as do all 2,100 independent real-bin checks. The controller
started all six full model fits at 19:00:14 UTC. The user's previously
authorized 30-minute read-only monitor now targets this separate H0 study,
with six-model completion gates and automatic pause after final reporting.
No production metrics are claimed before final model audits.

## 2026-09-21 — supervised DL workspace and reusable alpha15 GBDT bundles

Request: prepare the protein–ligand supervised TopoFormer folder structure,
record future hyperparameter ablations under datasets, and deliver Cornell
features/trained GBDT models with fitted normalization for direct use.

Added the reserved optional workflow namespace (no DL model/trainer API yet),
repository `dl/` contract/template, 18 planned one-factor configurations and
an explicit 2,844/711 selection split from refined-2016 after excluding all
CASF IDs. Pretraining is disabled; no experiment was launched. Added repository
`ml/` inference/export helpers, recorded runtime files, tests and source-archive
patterns. No mathematical, geometric, higher-dimensional or feature algorithms
changed. No PyTorch dependency was installed or added to the base package.

Cornell checksum synchronization matched all existing local scientific files.
Exported four standalone original GBDT pipelines, each with its embedded fitted
StandardScaler, plus separate scaler arrays and hashes/manifests. All 1,350
predictions match bitwise on Cornell through both scaler-array reconstruction
and the direct-use helper. A first export stopped at a strict rounding check;
the final exporter explicitly follows sklearn 1.9's float32 statistics casting.
Only the successful export is delivered.

Validation: 31 inference/architecture tests passed; independent local audit of
19,066 feature/record pairs, 26 original model artifacts, 326 frozen source files
and recomputed metrics passed. Local checks validate all 49 delivered bundle
files, four scalers, 18 configs, split membership and unchanged training counts.
Models were not unpickled under incompatible local dependencies. Remaining:
implement/test the compact DL model/trainer before future ablations; a fresh
local installation of the recorded GBDT runtime has not been exercised.

## 2026-09-21 — supervised TopoFormer plan revision 2

User fixed positional encoding/pooling to the original TopoFormer conventions,
batch size to 32 and dropout to 0.1, and requested a larger baseline. The
repository DL template and dataset experiment plan now contain **12** active
configurations over width, depth, attention heads, feed-forward multiplier,
learning rate and weight decay. Positions are frozen upstream 2D sine/cosine
on the 50×1 patch grid (width-first axis concatenation, zero CLS position);
CLS passes through a d-to-d tanh pooler and scalar projection. The baseline
is width256 / four blocks / eight heads / FF1024, about 3.37M trainable
parameters by analytic count. AdamW starts at lr1e-4 and weight decay1e-3,
with a 500-epoch cap, patience50 and ten warmup epochs; these remain untested
proposals. GBDT tree count does not define an equivalent transformer size.

Archived the previous unexecuted 18-config plan with checksums, preserving
its identity against the historical delivery receipt. No features, labels,
selection IDs, GBDT bundles or runtime model code changed. No DL training
was launched and pretraining remains disabled. Actual model/trainer work is
still pending. Updated application/package/dataset READMEs and the planned
implementation contract; no installed dependency or layer boundary changed.

## 2026-09-21 — Separate package application from protocol experiments

User request: retain one selected protein–ligand feature strategy/ML application
inside the six-module package and organize ablations outside TopoKit with unified
letter identifiers. Added installed `workflows.protein_ligand_prediction` for
FS-AN (15 Å, radii 0.1–5.0, 27,500 features), a frozen reference schema, independent
implementation receipt and guarded optional GBDT inference. Selected the existing
general-v2020R1 pipeline as the default companion model; its embedded scaler and
all weights remain unchanged outside the wheel. Refined checkpoints remain study
benchmarks. The repository entry points now delegate to the installed API.

Moved the seven historical workflow sources and the mixed 20/15 Å application
snapshot to `datasets/protein_ligand_prediction/experiments/protocol_ablation`,
verifying 215 source hashes. Collected 42 complete strategy cards (FS-A…FS-AP),
scoped legacy aliases, ten study indexes, 123 evaluation rows, separate RMSE/PCC
JSON tables and immutable payload links. FS-Y identifies original r4; FS-AJ the
native 20 Å reference; FS-AN the selected application. The planned TopoFormer
ablation is ST-08 on fixed FS-AN, with no new feature IDs for model hyperparameters.

Updated package/workflow/dataset READMEs, architecture, model selection, package
resources and distribution/test discovery. No builder/core or higher-dimensional
algorithm changed; historical scientific identities and results are intact.
Validation: 1,243 tests passed, 23 optional skips; clean wheel/sdist and isolated
wheel checks pass; molecular tensor conformance is bitwise; all 369 result metrics
recompute within 1e-12. No new training, remote jobs or publication. DL remains a
scaffold; companion model deserialization still requires its pinned runtime.

## 2026-09-22 — supervised TopoFormer implementation and general-v2020R1 GPU study

User requested installation and execution through Cornell WSL on two GPUs.
Added optional `dl` extra and `workflows.topoformer` model/config, original
position/pooling conventions, explicit token ordering and checksummed
weights/scaler/schema bundles. Dataset-side revision3 registers 12 initial
configs, general-only validation, seed confirmation, bounded joint comparisons
and three full refits. Feature FS-AN and all topology/higher-dimensional APIs
are unchanged. Tests cover ordering, positions, initialization, tiny overfit,
scaler isolation, round-trip prediction and lazy dependencies. GPU smoke and
production outcomes are recorded in the study receipts and VALIDATION.md.
Training completion and relative benchmark performance remain unestablished
until final audits pass.

## 2026-09-22 — Complete TopoFormer benchmarks and deliver audited bundles

Completed the user-authorized general-v2020R1 study and resumed delivery after
the user renewed SSH authentication. All23 fits and12 CASF evaluation rows passed
remote and independent local audits. Six layers was selected by validation RMSE;
three full models, fitted scalers, all ablation bundles, predictions and histories
are local. Updated package/workflow/dataset/study README status, ST-08 index,
VALIDATION.md and the final experiment report. No API, feature, mathematical
core, higher-dimensional implementation or runtime dependency changed.

Validation:232 artifact hashes,204 frozen source/input hashes,19,066 feature
record/tensor pairs,73,980 validation and2,700 CASF prediction rows; metrics,
sample SD, ensemble means, training membership and refit schedules verified.
Selection and full scalers independently recomputed with exactly matching means
and scales. Original remote completion and frozen inputs remain unchanged;
LOCAL_DELIVERY.json records copied_locally=true after verification. CASF sets
remain reused feature-selection benchmarks; report seed means and the three-model
ensemble separately from single-model GBDT. No additional training or tuning.

## 2026-09-22 — Add article-informed TopoFormer ablation settings

User requested pooling/epoch clarification and more candidates grounded in the
original article. Added eight standalone JSON candidates, a follow-up plan,
published supplement/source evidence and validation receipt under the completed
study's extensions directory. Candidates cover depth12, width512 with depth6,
source-default768/12/12 capacity, article-scale1024/12 with proposed16 heads, and
four explicit optimizer learning rates. Recorded the paper's incompatible printed
head count and the README base-rate scaling discrepancy rather than silently
correcting either. Updated README/index links and validation notes. Pooling,
positions, batch32, dropout0.1, features and split remain fixed. Original204 frozen
hashes and all completed artifacts are preserved; no package code, API, dependency,
remote jobs or automation changed. Checks validate settings only; no new scores.

## 2026-09-22 — Execute the approved article-informed TopoFormer candidates

User authorized trying all eight registered candidates. Added isolated dataset
study orchestration for read-only input/control reuse, resource profiling,
eight screens, top-three seed confirmations, conditional full refits and
independent validation/test audits. Preserved the original experiment and
unchanged numerical trainer; package APIs and scientific feature definitions
are unaffected. Updated workflow/package/dataset status and reactivated the
existing hourly monitor for this follow-up.

Validation:36 tests and all8 real-GPU batch32 float32 profiles passed. Exact
resume and numerical-code reuse checks passed; local206-file lock and204-file
parent lock plus67 reference artifacts verified. Corrected a profiler logging
error before production and retained the failed prelaunch evidence. Launched
one detached two-GPU controller and observed both first epochs. Final winning
configuration, CASF results and local output delivery are pending.

## 2026-09-22 — Add ESM-2/ChEMBL27 sequence GBDT modality

Request: add pretrained protein and ligand embeddings as a second binding-prediction modality using the existing refined-2007/2013/2016 and general-v2020R1 memberships and GBDT settings. Added lazy workflow sequence encoders, guarded model/scaler inference, fixed-setting fitting, optional dependencies and an attributed compact ChEMBL checkpoint loader. Copied pretrained assets and external FS-AQ preparation, scheduling, fitting and verification scripts into the dataset study. Protein pooling covers every chain window; ligand context/unknown-symbol and source-valence limitations are explicit receipts. Geometry and all higher-dimensional topology routes are unchanged. Validation: 38 targeted tests, original ChEMBL smoke parity, ESM CPU pooling smoke, full input/split audit. Production accuracy remains pending the queued inference and four audited fits.

## 2026-09-22 — Move FS-AQ ESM generation to the local Mac

User requested local Mac resources instead of waiting for remote GPUs. Added an external study controller and execution manifest, canceled only the verified idle ESM waiter/launcher, and preserved all completed ligand embeddings. Apple Metal FP32 passed CPU agreement and padding checks at representative and maximum window lengths (maximum absolute error 1.43e-6). The unchanged frozen numerical runner produces the same 1280-dimensional protein representation; original checksums, inputs, pooling and GBDT settings remain intact. CPU assembly and existing Cornell ML/local-delivery controllers retain their audits. Current READMEs now describe Mac inference.

## 2026-09-22 — Complete and deliver FS-AQ sequence baseline

Completed the requested ESM-2/ChEMBL27 sequence workflow: 19,066 embeddings,
four full fixed-setting GBDT models with embedded training-only scalers and
six CASF evaluations. Mac MPS protein inference, reused CPU ligand embeddings
and Cornell training completed without changing the frozen scientific code.
All outputs are independently verified locally in the external FS-AQ study;
dataset feature, model and result links resolve to those delivered artifacts.

Updated package/workflow/dataset READMEs, the experiment report and current
progress receipt with actual results and inference environment requirements.
Preserved pre-run strategy/source manifests and historical milestones.
Validation: 27 output hashes, all pooled rows, split memberships, labels,
scaler statistics and recomputed metrics passed local delivery; all 109
frozen source/input hashes remain intact. The topology baseline has lower
RMSE and higher PCC on all six comparisons and remains the representative.
No API, numerical-core, higher-dimensional topology or feature-definition
changes were made in this completion update. Sequence context/UNK and source
graph limitations remain documented in the input audit and final report.

## 2026-09-22 — Launch first-generation ESM / NMI-parameter ablation

Request: regenerate proteins with first-generation ESM and retrain one GBDT
per training set with NMI settings, leaving unmentioned parameters at defaults.
Added an isolated external FS-AR study with an ESM-1b adapter, pinned assets,
verified ChEMBL27 cache reuse, four-model Cornell orchestration and independent
local delivery. The estimator overrides are n_estimators=10000, max_depth=7,
min_samples_split=2, subsample=0.4 and learning_rate=0.005; random_state remains
None and the actual pre-fit RNG state is recorded. Training-only scaling and
all original memberships are retained. No ensemble or repeated-seed fits.

Updated package/workflow/dataset documentation and the external strategy
index. No installed API, mathematical core, topology feature or prior run
was changed. Validation: three orchestration/cache/parameter tests plus real
CPU/MPS residue-pooling/padding checks and scaler smoke fit passed. Launched
one detached Mac controller to generate 8,830 windows, then run four parallel
Cornell fits and local audits. Full production results remain pending.

## 2026-09-23 — Add user-confirmed ESM-2/CPZ sequence study

Added external FS-AS orchestration for ESM-2 t33/650M protein vectors and the
supplied ChEMBL27/PubChem/ZINC ligand checkpoint. Verified protein reuse, new
CPZ BOS embeddings, active-vocabulary audit, four single NMI-parameter GBDT/scaler
fits, six CASF evaluations and independent delivery checks preserve all original
memberships. Updated package/workflow/dataset notes and pinned companion assets.
Four regression tests, real checkpoint CPU/MPS checks and all 19,066 feature
rows passed. Production training/results remain pending. No installed scientific
API, topology calculation or prior study was changed.

## 2026-09-23 — Queue ESM-2/ChEMBL27-only with current NMI GBDT

Added external FS-AT as requested after the active CPZ study. It reuses the
verified FS-AQ ESM-2 (1,280) plus ChEMBL27-only (512) matrix and applies the
same five NMI GBDT overrides as FS-AS. One dependency-gated controller waits
for FS-AS's verified local completion, then runs four Cornell fits and six
evaluations, delivers model/scaler bundles and independently audits outputs.
Both original caches, all memberships and prior studies are preserved.

Updated package, workflow, dataset and protocol-catalogue documentation.
The existing monitor covers the queued follow-up; no new automation was
created. Five regression tests and all 19,066 pooled feature rows passed.
Full production training/results remain queued. No installed API, mathematical
core, representative topology strategy or shared environment changed.

## 2026-09-23 — Article-informed TopoFormer ablation complete and delivered

Completed the authorized eight-candidate extension: 14 validation fits and
three full refits, with 17 model/scaler bundles delivered locally through Cornell
WSL. Added the final report, separate RMSE/PCC comparisons, per-seed and ensemble
results, model usage notes and a repeatable local delivery audit; updated current
package/workflow/dataset status. A03 won the registered validation comparison
and used 102/104/92 full-refit epochs. It improves CASF-2007, while the preserved
six-layer control remains better on CASF-2013/2016 for RMSE and PCC. No package
API, numerical source, feature strategy, dependency, hyperparameter or default
changed; parent and unrelated study artifacts are preserved. Verified 188 output
hashes, all 17 bundles, 206 follow-up and 204 parent frozen files, 67 external
references, memberships, labels, training-only scalers, validation/CASF metrics,
sample SD and prediction ensembles. The prior raw-feature/scaler audit was
explicitly authenticated and reused. No new training or feature generation was
started during delivery. See the current validation note and experiment final report.

## 2026-09-23 — Deliver FS-AR ESM-1b/NMI GBDT results

Recorded completion and independent local delivery of 19,066 features, four
10,000-tree GBDT/scaler bundles and six evaluations. Updated the package,
workflow, sequence and dataset READMEs and experiment report with actual
results and artifact links; separate RMSE/PCC comparisons retain encoder/GBDT
and stochastic-fit limitations. Additional artifact review verified hashes,
reused ligand provenance and scaler scale arrays without local model unpickling.
No installed numerical API or frozen study code changed. FS-AS continues,
FS-AT retains its dependency gate, and the existing monitor remains active.

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

## 2026-09-23 — Finalize FS-AU sequence parameters; archive FS-AW

At the user's request, saved the NMI-inspired sqrt ESM-2 + CPZ configuration as
the final sequence profile, including all resolved parameters, four model/scaler
hashes and links to retained features, results and predictions. Added canonical
final and dataset/workflow selection manifests. Archived FS-AW previous-setting
results and parameters in place without deleting its artifacts. Updated package,
workflow, sequence, dataset, study and catalogue READMEs and validation notes.
No training code, scientific inputs, models, scalers, predictions or historical
receipts changed. Verified both source locks and all final audited payload hashes,
exact parameters, final links and archive retention. No retraining or package
API change was necessary. Existing model runtime requirements remain documented;
monitoring stays retired.

## 2026-09-23 — Select final A03 architecture; organize DL ablations and propose tuning

Registered the user-selected A03 protocol architecture (768 hidden units,
12 encoder layers/heads, FF width 3072; 86,071,297 parameters) in identical
workflow/dataset receipts and an explicit application config. Added the canonical
final model/scaler/result links and representative_topoformer entry point.
Organized both completed studies under ST-08: 20 candidate settings, 40 unique
fits and 24 CASF rows, one aggregate JSON plus separate RMSE/PCC and CSV tables.
All original artifacts remain archived in place; historical model/source receipts,
feature identities, GBDT/sequence profiles and generic constructor defaults are
unchanged. This is an application-profile/documentation change, not new numerical
model code or a claim that A03 wins every test or is faster per fit.

Prepared a future fixed-A03 optimization plan: explicit 16,648/1,850 split,
nine LR/batch configs, staged weight decay, common-validation training-fraction
diagnostic and three-seed/full-refit rules (at most 25 new fits). Optional earlier
cosine decay is a separate later proposal. No jobs or automations were started.
Validation: authenticated original deliveries/frozen files; rescored 131,166
prediction rows within 1e-12; verified profile identity, all nine configs, split
membership/test exclusion, fit budget and new documentation links. Batch resource
profiles and proposed performance gains remain untested until a future run.

## 2026-09-23 — Execute approved A03 optimizer tuning on two GPUs

User approved the registered LR/batch/WD plan and reduced the maximum to200
 epochs. Added an isolated dataset execution with54 preregistered primary and
 diagnostic configs, a fixed16648/1850 split,200-epoch cosine horizon, automated
 seed confirmation, paired training-fraction comparison, three full refits and
 validation/CASF/output audits (23–25 actual fits). Reused authenticated features
 and parent raw/full caches without downloads or changes. Preserved the original
 500-epoch proposal and completed models/results. Updated current package,
 workflow, dataset and ST-08 documentation and launch receipts.

GPU validation:42 tests and9 memory/determinism profiles passed. Controller
 launched17:58:06UTC, two active GPU trials. Frozen source has257 files with
 SHA256189a9f277147d4cdbf5aeb7440283e58c6a73449b2744e24dbc42f543a5bf81f.
 A missing preflight-helper import was corrected before training and preserved
 in the prelaunch archive. Final results remain pending; architecture is fixed,
 optimization is experimental, and CASF is not used for selection.


## 2026-09-23 — Final FS-AN GBDT hyperparameters and three-seed consensus

At the user's request, retained the previous modeling parameters for historical
feature ablations and selected the transferred CPZ parameters for final topology
ML: learning_rate .005, min_samples_split 2, subsample .4, 10,000 depth-7 trees
and sqrt sampling. Trained seeds 0/1/2 for each of four original memberships;
final predictions average all three pK outputs. The 12 fitted pipelines, 12
portable scalers, 18 seed evaluations and six consensus evaluations are saved
under the external final/topology_gbdt artifact folder.

Updated the installed workflow prediction adapter and model profile, current
README/selection entry points, CLI documentation, architecture/ML contracts and
inference tests. The API accepts the final ensemble manifest and legacy complete
single-model bundles, keeps scaling inside each pipeline, validates all members
and preserves the output CSV format. Default aliases select the general-v2020R1
ensemble. Feature values, geometry, L0/higher-dimensional algorithms and historical
ablation parameters/results are unchanged. Optional ML dependencies remain lazy;
no weights or fitted scalers enter the wheel.

Validation: 45 inference/workflow/architecture/CLI tests passed; all 19,066
original tensors and records, exact memberships/targets and 12 scaler arrays
were independently audited. Saved-model reloads and the public prediction module
reproduced every one of the 1,350 final predictions bitwise. Compatible-runtime
wheel validation and authorized superseded-pipeline retirement are recorded in
the final artifact folder's WHEEL_AUDIT.json and RETIREMENT receipts. The final
ensemble improves the old general-set configuration on all three CASF tests,
while the previous new-parameter single stochastic fit has slightly better
general-set scores. No best-seed selection or ensemble-weight tuning is used.


## 2026-09-23 — Development copy, public guides and reusable feature recipes

Request: preserve the current package as `topokit_dev`, standardize current
software documentation and basic usage, highlight the updated notebook tutorials,
provide about ten protein–ligand examples in both copies, and add agent/skill
recipes for topology and the selected pretrained sequence encoders.

Created a full independent working copy before modification (4,831 files,
305,304,747 bytes at copy time). `topokit_dev` is the development copy and
`topokit` is the synchronized public release candidate; both install as
`topokit`. No repository, package or data was published. Previous root/workflow/
sequence README contents are retained in named history files.

Changed root and workflow READMEs, AGENTS guides, topology/sequence recipe JSON,
notebook links, MANIFEST and CI distribution checks. Added ten compact diverse
PDB/MOL2 pairs with a portable CSV manifest, exact source hashes and expected
feature metadata under `examples/protein_ligand` in both folders. Corrected one
stale rerun filename in the fixed-object notebook; notebook code, numerical
outputs and scientific content were preserved.

Added the repository-only topology manifest exporter. It composes the existing
FS-AN selection/compute API, exports canonical schema bytes and per-sample
provenance, reports input warnings and errors, and writes a combined matrix only
for a complete batch. Added a sequence extraction CLI with explicit chains/SMILES,
local asset verification and raw embedding receipts, plus two repository skills.

The sole installed API addition is `SequenceEncoder(ligand_profile="cpz")`,
with pinned CPZ checkpoint/dictionary hashes and a distinct embedding recipe ID.
The historical `chembl27` default and FS-AQ GBDT helpers remain unchanged. No
mathematical core, FS-AN geometry, crop, channel, spectrum or statistic changed;
optional ML/DL dependencies remain lazy. No model was trained or downloaded.

Validation: 1,279 tests passed, 23 skipped and one external-weight inference test
was deliberately deselected, with the supervised TopoFormer test module excluded;
architecture and relevant historical sequence tests are included. Five new
exporter tests cover exact API equality, canonical schema, prediction-loader
compatibility, manifest resolution, explicit partial failure and resource errors.
Fourteen new sequence tests cover profile identity, trusted hash guards, no-load
validation and output receipts. All ten real pairs pass direct extraction and
an independent skill-forward batch run; output is finite float32 (10,27500),
matching all per-sample (10,50,55) tensors in order. All three notebooks validate
as nbformat4; both skill files pass the skill validator. Seven actual local
ESM-2/CPZ asset hashes were checked without importing or loading the models.
Distribution and installed-wheel results are recorded in VALIDATION.md.

Remaining limits: the input structures retain source-data terms and unresolved
redistribution provenance; MIT covers the code, not those structures. Actual
sequence model inference and the advertised full OS/Python CI matrix were not
executed locally for this change. Upstream model pages were checked, but the
external CPZ download archive was not downloaded or validated. The optional
trained affinity models remain external assets; the sequence runner does not
implement selected FS-AU model inference. Fixed-cohort historical H0/deletion
controllers retain their original scope.


## 2026-09-24 — First GitHub source and release publication

At the user's explicit request, prepared the public package for
`git@github.com:ChenDdon/TopoKit.git` and release tag `v0.3.0`, preserving the
remote initial commit while removing its one-byte placeholder `tempfile`.
Added canonical repository/issue/release URLs and clone/install instructions,
release notes, reusable CI and a GitHub release workflow. Only a passing
six-environment platform matrix can publish the built wheel, source archive
and SHA256SUMS. Existing release assets are not overwritten.

Removed regenerable caches and stale local build metadata from the public
copy. Generated research outputs, pressure transfer/history artifacts and
machine-specific status remain local and are ignored/pruned from publication.
The development copy is unchanged. Credential-pattern and notebook metadata
review found no credential material in the intended publication files.

Scientific/API impact: none. The FS-AN recipe, explicit CPZ profile, examples,
notebook code and optional dependency boundaries retain their validated behavior.
The platform/release results are recorded by GitHub Actions for the release tag;
local archive and test evidence are retained outside the repository. Structure
inputs keep their source provenance and terms rather than inheriting the MIT
code license. Full pretrained sequence inference is outside these release checks.

## 2026-09-24 — Preserve sample bytes in Windows clones

The initial GitHub matrix exposed automatic line-ending conversion in Windows
checkouts: all source tests passed, but packaged sample hashes did not match
SOURCE.json. Added a narrowly scoped `.gitattributes` rule disabling conversion
for the checksummed structures, included it in source distributions, and kept
strict distribution hashes with more informative failures. README and validation
notes explain this cross-platform guarantee. No runtime API, scientific input,
expected hash, feature recipe or notebook changed. Linux/macOS and clean local
archive checks passed; the subsequent GitHub matrix verifies the Windows fix.
