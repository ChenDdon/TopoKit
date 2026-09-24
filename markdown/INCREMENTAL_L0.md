# Incremental ordinary L0 construction

For an explicit sequence-hyperdigraph filtration in which every edge's two
singleton faces are present no later than that edge, ordinary L0 is

\[
L_0(s)=\sum_{b_e\leq s}(e_i-e_j)(e_i-e_j)^\top.
\]

The birth coordinate controls inclusion only. It is not a matrix weight.
Ordered reciprocal edges are separate incidence columns and contribute twice.
No tail/head directed-hypergraph or normalized/directed random-walk Laplacian
is substituted for the native embedded-chain definition.

## Public API

```python
from topokit.core.hyperdigraph import FilteredHyperdigraph, L0Sweep

filtration = FilteredHyperdigraph(
    ("a", "b", "c"),
    [(("a", "b"), 0.5), (("b", "c"), 1.0)],
    include_all_vertices=True,
)
sweep = L0Sweep(filtration, max_dense_entries=4_000_000)
L_at_half = sweep.matrix_at(0.5)   # Matrix only; no eigensolver.
L_at_one = sweep.matrix_at(1.0)    # Adds just the newly born edge.
labels = sweep.basis_at(1.0)      # Singleton hyperedges in matrix row order.
spectrum = sweep.laplacian(1.0)  # Optional existing symmetric eigensolver.
assert sweep.diagnostics["edges_inserted"] == 2
```

`L0Sweep` accepts a native `FilteredHyperdigraph` or a hyperdigraph `Topology`.
It consumes only the supplied object, never geometry or point clouds. Scales
must be finite, nondecreasing and within the wrapper's filtration domain.
Equal scales are allowed and do not reinsert edges. Inclusion is `birth <= scale`;
no epsilon is added. Invalid scales do not advance the accumulator.

Each edge performs four accumulated updates: +1 at both endpoint diagonals,
and -1 at both symmetric off-diagonal positions. Repeated endpoints and
reciprocal edges are accumulated correctly. The vertex indexing is fixed once
in ambient order; late singleton births enter the returned basis at their
actual birth. Ambient vertices without explicit singleton hyperedges are not
silently inserted. Every returned `matrix_at` matrix and every spectrum's
requested matrix is an independent copy, so later updates cannot alter it.
An instance is mutable and must not be advanced concurrently.

Complete eigenvalue-only queries retain the molecular workflow's existing
incident-vertex solve with exact isolated-zero restoration. Full eigenvector
and partial-spectrum queries use the full active basis and retain the original
solver, residual, completeness and numerical-nullity semantics. Matrix-only
queries perform no eigensolving, and no component decomposition or approximate
spectral method is introduced by this change.

## Eligibility and higher dimensions

`L0Sweep.supports(obj)` checks that every edge endpoint has a singleton birth
at or before the edge birth. The explicit sweep rejects a missing/late face.
Such native hyperdigraphs remain valid and use the general embedded-chain
Laplacian, including its missing-face constraint projection. Higher hyperedges
may be present in an eligible object: they do not enter ordinary L0 and remain
available for the unchanged L1, L2, L3 and higher calculations.

`workflows.laplacian_series` automatically uses the sweep for eligible ordinary
hyperdigraph L0, including a mixed-degree series. Higher degrees still call
the existing core. An explicit `backend="reference"`, a missing-face
filtration, or a full-sweep resource limit selects the original workflow path.
Simplicial and interaction routes are unchanged.

Genuine two-scale persistent `L0(start,end)` can involve late-vertex chain
constraints and is not obtained by these edge updates. `persistent_laplacian`
and `laplacian_series(mode="persistent")` continue to use their existing
algorithms in every degree.

## Independent vertex deletion (2026-09-18)

```python
deleted = sweep.vertex_deleted_laplacian(1.0, "b", return_matrix=True)
original = sweep.laplacian(1.0, return_matrix=True)  # remains undeleted
```

The new optional operation removes one active singleton identified by its
ambient vertex ID, together with all its incident edge columns. It returns
the complete ordinary L0 spectrum on the remaining singletons. Every call
starts from the same undeleted sweep state at that scale: deletions are
independent, and isolated remaining vertices are retained. An absent or
not-yet-born vertex raises `ValueError`; scales remain nondecreasing. A sole
active vertex can be deleted, producing a complete empty operator. The
metadata records the removed ID and remaining singleton/edge counts.

The private `delete_vertex_l0` kernel takes the principal submatrix and adds
the removed column's negative off-diagonal entries to its diagonal. This
subtracts the incident-edge degree contributions, including multiplicity
from reciprocal ordered edges. A principal submatrix without this correction
is not the induced graph Laplacian. The original matrix is never mutated.
Matrix/eigenvector return options and numerical nullity use the existing
spectral machinery. This API does not compute vertex-deleted higher-degree
or two-scale persistent operators and does not change their existing APIs.

Tests independently assemble signed incidence products after each deletion,
including random filtrations, reciprocal edges, late singletons, mixed IDs,
isolates, empty operators, full reference reconstruction and input immutability.
The new Cornell molecular application calls this API instead of rebuilding
every graph object. Its connected-component and graph-twin reuse remains in
the application, and its full spectra are checked against independent D−A
oracles. See the [experiment recipe](../workflows/protein_ligand_prediction_version2/README.md).

## Resources and computational cost

The accumulator allocates one dense buffer for the final explicit singleton
basis: N squared entries, guarded before allocation by `max_dense_entries`.
Its edge schedule is guarded by `max_sparse_entries` on the singleton count
and twice the number of directed edges, retaining the incidence-storage bound.
These limits cover the sweep's entire supplied filtration, including later
events. The generic workflow can fall back to individual snapshots when only
the requested snapshots fit the budget; it does not drop vertices or edges.

After schedule preparation, each edge is inserted once rather than revisited
at every later scale. Edge-update work is O(E); matrix initialization remains
O(N squared), and returning S separate dense snapshots costs O(S N squared).
The full eigenvalue calculation and exact alpha geometry have separate costs.
Keeping all returned matrices alive also retains all those matrix copies.

## Alpha feature application and provenance

The alpha-ablation engine 1.2.0 constructs one explicit edge filtration per
channel and calls the public L0 sweep. It retains the original exact GUDHI
3.13.0 alpha births, scale/2 radius convention, per-channel atom selection,
internal-edge exclusion, zero tolerance, isolated-zero handling and summaries.
It also retains the existing reuse of unchanged edge sets and shared channel
results across revision masks. No ML settings or benchmark splits change.

New source hashes and the explicit `l0_assembly` schema field yield new recipe
IDs. The completed 2016 tensors/models and immutable AWS 1.1.0 source remain
historical evidence. They are not relabelled or overwritten, and no new full
AWS production/training study is launched by this package update. New runs
must use fresh source/output locations; the existing AWS helper defaults still
identify the frozen completed study.

## Validation

The package and all molecular workflow suites pass **1,596 tests**, with
**10 skips** and two existing intentional truncation warnings. The new tests
compare matrices against independently assembled signed incidence products,
including random filtrations, reciprocal edges, tied and boundary births,
late vertices, absent zero chains, mixed vertex IDs and snapshot independence.
They check invalid queries, budgets, partial/eigenvector semantics, and ensure
matrix-only construction never calls an eigensolver. Mixed-degree L0-L3 and
genuine persistent results match the reference backend. Existing independent
full-deletion-matrix audits also cover non-face-closed higher-dimensional cases.

The reproducible [construction benchmark](../examples/benchmark_incremental_l0.py)
uses 50 scales, fixed synthetic 3D point clouds and five alternating-order
timing repeats, after a warm-up/equality check. Both methods return dense
matrix copies; exact alpha geometry and eigensolving are excluded.

| Support | Vertices | Edges | General assembly median | Incremental median | Ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Alpha | 64 | 303 | 22.658 ms | 0.675 ms | 33.5x |
| Alpha | 256 | 1,580 | 97.489 ms | 3.198 ms | 30.5x |
| Distance | 64 | 582 | 25.654 ms | 0.823 ms | 31.2x |
| Distance | 256 | 8,768 | 242.524 ms | 6.124 ms | 39.6x |

All 200 scale-specific matrices are bitwise equal to the existing general
embedded-operator construction. These are local construction-only timings,
not measured end-to-end molecular or AWS throughput improvements.
[Raw benchmark receipt](../examples/output/incremental_l0_20260915/benchmark.json).

The frozen 1.1.0 and new 1.2.0 feature engines also produce identical complete
tensors for all seven strategies on four molecular complexes: 1a30, 4ej8,
1hps and 4fys. All **28 tensors** are bitwise equal after float32 conversion,
with **zero maximum absolute difference** in the float64 summaries. This
includes a previously problematic alpha geometry case and a large selection
of 3,535 protein atoms plus 110 ligand atoms. These are representative
equivalence checks, not a rerun of the full 4,057-complex study.
[Molecular receipt](../examples/output/incremental_l0_20260915/molecular_equivalence.json).

The built wheel passes the installed-package smoke check from an isolated
interpreter outside the repository, including the new sweep API, all three
topology routes, higher-dimensional analysis and persistent queries. Legacy
research packages and optional geometry engines are blocked during this
runtime check. The validation receipt records the wheel hash and source hashes.
[Validation receipt](../examples/output/incremental_l0_20260915/validation.json).
