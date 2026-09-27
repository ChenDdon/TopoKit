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

Complete eigenvalue-only queries use an incident-vertex solve with exact
isolated-zero restoration. Full eigenvector and partial-spectrum queries use
the full active basis and the core solver's residual, completeness and
numerical-nullity semantics. Matrix-only queries perform no eigensolving.
The sweep uses neither component decomposition nor approximate spectral
methods.

## Eligibility and higher dimensions

`L0Sweep.supports(obj)` checks that every edge endpoint has a singleton birth
at or before the edge birth. The explicit sweep rejects a missing/late face.
Such native hyperdigraphs remain valid and use the general embedded-chain
Laplacian, including its missing-face constraint projection. Higher hyperedges
may be present in an eligible object: they do not enter ordinary L0 and remain
available for L1, L2, L3 and higher calculations.

`workflows.laplacian_series` automatically uses the sweep for eligible ordinary
hyperdigraph L0, including a mixed-degree series. Higher degrees call
the general hyperdigraph core. An explicit `backend="reference"`, a missing-face
filtration, or a full-sweep resource limit selects per-scale core evaluation.
Simplicial and interaction series use their respective core routes.

Genuine two-scale persistent `L0(start,end)` can involve late-vertex chain
constraints and is not obtained by these edge updates. `persistent_laplacian`
and `laplacian_series(mode="persistent")` use the persistent core algorithms
in every degree.

## Independent vertex deletion

```python
deleted = sweep.vertex_deleted_laplacian(1.0, "b", return_matrix=True)
original = sweep.laplacian(1.0, return_matrix=True)  # remains undeleted
```

The optional operation removes one active singleton identified by its
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
Matrix/eigenvector return options and numerical nullity use the core spectral
machinery. This API does not compute vertex-deleted higher-degree or two-scale
persistent operators.

Tests independently assemble signed incidence products after each deletion,
including random filtrations, reciprocal edges, late singletons, mixed IDs,
isolates, empty operators, full reference reconstruction and input immutability.

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
