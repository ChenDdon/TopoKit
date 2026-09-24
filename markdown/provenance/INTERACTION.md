# Interaction reference-core provenance

Migrated from the local `Interaction_Topo/src/interaction_topology` version
0.2.0 reference implementation. The migration retains relative imports,
ordered tensor quotient definitions, GF(2) barcode reduction, signed real
boundaries, and genuine two-time persistent Laplacians. The original source
folder was not edited. Its mathematical reference tests are rebased under
`tests/legacy_interaction`.

The version 0.2 architecture splits the former route facade without retaining
the old `topokit.interaction` or `topokit._core.interaction` paths:

- `topokit.builders.interaction` constructs explicit two-factor objects,
  manages point identity, geometric factor construction, start-stage clamping,
  and construction resource preflight.
- `topokit.core.interaction` analyzes defined interaction objects and exposes
  common results and labelled eigenvectors. It does not import builders.
- `topokit.core._interaction` contains the explicit factor/chain types,
  quotient boundary construction, barcode reduction, and Laplacian algorithms.
- `topokit.workflows.interaction` owns the dimension-dictionary and explicit
  factor construction-plus-analysis convenience workflows.

Geometric factor construction calls `topokit.builders._simplicial`; it is
never performed inside the interaction mathematical core.
No path-complex or path-homology module is imported or required.

The original package does not declare a license. Redistribution remains
subject to the project owner's license and provenance review.
