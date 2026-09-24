# Simplicial core provenance

This internal implementation was mechanically migrated from the local
`simplicial_topo/src/simplicial_topo` research reconstruction. Its original
directory is retained unchanged. Runtime imports are wholly within `topokit`;
the original sibling package is not required.

The source implements face-closed filtered simplicial complexes, prime-field
homology and persistence, real Hodge Laplacians, and inclusion-persistent
Laplacians. NumPy and SciPy provide numerical linear algebra, Qhull Delaunay
triangulation, sparse storage and KD trees. No external TDA package is imported.

The original mathematical validation suite is retained under
`tests/legacy_simplicial`, with imports rebased to the six-layer architecture.
The public analysis adapter is `topokit.core.simplicial`. Point-cloud/graph
construction is in `topokit.builders.simplicial`, with the actual geometric
algorithms in `topokit.builders._simplicial`. Convenience composition is in
`topokit.workflows.simplicial`; none is imported by this mathematical core.

Changes beyond mechanical migration are deliberately small: the native
resource-limit exception also inherits the toolkit resource-limit exception;
alpha construction rejects an excessive unique-vertex count before Qhull;
and edge circumcenters use their exact midpoint formula instead of a numerical
least-squares solve. Higher-dimensional circumcenters and coface propagation
retain the original algorithm.

Alpha geometry is **unweighted and floating-point**. PointCloud weights are
retained by the public adapter but do not define power-distance/weighted-alpha
geometry. This implementation does not claim exact geometric predicates or
performance parity with GUDHI/CGAL.
