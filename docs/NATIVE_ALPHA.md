# Native alpha construction

TopoKit constructs unweighted alpha filtrations with its own implementation
in `builders/_alpha_native.py`. NumPy and SciPy are sufficient. Neither
feature generation nor the native builder imports GUDHI. The explicit
`backend="gudhi_exact"` adapter is optional for reference calculations and
independent comparisons; it is never an automatic fallback.

```python
from topokit.builders.simplicial import from_points

# Public max_dimension is the intended analysis degree q.
# q=0 constructs vertices AND correctly filtered edges.
topology = from_points(points, complex_type="alpha", backend="native",
                       max_dimension=0, filtration_range=(0, 5.0**2))
```

## Geometry and units

SciPy/Qhull provides candidate Delaunay simplices. TopoKit implements the
circumsphere calculations, empty-ball tests, degeneracy handling, coface
propagation, dimension/cutoff extraction and conversion to its native object.
The native builder uses NumPy, SciPy and Qhull primitives without requiring
an external alpha-construction library.

For simplex vertices `p0,...,pk`, let the rows of `E` be `pi-p0`.
The minimum circumsphere center is

`c = p0 + E.T @ solve(E @ E.T, diag(E @ E.T)/2)`.

Its squared radius is `rho² = ||c-p0||²`. A simplex is Gabriel when no
selected point lies strictly inside this sphere. Its alpha birth is the
minimum of its own squared circumradius, if Gabriel, and the births inherited
from its immediate cofaces. A non-Gabriel simplex inherits a coface birth.
Vertices have birth zero. All geometric decisions use the whole selected
cloud before application-specific cross-edge masks.

Consequently an obtuse triangle can delay an edge beyond `length²/4`.
For `(-1,0), (1,0), (0,0.5)`, the long edge and triangle both enter at
`1.5625`, although the edge midpoint radius squared is `1`. Keeping only
short Delaunay edges would give a different filtration. The definition agrees
with the standard [alpha filtration rule](https://gudhi.inria.fr/python/3.10.1/alpha_complex_ref.html).

The native filtration coordinate is **squared radius**, in squared coordinate
units. The Topo molecular workflow uses 50 radii `r=0.1,0.2,...,5.0 Å`
and includes an edge when `birth <= r²`. The returned birth is already
squared; do not square it again. No coordinate weights enter alpha geometry.

## Efficiency and higher dimensions

Circumspheres and neighbor queries are batched. Edges use midpoints;
3D triangles and tetrahedra use vector formulas. Compact simplex arrays and
indexed minimum reductions propagate births through the coface closure. For L0,
only vertices and edges are materialized in the returned object. Higher cofaces
still determine their correct births. Geometry is built once per channel; `L0Sweep`
inserts edge events once and reuses unchanged spectra across sample scales.

The same builder supports full higher-dimensional complexes. Dimension
truncation happens **after** coface propagation. Homology, persistence, boundary
matrices and Laplacians are computed by the separate mathematical cores. Requesting
analysis degree `q` retains `q+1` simplices by default, preserving the
upper-boundary contribution and available persistence deaths.

## Precision, degeneracy and limits

This is a floating-point implementation with rational repair, **not a claim
to reproduce CGAL/GUDHI's exact-predicate kernel for every possible input**.
Metadata records `geometry_backend="native"`, the engine version and
the precision description. Extended precision is used where NumPy's platform
supports it; stored births remain float64. Ordinary roundoff can differ from
GUDHI, including near an exactly chosen observation boundary.

Ill-conditioned circumspheres and ambiguous empty-ball tests use Python
`Fraction` arithmetic on the exact supplied binary floating-point values.
Uncertain affine rank is checked rationally, so a very thin full-dimensional
cloud is not silently flattened. A basis projection is used only to propose
a triangulation for genuinely lower-dimensional clouds; circumspheres and
empty-ball checks use the original coordinates.

Candidate maximal cells are checked for empty circumspheres, cospherical
consistency and omitted vertices. Near-cospherical cavities are repaired locally:
only candidate cells among the affected sphere's sites are enumerated, then
checked against **all original points**. Manifold face incidence and supporting
boundary planes are checked after replacement. A flat candidate uses its facet
neighbors to define the local repair. If Qhull itself fails or omits vertices,
bounded whole-cloud enumeration is available. Rational decisions resolve
uncertain cases. At most **300,000 candidate cells** are permitted per repair;
exceeding that limit raises `GeometryError`. This is a repair within
the native implementation, not a switch to an external alpha backend.

Cospherical ties use symbolic lifting of vertex heights, ordered by the
original coordinates (largest lexicographic coordinate has the dominant
positive lift), to select one consistent triangulation. No coordinate or alpha radius is
perturbed. Delaunay triangulations are not unique in such cases, and the
generic higher-dimensional choices are not a universal GUDHI-compatibility
promise. Molecular equivalence is measured explicitly. The native implementation
uses its own barycentric lifting predicate; it does not link CGAL. SciPy documents its
[Qhull precision and omitted-point behavior](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html).

Alpha builders default to
`duplicates="merge"`: check finite input coordinates first, retain each exact
location once, and keep its first original row as the stable vertex label.
Both native and optional GUDHI backends use the same input rule.
`duplicates="error"` is available when exact duplicate coordinates must be rejected. Metadata
records the policy, original/unique/removed point counts and
`original_to_vertex`; the public builder also maps original stable IDs.
`PointCloud.unique_coordinates()` returns an independent unique subset,
preserving first-row IDs, weights, element labels and documented row-aligned
reader metadata. Source records remain attached as provenance. Selected bonds
are the induced subset, not lifted or rewired. Exact equality includes signed
zero; finite near-coincident values remain separate. No jitter, rounding or
coordinate averaging is applied.

The molecular workflow calls this operation on the complete selected atom
list **before splitting into element channels**, in cropped-protein then
ligand order. First-row element/component labels win even when conflicting.
Original ligand sites still define the 15 Å crop. Removed atoms do not add
vertices, edges, eigenvalues or multiplicity weights, including at scale zero.
A component/element made empty follows the absent-channel zero rule.
JSON receipts list every removal and retained counterpart. Removed atoms are
not restored through edge lifting.

`max_simplices` counts the full Delaunay closure, even for an L0-only result.
Known vertex and single-simplex lower bounds are checked before construction;
closure growth is checked before further expansion. This limit is not a cap
on Qhull's transient allocation. `geometry_tolerance` remains accepted and
validated for API compatibility; it does not enlarge empty balls. Numerical
uncertainty routes to the rational checks instead.

See the [Topo workflow guide](../workflows/protein_ligand_prediction/README.md)
for the complete molecular recipe and output schema.
