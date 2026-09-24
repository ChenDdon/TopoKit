# Native alpha construction (2026-09-17)

The complete native benchmark finished on 2026-09-17: all 19,066 features,
four models and six evaluations are copied locally and audited. The actual
19,063 native-1.1.0 plus three native-1.1.1 generation records remain intact
under the verified compatibility inventory. Superseded production runs are
archived with relocation/hash ledgers; current outputs use
`final_alpha_l0_unique_repair`. [Final results](../../datasets/protein_ligand_prediction/results/final_alpha_l0_unique_repair/RESULTS.md).
Older execution-status paragraphs below are historical checkpoints.


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


TopoKit constructs unweighted alpha filtrations with its own implementation
in `builders/_alpha_native.py`. NumPy and SciPy are sufficient. Neither
feature generation nor the native builder imports GUDHI. The explicit
`backend="gudhi_exact"` adapter remains optional for historical replay and
independent comparisons; it is never an automatic fallback.

```python
from topokit.builders.simplicial import from_points

# Public max_dimension is the intended analysis degree q.
# q=0 constructs vertices AND correctly filtered edges.
topology = from_points(points, complex_type="alpha", backend="native",
                       max_dimension=0, filtration_range=(0, 4.9**2))
```

## Geometry and units

SciPy/Qhull provides candidate Delaunay simplices. TopoKit implements the
circumsphere calculations, empty-ball tests, degeneracy handling, coface
propagation, dimension/cutoff extraction and conversion to its native object.
This replaces the alpha-construction dependency; it does not reimplement
NumPy, SciPy or Qhull.

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
units. The molecular workflow reports `s=0,.2,...,9.8 Å`, defines radius
`r=s/2`, and includes an edge when `birth <= r²`. The returned birth is already
squared; do not square it again. No coordinate weights enter alpha geometry.

## Efficiency and higher dimensions

Circumspheres and neighbor queries are batched. Edges use midpoints;
3D triangles and tetrahedra use vector formulas. Compact simplex arrays and
indexed minimum reductions replace the previous temporary full simplex tree
and Python loop over every circumsphere. For L0, only vertices and edges are
materialized in the returned object. Higher cofaces still determine their
correct births. Geometry is built once per channel; the existing `L0Sweep`
inserts edge events once and reuses unchanged spectra across sample scales.

The same builder supports full higher-dimensional complexes. Dimension
truncation happens **after** coface propagation. No homology, persistence,
boundary matrix, L1/L2, or interaction-core algorithm changed. Requesting
analysis degree `q` retains `q+1` simplices by default, preserving the
upper-boundary contribution and available persistence deaths.

## Precision, degeneracy and limits

This is a floating-point implementation with rational repair, **not a claim
to reproduce CGAL/GUDHI's exact-predicate kernel for every possible input**.
Metadata records `geometry_backend="native"`, engine version `1.1.0`, and
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

As explicitly requested on 2026-09-17, alpha builders now default to
`duplicates="merge"`: check finite input coordinates first, retain each exact
location once, and keep its first original row as the stable vertex label.
Both native and optional GUDHI backends use the same input rule.
`duplicates="error"` remains available for strict historical replay. Metadata
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
Original ligand sites still define the 20 Å crop. Removed atoms do not add
vertices, edges, eigenvalues or multiplicity weights, including at scale zero.
A component/element made empty follows the existing absent-channel zero rule.
JSON receipts list every removal and retained counterpart. This is unique-atom
selection, not the previously discussed all-atom edge-lifting alternative.

`max_simplices` counts the full Delaunay closure, even for an L0-only result.
Known vertex and single-simplex lower bounds are checked before construction;
closure growth is checked before further expansion. This limit is not a cap
on Qhull's transient allocation. `geometry_tolerance` remains accepted and
validated for API compatibility; it no longer enlarges empty balls. Numerical
uncertainty routes to the rational checks instead.

## Molecular recipe and historical results

The active workflow is version **1.2.0**, schema **3**, using native alpha
engine **1.1.0**. Its 20 Å crop, 55 channels, 50 scales, 10 summaries, L0
operator, 27,500 feature coordinates, manifests and ML settings are unchanged.
The user-authorized unique-atom policy changes duplicate-containing inputs;
engine and recipe identities change explicitly. Existing GUDHI tensors and
models retain their original records and hashes; they are not relabeled as
native outputs. Generation rejects mixed recipes.

The future native controller and AWS helpers use `final_alpha_l0_unique`
output roots and a new source snapshot. The existing `final_alpha_l0` results
remain the completed historical benchmark. No mass regeneration or model
retraining is triggered by installing or importing the new backend.

`workflows/protein_ligand_prediction/validate_native_alpha.py` is a read-only
migration auditor. It checks source-structure and reference-tensor hashes,
then compares recomputed tensors bitwise. Its optional `--oracle` mode checks
all selected-cloud edges against GUDHI exact geometry and their inclusion at
every one of the 50 scales. GUDHI is required only for that optional mode.
Current measurements and audit coverage are recorded in
[VALIDATION.md](VALIDATION.md) and the dataset experiment folder
`experiments/native_alpha_validation_20260917/`.


Historical engine-1.0.0 validation: all 4,057 r4 complexes, 106,086 channels and 228,691,848
edges agree with GUDHI's scale-sampled filtration; 92/92 final AWS tensors are
bitwise identical. Alpha construction is 1.998× faster in that paired audit.
The full local/AWS suites pass 1,134/1,135 tests. See the
[completed audit report](../../datasets/protein_ligand_prediction/experiments/native_alpha_validation_20260917/REPORT.md)
for timing scope, source hashes and cross-platform roundoff reconciliation.
