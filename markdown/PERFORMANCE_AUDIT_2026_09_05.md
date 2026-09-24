# TopoKit algorithm and efficiency audit — 5 September 2026

## Assessment

**TopoKit has substantial opportunities to run faster without changing its
defined topology or finite-field answers. Start with eliminating repeated work,
using the requested degree throughout the computation, and exposing specialized
algorithms that the native kernels already implement.** A wholesale replacement
of the three mathematical cores is neither necessary nor justified by this audit.

Reviewed: the six-layer architecture and scientific contracts; simplicial
alpha/Rips/flag construction, homology and persistence; sequence-hyperdigraph
embedded chains and persistence; two-factor interaction construction and
persistence; ordinary and persistent Laplacians; workflow composition; readers,
feature processing and serialization. Separate reviews covered each topology
family, with direct profiling and output comparisons. External sources were
checked for geometry queries, persistence optimizations and spectral methods.

The existing suite passed **621 tests**, with the two documented construction-cap
warnings. This supports the current tested behavior; it is not a proof of
correctness for every input. Alpha already uses floating-point geometry and
spectral routines already use numerical rank/zero tolerances.

**No files under `src/topokit` were changed.** The changes accompanying this
report are documentation, example-only benchmark prototypes, and their results.
The prototypes are evidence for future implementation, not installed backends.
The checkout has no Git repository metadata, so this is a source inspection and
benchmark audit rather than a Git diff review. Numerical source checksums are
recorded in the spectral benchmark JSON.

## What “unchanged accuracy” requires

There are two different acceptance levels:

1. **Exact reuse or discrete arithmetic:** identical cells, filtration grades,
   coefficient field, interval multiplicities, and observable ordering/provenance
   where the API exposes them. Caching an already computed boundary and reducing
   only the degrees necessary for a query belong here. Work-count diagnostics
   may legitimately change when fewer operations are performed.
2. **Equivalent floating-point algorithms:** the same mathematical operator or
   geometric definition, verified within the existing numerical contract.
   Reordering matrix products can change rounding. Compare eigenpair residuals,
   numerical nullity and repeated-eigenspace projectors, not individual repeated
   eigenvectors or bitwise eigenvalue equality. SciPy explicitly documents that
   even requesting eigenvectors can alter rounding near zero
   ([`eigh` documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html)).

Prefer the first group. A faster method must retain q+1 boundary information for
Hq/Lq, exact weight-tie orientations, overlap identities, filtration endpoint
inclusion, field choice, and the metric on embedded/quotient chains. Smaller
cutoffs, pruning, subsampling, float32, approximate neighbors, or silently
replacing full spectra with partial spectra do not satisfy this request.

## Measured opportunities

Local Python 3.12.2, NumPy 1.26.4, SciPy 1.13.1, macOS ARM64. Final runs used one
OpenBLAS thread. Values are repeated-run medians on a shared desktop, not release
performance guarantees. Timed scope differs by row; speedups cannot be multiplied
or extrapolated to the whole package. Memory figures below describe selected
arrays or Python allocations, not process-wide peak RSS.

| Priority / operation | Workload | Current → prototype | Verified preservation |
| --- | --- | --- | --- |
| High: retain transformed hyperdigraph boundaries | Equal-weight, 24-point 3D construction; 19,938 degree-3 sequences | Chain assembly **608 → 274 ms; 2.22×** | Entire filtered-chain record exactly equal |
| High: stop interaction persistence at requested q+1 | H0 query on prebuilt H2 interaction chain of two 24×24 triangulated grids; 61,325 cells | Query **211 → 6.47 ms; 32.7×** | H0 interval objects, including diagonal pairs |
| High: cache point-ID lookups during overlap validation | Two factors with 4,096 points and 4,096 overlap pairs | Helper **789 → 5.39 ms; 146×** | Exact overlap and global-ID maps |
| High: use interaction L0 diagonal directly | Prebuilt 1,000×1,000 diagonal operator | Solver **5.89 → 0.0040 ms** | Bitwise-equal full eigenvalues; independent degree-count check |
| Medium: batch exact alpha nearest queries | Native full alpha builder, 500 3D points, 13,709 simplices | Build **388 → 267 ms; 1.45×** | Exact filtration and metadata on tested cases |
| Medium: avoid canonicalizing trusted internal simplex tuples again | Complete 27-vertex flag expansion through dimension 3 | Graph setup + expansion **80.9 → 28.5 ms; 2.84×** | Filtration, GF(2/3/5) homology and interval provenance |
| Medium: retain one interaction spectral engine per series | 12-point/24-point factors, nine scales, L0–L2 | Spectral series **505 → 443 ms; 1.14×** | Bitwise eigenvalues; basis, nullity and metadata equal |
| Conditional: existing compact hyperdigraph H1 reducer | Strictly score-ordered 50 vertices, complete supplied support | Analysis **185 → 9.44 ms; 19.6×** | Exact interval multiset using original stored edge births |

Additional measurements: existing-barcode queries give the same Betti tuples
while reducing the time for **persistence plus three Betti queries** by 5.17×
(simplicial), 1.62× (hyperdigraph), and 3.96× (interaction). These measurements
exclude spectra and do not replace chain diagnostics or representative output.
Trusted-tuple simplicial validation retained its checks and improved 8.07×.
Rolling cached interaction validation improved approximately 2.4–2.7× on the
larger fixtures, at the expense of additional Python memory.

For context, the existing 24-point demonstration's build / analysis medians were
10.8 / 10.8 ms for simplicial, 55.8 / 1,045 ms for hyperdigraph, and 42.1 / 329 ms
for interaction. Analysis includes persistence and H0–H2/L0–L2 at three scales;
it excludes file export and plotting. Interval counts remain **47 / 110 / 99**.
These routes have different definitions and matrix sizes, so this is a profile
of the supplied example rather than a comparison of equivalent computations.

## Findings and implementation guidance

### 1. Preserve work already performed by embedded-chain reduction

In [`chain_complex.py`](../src/topokit/core/_hyperdigraph/chain_complex.py),
`_filtered_column_compatible_basis` (lines 178–220) computes a transformed
generator together with its reduced ambient boundary, then returns only the
generator and its birth. `_express_boundaries` computes that boundary again
through `apply_columns` (line 378), traversing all selected original columns.
Large cancellation combinations make this reconstruction expensive.

Return the corresponding exact boundary along with each generator, convert the
row-priority indexing back to ambient face indexing once, and reuse it when
expressing the boundary in the lower Omega basis. Preserve the sort order,
births, coordinate solver, and existing validation. The prototype achieved 2.22×
faster **chain construction**, including the existing checks, and matched the
complete chain dataclass, with 60 additional arbitrary-hyperedge equivalence
cases. This benefits the default equal-weight bidirected
case; it is not limited to acyclic strict-weight data.

A production implementation should pass aligned boundary tuples explicitly
rather than use the prototype's temporary dictionary/monkeypatch. Account for
additional retained bitsets and release intermediates degree by degree. The
memory benefit or cost has not been benchmarked here. Larger stress tests and
focused regression coverage are still required before deployment.

### 2. Honor the requested interaction degree during reduction

[`core/interaction.py`](../src/topokit/core/interaction.py), lines 68–80, checks
`max_dimension` but calls the native reducer on the whole prebuilt chain, then
filters the intervals. The native reducer at
[`persistence.py:128`](../src/topokit/core/_interaction/persistence.py) processes
through `chain.max_homology_dimension + 1` regardless of the lower public query.

Pass the requested maximum into the native computation and reduce only through
q+1. Do not mutate or truncate the supplied object. Since degrees above q+1
cannot change H0…Hq, this removes unnecessary work without changing those
answers. The 32.7× example is specifically an H0 query on an already constructed
H2 chain; an H2 query receives no benefit from this particular change. A dense
18-vertex factor fixture showed an even larger 530× query reduction, but that
special case should not be advertised as a typical package speedup. Diagnostics
should accurately report the degrees actually reduced.

### 3. Reuse barcodes, snapshots, and boundary workspaces

[`workflows/analysis.py:43`](../src/topokit/workflows/analysis.py) computes bars
once but subsequently recomputes homology for every scale. Interaction homology
at [`core/interaction.py:96`](../src/topokit/core/interaction.py) itself computes
the same complete barcode again, solely to count active intervals.

When only Betti numbers are needed, the public API already supports:

```python
bars = core.persistence(topology, max_dimension=2)
betti_by_scale = {t: bars.betti_at(t) for t in observation_scales}
```

Use raw persistence with the intended field and no positive-length interval
filter. Barcode counts follow `birth <= t < death`. Do not infer real Betti
numbers from GF(2) bars, nor claim this reproduces representatives or every
homology diagnostic.

Also, [`core/__init__.py:35`](../src/topokit/core/__init__.py) dispatches each
Laplacian degree independently. The simplicial adapter creates a snapshot for
each degree; its homology adapter constructs the same snapshot twice (lines
85–87). Hyperdigraph snapshots and adjacent-degree boundary parts are similarly
repeated. Build a snapshot and its degree-indexed boundaries once per operation.

The interaction native `InteractionLaplacianEngine` already caches signed
boundaries, but [`core/interaction.py:110`](../src/topokit/core/interaction.py)
creates a fresh engine for every degree/scale. The measured engine-reuse
prototype leaves all outputs identical. Prefer an operation-scoped workspace
owned by the core and orchestrated by workflows. An unbounded global cache is
inappropriate: native simplicial objects and result metadata can be mutable,
and large sparse matrices must be released predictably.

### 4. Keep the diagonal interaction L0 specialization through the public API

The native [`laplacian.py:546`](../src/topokit/core/_interaction/laplacian.py)
already recognizes diagonal L0, and its native `spectrum` method at line 313
sorts the diagonal directly. The public adapter instead converts it to a generic
`LinearOperator`, and [`_spectral.py:63`](../src/topokit/core/_spectral.py)
materializes it using an n×n identity before dense eigensolving.

For a diagonal operator, the complete eigenvalues are just its sorted diagonal:
O(n log n) time and O(n) storage. A 1,000-entry diagonal occupies 8 KB, whereas
just one dense matrix occupies 8 MB. A 2,100-vertex zero-L0 example is currently
refused by the public workspace estimate although its full exact eigenvalue
array needs only 16.8 KB. The resource guard is doing its documented job; the
unnecessary generic backend causes the avoidable refusal.

Expose the diagonal specialization in the shared result adapter, keeping its
existing tolerance, completeness, basis labels and sorting semantics. Do not
blindly delegate every option to native `spectrum`: it uses different zero
cleaning conventions and lacks the public eigenvector interface. Full explicit
eigenvectors still require O(n²) output; partial vectors require O(nk). Budget
those requests separately. This is a solver-kernel improvement; basis-label and
operator-construction costs remain.

### 5. Remove quadratic overlap lookup construction

[`builders/interaction.py:68`](../src/topokit/builders/interaction.py) evaluates
`a.index` and `b.index` inside the overlap-pair loop.
[`PointCloud.index`](../src/topokit/data.py) reconstructs an entire ID dictionary
on every access. For m pairs, this costs O(m(nA+nB)) before geometry begins.

Assign the two dictionaries once before the loop. The work becomes
O(nA+nB+m), with unchanged ID membership, one-to-one validation, errors and
ordering. The timing includes constructing those dictionaries. The benchmark
is for explicit two-cloud overlap validation, not full geometry or the one-cloud
self-interaction path. This is a small, low-risk first implementation change.

### 6. Batch alpha queries and use trusted internal tuple access

At [`builders/_simplicial.py:230`](../src/topokit/builders/_simplicial.py), the
alpha builder calls `cKDTree.query` separately for every circumcenter. Collect
centers in bounded chunks and call the same exact query on each chunk. Keep
`eps=0`, Euclidean distance, the original least-squares and midpoint arithmetic,
scalar squaring, Gabriel tolerance, and descending coface propagation order.
SciPy explicitly accepts arrays of query points
([`cKDTree.query`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query.html)).

The 16.6× nearest-query kernel improvement becomes **1.45× for the full native
alpha builder**, because Qhull, circumcenters, propagation and validation remain.
The prototype compared 132 additional alpha cases and 168 exact/adjacent-float
cutoff cases, including affine rank deficiency, duplicate policies, cospherical
ties, small perturbations, and skeleton caps. This preserves the current
floating-point contract; it does not add exact geometric predicates.

Flag expansion at line 46 repeatedly calls `tree.filtration` on already
canonical internal tuples. `SimplexTree.validate` similarly canonicalizes known
facets again. Use a private trusted lookup while retaining validation at public
input boundaries and all closure/filtration checks. Cache per-degree simplex
lists with proper invalidation if further profiles justify it. Union-find also
gives exact signed graph-incidence rank for finite-field d1; the measured 10,000
vertex path rank kernel improved 2.10×, but its absolute cost was only milliseconds.

### 7. Dispatch specialized hyperdigraph H0/H1 only when its assumptions hold

The hyperdigraph pipeline builds the explicit Omega chain before reaching its
low-dimensional persistence logic. For a face-compatible one-skeleton, H0 can
use vertex/edge events directly with union-find; the 8,000-vertex ring probe
improved from 82.2 to 3.05 ms. The prototype assumes vertices are born at zero,
labels are contiguous integers, edge births are nonnegative, and diagonal
intervals are not requested.
Missing or later-born endpoints need the existing embedded-chain fallback or a
separately proved extension; they cannot be normalized away.

The private compact score-directed backend already handles a useful H1 subset.
Its eligibility must establish a strict total score order and the complete
distinct-vertex consecutive-edge sequence recipe with maximum-edge births.
Reuse recorded edge values exactly. Public integration must also respect the
backend's index limits, zero-born singleton policy and output/provenance
contracts, and must stay within the six-layer dependency rules. The native
specialization is informed by [Dey, Li and Wang](https://arxiv.org/abs/2001.09549);
its eligibility for this sequence-hyperdigraph representation must still be
established from the actual object.

This is **not a replacement for the default equal-weight bidirected route** or
arbitrary explicitly supplied hyperedges. The audit includes a diamond example
where replacing sequence topology with a clique complex changes an H1 death
from 2 to infinity. The broader embedded-chain definition is described by
[Chen et al.](https://arxiv.org/abs/2304.00345). Likewise, graph union-find must not be substituted for the
two-factor interaction quotient's H0.

### 8. Make sparse storage remain sparse through spectral computation

Simplicial [`laplacian.py:35`](../src/topokit/core/_simplicial/laplacian.py)
always forms `Bq.T @ Bq + B(q+1) @ B(q+1).T`, even for a partial spectrum with
no matrix export. Sparse boundaries can yield dense Gram products. In a
700-edge star, 1,400 boundary nonzeros produced 490,000 Laplacian nonzeros:
approximately 22.4 KB versus 5.88 MB of CSR arrays.

Use `x -> Bq.T(Bq x) + B(q+1)(B(q+1).T x)` for existing partial-spectrum requests
when no explicit matrix is requested. The probe's matrix-vector difference was
4.4e-14; this is mathematical equivalence with different rounding, not bitwise
identity. SciPy supports this interface
([`LinearOperator`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html)).
No iterative-solver runtime improvement was measured, and a complete general
spectrum still has a different cost from a partial spectrum. Move predictable
full-spectrum resource checks before expensive Gram assembly.

For genuine persistent simplicial Laplacians, the existing row-space SVD
projection is mathematically appropriate. It currently densifies a global
constraint matrix and correction (lines 87–98), even if `k` is supplied later.
Component-wise projection and matrix-free correction, following the existing
interaction engine design, merit a second phase. Preserve the current global
rank cutoff if splitting an SVD: component-local relative thresholds can change
rank decisions. Avoid casually replacing the SVD with normal equations, which
can worsen conditioning. The relationship to Schur complements is established
in [Mémoli, Wan and Wang, Persistent Laplacians](https://arxiv.org/abs/2012.02808).
This larger change was inspected but not benchmarked or implemented here.

### 9. Secondary opportunities and methods already in place

Interaction reduction can use descending-degree clearing without changing
finite-field barcodes. Prototypes retained zero columns and death pairs across
129 backend/empty/randomized comparisons. Measured improvements ranged from
approximately 1.0× to 1.4×, so degree restriction has higher priority. The
`auto` switch at 8,192 rows ignores column density: integer XOR was 1.85× faster
on one large fixture, but used more Python memory. Use a memory-aware decision,
not a universal switch to integer storage. Cached validation must retain
missing-face, birth-order and boundary-squared checks; the larger validation
rolling-cache prototypes used about 0.27–1.77 MB peak Python allocation, after
avoiding storage of top-degree boundaries that will not be reused.

Hyperdigraph unit vectors represented as `1 << index` also have a storage
pitfall: their aggregate Python integer payload is quadratic in the number of
basis vectors, even for an identity basis. The 8,000-vertex ring allocated about
8.95 MB for Omega integer payloads alone. Symbolic identity blocks and adaptive
sparse/dense columns could reduce this, but require a larger representation
change. Compute adjacent-degree ranks once rather than repeatedly as well;
profiles put this below boundary reconstruction in priority.

Simplicial persistence already has clearing, union-find H0 and GF(2) integer
XOR; Rips already uses exact radius-neighbor construction for finite cutoffs.
Interaction already uses overlap postings instead of a blind Cartesian product,
prefix filtration slicing, component SVD projection and matrix-free operators.
Hyperdigraph already has specialized missing-face bases and incidence-kernel
projection. These should be retained. Ripser's cohomology, clearing and implicit
matrix techniques are useful references for future Rips-specific work, but
Ripser is not a drop-in engine for all TopoKit families
([official implementation](https://github.com/Ripser/ripser),
[Bauer's paper](https://link.springer.com/article/10.1007/s41468-021-00071-5)).

Readers, visualization and optional ML were not dominant targets in the numerical
profiles. For large exported collections, `serialization.to_jsonable` creates
nested lists and `save_result` materializes a pretty-printed JSON string; a
separate versioned binary-array/streaming export could reduce memory. Barcode
vectorization rebuilds names/schema and scans intervals for each requested
degree; sharing immutable fitted schema and grouping intervals once are
reasonable batch improvements. Neither was benchmarked. Preserve typed IDs,
NaN/infinity encoding, bin endpoints, overflow channels, and training-fit scope.
Do not remove copies that protect results from user callbacks merely to save time.

## Recommended implementation sequence and acceptance

1. **First changes:** overlap dictionaries; interaction requested-degree
   reduction; public diagonal spectra; existing-barcode Betti queries and
   operation-scoped spectral reuse. These have clear invariants and limited
   implementation scope.
2. **Next:** retained hyperdigraph boundaries; batched alpha queries; trusted
   internal simplex access. Add focused differential cases around every
   representation and filtration branch before switching defaults.
3. **Then:** eligible H0/H1 dispatch; memory-aware reduction selection;
   matrix-free simplicial and component persistent-Laplacian backends. Keep
   reference fallbacks until numerical/eligibility coverage is adequate.

For discrete changes, compare complete ordered filtrations, interval
multiplicities, supported prime fields, and provenance; retain validation of
the chain identity. For spectral changes, compare the same operator in the
same metric, eigenvalue errors under existing tolerances, residuals and nullity;
test empty spaces, singular constraints, repeated/near-zero modes, and s=t
agreement with ordinary Laplacians. Benchmark end-to-end construction,
analysis, and peak memory separately. Test that resource failures remain
explicit and never silently alter the mathematical input.

The supplied examples and deterministic synthetic fixtures are the evidence
base here. Representative production point counts, dimensions, density, weight
tie frequency and scale schedules are still needed to establish deployment
speedups. General high-order sequence/flag/interaction growth remains
combinatorial; preserving all requested cells places an unavoidable lower bound
on memory and runtime.

## Reproduction and artifacts

From the TopoKit checkout, use the local scientific Python environment:

```bash
python -m pytest -q
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python examples/performance_audit/audit_simplicial.py
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python examples/performance_audit/audit_hyperdigraph.py
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python examples/performance_audit/audit_interaction.py
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python examples/performance_audit/audit_spectral.py
```

Scripts and result files are under [`examples/performance_audit/`](../examples/performance_audit/).
Recorded results: [`simplicial_benchmarks.json`](../examples/performance_audit/simplicial_benchmarks.json),
[`hyperdigraph_audit.json`](../examples/performance_audit/hyperdigraph_audit.json),
[`interaction_results.json`](../examples/performance_audit/interaction_results.json),
[`spectral_results.json`](../examples/performance_audit/spectral_results.json).
The text profiles identify cumulative hotspots; instrumented profile times are
not used as unprofiled benchmark times. Source links/line numbers refer to the
audited 0.3.0 checkout and will need refreshing after implementation changes.


## Later AWS pressure-test evidence — 2026-09-05

The [completed AWS study](../examples/pressure_test/results/aws_2026-09-05/FINDINGS.md)
measures the renamed production algorithms without the prototypes above. It
contains 358 cases (282 successful, 76 explicit guards), with separate q0/q1/q2
and compact H0/H1 matrices. The 30,000-point alpha H0 case spent 92.323 s in
construction and 0.920 s in reduction, supporting the construction-work priority.
A 128-vertex hyperdigraph persistent L0 case exposed a 42,276,004-entry identity
space allocation guard despite a 128-dimensional final basis. The findings
describe the exact equal-singleton-space shortcut and its necessary metadata
and basis-order conditions as a future candidate. No production optimization
was introduced by the pressure study, and its single-seed timings are not
universal capacity or speedup claims.
