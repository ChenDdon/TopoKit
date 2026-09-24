# Initial integration and validation report

Historical 0.1.0 snapshot. File paths and dataset packaging below describe that
version, not the current six-layer layout. See ../MIGRATION_0_2.md and
../VALIDATION.md for the current implementation.

Date: 2026-09-03. Package version: 0.1.0.

## Outcome

The first independently installable `topokit` implementation is working. All
three mathematical routes run from the shared point-cloud input through H0–H2,
ordinary L0–L2 snapshots, numerical features, exports, and visualization. This
is the initial integrated development milestone, not a claim that the entire
planned molecular/materials toolkit or its protocol paper is complete.

The original `simplicial_topo`, `hyperdigraph_topo`, and `Interaction_Topo`
directories remain unchanged as migration references. New development belongs
in `topokit`. The installed package does not import those sibling directories
or depend on a general TDA platform.

## Implemented organization

* `data.py` and `io.py`: validated point coordinates, stable IDs, assigned
  weights, thin file readers, and the packaged coordinate-only fixture.
* `simplicial.py`, `hyperdigraph.py`, and `interaction.py`: separate public
  mathematical facades over distinct internal cores. Static objects and
  point-cloud builders remain separate operations.
* `api.py`, `results.py`, and `workflows.py`: common construction/analysis
  entry points, typed result records, labelled bases, and workflow metadata.
* `features.py` and `serialization.py`: numerical barcode features and portable,
  versioned output. Infinite intervals and out-of-range values are explicit;
  feature construction does not alter raw bars.
* `visualization.py`: points, simplices, directed hyperedges, interaction factor
  views, barcodes, spectra, eigenvector coefficients, and feature heatmaps.
* `examples/`, `tests/`, and `docs/`: static examples, point-cloud workflow,
  ML-ready feature handoff, migrated regressions, public API tests, scientific
  contracts, and an adoption/development roadmap.

All requested initial semantics are retained: unweighted alpha; assigned
weights used only for hyperdigraph orientation; both directions for equal
weights; independently constructed two-factor interaction with explicit
overlap; start-stage clamping with raw-birth provenance; ordinary snapshot
Laplacians by default; and optional genuine two-scale persistent Laplacians.
Eigenvectors and matrix export are opt-in. There is no path-homology module.

## Validation performed

### Source suite

`python -m pytest -q`: **354 passed**, one intentional warning, in 10.74 seconds
on this local Python 3.12 environment. The warning comes from a regression case
that deliberately requests a truncated simplex skeleton; the API correctly
warns that H1 deaths and the L1 upper term may be omitted.

Coverage includes:

* The migrated native-core tests and new facade/dispatch tests.
* Reciprocal equal-weight edges, explicit directed objects, relabelling,
  overlap correspondence, independent factor filtrations, and start stages.
* Boundary/chain identities, higher-degree toy cases, full and partial spectra,
  labelled eigenvectors, and two-scale persistent operators.
* An independent 21-test spectral audit. It assembles deletion matrices
  independently and compares ordinary and persistent operators through degree
  3, including induced chain metrics and signed-component kernel projectors.
* Input validation, coefficient and scale metadata, immutable feature schemas,
  result round trips, resource guards, and the three-route end-to-end workflow.

Tests are evidence for the tested definitions and cases, not a proof of every
algorithm on every possible input.

### Installed wheel

The package was built as `dist/topokit-0.1.0-py3-none-any.whl` and installed into
a temporary virtual environment. Tests ran outside the source checkout with
Python isolated mode (`-I`). The environment reused installed NumPy/SciPy and
other scientific dependencies; it was not an independent clean operating-system
installation.

The wheel smoke test explicitly blocks imports from all three original package
names and from common external TDA packages. All three routes pass, including
the packaged dataset and output writing. No global package installation or
public package upload was performed.

### Numerical fixture and figures

`artifacts/demo/summary.json` reports all three routes as passed. The fixture
contains the supplied 24 coordinates with generic point IDs and uniform
assigned weights. For interaction, the first 12 selected IDs form one cloud;
the full cloud forms the second, with the selected IDs identified as overlap.
No molecular interpretation enters construction or analysis.

| Route | Scale convention | Raw interval count, H0–H2 | Feature-vector length |
| --- | --- | ---: | ---: |
| Simplicial | squared coordinate radius | 47 | 474 |
| Hyperdigraph | coordinate distance | 110 | 474 |
| Interaction | squared coordinate radius | 99 | 474 |

Selected H2/L2 checks demonstrate nontrivial higher-degree outputs:

| Route and snapshot | GF(2) Betti 2 | Real L2 nullity | L2 basis size |
| --- | ---: | ---: | ---: |
| Simplicial, scale 3 | 1 | 1 | 132 |
| Hyperdigraph, scale 2 | 18 | 18 | 84 |
| Interaction, scale 1 | 6 | 6 | 66 |

These are different mathematical constructions and differently calibrated
scales, not interchangeable descriptors. Agreement between Betti numbers over
GF(2) and real nullity in these checks is not a universal identity across fields.

The demo writes raw interval CSV, result JSON, feature JSON/NumPy, and five PNG
views per route. Representative exported figures were visually inspected and
clipped axis labels were corrected. Interaction visualization deliberately
shows the factor complexes and overlap, not an invented geometric embedding
of the full quotient-chain object.

## Issues found and addressed

* The initial higher-order hyperdigraph spectral route had costly dense
  intermediate work. A structural sparse embedded-operator implementation now
  avoids that intermediate, with comparisons against independently assembled
  reference operators. The full fixture L2 at the final filtration has 1,674
  eigenvalues and succeeds locally.
* Sparse-product guards initially checked some storage only after allocation.
  Symbolic support checks now reject excessive products before multiplication;
  regression tests cover ordinary and persistent calculations.
* Public results had inconsistent units/residual labels between cores. Common
  metadata is now aligned while preserving each route's distinct scale units.
* Feature processing could previously accept uncomputed dimensions or a changed
  fitted bin schema. It now rejects these cases and incompatible scale units.
* Partial spectral solves now validate symmetry and handle integer sparse
  matrices. Partial spectra never report an inferred complete nullity.

No remaining blocking issue was found in the completed local validation.

## Limitations and release gates

1. **Numerical alpha geometry.** Construction uses SciPy/Qhull and floating-point
   geometry, not certified exact predicates. Degenerate or omitted-point cases
   must fail explicitly rather than add random jitter. Tiny bars are retained.
   GUDHI-level speed or robustness has not been established; a systematic
   reference comparison and benchmark remains a priority.
2. **High-dimensional cost.** Generic higher-degree calculations are available
   where supported by the object, but combinatorial growth remains substantial.
   Degree q requires the appropriate q+1 boundary information. Resource guards
   prevent known excessive allocations, not every possible out-of-memory case.
   Full spectra and some pairwise operations still require budgeted dense work.
3. **Platforms.** Local validation used Python 3.12 on this Mac. Linux/macOS/
   Windows CI for Python 3.10 and 3.12 is configured but has not run remotely.
4. **Scientific validation.** Broader published-definition comparisons,
   degeneracy tests, benchmark datasets, and protocol case studies are still
   required before publication-readiness claims.
5. **Domain and ML scope.** Weighted alpha, path topology, multiparameter
   persistence, periodic systems, physics-informed domain recipes, batch/HPC
   execution, and predictive-performance experiments are not part of this
   initial milestone. Fixed-feature export is an ML handoff, not a trained model.
6. **Public release.** Toolkit code has the authorized MIT license. Original
   coordinate provenance is retained separately; data redistribution metadata
   should be checked before packaging a public scientific release. Repository
   hosting, package upload, DOI registration, and submission have not occurred.

## How to run and what comes next

From the `topokit` directory:

```bash
python -m pip install -e '.[plot,test]'
python -m pytest -q
python -m topokit demo --output artifacts/demo --plots
python examples/static_objects.py
python examples/features_to_ml.py
```

For repeatable local performance measurements, explicitly control BLAS threads
and record that setting along with the package/environment versions. One demo
run is not a performance benchmark.

The next engineering priorities are systematic alpha validation/performance
work, larger and higher-dimensional regression fixtures, a documented
capability matrix, and reproducible batch/checkpoint workflows. Domain-specific
builders and physics-informed features should then be introduced as separate
recipes, followed by labelled molecular/material case studies and adoption
materials for the protocol paper. See [ROADMAP.md](ROADMAP.md) and
[CONTRACTS.md](CONTRACTS.md).
