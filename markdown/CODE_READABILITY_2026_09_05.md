# Code naming and readability review — 2026-09-05

Request: make the overall code easier to understand, using meaningful variable
names and reasonable abbreviations, before running pressure tests on AWS.

All 70 package Python files were reviewed. Local identifiers changed in 45
files; already clear modules were retained. This review changes naming and
short explanatory comments only. It does not implement the earlier efficiency
audit's optimization prototypes.

## Naming policy

Use names that identify the scientific or computational role: source/target
scale, boundary rows, simplex birth, surviving component, or eigenvalues.
Retain short conventional indices (`i`, `j`, `q`), `np`, `ax`, and clear
mathematical notation within a small documented scope. Avoid both opaque
temporary names and long prose-like names.

Existing function/method parameter names, public attributes, dictionary keys,
result schemas, and import names are retained for compatibility. In particular,
a public caller using an existing keyword receives the same behavior.

| Area | Before | After |
| --- | --- | --- |
| Alpha geometry | `coords`, `radius2` | `affine_coordinates`, `squared_radius` |
| Simplicial persistence | `paired`, `clear` | `death_partners`, `clearing_candidates` |
| Persistent boundaries | `a`, `b` | `source_boundary`, `outside_boundary` |
| Hyperdigraph construction | `raw`, `records` | `raw_birth_records`, `effective_birth_records` |
| Sparse hyperdigraph constraints | `mr`, `mc`, `mv` | `missing_rows`, `missing_columns`, `missing_values` |
| Persistent operators | `first`, `last` | `source_snapshot`, `target_snapshot` |
| Spectral workflows | `a`, `b`, `budget` | `source_scale`, `target_scale`, `snapshot_budget` |
| Barcode feature bins | `birth`, `death`, `outside` | `birth_axes`, `death_axes`, `overflow_counts` |
| Spectral summaries | `selected`, `threshold` | `selected_eigenvalues`, `zero_threshold` |
| Interaction plotting | `a`, `b` | `factor_cloud_a`, `factor_cloud_b` |

The same principles were applied to interaction chain construction, GF(2)
reduction, overlap mappings, readers, serialization, and shared validation.
Short comments explain component roots versus surviving births, clearing
pairs, persistent boundary blocks, and duplicate-point mappings.

## Preservation evidence

A complete pre-change source snapshot was taken before any edit. The
structural audit compares Python syntax trees with consistent local renames
reversed. Across 45 changed files, **1,590 identifier references** differ;
the remaining tree structure is unchanged. Parameters, attributes, constants,
keys, operations, function calls, and loop/reduction ordering are protected.

Ten deterministic small scenarios cover simplicial alpha/Rips; interaction
alpha/Rips with full and half overlap; and Delaunay/complete-support
hyperdigraphs with distinct/equal weights. The isolated before/after run checks
filtrations, cells, metadata, diagonal-inclusive intervals, homology snapshots,
full ordinary spectra, and genuine/equal-stage persistent spectra through
degree two. **190 comparisons match exactly, including 268 bitwise-identical
numerical arrays.** There are also 110 internal consistency checks for each
revision. This is bounded regression evidence, not a proof for every input.

The original 763-test suite passes with the same two expected explicit-skeleton
truncation warnings. New harness/equivalence checks and AWS results are recorded
in [VALIDATION.md](VALIDATION.md) when complete.

One historical example, `examples/performance_audit/audit_simplicial.py`, used
source-string extraction for a prototype. Its extraction strings were adapted
to the clearer local names; 30 exact prototype-equivalence cases pass. Stored
historical benchmark results were preserved. Hyperdigraph and spectral audit
prototypes also pass small compatibility checks without editing their results.

## Pressure testing

The independent suite lives in `examples/pressure_test/`; it measures the
renamed production algorithms with explicit point counts, settings, resource
limits, stage timings, CPU, peak RSS, and a 600-second wall limit per case.
Actual native computations are used, not the earlier audit's monkeypatched
prototypes. See the [pressure-test protocol](../examples/pressure_test/README.md).
