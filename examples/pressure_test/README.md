# TopoKit pressure tests

This folder contains reproducible pressure tests for the installed TopoKit
algorithms, their AWS run records, and evidence for the preceding readability
refactor. Historical sibling-package benchmark results are not reused as
measurements of this package.

## Scientific matrix

The canonical main matrix has eight profiles, each tested for requested
degrees 0, 1, and 2 with persistence, ordinary Laplacians, and two-scale
persistent Laplacians. Every canonical case constructs through q+1; Hq deaths
and the up-Laplacian are not removed to make a case fit. A persistence request
for q returns H0 through Hq; a spectral request measures Lq. Persistence uses
GF(2); spectra use the existing real-valued operators and numerical tolerances.
Low-dimensional persistence has its own scaling sizes, separate from complete
L0/L1 spectra and from H2/L2.

The following sizes apply to **ordinary and persistent Laplacians**. Their q2
column also gives the unchanged H2 persistence sizes. `PROFILES` in `cases.py`
is the source of these settings.

| Profile | Interpretation | Spectral point counts for q0 / q1 / q2 |
| --- | --- | --- |
| Simplicial alpha | Unweighted alpha; squared-radius births | 1000/5000/20000; 128/512/2048; 32/128/512 |
| Simplicial Rips | Complete distance support, no cutoff; edge-distance births | 256/1024/4096; 32/64/128; 16/32/64 |
| Hyperdigraph Delaunay, distinct weights | Delaunay support, one orientation per pair | 512/2048/8192; 32/128/512; 16/32/64 |
| Hyperdigraph Delaunay, equal weights | Same support, both orientations | Same sizes |
| Hyperdigraph complete, distinct weights | Every pair, strict score-directed DAG | 128/512/2048; 16/32/64; 8/16/32 |
| Hyperdigraph complete, equal weights | Every pair, both orientations | Same sizes |
| Interaction alpha | Two independent alpha factors, full explicit ID overlap | 128/512/2048; 32/128/512; 16/64/256 |
| Interaction Rips | Two independent Rips factors, full explicit ID overlap | 32/128/512; 12/32/64; 8/16/32 |

`PERSISTENCE_SIZES` in `cases.py` overrides only the canonical main matrix's
H0 and H1 persistence sizes:

| Profile | H0 point counts | H0–H1 point counts |
| --- | --- | --- |
| Simplicial alpha | 1000 / 10000 / 30000 | 512 / 2048 / 8192 |
| Simplicial Rips | 256 / 1024 / 1536 | 32 / 128 / 192 |
| Hyperdigraph Delaunay, distinct weights | 1024 / 8192 / 32768 | 128 / 1024 / 8192 |
| Hyperdigraph Delaunay, equal weights | 512 / 4096 / 16384 | 64 / 256 / 1024 |
| Hyperdigraph complete, distinct weights | 128 / 384 / 768 | 24 / 64 / 128 |
| Hyperdigraph complete, equal weights | 96 / 256 / 512 | 16 / 40 / 64 |
| Interaction alpha | 512 / 4096 / 16384 | 64 / 256 / 1024 |
| Interaction Rips | 128 / 512 / 1024 | 16 / 64 / 112 |

There are 216 canonical main cases for one repetition. The saved
`matrices/pressure_q0.json`, `pressure_q1.json`, and `pressure_q2.json` contain
72 cases each; together they are exactly `matrices/pressure.json`. The
separation allows low-dimensional measurements to finish before H2/L2 work.
Each three-size series includes a stress point that may reach a guard or
process limit; these are planned attempts, not established feasible capacities.

The original `smoke` matrix has six points per cloud and covers all 72
profile/operation/degree combinations. The 54-case `supplemental` matrix uses
the middle spectral-profile size, requests eight eigenvalues explicitly, and
uses half overlap for interaction, including interaction persistence. Partial
spectra are never substituted automatically when a full-spectrum case fails.
A request for eight modes can still return a complete spectrum when the
operator has at most eight dimensions; the actual `complete` flag and basis
size are recorded.

Delaunay-supported hyperdigraphs are not alpha complexes, and complete-support
hyperdigraphs are not Rips complexes. Their higher cells are ordered
distinct-vertex sequences supported by consecutive directed edges, with
maximum consecutive-edge-distance birth. No directed-clique replacement is
made. Point weights only affect hyperedge orientation.

Small dense cases can contain far more high-dimensional cells than large
sparse clouds. Point count, support edges, directed edges, cells by degree,
factor sizes, and actual operator basis sizes are recorded separately.
Interaction point counts mean points **per factor**, with two independent
clouds and explicit full/half overlap pairs.

## Existing low-dimensional algorithms and the compact route

The canonical builder/core route records the algorithms the package actually
selects:

- Simplicial H0 uses union-find. Higher degrees use dimension-ordered boundary
  reduction with clearing and GF(2) bitsets. A highest requested degree of one
  still requires triangle reduction; H1 is not a union-find computation.
- Canonical hyperdigraph persistence first builds the filtered Omega chain.
  When every edge is born no earlier than its singleton faces, its native
  H0/H1 route uses union-find and fundamental-cycle coordinates with native
  Omega2 boundaries (`modified_dlw_native_omega2`). These point-cloud fixtures
  satisfy that condition, including the equal-weight cases. H2 uses ordinary
  filtered-boundary reduction.
- Interaction persistence uses native GF(2) boundary reduction. Its automatic
  backend chooses integer bitsets for row counts up to 8192 and sparse sets
  above that threshold; it does not automatically use simplicial union-find
  or clearing.

Persistence shortcuts do not automatically accelerate Laplacian spectra.
Full spectral cases retain the real operator construction and complete
eigensolve. Interaction L0 has an existing diagonal operator construction,
but the public full-spectrum route still passes through the generic spectral
solver. Algorithm and solver diagnostics, basis size, and completeness are
recorded so a cheap operator kernel is not confused with a cheap complete
analysis.

A separate suite measures the existing public workflow
`topokit.workflows.hyperdigraph.compute_persistence_from_point_cloud` using
`omega_backend="auto"`, `reduction_backend="native_h1"`, and
`storage_backend="numpy"`. This compact route is eligible here only for
**distinct direction weights and H0/H1 requests** on Delaunay or complete
support, with no distance cutoff. It keeps packed vertex/edge arrays and
avoids an explicit collection of directed two-paths. Its H0 backend is
`score_dag_union_find_early_stop`; H1 uses implicit triangle/quadrangle boundary
families (`modified_dlw_implicit_two_family`). Equal-weight, H2, and spectral
cases remain in the canonical matrix; no fallback between routes is hidden.

`COMPACT_SIZES` declares these additional cases:

| Compact support | H0 point counts | H0–H1 point counts |
| --- | --- | --- |
| Delaunay | 5000 / 20000 / 60000 | 1000 / 5000 / 20000 |
| Complete | 1000 / 3000 / 10000 | 300 / 1000 / 5000 |

`matrices/compact_smoke.json` contains four eight-point cases;
`matrices/compact_pressure.json` contains 12 cases. Both use the explicit
`compact_hyperdigraph` execution route. Stored vertex/edge counts exclude
implicit two-paths and are labelled accordingly; they are not interchangeable
with materialized canonical hyperedge counts.

## Reproducible inputs and observations

`numpy.random.default_rng(20260905).random((n, 3))` generates unit-cube
coordinates. Interaction factors use successive independent draws from the
same generator. Distinct weights are `arange(n)` and equal weights are
`ones(n)`. Cases at the same size/settings use identical inputs across
operations and repetitions; no benchmark-specific coordinate perturbation
or topology pruning is applied. Each record contains input and source hashes.

Persistence consumes all actual filtration events. Spectral observations are
selected independently in native units: source and target are the entries at
`floor(0.50*(m-1))` and `floor(0.80*(m-1))` of the sorted positive unique
finite births. Ordinary spectra use the target; persistent spectra use both.
Exact thresholds, indices, and active cells are recorded. An object with no
positive births uses zero and records that fallback. The before/after
validation also checks equal-stage persistent versus ordinary operators.
All vertices in these generated cases are born at zero. Consequently the
two-scale L0 fixtures have the same degree-zero domain at both stages and
equal the ordinary target L0; they do not exercise vertices appearing between
the stages.

## Measurement protocol

Each measured case runs in a fresh process group, sequentially, with one
BLAS/OpenMP thread. The **600-second wall limit applies to each case**, from
process launch through interpreter imports, input generation, support
preparation, construction, observation selection, analysis, validation, and
numerical export. Controller-only preparation of the input JSON is excluded.
Canonical stage timings separately identify generation, support, construction,
analysis, and export. The compact public API exposes one combined
`construction_and_analysis` timing, including support geometry, packed edge
construction/sorting, and persistence. Its canonical equivalence probe is
timed separately; both are included in the total case wall limit. A result
first observed after the deadline is conservatively classified as a timeout;
the observed kill/wait overhead is also recorded.

Each successful compact case includes a check against the canonical route on
the same seeded coordinates and support policy at `min(n, 8)` points. For a
larger case this is an independently rebuilt small prefix, not verification
of all its intervals or the induced subgraph of its full Delaunay support.
The comparison records interval counts/degrees, infinite-endpoint masks,
exact endpoint hashes, differences, and a separate tolerance verdict
(`rtol=1e-12`, `atol=1e-14`). Distance evaluation order can change final float
bits; endpoints are not rounded. Passing this bounded probe is limited
equivalence evidence, not a general proof of numerical accuracy or a claim
that large-case endpoints match bitwise.

On Linux the parent samples current `/proc/PID/status` RSS every 0.05 seconds
and applies a 16-GiB RSS ceiling. A 22-GiB virtual-address-space limit provides
a second bound. `wait4` captures user/system CPU and peak RSS for the entire
child lifetime, including killed workers. Peak RSS is reported in MiB/GiB,
with platform-specific `ru_maxrss` units converted to bytes. Sampled RSS and
the OS high-water mark are distinct fields. No unrelated AWS process is killed.

The canonical route's explicit budgets are recorded in its cases: one million
simplices/hyperedges, two million interaction-cell witness upper bound,
4-GiB hyperdigraph chain bound, 16 million dense entries, 4-GiB interaction
dense-workspace bound, and 20 million hyperdigraph sparse entries. Only limits
supported by each public API are passed. These guards remain active; raising
them does not imply that every case fits memory. Defaults for ordinary package
users are unchanged.

The compact workflow exposes none of those canonical allocation/reduction
guard arguments. Its cases therefore have `limits={}` and rely on the parent
wall/RSS/address-space limits, plus intrinsic packed-storage caps of 65,535
vertices and 2,147,483,647 edges. These are process-limited measurements, not
measurements protected by the canonical one-million-hyperedge or chain guards.
The separate eight-point canonical probe uses the canonical hyperedge/chain
guards and records them independently.

Terminal statuses distinguish success, package resource-limit rejection,
memory failure, timeout, RSS stop, numerical-check failure, application error,
and controller/worker failure. A guard rejection is not a timed successful
analysis. A timeout is a censored observation, not a completed 600-second
runtime. Large cases are attempted independently; failures do not trigger
hidden cutoffs, lower degrees, alternate backends, or partial eigenvalues.

Per-case checkpoints are replaced atomically after each stage. Case locks are
inherited by workers, preventing overlapping attempts even if a controller
exits unexpectedly. Abandoned logs/checkpoints are preserved before a new
attempt. Resume requires the same case matrix, Python interpreter, source
hashes, and limits. Actual numerical outputs are saved separately from timing
summaries; matrices/eigenvectors are not requested by pressure cases.

## Run

```bash
# Run from the package root with the desired scientific Python environment.
PYTHONPATH=src python examples/pressure_test/run_pressure.py \
  --suite smoke --output examples/pressure_test/results/local_smoke

PYTHONPATH=src python examples/pressure_test/run_pressure.py \
  --matrix examples/pressure_test/matrices/pressure_q0.json \
  --wall-seconds 600 --rss-gib 16 --address-gib 22 \
  --output examples/pressure_test/results/aws_2026-09-05/pressure_q0

PYTHONPATH=src python examples/pressure_test/run_pressure.py \
  --matrix examples/pressure_test/matrices/compact_smoke.json \
  --wall-seconds 600 --rss-gib 16 --address-gib 22 \
  --output examples/pressure_test/results/local_compact_smoke
```

Use `--profiles` to select declared profiles, `--repeats` for independent fresh
process repetitions on the same input, and `--matrix` for a saved explicit case
list. Degree-split and compact runs use `--matrix`; they are not `--suite`
names. `--max-cases` pauses after a chosen number of new cases; `--resume`
continues an identical run. The suite's total duration can exceed ten minutes
because the limit applies separately to each case.

The user has authorized the source upload and AWS execution. The entry point is:

```bash
bash examples/pressure_test/run_aws.sh \
  /absolute/path/to/venv/bin/python /absolute/path/to/results
```

It runs the full tests and isolated before/after verification, then the saved
matrices in this order: `smoke`, `compact_smoke`, `compact_pressure`, `pressure_q0`, `pressure_q1`,
`pressure_q2`, `supplemental`. Each measured case has its own 600-second limit.
The script retains logs and installed versions and performs no upload or
instance provisioning itself. The report generator
`summarize.py RUN_DIRECTORY --output RESULTS.md` verifies numerical-file hashes
and reports completed, failed, censored, and unrun cases separately.

Files in each run directory:

- `manifest.json`: complete case matrix, source hashes, machine/runtime, limits,
  progress, and final status counts.
- `results.csv` and `results.jsonl`: readable table and complete raw records.
- `cases/<case_id>/`: case configuration, stage checkpoints, terminal record,
  stdout/stderr, returned numerical arrays, and traceback when applicable.

Run reports must separate local calibration from AWS measurements, identify
single-trial timings where repeats were not performed, and retain failed
cases. No generalized capacity limit or algorithm-speedup claim follows from
one seeded cloud or one machine.

## Readability checks

```bash
python examples/pressure_test/check_readability.py \
  --baseline-src /path/to/pre-change/src --current-src src \
  --output examples/pressure_test/readability/verification.json
```

The structural audit permits only consistent local-identifier renames and
comments/docstrings. Function arguments, attributes, strings/keys, numerical
expressions, calls, and statement order are protected. Isolated processes
compare baseline/current scientific outputs on seeded small examples of all
three families. Their complete payloads and individual comparisons are saved,
including bitwise-array equality and independently labelled tolerance checks.

The completed AWS before/after verification passed all 190 scientific-output
comparisons, including 268 bitwise-identical arrays. This bounded refactor check
is separate from pressure measurements and compact-versus-canonical probes.
All seven AWS suite reports are now complete and verified.

See `markdown/CODE_READABILITY_2026_09_05.md` and `markdown/VALIDATION.md` for
the completed refactor and validation record.


## Completed AWS run, 2026-09-05

All seven suites completed: **358 terminal cases, 282 successes, 76 explicit
resource guards**, with all 282 numerical artifacts verified. There were no
measured timeouts, memory stops, application errors, or numerical-check failures.
These totals include 76 small smoke cases; the pressure/supplemental subset has
282 cases, with 206 successes and 76 guards. Actual runtime remained below
166.504 seconds per case; observed peak RSS reached 2,138.7 MiB. Declared native
guards can bind before the wall or process memory ceilings.

Read [findings](results/aws_2026-09-05/FINDINGS.md), the
[complete overview](results/aws_2026-09-05/OVERVIEW.md), and
[combined CSV](results/aws_2026-09-05/combined.csv). The standalone
`aggregate_aws.py PARENT_DIRECTORY` regenerates `OVERVIEW.md`, `combined.csv`,
and `combined.json` from local per-suite evidence while verifying artifacts.
It never contacts AWS or runs computations; missing suites remain visibly unrun.
