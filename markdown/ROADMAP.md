# Development and adoption roadmap

## Completed foundation, followed by remaining release work

1. Freeze point IDs, filtration semantics, mathematical namespaces, and results.
2. Migrate and regress the three native cores in an independently installable
   package. Preserve the original source snapshots and notices.
3. Add point-cloud adapters: alpha, tie-aware sequence hyperdigraphs, and two
   independently filtered alpha factors with explicit overlap.
4. Exercise static toy objects and a complete point-cloud route through H2/L2,
   numerical export, fixed-bin features, and visualization for all three cores.
5. Validate optional eigenvectors and genuine persistent Laplacians on small
   cases; record resource limitations and supported optimized backends.
6. Verify a built wheel outside the source checkout, then publish versioned
   installation instructions, examples, result-schema documentation, and tests.

The initial integrated implementation and local wheel validation are complete.
Version 0.2 additionally reorganizes all source into six layers, moves demo data
and execution into examples, and centralizes the current documentation/change
trace in this folder. Public hosting/publication is still a separate action.
Version 0.3 adds sourced XYZ weight enrichment, configurable bond/graph/flag/
digraph builders, paired interaction schedules and endpoint operators, fitted
dataset-wide feature schemas, molecular/crystallographic coordinate readers,
native JSON point-cloud interchange, and scoped spectral summaries including
variance, moments, and Laplacian energy. A complete small configured example is available at
`examples/configured_pipeline.py`; the 24-point coordinate-only example remains
the larger three-route regression.

## Next engineering milestones

* Profile alpha construction and spectral work independently. Optimize measured
  bottlenecks with bulk face construction, geometry caching/batching, and then
  compiled kernels only if justified. Benchmark against optional external
  references with comparable precision settings; no unmeasured speed claims.
* Expand degeneracy, higher-dimensional, relabelling, and matrix-free tests.
* Add batch APIs/checkpointing and HPC scripts using the same library. Record
  construction counts, peak memory, timing, versions, and failures per object.
* Broaden feature encoders and validated physical/domain recipes without moving
  parsing or chemistry into the mathematical cores.
* Extend the molecular/material layer with multi-record collection APIs,
  periodic/symmetry-aware builders, and physics-informed featurization as
  separate, explicitly validated work. The current readers intentionally stop
  at one coordinate record and never infer periodic topology.

## Adoption and protocol-paper readiness

* A five-minute quick start, small static examples, one complete numerical
  point-cloud tutorial, and documented recovery from resource/geometry errors.
* Reproducible config files, versioned fixtures, golden numerical outputs,
  per-dataset provenance, archived environments, and release notes.
* A capability matrix specifying object, operation, coefficients, scale units,
  dimensions, output basis, and limitations. Never conflate missing support with
  a zero scientific result.
* Protocol case studies added only after scientific recipes and external
  datasets are selected. Include expected outputs, troubleshooting, runtime
  ranges, and comparison/ablation evidence.
* ML demonstrations require labelled datasets and appropriate train/test splits;
  a single point cloud establishes usability, not prediction performance.
* Package hosting, public repository creation, DOI registration, and journal
  submission are separate release actions requiring explicit user direction.
