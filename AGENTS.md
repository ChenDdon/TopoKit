# topokit maintenance rules

This package has six scientific layers: readers, builders, core,
postprocessing, visualization, and workflows. Read README.md and
markdown/ARCHITECTURE.md before changing their boundaries.

For every software change:

1. Update the root README.md so its introduction, API, scope, and status stay current.
2. Append a dated notice to markdown/CHANGELOG.md covering request, changed
   layers/files, API or scientific impact, validation, and unresolved concerns.
3. Update affected notation/contracts/migration notes under markdown/. Keep
   historical records intact; current validation belongs in markdown/VALIDATION.md.
4. Add/run relevant tests, including architecture tests for dependency changes.
5. Keep datasets, demo-only loaders/scripts, and generated example output in
   examples/, not src/topokit. Wheels must not bundle them.

Core analysis accepts explicit well-defined objects. Geometry, file parsing,
feature computation, visualization, and ML do not belong in mathematical cores.
Do not introduce silent perturbation, pruning, feature imputation, field changes,
or train/test splitting. Optional ML and plotting dependencies must remain lazy.

Keep the original sibling research packages as migration references. Active
development is in the sibling `topokit_dev` copy. Do not publish packages, repositories, datasets,
or manuscripts without explicit user direction.

## Repository workflows and working copies

This is the public TopoKit repository. In the maintainer's local workspace,
the sibling `topokit_dev` directory is the working development copy; both install as `topokit`.
Use separate virtual environments or explicitly reinstall the intended copy.
Do not assume that editing one directory updates the other. When synchronizing
a release, compare source/tests/recipes and preserve user edits and historical
documentation; do not copy generated outputs into the runtime package.

For user feature-generation tasks, read [workflows/AGENTS.md](workflows/AGENTS.md).
The repository provides separate skills for fixed FS-AN topology features and
ESM-2 + CPZ sequence embeddings. These instructions compose the public APIs;
they do not authorize retraining or changing scientific defaults. Example
complexes are under `examples/protein_ligand`, with byte-level provenance and
source-data terms distinct from the MIT code license.
