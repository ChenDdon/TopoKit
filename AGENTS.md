# topokit maintenance rules

This package has six scientific layers: readers, builders, core,
postprocessing, visualization, and workflows. Read README.md and
docs/ARCHITECTURE.md before changing their boundaries.

For every software change:

1. Update the root README.md so its introduction, API, scope, and status stay current.
2. Update the Unreleased section of CHANGELOG.md with user-visible changes,
   API or scientific impact, relevant validation and known limitations.
3. Update the affected maintained references under docs/. Record validation
   evidence in the changelog and automated checks. Preserve already
   released notes under release-notes/ unchanged.
4. Add/run relevant tests, including architecture tests for dependency changes.
5. Keep tutorial inputs and demos in examples/, regression-only data in
   tests/fixtures/, and generated example output in examples/output/. Do not put
   these files in src/topokit or bundle them in wheels.

Core analysis accepts explicit well-defined objects. Geometry, file parsing,
feature computation, visualization, and ML do not belong in mathematical cores.
Do not introduce silent perturbation, pruning, feature imputation, field changes,
or train/test splitting. Optional ML and plotting dependencies must remain lazy.

## Repository workflows

Use a separate virtual environment for each checkout, or explicitly reinstall
the intended checkout before testing. Preserve user edits and do not commit
generated outputs, credentials, external model weights or research datasets.
Publishing packages, repositories or datasets requires explicit direction
from the maintainer.

For user feature-generation tasks, read [workflows/AGENTS.md](workflows/AGENTS.md).
The repository provides separate skills for protein–ligand and protein–protein
Topo features, and for protein–ligand ESM-2 + CPZ sequence embeddings. The public
PPI workflow contains the selected topology-only recipe and optional inference;
research runs, training controllers and sequence comparisons remain outside it. These instructions compose the public APIs;
they do not authorize retraining or changing scientific defaults. Example
complexes are under `examples/protein_ligand` and `examples/protein_protein`,
with byte-level provenance and
source-data terms distinct from the MIT code license.

Use **Topo** or **topology features** in user-facing text. Internal recipe IDs,
schemas, hashes and model contracts remain unchanged by display-name edits.

Use **simplicial complexes**, **hyperdigraphs** (or **topological hyperdigraphs**)
and **interaction complexes** as the mathematical family names. Describe the
current interaction implementation's two-factor limit separately in API
contracts. Keep feature dimensions, encoder names and detailed recipes in
workflow guides; the root README covers the toolkit and links to those guides.
Public usage guides state current interfaces and requirements without internal
development narratives or old-versus-new recipe comparisons.
