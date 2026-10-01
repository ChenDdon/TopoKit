# Changelog

## Unreleased

- Standard mathematical names are simplicial complexes, hyperdigraphs and
  interaction complexes. The package overview links to detailed application
  recipes and lists four method references with DOI links. Workflow guides
  describe supported usage, required assets and output contracts.

- Make PPI test fixtures portable across Windows and Unix line endings while
  retaining strict canonical-byte checks for exported feature schemas.
- Include ten compact protein–protein examples with exactly two chains, A and B,
  one per partner, plus a manifest, usage guide and byte-level provenance.
  All ten produce finite 5,040-value Topo features with the default recipe.
  Source archives include the structures; wheels remain data-free.
- Add the topology-only protein–protein workflow: the selected unaugmented
  sparse-radius recipe produces 5,040 features with a 20 Å Cα interface crop,
  14 alpha radii from 1.0 to 7.5 Å and 36 ordered atom-category channels.
  Provide an installed API, portable manifest exporter, default recipe and
  assistant skill, without experiment directories or sequence results.
- Provide optional three-seed GBDT inference for signed binding free energy
  in kcal/mol, with pinned external model identities, runtime checks and
  feature-store validation. Trained pipelines remain separate assets.
- Use **Topo** for protein–ligand topology features and **ESM-2 + CPZ** for
  sequence embeddings in user guides. Numerical recipes, internal compatibility
  identifiers, schemas and model contracts remain unchanged.
- Provide concise installation, usage and optional-model summaries, portable
  contributor instructions, and current API/scientific references under `docs/`.
- Keep training settings in recipes and model requirements, evaluation and
  limitations in model cards. Remove internal execution and delivery records,
  stale research commands and redundant strategy records from public guides.
- Retain three tutorial notebooks, useful demos, ten example complexes and
  default feature recipes. Move regression-only data to `tests/fixtures/` and
  remove fixed-cohort research controllers and generated outputs from the
  public distribution.

Validation: the non-DL regression suite passes (1,095 passed, 22 skipped,
one deselected, plus 240 subtests), including 55 PPI checks. Wheel and source
archive checks pass, including installed PPI extraction, source-archive PPI
export and Topo extraction for all ten protein–ligand examples. PPI sparse
features match the retained original columns exactly. Separate replay with
the three original GBDT pipelines reproduces saved predictions within
1e-12 kcal/mol, including a structure-to-prediction replay. Optional pretrained
encoders and other supervised models remain outside this validation scope.

## 0.3.0 — First GitHub release

- Six composable layers for simplicial, sequence-hyperdigraph and two-factor
  interaction topology, with homology, persistence, Laplacians and features.
- Native NumPy/SciPy alpha construction and the 27,500-feature protein–ligand
  Topo recipe, with a manifest exporter and ten example complexes.
- Three tutorial notebooks, optional ESM-2 + CPZ sequence embeddings and agent
  workflow guides. Pretrained weights and fitted models remain separate assets.
- Wheel/source distribution checks and Linux, macOS and Windows CI for
  Python 3.10 and 3.12, including isolated installed-wheel smoke tests.

See the [original release notes](release-notes/v0.3.0.md) and
[GitHub release](https://github.com/ChenDdon/TopoKit/releases/tag/v0.3.0) for the
historical release contents and validation scope.
