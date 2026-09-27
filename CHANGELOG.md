# Changelog

## Unreleased

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

Validation: the non-DL regression suite passes (1,040 passed, 22 skipped,
one deselected), and wheel/source archive checks pass, including Topo feature
extraction for all ten example complexes. Optional pretrained encoders and
supervised models require separate dependencies and external assets; their
full inference is outside this validation scope.

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
