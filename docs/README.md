# Reference documentation

Start with the [package README](../README.md) for installation, basic usage and
[tutorial notebooks](../README.md#tutorial-notebooks). The
[workflow guides](../workflows/README.md) cover Topo protein–ligand and protein–protein
features, optional affinity models and protein–ligand sequence embeddings.

| Reference | What it explains |
| --- | --- |
| [Architecture](ARCHITECTURE.md) | Six layers, dependency boundaries and extension points |
| [Scientific contracts](CONTRACTS.md) | Input identity, construction, filtration units, spectra and feature schemas |
| [Notation](NOTATION.md) | Mathematical symbols and interpretation of output records |
| [Visualization](VISUALIZATION.md) | Object/result plots, stationary views, styling and exports |
| [Native alpha](NATIVE_ALPHA.md) | Geometry, squared-radius births, degeneracy and resource limits |
| [Incremental L0](INCREMENTAL_L0.md) | Ordinary hyperdigraph L0 sweeps and independent vertex deletion |
| [Reference constants](REFERENCE_DATA.md) | Pauling electronegativity snapshot and attribution |

Changes are summarized in the [changelog](../CHANGELOG.md). The
[v0.3.0 release notes](../release-notes/v0.3.0.md) retain the wording shipped
with that release. Current automated checks are available in
[GitHub Actions](https://github.com/ChenDdon/TopoKit/actions).

Examples have their own [data inventory](../examples/data/README.md) and
[protein–ligand provenance](../examples/protein_ligand/README.md). Code, data
and model weights have distinct licensing terms; see [NOTICE.md](../NOTICE.md).
