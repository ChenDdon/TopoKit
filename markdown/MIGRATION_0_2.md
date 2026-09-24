# Migration from 0.1 to 0.2

Version 0.2.0 deliberately changes the early development layout. No old
root-level topology modules or duplicate native kernels are kept as shims.

| Previous location/API | Current location/API |
| --- | --- |
| `topokit.io.read_csv/read_xyz` | `topokit.readers.read_csv/read_xyz`, or extensible `read`; current reader dispatch also includes native point-cloud JSON, PDB, MOL2, SDF, MOL, PDBQT, and CIF |
| `topokit.io.demo_point_cloud` | example-only `examples/data_fixture.py:load_demo_cloud` |
| `src/topokit/datasets/` | `examples/data/`; not installed in the wheel |
| `topokit.simplicial.from_points/from_graph` | `topokit.builders.simplicial` |
| `topokit.hyperdigraph.from_points/from_digraph` | `topokit.builders.hyperdigraph` |
| `topokit.interaction.from_points/from_complexes` | `topokit.builders.interaction` |
| `topokit.<route>.homology/persistence/laplacian/persistent_laplacian` | `topokit.core.<route>` or common `topokit.core` dispatch |
| Native `topokit._core` | Private `topokit.core._<route>`; construction helpers extracted to builders |
| `topokit.features` | `topokit.postprocessing` (barcodes and new spectral summaries) |
| `topokit.visualization` single file | Same import name, now a separate package |
| `topokit.workflows` single file | Same import name, now a composition package with optional `ml` |
| `topokit.demo` / `python -m topokit demo` | `python examples/point_cloud.py --plots` |
| `topokit.demo.compact_analysis` | `topokit.workflows.compact_analysis` |
| `artifacts/demo/` | `examples/output/` |
| `docs/` and native PROVENANCE.md files | `markdown/`, `markdown/history/`, `markdown/provenance/` |

Root convenience `topokit.from_points`, `homology`, `persistence`, `laplacian`,
and `persistent_laplacian` remain thin delegates. Prefer explicit layer imports
in new examples and applications. Root import no longer eagerly loads builders.
The route modules listed above are lazy attributes of the `builders` and `core`
parent packages, so parent-first access is supported without relying on a prior
submodule import. Explicit imports such as
`from topokit.builders import simplicial` remain fully supported.
The portable numerical result schema remains version 1; native Python object
pickles were never the supported interchange format.

Historical native point-cloud/adjacency convenience constructors have become
builder functions. Construct-and-compute helpers live under `workflows/<route>`.
Object containers still accept explicit cells/filtrations and validate them;
they do not parse scientific files or choose a geometric construction.

Later 0.3 additions remain under those layer names. Use
`topokit.postprocessing.spectral_variance`, `spectral_moment`/
`spectral_moments`, and `laplacian_energy` for the new predefined spectral
information. The last is centered full-spectrum energy, not the older
uncentered `spectral_energy`; it equals classical graph Laplacian energy for a
combinatorial graph L0 and is a declared generalization elsewhere.
