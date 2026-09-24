# topokit documentation and trace

The root README is the current introduction. This folder contains the detailed
records; no scientific notation, migration reports, or datasets belong in
`src/topokit`.

For current user workflows, start with the [standard package guide](../README.md),
[FS-AN manifest exporter](../workflows/protein_ligand_prediction/README.md),
[ESM-2 + CPZ recipe](../workflows/protein_ligand_prediction/sequence/README.md),
and [agent instructions](../workflows/AGENTS.md). The three notebook tutorials
are linked from the package guide; ten real example pairs and checksums are in
[examples/protein_ligand](../examples/protein_ligand/README.md).

* [ARCHITECTURE.md](ARCHITECTURE.md): six layers and dependency boundaries.
* [NOTATION.md](NOTATION.md): mathematical notation and output interpretation.
* [CONTRACTS.md](CONTRACTS.md): identity, filtration, algebra, and numerical rules.
* [CONTRACTS_0_3.md](CONTRACTS_0_3.md): explicit bonds, paired schedules, weights, fitted bins, and spectral defaults.
* [REFERENCE_DATA.md](REFERENCE_DATA.md): offline elemental-property values and source provenance.
* [CHANGELOG.md](CHANGELOG.md): append-only change notices and validation trace.
* [MIGRATION_0_2.md](MIGRATION_0_2.md): old-to-new API and file locations.
* [VALIDATION.md](VALIDATION.md): current validation status and known limitations.
* [VISUALIZATION.md](VISUALIZATION.md): object views, reusable dimensional blocks, style/camera controls, semantics, and vector export.
* [CODE_READABILITY_2026_09_05.md](CODE_READABILITY_2026_09_05.md): local-name refactor and exact before/after evidence.
* [Pressure-test protocol](../examples/pressure_test/README.md): generated-case matrix, 600-second wall limits, AWS state, and raw measurements.
* [AWS pressure findings](../examples/pressure_test/results/aws_2026-09-05/FINDINGS.md): completed 358-case results, verified numerical evidence, separate low/high dimensions, and optimization candidates.
* [PERFORMANCE_AUDIT_2026_09_05.md](PERFORMANCE_AUDIT_2026_09_05.md): algorithm review, measured exact-result optimization prototypes, numerical safeguards, and implementation priorities.
* [ROADMAP.md](ROADMAP.md): remaining engineering, adoption, and paper milestones.
* `provenance/`: implementation migration attribution.
* `history/`: historical reports and original source checksums; not the current API.

For every source change, update the root README (including changed status or
limitations) and append a dated CHANGELOG entry: request, files/layers affected,
scientific/API impact, tests run, and unresolved concerns. A template is included
in the changelog. Never overwrite historical validation claims with new counts.

- [Final protein–ligand topology ML](PROTEIN_LIGAND_ML_FINAL.md): unchanged FS-AN features, separate historical feature-selection parameters, final GBDT hyperparameters, three-seed consensus and model/scaler retention.
- [Protein–ligand notation and historical migration](PROTEIN_LIGAND_FINAL.md): original r4 notation and September 16 feature grid, refined/CASF memberships, and updated v2020R1 naming; current selection is linked explicitly.
