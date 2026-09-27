# Scientific reference constants

## Pauling electronegativity

The offline `readers.atomic_properties.PAULING_ELECTRONEGATIVITY` mapping uses
the `Symbol` and `Electronegativity` columns of the [PubChem periodic-table JSON](https://pubchem.ncbi.nlm.nih.gov/rest/pug/periodictable/JSON),
retrieved on 2026-09-03. PubChem explicitly identifies these values as the
[Pauling scale](https://pubchem.ncbi.nlm.nih.gov/periodic-table/electronegativity).
Provider: PubChem, NCBI, US National Library of Medicine, NIH.

SHA-256 of the retrieved complete JSON response:
`ff0f75976583b8a6493b18d0b42e83e1b68bd6e112b45d33578d98494ed5d321`.
The table has 118 element keys. Numeric values are transcribed without
rounding or mixing sources; blank entries become `None`. These are a source
snapshot, not a claim that every listed value is experimentally determined.
For example, C = 2.55, B = 2.04, N = 3.04, and Ne has no supplied value.

`assign_element_weights(cloud)` is an explicit reader-layer enrichment step.
It preserves coordinates and IDs, returns a new `PointCloud`, and records
source provenance. The XYZ reader itself leaves weights uniform. Unknown
symbols and missing values raise an error: no zero, average, or guessed value
is substituted. Finite user overrides are supported and recorded; overridden
numbers must not be presented as PubChem measurements. A caller can use
`PointCloud(..., weights=...)` directly for arbitrary non-elemental weights.

The mapping is a read-only scientific reference constant shipped as Python
code, not a test dataset. Example structures and generated outputs remain
under `examples/`. There is no runtime network request. Updating this snapshot
requires verifying the source, recording its retrieval date/checksum, and
updating tests and the change log.

`examples/data/data_material_from_cif.json` embeds the corresponding Pauling
values for each of its 29 element-labelled sites as explicit example weights.
The JSON reader preserves those numbers; it does not look them up, infer them,
or apply this table implicitly.
