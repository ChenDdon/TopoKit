# Six-layer architecture

The numerical path is `readers → builders → core → postprocessing`.
Workflows compose these public layers; visualization consumes existing inputs,
objects and results. Shared data/result records, serialization, exceptions and
validation are infrastructure rather than another scientific layer.

| Layer | May use | Responsibility and boundary |
| --- | --- | --- |
| `readers` | Shared data records and format parsers | Parse coordinates and preserve identity/metadata; never select topology |
| `builders` | Data records, object models and geometry helpers | Construct explicit objects or filtrations; never compute homology or ML features |
| `core` | Object models, chain algebra and numerical primitives | Analyze supplied objects; no parsing, geometry construction, plotting or ML |
| `postprocessing` | Numerical result records | Transform bars and spectra; never rebuild topology |
| `visualization` | Data/results and plotting primitives | Display supplied records without recomputing or mutating them |
| `workflows` | Public layers and optional estimators | Compose operations; never copy mathematical algorithms |

Private kernels live under `core/_simplicial`, `core/_hyperdigraph` and
`core/_interaction`. Their public adapters return common result records while
preserving the distinct mathematical objects. The `builders` and `core`
parents load their `simplicial`, `hyperdigraph` and `interaction` route modules
lazily. A core remains usable without importing readers, plotting or ML.

## Readers and builders

A reader returns `PointCloud` with stable IDs, coordinates, weights, units and
available scientific metadata. Raw molecular readers preserve elements,
charges and declared bonds; they do not infer topology or assign chemical
weights. `PointCloud.subset` updates current counts and induced connectivity
while retaining source records. The optional explicit
`readers.assign_element_weights` helper uses the attributed
[Pauling table](REFERENCE_DATA.md).

Add a format with `readers.register_reader(name, callable, extensions=(... ,))`,
or use an isolated `ReaderRegistry`. Format semantics and input validation
belong to the reader. Symmetry expansion, periodic image construction and
physical feature rules are not inferred from stored metadata.

Builders produce a `Topology` containing a defined native object, its point
records and construction metadata. Use `builders.register_builder` for a new
recipe within a supported family; a new mathematical family needs its own
core and dispatch support. Inserting explicitly supplied cells or constructing
a chain basis is object/algebra work; alpha, Rips and directed support selection
are builder work.

The [native alpha builder](NATIVE_ALPHA.md) implements circumspheres, empty-ball
checks, degeneracy repair and coface propagation using NumPy/SciPy primitives.
The optional GUDHI backend is explicit and lazy. Neither backend performs
homology or feature encoding.

## Core and postprocessing

Ordinary and genuine two-scale persistent Laplacians remain separate APIs.
The [incremental L0 sweep](INCREMENTAL_L0.md) consumes an explicit filtration;
it does not choose geometric support or change alpha births. Its guarded
ordinary-L0 optimization leaves higher-degree and persistent algorithms in
their respective cores.

Postprocessing accepts persistence or spectrum records. Barcode bins, spectral
statistics, moments and entropy are numerical transformations. Named custom
callbacks receive copies of the supplied eigenvalues. Complete empty operators,
partial spectra, numerical-zero tolerance and missing values have explicit
[contracts](CONTRACTS.md); no failed computation is silently imputed.

## Visualization and general workflows

Plots are views of authoritative records. Object views select existing cells;
barcode and Betti plots consume recorded intervals; spectral curves consume
already computed summaries. Interaction views display the two factors and
overlap rather than claiming a geometric embedding of their quotient chain.
Plotting imports remain optional and lazy. See [visualization](VISUALIZATION.md).

`workflows.analyze`, `analyze_stationary` and `laplacian_series` compose public
analysis and postprocessing APIs. `workflows.ml` fits estimators only on
caller-supplied training features and targets; it does not invent labels,
split datasets or train during topological analysis.

## Protein–ligand applications

The installed `workflows.protein_ligand_prediction` module supplies the fixed
**Topo** encoder and guarded optional GBDT inference. The encoder calls public
readers, alpha construction and ordinary Hyperdigraph L0. It retains the
15 Å crop, 50 radii, 55 channels and ten statistics producing 27,500 features.
The repository manifest exporter manages input paths, hashes, diagnostics and
output records. These concerns stay outside mathematical cores.

The selected topology predictor averages three external GBDT members, each
with its own fitted scaler. The ESM-2 + CPZ sequence recipe is a separate
1,792-feature modality; its general-set predictor also averages three seeds.
The optional supervised TopoFormer application consumes Topo features using
the selected A03 architecture and its separately recorded training recipe.
Feature extraction, sequence encoding and affinity prediction have different
inputs and assets; see the [workflow index](../workflows/README.md).

PyTorch, Transformers, RDKit, scikit-learn and plotting dependencies remain
lazy. Trained models, encoder weights, dataset memberships, training controllers
and historical study outputs are external assets. The wheel includes library
code and small runtime schemas; source distributions also contain user scripts,
tutorials, examples, skills, tests and this reference documentation. Examples
and generated outputs belong under `examples/`, never `src/topokit`.
