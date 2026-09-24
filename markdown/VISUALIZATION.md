# Object visualization, stationary representations, and reusable building blocks

TopoKit displays explicit simplicial cells with filled triangular faces and
sequence hyperedges with colored ribbons and directional arrows. Dark nodes,
thin outlines, optional labels, and orthographic views support small scientific
schematics. Object and single-block functions return Matplotlib axes; the
gallery returns a figure and a flat array of axes. Plotting never computes
homology, changes a filtration, or saves/shows a figure implicitly.

Install the optional plotting dependency with `pip install -e '.[plot]'`.
Matplotlib remains lazy: importing `topokit.visualization` does not import it.

## Persistence and spectral-result curves

The result-curve functions display records supplied by the caller. They do not
call a builder, recompute persistence or a Laplacian, or calculate spectral
summary statistics inside the visualization layer.

```python
from topokit import postprocessing, visualization as viz

# `bars`, `scales`, and `spectra` were computed by the application.
viz.plot_barcodes(bars, mark_initial_stage=False)
viz.plot_betti_curves(bars, scales, dimensions=(0, 1))

summaries = postprocessing.summarize_spectra(
    spectra,
    statistics=("min", "max", "mean", "laplacian_energy"),
    positive_only=False,
)
summary_axis, energy_axis = viz.plot_spectral_summary_series(
    summaries,
    scales=scales,
)
```

`plot_barcodes(result, ..., mark_initial_stage=True)` retains the default
open-circle cue for intervals already present at the recorded filtration
start. This is useful when the left endpoint is only a truncation boundary.
Set `mark_initial_stage=False` when the displayed births are known to be exact,
as in the tutorial's H0 filtrations whose vertices are born at zero. The option
changes markers only; it never changes interval coordinates. Right-pointing
markers continue to denote infinite death in the supplied record; in a bounded
filtration this is right-censoring at the recorded cutoff.

`plot_betti_curves(result, scales, dimensions=...)` queries the supplied
`PersistenceResult` at a finite, strictly increasing caller-defined scale grid.
The grid must remain inside the result's recorded domain. It counts the stored
intervals through `PersistenceResult.betti_at`; it does not rerun persistent
homology or turn a birth-death histogram into a Betti curve.

`plot_spectral_summary_series(summaries, ...)` accepts an ordered sequence of
aligned `VectorResult`-like summaries. Supply one strictly increasing scale per
record explicitly, or retain one in each record's metadata. By default the
primary axis shows `min`, `max`, and `mean`, and a secondary axis shows
`laplacian_energy`. Undefined NaN values remain visible as gaps. The helper
requires all records to share one ordered schema and, when provided, one
Laplacian dimension; it only displays their existing values and returns both
axes.

### Interpreting the ordinary snapshot spectra

The molecular-format tutorial includes a short ordinary-Hodge primer. For a
real chain complex,

\[
L_q=\partial_q^*\partial_q+\partial_{q+1}\partial_{q+1}^*,
\]

so every eigenvalue is nonnegative. Numerical zero modes describe the harmonic
kernel over the operator's coefficient field. Small positive modes are the
lowest non-harmonic Hodge energies and can be sensitive to global coupling or
bottlenecks; large modes describe high-Hodge-energy, more local variation.
Terms such as “soft,” “stiff,” “flexible,” or “interaction strength” are only
structural analogies unless the chosen weights and units encode the relevant
physics. The tutorial's values are not molecular energies.

Its six-scale curves are a series of independently evaluated ordinary
Laplacians \(L_q(r)\). They are not spectra of the genuine two-scale operator
returned by `core.persistent_laplacian(start=s, end=t)`. Request the latter
explicitly, or use
`workflows.laplacian_series(mode="persistent", scale_pairs=[(s, t)])`.

### Descriptor reference

For a complete spectrum \(\lambda_1,\ldots,\lambda_n\), the four displayed
curves are:

| Curve | Summary name | Definition |
| --- | --- | --- |
| Minimum | `min` | \(\min_i \lambda_i\) |
| Maximum | `max` | \(\max_i \lambda_i\) |
| Mean | `mean` | \(n^{-1}\sum_i \lambda_i\) |
| Centered Laplacian energy | `laplacian_energy` | \(\sum_i\lvert\lambda_i-\bar{\lambda}\rvert\) |

The tutorial requests `positive_only=False`, so its minimum, maximum, and mean
include numerical zero modes. Centered Laplacian energy always uses every mode
of a complete spectrum. Its full reference table distinguishes these additional
built-in or settings-derived features from custom calculations:

| Descriptor | TopoKit status |
| --- | --- |
| Numerical zero multiplicity | Built in as `zero_count` |
| Positive-mode count | Built in as `positive_count` |
| Smallest positive eigenvalue | `min` with `positive_only=True` |
| Maximum, sum, mean, and median | Built in as `max`, `sum`, `mean`, and `median` |
| Variance and population standard deviation | Built in as `variance` / `spectral_variance` and `std` |
| Raw population moments | `moment_1` through `moment_4`; use `spectral_moment` in a custom mapping for other orders |
| Uncentered spectral energy | Built in as `energy`; the array API is `spectral_energy` |
| Centered Laplacian energy | Built in as `laplacian_energy` for a complete spectrum |
| Trace-normalized spectral entropy | Built in as `spectral_entropy`; alternative normalization is custom |
| Non-median quantiles, IQR, and unnormalized power sums | Custom scalar callables |
| Inverse positive-spectrum sum and log pseudo-determinant | Custom positive-spectrum callables |
| Heat trace at a chosen diffusion time | Custom full-spectrum callable |

Here `spectral_energy` is the uncentered sum of absolute eigenvalues; it is not
the centered `laplacian_energy` shown in the tutorial. In a `statistics=`
sequence, its built-in name is `energy`; `spectral_energy` is the public
array-level helper. Built-in `spectral_entropy` returns zero for a zero or empty
selected spectrum and locally clips accepted tolerance-small negative modes to
zero before trace normalization.

### Custom scalar summaries

The notebook provides an executable custom-mapping example. A minimal version
is:

```python
import numpy as np
from topokit import postprocessing

def inverse_positive_sum(values):
    return float(np.sum(1.0 / values)) if values.size else np.nan

custom = postprocessing.summarize_spectrum(
    spectrum,
    statistics={
        "positive_q25_linear": lambda values: (
            float(np.quantile(values, 0.25, method="linear"))
            if values.size else np.nan
        ),
        "inverse_positive_sum": inverse_positive_sum,
    },
    positive_only=True,
)
```

Every mapping value receives its own copy of the one-dimensional eigenvalue
selection and must return one real scalar. `positive_only=True` supplies modes
strictly above the resolved numerical-zero tolerance; `False` supplies every
mode in the spectrum record. The choice applies to the whole summary call,
apart from built-ins with documented full-spectrum behavior such as
`zero_count` and `laplacian_energy`. Put parameters in stable feature names—for
example, `positive_q25_linear` or `heat_trace_t1`—because callback internals are
not encoded automatically. Undefined and non-finite values remain explicit;
TopoKit does not impute them.

### Across-filtration curve descriptors

Area under the curve, mean and spread over scale, sampled range, total
variation, extremum location, and the largest sampled slope operate on a
sequence of per-scale summary values, not on one `SpectrumResult`. They are
therefore separate application-level calculations, after `summarize_spectra`,
rather than custom `summarize_spectrum` callbacks. Their values depend on the
scale grid and the declared interpolation rule. In particular, a Betti-curve
integral is exact only when the grid contains every relevant birth and death
event and the integration rule respects its step convention.

## Stationary representation API

The stationary API provides the compact two-dimensional views used by the
point-cloud tutorial without requiring each notebook to define its own plot
functions. It consumes an existing `Topology`, selects explicit cells at one
inclusive scale, and never constructs or mutates topology.

```python
from itertools import combinations
from topokit import PointCloud, builders, workflows, visualization as viz

cloud = PointCloud(
    [[0.0, 0.0], [0.4, 0.0], [0.2, 0.3]],
    ids=("a", "b", "c"),
)
obj = builders.from_points(
    cloud,
    kind="simplicial",
    max_dimension=1,
    complex_type="flag",
    bonds=tuple(combinations(cloud.ids, 2)),
    cutoff=0.55,
    max_scale=0.55,
)
style = viz.StationaryStyle()

axis = viz.plot_stationary_simplicial(
    obj,
    scale=0.55,
    proximity_radius=0.275,
    style=style,
)
result = workflows.analyze_stationary(
    obj,
    scale=0.55,
    dimensions=(0, 1),
)
figure, axes = viz.plot_stationary_diagnostics(result, style=style)
```

The representation-specific functions have distinct display contracts:

| Function | Display contract | Return value |
| --- | --- | --- |
| `plot_stationary_graph` | Explicit points and 1-simplices only; no filled faces or proximity circles | axis |
| `plot_stationary_simplicial` | Explicit 0-, 1-, and 2-simplices; optional exact-radius circles | axis |
| `plot_stationary_hyperdigraph` | Directed 0-, 1-, and 2-hyperedges; deduplicated consecutive oriented segments and optional weight colors | axis |
| `plot_stationary_interaction` | Two retained factor complexes with explicit overlap rings; no inferred quotient embedding | `(figure, axes)` |
| `plot_stationary` | Dispatcher selected from the object kind or an explicit `representation=` | representation-specific value |

`StationaryStyle` stores a restrained blue/green/orange palette, cividis weight
colors, circle styling, node sizes, widths, and arrow curvature. It does not
modify global Matplotlib settings. `proximity_radius=` draws one background
circle with exactly that radius around each point in the simplicial,
Hyperdigraph, and interaction views. It is optional because the radius can use
different native units from the topology's stored filtration scale.

In a stationary Hyperdigraph view, each consecutive orientation is drawn at
most once within the directed-1 and directed-2 display layers. The directed-2
segment is a wider translucent orange underlay and the directed-1 segment is a
narrow blue arrow on the same curve. `display_counts={2: 14}` chooses a stable,
evenly spaced display subset; `display_hyperedges={2: cells}` accepts an
explicit subset already present at the requested scale. Both affect display
only. All supplied directed hyperedges remain available to homology and
Laplacian calculations.

`workflows.analyze_stationary` performs the separate numerical composition. It
returns a `StationaryResult` containing GF(2) homology through the largest
requested degree, complete ordinary Laplacian spectra for the requested
degrees, and named minimum, maximum, mean, numerical-zero-count, and spectral-
energy summaries by default. `plot_stationary_diagnostics` consumes that
result; the plotting layer does not call the core. Request only
`dimensions=(0,)` for a graph and use `display_dimensions=(0, 1)` to retain an
explicitly empty L1 panel in a common four-panel layout.

Animation orchestration is intentionally outside the package. A notebook may
call a stationary renderer repeatedly at changing scales from a Matplotlib
animation callback, then choose its own frame schedule, annotations, and GIF
writer. This keeps the reusable API static while giving animated and stationary
figures the same object semantics.

## Reusable blocks

```python
from topokit import visualization as viz

# One block; reuse ax=... to compose it into an existing figure.
axis = viz.plot_simplex_block(3)
viz.save_figure(axis, "examples/output/tetrahedron.svg")

# Independent ordered hyperedge; order changes arrows, not vertex positions.
axis = viz.plot_hyperdigraph_block(3, order=(3, 0, 2, 1))

# A row of dimensional blocks, suitable for later reuse in other figures.
figure, axes = viz.plot_building_blocks("hyperdigraph", dimensions=(0, 1, 2, 3))
viz.save_figure(figure, "examples/output/hyperdigraph_blocks.pdf")
```

`plot_simplex_block` and `plot_hyperdigraph_block` accept a nonnegative
`dimension`, optional `points`, `ids`, `ax`, and the rendering options below.
The hyperdigraph block also accepts `order`, a permutation of its IDs.
Custom coordinates must have exactly `dimension + 1` rows in two or three
dimensions. The default labels are integers from zero to the dimension.

| Dimension | Default layout | Simplicial depiction | Hyperdigraph depiction |
| --- | --- | --- | --- |
| 0 | Point | Vertex | Singleton hyperedge |
| 1 | Unit segment | Edge | One arrow |
| 2 | Equilateral triangle | Filled triangle | Two consecutive arrows |
| 3 | Regular tetrahedron | Four triangular faces | Three consecutive arrows |
| Above 3 | Regular polygon | Labelled schematic 2-skeleton | Labelled schematic ordered sequence |

The higher-dimensional layouts are diagrams, not geometric embeddings of
simplex interiors. Canonical block coordinates are illustrative and carry no
experimental meaning. Blocks default to `view="projected"` and an azimuth of
−90 degrees for a clear tetrahedron silhouette. The gallery accepts an existing
`axes` array with one axis per requested dimension, all on the same figure;
it preserves dimension order and does not clear existing artists. Use individual
block calls when panels need different IDs, coordinates, or sequence orders.

## Existing TopoKit objects

```python
from topokit import readers, builders, visualization as viz

cloud = readers.read("examples/data/point_cloud_24.csv")
obj = builders.from_points(cloud, kind="simplicial", max_dimension=2)
axis = viz.plot_simplicial_complex(obj, view="projected", labels=True)
viz.save_figure(axis, "examples/output/simplicial_object.svg")

# The same dispatcher accepts a hyperdigraph Topology.
axis = viz.plot_topology(obj, scale=3.0, dimensions=(0, 1, 2), view="projected")
```

`plot_simplicial_complex(obj, ...)`, `plot_hyperdigraph(obj, ...)`, and
`plot_topology(obj, ...)` use the coordinates attached to a `Topology`.
Native objects without attached coordinates require an explicit
`cloud=PointCloud(points, ids=...)`. Native labels must match those IDs;
builder-provided simplicial vertex-ID maps are respected.

`scale=None` displays all supplied cells. A finite `scale` includes cells with
birth at or below that threshold and must lie inside the recorded construction
domain. A static hyperdigraph has no filtration to query and rejects `scale`.
`dimensions=2` selects exactly degree two; `(0, 1, 2)` selects those three
degrees. This is a display selection: the original q+1 cells, filtration, and
core computations remain intact. Only vertices referenced by selected cells
are displayed; include degree zero to retain isolated vertices born by the
selected scale.

Interaction quotient chains have no inferred geometric embedding. Their
existing `plot_interaction_factors` function continues to show the two factors
and explicit overlap separately.

## Explicit cells and ordered sequences

```python
from topokit import PointCloud, visualization as viz

cloud = PointCloud([[0, 0], [1, 0], [0.5, 0.9]], ids=("a", "b", "c"))
axis = viz.plot_simplices(cloud, [("a", "b", "c")], labels=True)
axis = viz.plot_hyperedges(cloud, [("a", "b", "c"), ("b", "a")], labels=True)
```

These lower-level functions accept either an iterable of cells or
`{dimension: cells}`. They retain all points in the supplied cloud, including
isolated coordinates. IDs are stable strings or integers, not implicit array
indices. Each cell has distinct vertices; hyperedge tuple order is significant.

For a supplied simplex, its edges and triangular faces are displayed once.
Three supplied edges alone do not create a filled triangle. Shared faces use
the color of their highest-dimensional supplied parent. Cells above degree
three display only their edges and triangular faces.

For an ordered q-hyperedge `(v0, ..., vq)`, the view has exactly q arrows:
`v0 → v1 → ... → vq`. It does not draw every pair as a directed flag complex,
insert deletion faces, or assert that the displayed sequence is a chain-group
basis element. Endpoint markers locate vertices; they do not assert the
presence of separate singleton hyperedges. Color denotes hyperedge degree.
Repeated and reciprocal segments occupy separate curved display lanes. These
offsets improve readability without changing the coordinates or sequence.
Very crowded views can still obscure individual hyperedges, especially when
several same-degree sequences share support; select an explicit subset or use
separate panels when their individual identity matters.

## Camera, styling, and export

```python
style = viz.PlotStyle(node_size=60, face_alpha=0.4, ribbon_width=10,
                      ribbon_alpha=0.5, arrow_size=14, label_size=10)
axis = viz.plot_simplex_block(3, style=style, colors={3: "#517EA5"},
                              view="projected", elev=25, azim=-75)
viz.save_figure(axis, "examples/output/custom_block.png", dpi=300)
```

| Option | Behavior |
| --- | --- |
| `view="auto"` | Retain 2D coordinates on a 2D axis and 3D coordinates on a rotatable 3D axis; object-view default |
| `view="projected"` | Orthographically project 3D coordinates to a 2D axis; leave 2D coordinates unchanged |
| `view="3d"` | Use an orthographic 3D axis; pad 2D coordinates with z=0 |
| `elev`, `azim` | Camera angles in degrees; default object camera is 18, −65 |
| `labels=True` | Show stable point IDs; default for blocks, off for object views |
| `show_axes=True` | Display coordinate axes and declared units; off by default |
| `legend=True` | Display dimension-color legend; off by default |
| `colors={q: color}` | Override dimension colors for simplex faces/edges or hyperedge ribbons |
| `style=PlotStyle(...)` | Set node sizes/colors, face/ribbon opacity, edge/arrow width, curvature, labels, and padding |
| `color_by_weights=True` | Color coordinate markers by supplied weights; dark markers are the new default |
| `title`, `ax` | Customize the title and compose on an existing compatible 2D/3D axis |

Projected positions may overlap or hide an edge at some camera angles. Change
the camera or use 3D rotation; the renderer never jitters scientific coordinates.
Transparent 3D faces and screen-space arrow curves are schematic rendering
choices, not volumetric occlusion guarantees. Coordinates above three ambient
dimensions require an explicit caller-supplied projection.

`save_figure(figure_or_axis, path, dpi=300, transparent=False)` exports using the
filename extension and returns a `Path`. PNG, SVG, and PDF are verified. SVG
labels remain editable text; PDF uses embedded TrueType fonts. The helper uses
local Matplotlib settings and keeps the figure open. As usual, output files at
the given path are replaced. Call `plt.show()` or `plt.close(figure)` yourself
when needed. No plotting call changes global `rcParams`.

Existing `plot_simplices`/`plot_hyperedges` import paths remain available. Their
visual defaults now use clean axes, dark nodes, filled simplex faces, and real
directional arrowheads. Use `show_axes=True, color_by_weights=True` when you
want coordinate axes and weight-colored markers. Numerical APIs and defaults
are unchanged.

## Bounded rendering and validation

Object rendering is intended for selected views, not unrestricted rendering of
all high-order cells. Exceeding a budget raises an error; cells are never
silently sampled or dropped.

| Guard | Default | Counts |
| --- | --- | --- |
| `max_cells` | 1000 simplices; 500 hyperedges | Supplied/selected cells, including singleton cells and duplicates |
| `max_primitives` | 20,000 | Sum of edge and triangle incidences before simplex deduplication, or two primitives per hyperedge segment (ribbon + arrow) |
| `max_scan_cells` | 1,000,000 | Object-adapter records inspected, including records discarded by scale filtering |
| `max_vertices` | 32 | Vertices per building block, checked before layout allocation |
| `max_blocks` | 16 | Panels in one building-block gallery |

Raise a guard explicitly only for a view appropriate to available resources.
Simplicial edges/faces use batched Matplotlib collections and cached local ID
lookups. Hyperedge artist count grows linearly in the total displayed sequence
length. Shared simplex primitives are deduplicated; arrow occurrences are
retained because combining them would conceal supplied sequence information.

## Runnable gallery and checks

```bash
python examples/visualization_gallery.py
```

This writes `building_blocks`, `objects_projected`, and `objects_3d` in PNG,
SVG, and PDF under `examples/output/visualization_gallery/`. All coordinates in
this gallery are illustrative. The source is
[examples/visualization_gallery.py](../examples/visualization_gallery.py).

Regression checks cover consecutive direction, reciprocal curves, absence of
clique filling, shared faces, stable IDs, inclusive filtration selection,
unchanged inputs, lazy imports, resource guards, and rendered exports.
The generated PNGs were visually inspected for readable labels, distinct
arrows, tetrahedron visibility, alignment, and clipping. See
[VALIDATION.md](VALIDATION.md) for the tested environment and current results.

Rendering primitives follow the official Matplotlib interfaces for
[FancyArrowPatch](https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.FancyArrowPatch.html)
and [Poly3DCollection](https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.mplot3d.art3d.Poly3DCollection.html).
