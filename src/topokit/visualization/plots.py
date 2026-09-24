"""Optional static views consuming the same numerical objects as analysis.

Matplotlib is imported only when plotting. Higher-dimensional cells are shown
as incidence/face views, not claimed to have a unique geometric embedding.
"""

import math
from numbers import Integral
import numpy as np
from ..data import as_point_cloud
from ..results import PersistenceResult


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError("Install topokit[plot] to use visualization") from error
    return plt


def _point_axes(cloud, ax=None):
    ambient_dimension = cloud.points.shape[1]
    if ambient_dimension not in (2, 3):
        raise ValueError("Point views need 2D/3D coordinates; project explicitly before plotting")
    if ax is None:
        figure = _plt().figure(figsize=(6.6, 5.4))
        ax = figure.add_subplot(projection="3d" if ambient_dimension == 3 else None)
        if ambient_dimension == 3:
            figure.subplots_adjust(left=0.02, right=0.84, bottom=0.12, top=0.90)
    if ambient_dimension == 3 and not hasattr(ax, "get_zlim"):
        raise ValueError("3D points require a 3D axis")
    return ax


def plot_points(cloud, *, ax=None, labels=False, color_by_weights=True, title="Point cloud"):
    cloud = as_point_cloud(cloud)
    ax = _point_axes(cloud, ax)
    points = cloud.points
    coordinates = [points[:, i] for i in range(points.shape[1])]
    ax.scatter(*coordinates, c=cloud.weights if color_by_weights else "#176d9c", cmap="viridis" if color_by_weights else None, s=34, zorder=5)
    if labels:
        for label, point in zip(cloud.ids, points):
            ax.text(*point, str(label), fontsize=7)
    ax.set_xlabel("x", fontsize=9, labelpad=4)
    ax.set_ylabel("y", fontsize=9, labelpad=4)
    ax.tick_params(labelsize=8)
    if points.shape[1] == 3:
        ax.set_zlabel("z", fontsize=9, labelpad=5)
        if len(cloud):
            ax.set_box_aspect(np.maximum(np.ptp(points, axis=0), 1e-8))
    else:
        ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title, fontsize=12)
    return ax


# Backward-compatible imports; object rendering lives in its own module.
from .objects import plot_simplices, plot_hyperedges


def plot_interaction_factors(cloud_a, cloud_b, cells_a, cells_b, *, overlap_pairs, max_cells=1000):
    """Two factor views with shared points highlighted; not a tensor-cell embedding."""
    factor_cloud_a, factor_cloud_b = as_point_cloud(cloud_a), as_point_cloud(cloud_b)
    figure = _plt().figure(figsize=(12, 5.5))
    axes = [figure.add_subplot(1, 2, i+1, projection="3d" if factor_cloud.points.shape[1] == 3 else None)
            for i, factor_cloud in enumerate((factor_cloud_a, factor_cloud_b))]
    for axis, cloud, cells, side in zip(axes, (factor_cloud_a, factor_cloud_b), (cells_a, cells_b), (0, 1)):
        plot_simplices(cloud, cells, ax=axis, max_cells=max_cells, title=f"Interaction factor {side+1}")
        overlap_ids = [pair[side] for pair in overlap_pairs]
        if overlap_ids:
            overlap_coordinates = cloud.points[[cloud.index[label] for label in overlap_ids]]
            axis.scatter(*overlap_coordinates.T, s=80, facecolors="none", edgecolors="#d25d30", linewidths=1.5)
    figure.suptitle("Two factors; orange rings identify the explicit overlap", fontsize=11)
    figure.subplots_adjust(left=0.02, right=0.91, bottom=0.1, top=0.85, wspace=0.12)
    return figure, axes


def plot_barcodes(result, *, ax=None, min_persistence=0.0, title="Persistence intervals",
                  mark_initial_stage=True):
    """Plot persistence intervals, optionally marking left-truncated births.

    Open circles indicate intervals already present at the recorded filtration
    start. Set ``mark_initial_stage=False`` when ordinary bar starts are the
    intended display, for example when H0 vertices are known to be born exactly
    at the initial scale.
    """
    if not isinstance(mark_initial_stage, (bool, np.bool_)):
        raise ValueError("mark_initial_stage must be a boolean")
    if ax is None:
        _, ax = _plt().subplots(figsize=(7, 4))
    bars = sorted((bar for bar in result.intervals if bar.lifetime >= min_persistence),
                  key=lambda bar: (bar.dimension, bar.birth, bar.death))
    finite_endpoints = [x for bar in bars for x in (bar.birth, bar.death) if math.isfinite(x)]
    low = min(finite_endpoints, default=0.0)
    high = max(finite_endpoints, default=1.0)
    span = max(high-low, 1.0)
    essential_end = high + 0.08 * span
    palette = _plt().get_cmap("tab10")
    labelled_dimensions = set()
    for row, bar in enumerate(bars):
        color = palette(bar.dimension % 10)
        end = bar.death if math.isfinite(bar.death) else essential_end
        ax.plot((bar.birth, end), (row, row), color=color, linewidth=1.8,
                label=f"H{bar.dimension}" if bar.dimension not in labelled_dimensions else None)
        labelled_dimensions.add(bar.dimension)
        if mark_initial_stage and bar.at_initial_stage:
            ax.plot(bar.birth, row, marker="o", markerfacecolor="white", markeredgecolor=color, markersize=3)
        if not math.isfinite(bar.death):
            ax.plot(end, row, marker=">", color=color, markersize=4)
    ax.set_xlabel(f"Filtration [{result.metadata.get('scale_units', 'native units')}]")
    ax.set_ylabel("Interval index")
    ax.set_title(title)
    if labelled_dimensions:
        ax.legend(frameon=False)
    ax.set_xlim(low-0.03*span, essential_end+0.03*span)
    ax.spines[["top", "right"]].set_visible(False)
    ax._topokit_view = {
        "kind": "barcodes", "interval_count": len(bars),
        "initial_stage_markers": bool(mark_initial_stage),
    }
    return ax


def _plot_scales(scales):
    if isinstance(scales, (str, bytes)):
        raise ValueError("scales must be a finite increasing sequence")
    try:
        values = tuple(scales)
    except TypeError as error:
        raise ValueError("scales must be a finite increasing sequence") from error
    if (not values or any(isinstance(value, (bool, np.bool_)) for value in values)):
        raise ValueError("scales must be a finite increasing sequence")
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("scales must be a finite increasing sequence") from error
    if (result.ndim != 1 or not np.all(np.isfinite(result))
            or np.any(np.diff(result) <= 0)):
        raise ValueError("scales must be a finite strictly increasing sequence")
    return result


def _plot_dimensions(result, dimensions):
    if dimensions is None:
        return tuple(range(result.max_dimension + 1))
    if isinstance(dimensions, Integral) and not isinstance(dimensions, (bool, np.bool_)):
        dimensions = (int(dimensions),)
    elif isinstance(dimensions, (str, bytes, set, frozenset)):
        raise ValueError("dimensions must be an ordered collection of available degrees")
    else:
        try:
            dimensions = tuple(dimensions)
        except TypeError as error:
            raise ValueError("dimensions must be an ordered collection of available degrees") from error
    if (not dimensions
            or any(isinstance(q, (bool, np.bool_)) or not isinstance(q, Integral)
                   or q < 0 or q > result.max_dimension for q in dimensions)
            or len(set(dimensions)) != len(dimensions)):
        raise ValueError("dimensions must be unique degrees available in the persistence result")
    return tuple(int(q) for q in dimensions)


def plot_betti_curves(result, scales, *, dimensions=None, ax=None,
                      title="Betti curves", xlabel=None):
    """Plot Betti step curves by querying one supplied persistence result."""
    if not isinstance(result, PersistenceResult):
        raise TypeError("result must be a PersistenceResult")
    scale_values = _plot_scales(scales)
    degrees = _plot_dimensions(result, dimensions)
    betti = np.asarray([result.betti_at(scale) for scale in scale_values], dtype=float)
    if ax is None:
        _, ax = _plt().subplots(figsize=(6, 4))
    palette = _plt().get_cmap("tab10")
    line_styles = ("-", "--", "-.", ":")
    for index, degree in enumerate(degrees):
        ax.step(
            scale_values, betti[:, degree], where="post", label=f"H{degree}",
            color=palette(degree % 10), linestyle=line_styles[index % len(line_styles)],
        )
    units = result.metadata.get("scale_units", "native units")
    ax.set_xlabel(xlabel or f"Filtration [{units}]")
    ax.set_ylabel("Betti number")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax._topokit_view = {
        "kind": "betti_curves", "dimensions": degrees,
        "scales": tuple(float(value) for value in scale_values),
    }
    return ax


def plot_spectral_summary_series(
        summaries, *, scales=None, statistics=("min", "max", "mean"),
        energy="laplacian_energy", ax=None, energy_ax=None,
        title=None, xlabel=None):
    """Plot already-computed spectral summaries across filtration scales.

    ``summaries`` is an ordered sequence of result objects with aligned
    ``names`` and ``values`` fields, such as the output of
    ``postprocessing.summarize_spectra``. This function only displays those
    supplied numbers; it does not compute spectra or summary statistics.
    Undefined NaN values are retained as visible gaps. The optional ``energy``
    statistic is drawn on a secondary y-axis and both axes are returned.
    """
    if isinstance(summaries, (str, bytes)):
        raise ValueError("summaries must be a nonempty ordered result sequence")
    try:
        summaries = tuple(summaries)
    except TypeError as error:
        raise ValueError("summaries must be a nonempty ordered result sequence") from error
    if not summaries:
        raise ValueError("summaries must be a nonempty ordered result sequence")

    names = tuple(getattr(summaries[0], "names", ()))
    if (not names or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)):
        raise ValueError("spectral summaries must have unique nonempty names")
    short_names = tuple(name.rsplit(":", 1)[-1] for name in names)
    if len(set(short_names)) != len(short_names):
        raise ValueError("spectral summary statistic names must be unique")
    name_index = {name: index for index, name in enumerate(short_names)}

    rows = []
    for summary in summaries:
        if tuple(getattr(summary, "names", ())) != names:
            raise ValueError("spectral summaries must share one ordered schema")
        try:
            raw_values = summary.values
            if np.iscomplexobj(raw_values):
                raise ValueError("spectral summary values must be real scalars")
            values = np.asarray(raw_values, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("spectral summary values must be real scalars") from error
        if values.shape != (len(names),) or np.any(np.isinf(values)):
            raise ValueError("spectral summary values must be scalar, finite, or NaN")
        rows.append(values)
    values = np.asarray(rows, dtype=float)

    if isinstance(statistics, str):
        statistics = (statistics,)
    elif isinstance(statistics, (bytes, set, frozenset)):
        raise ValueError("statistics must be a nonempty ordered collection")
    else:
        try:
            statistics = tuple(statistics)
        except TypeError as error:
            raise ValueError("statistics must be a nonempty ordered collection") from error
    if (not statistics or any(not isinstance(name, str) or name not in name_index
                              for name in statistics)
            or len(set(statistics)) != len(statistics)):
        raise ValueError("statistics must be unique names present in every summary")
    if energy is not None and (not isinstance(energy, str) or energy not in name_index):
        raise ValueError("energy must be None or a statistic present in every summary")
    if energy in statistics:
        raise ValueError("energy must not duplicate a primary statistic")

    if scales is None:
        try:
            scales = [summary.metadata["scale"] for summary in summaries]
        except (AttributeError, KeyError, TypeError) as error:
            raise ValueError("provide scales or summary metadata with one scale each") from error
    scale_values = _plot_scales(scales)
    if len(scale_values) != len(summaries):
        raise ValueError("scales must contain one value per spectral summary")

    dimension_values = [
        getattr(summary, "metadata", {}).get("dimension") for summary in summaries
        if getattr(summary, "metadata", {}).get("dimension") is not None
    ]
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
           or value < 0 for value in dimension_values):
        raise ValueError("spectral summary dimensions must be nonnegative integers")
    dimensions = set(dimension_values)
    if len(dimensions) > 1:
        raise ValueError("spectral summaries must describe one Laplacian dimension")
    dimension = next(iter(dimensions), None)
    if ax is None:
        if energy_ax is not None:
            raise ValueError("energy_ax requires an explicit primary ax")
        _, ax = _plt().subplots(figsize=(6, 4))
    if energy_ax is not None and energy_ax.figure is not ax.figure:
        raise ValueError("ax and energy_ax must belong to the same figure")

    labels = {"min": "minimum", "max": "maximum", "mean": "mean"}
    colors = ("#176d9c", "#4b8b6f", "#715a9a", "#8a6d3b")
    line_styles = ("-", "--", ":", "-.")
    lines = []
    for index, statistic in enumerate(statistics):
        line, = ax.plot(
            scale_values, values[:, name_index[statistic]], marker="o",
            markersize=3, color=colors[index % len(colors)],
            linestyle=line_styles[index % len(line_styles)],
            label=labels.get(statistic, statistic.replace("_", " ")),
        )
        lines.append(line)

    energy_line = None
    if energy is not None:
        energy_ax = ax.twinx() if energy_ax is None else energy_ax
        energy_line, = energy_ax.plot(
            scale_values, values[:, name_index[energy]], color="#c66b2b",
            linewidth=1.2, label=energy.replace("_", " "),
        )
        energy_ax.set_ylabel(energy.replace("_", " ").capitalize(), color="#c66b2b")
        energy_ax.tick_params(axis="y", colors="#c66b2b")
        energy_ax.spines["right"].set_visible(True)

    metadata = getattr(summaries[0], "metadata", {})
    units = metadata.get("scale_units", "native units")
    ax.set_xlabel(xlabel or f"Filtration [{units}]")
    ax.set_ylabel("Eigenvalue summary")
    ax.set_title(title or (
        f"L{dimension} spectral summaries" if dimension is not None
        else "Laplacian spectral summaries"
    ))
    ax.spines[["top", "right"]].set_visible(False)
    legend_lines = lines + ([] if energy_line is None else [energy_line])
    ax.legend(legend_lines, [line.get_label() for line in legend_lines],
              loc="best", frameon=False)
    ax._topokit_view = {
        "kind": "spectral_summary_series", "dimension": dimension,
        "statistics": tuple(statistics), "energy": energy,
        "scales": tuple(float(value) for value in scale_values),
    }
    return ax, energy_ax


def plot_spectrum(result, *, ax=None):
    if ax is None:
        _, ax = _plt().subplots(figsize=(6, 4))
    ax.plot(np.arange(len(result.eigenvalues)), result.eigenvalues, ".", color="#176d9c", markersize=4)
    suffix = "full" if result.complete else "partial"
    ax.set_title(f"{result.kind.capitalize()} L{result.dimension} spectrum ({suffix})")
    ax.set_xlabel("Sorted eigenvalue index")
    ax.set_ylabel("Eigenvalue")
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def plot_features(result, *, dimension=0, ax=None):
    degrees = result.metadata["dimensions"]
    if dimension not in degrees:
        raise ValueError("dimension is absent from the feature schema")
    if ax is None:
        _, ax = _plt().subplots(figsize=(5, 4))
    index = degrees.index(dimension) if isinstance(degrees, tuple) else list(degrees).index(dimension)
    births, deaths = result.metadata["birth_edges"], result.metadata["death_edges"]
    if isinstance(births, dict):
        births = births[dimension]
    if isinstance(deaths, dict):
        deaths = deaths[dimension]
    mesh = ax.pcolormesh(births, deaths, result.finite_histograms[index].T, shading="flat", cmap="Blues")
    ax.figure.colorbar(mesh, ax=ax, label="Count" if not result.metadata["normalized"] else "Fraction")
    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    ax.set_title(f"H{dimension} finite birth–death histogram")
    return ax


def plot_eigenvector(result, index=0, *, ax=None):
    """Display coefficients in the exported basis, not an invented cycle map."""
    if result.eigenvectors is None:
        raise ValueError("Recompute with return_eigenvectors=True")
    if index < 0 or index >= result.eigenvectors.shape[1]:
        raise ValueError("eigenvector index is out of range")
    if ax is None:
        _, ax = _plt().subplots(figsize=(7, 4))
    vector = result.eigenvectors[:, index]
    ax.plot(np.arange(len(vector)), vector, ".", color="#b05e35")
    ax.axhline(0, linewidth=0.5, color="0.6")
    ax.set_xlabel("Exported basis index")
    ax.set_ylabel("Coefficient")
    ax.set_title(f"L{result.dimension} eigenvector {index}; λ={result.eigenvalues[index]:.5g}")
    return ax
