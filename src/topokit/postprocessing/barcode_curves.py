"""One-dimensional barcode counts on explicit bins, without topology calls."""
from copy import deepcopy
import math
import numpy as np

from ..exceptions import ResourceLimitError
from ._barcode_bins import edges, feature_budget, finite_real, source_schema, validate_result
from ._metadata import source_semantics
from .spectra import VectorResult


def barcode_bin_counts(result, *, bin_edges, dimensions=None, mode="overlap",
                       max_features=1_000_000):
    """Count positive-length persistence intervals in each explicit bin.

    Intervals have the usual [birth, death) convention. ``overlap`` counts
    each bar meeting a positive-width part of [left, right); ``cover`` counts
    bars present throughout [left, right). Neither is point sampling.
    ``death`` instead counts finite deaths, using left-closed/right-open
    histogram bins, with the final right endpoint included.

    Essential (+infinity-death) bars contribute to overlap/cover counts but
    not death counts. Zero-length intervals never contribute. No endpoint
    rounding, normalization, learned bins, or replacement of infinite deaths
    is performed. Bins must lie within any declared observation window.
    Bars outside the explicit bins add no extra overflow columns. Output is
    flattened degree first, then bin; n+1 boundaries give n features/degree.
    """
    degrees = validate_result(result, dimensions)
    if mode not in ("overlap", "cover", "death"):
        raise ValueError("mode must be 'overlap', 'cover', or 'death'")
    axis = edges(bin_edges, "bin_edges")
    count = len(axis) - 1
    budget = feature_budget(max_features)
    if len(degrees) * count > budget:
        raise ResourceLimitError("barcode bin count exceeds max_features; no counts allocated")
    start = result.metadata.get("filtration_start")
    end = result.metadata.get("filtration_end")
    if start is not None and axis[0] < finite_real(start, "filtration_start"):
        raise ValueError("bin_edges begin before the recorded filtration_start")
    if end is not None and end != math.inf:
        if axis[-1] > finite_real(end, "filtration_end"):
            raise ValueError("bin_edges extend beyond the recorded filtration_end")

    values = np.zeros((len(degrees), count), dtype=np.int64)
    for row, degree in enumerate(degrees):
        bars = [bar for bar in result.intervals
                if bar.dimension == degree and bar.death > bar.birth]
        if not bars:
            continue
        births = np.asarray([bar.birth for bar in bars], dtype=float)
        deaths = np.asarray([bar.death for bar in bars], dtype=float)
        if mode == "death":
            values[row] = np.histogram(deaths[np.isfinite(deaths)], bins=axis)[0]
            continue
        if mode == "overlap":
            first = np.searchsorted(axis, births, side="right") - 1
            stop = np.searchsorted(axis, deaths, side="left")
        else:
            first = np.searchsorted(axis, births, side="left")
            stop = np.searchsorted(axis, deaths, side="right") - 1
        first = np.clip(first, 0, count)
        stop = np.clip(stop, 0, count)
        occupied = first < stop
        # Range additions avoid allocating a bars-by-bins boolean matrix.
        changes = np.zeros(count + 1, dtype=np.int64)
        np.add.at(changes, first[occupied], 1)
        np.add.at(changes, stop[occupied], -1)
        values[row] = np.cumsum(changes[:-1])

    policy = "omit_infinite_deaths" if mode == "death" else "count_without_replacing_infinite_deaths"
    schema = {"source": source_schema(result), "representation": "barcode_bin_counts",
              "dimensions": degrees, "bin_edges": axis.copy(), "mode": mode,
              "interval_convention": "[birth, death)", "zero_length_policy": "exclude",
              "essential_policy": policy, "window_policy": "restrict_to_explicit_bins",
              "final_death_endpoint_included": mode == "death"}
    metadata = {**source_semantics(result, field=result.field),
                **deepcopy({key: value for key, value in schema.items() if key != "source"}),
                "schema": schema, "learned_from_data": False, "fit_scope": "explicit",
                "flatten_order": "degree_then_bin", "grid_shape": values.shape}
    names = tuple(f"H{q}:barcode_{mode}:bin[{i}]" for q in degrees for i in range(count))
    return VectorResult(values.ravel().astype(float), names, metadata)
