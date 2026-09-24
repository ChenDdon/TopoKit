"""Barcode vectors and optional per-degree grids, independent of topology/ML."""
from copy import deepcopy
from dataclasses import dataclass, field
import numpy as np
from ._metadata import source_semantics
from ._barcode_bins import (
    check_schema, collection_maximum, compact_edges, default_minimum,
    edges as _edges, finite_real, resolve_axes, source_schema, stack_or_tuple, validate_result,
)


@dataclass(frozen=True)
class FeatureResult:
    values: np.ndarray
    names: tuple[str, ...]
    finite_histograms: np.ndarray | tuple
    essential_histograms: np.ndarray | tuple
    out_of_range: np.ndarray
    metadata: dict = field(default_factory=dict)

    @property
    def grids(self):
        """Finite 2D grids by degree; values still includes all extra channels."""
        return {q: self.finite_histograms[i] for i, q in enumerate(self.metadata["dimensions"])}


def histogram_features(result, *, birth_edges=None, death_edges=None, dimensions=None,
                       min_persistence=0.0, normalize=False, min_value=None,
                       max_value=None, step=0.1, max_features=1_000_000):
    """Birth/death histograms flattened by default, plus honest extra channels.

    Explicit edges override min_value/max_value/step. Each axis may instead map
    degree to an array or a specification with edges or range parameters. The
    final bin ends exactly at max_value. The default minimum is max(0, recorded
    filtration_start). A one-result call never infers its own maximum: supply
    finite max_value/edges, or use PersistenceVectorizer.fit on a collection.

    Finite intervals, essential births, and finite/essential overflow counts
    stay separate. Unequal grids are tuples without padding; shared-grid calls
    keep the existing stacked arrays. FeatureResult.grids exposes optional 2D
    views. Filtering/normalization never edits the original interval records.
    """
    degrees = validate_result(result, dimensions)
    cutoff = finite_real(min_persistence, "min_persistence")
    if cutoff < 0:
        raise ValueError("min_persistence must be finite and nonnegative")
    # Each axis mapping holds bin edges by homology degree, not barcode endpoints.
    birth_axes, death_axes, _ = resolve_axes(degrees, birth_edges=birth_edges, death_edges=death_edges,
        min_value=min_value, max_value=max_value, step=step, default_min=default_minimum(result),
        max_features=max_features)
    histograms, essential_histograms = [], []
    overflow_counts = np.zeros((len(degrees), 2), dtype=float)
    vectors, names = [], []
    for index, q in enumerate(degrees):
        bars = [bar for bar in result.intervals if bar.dimension == q and bar.lifetime >= cutoff]
        finite_intervals = [(bar.birth, bar.death) for bar in bars if np.isfinite(bar.death)]
        essential_births = [bar.birth for bar in bars if np.isposinf(bar.death)]
        grid = np.zeros((len(birth_axes[q])-1, len(death_axes[q])-1), dtype=float)
        essential_histogram = np.zeros(len(birth_axes[q])-1, dtype=float)
        if finite_intervals:
            finite_coordinates = np.asarray(finite_intervals)
            grid = np.histogram2d(finite_coordinates[:, 0], finite_coordinates[:, 1], bins=(birth_axes[q], death_axes[q]))[0]
            overflow_counts[index, 0] = len(finite_intervals) - grid.sum()
        if essential_births:
            essential_histogram = np.histogram(essential_births, bins=birth_axes[q])[0].astype(float)
            overflow_counts[index, 1] = len(essential_births) - essential_histogram.sum()
        if normalize and bars:
            grid /= len(bars)
            essential_histogram /= len(bars)
            overflow_counts[index] /= len(bars)
        histograms.append(grid)
        essential_histograms.append(essential_histogram)
        vectors.extend(grid.ravel())
        names.extend(f"H{q}:birth[{i}]:death[{j}]" for i in range(grid.shape[0]) for j in range(grid.shape[1]))
        vectors.extend(essential_histogram)
        names.extend(f"H{q}:essential:birth[{i}]" for i in range(len(essential_histogram)))
        vectors.extend(overflow_counts[index])
        names.extend((f"H{q}:finite_out_of_range", f"H{q}:essential_out_of_range"))
    feature_schema = {"source": source_schema(result), "birth_edges": compact_edges(birth_axes),
              "death_edges": compact_edges(death_axes), "dimensions": degrees,
              "min_persistence": cutoff, "normalized": bool(normalize),
              "essential_policy": "separate_birth_histogram", "overflow_policy": "separate_counts"}
    metadata = {**source_semantics(result, field=result.field),
        "representation": "birth_death_histogram", "dimensions": degrees,
        "birth_edges": compact_edges(birth_axes), "death_edges": compact_edges(death_axes),
        "min_persistence": cutoff, "normalized": bool(normalize),
        "essential_policy": "separate_birth_histogram", "overflow_policy": "separate_counts",
        "learned_from_data": False, "fit_scope": "explicit", "schema": feature_schema,
        "grid_shapes": {q: histograms[i].shape for i, q in enumerate(degrees)},
        "flatten_order": "degree_then_birth_then_death_then_essential_then_overflow",
        "bin_ranges": {q: {"birth": (float(birth_axes[q][0]), float(birth_axes[q][-1])),
                            "death": (float(death_axes[q][0]), float(death_axes[q][-1]))} for q in degrees}}
    return FeatureResult(np.asarray(vectors, dtype=float), tuple(names),
                         stack_or_tuple(histograms, (0, 0, 0)),
                         stack_or_tuple(essential_histograms, (0, 0)), overflow_counts, metadata)


class PersistenceVectorizer:
    """Fit one finite bin schema on supplied training persistence results.

    Absent edges/max_value, infer the maximum once from all finite selected
    births/deaths and declared finite filtration ends. It is shared across
    samples and degrees; per-degree explicit bins/ranges override it. Minimum
    defaults to max(0, the common filtration_start). No positive finite range
    means explicit edges or max_value is required; infinity never becomes a bin.

    fit_scope='training' is the default; 'descriptive' explicitly records use of
    a full descriptive collection. Callers supply the collection; no split is
    inferred. Transform never recalibrates, even for overflow. Route, field,
    definition, units, and filtration semantics must match on fit/transform.
    """
    def __init__(self, *, birth_edges=None, death_edges=None, dimensions=(0, 1, 2),
                 min_persistence=0.0, normalize=False, min_value=None, max_value=None,
                 step=0.1, fit_scope="training", max_features=1_000_000):
        self.birth_edges = birth_edges
        self.death_edges = death_edges
        self.dimensions = dimensions
        self.min_persistence = min_persistence
        self.normalize = normalize
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        self.fit_scope = fit_scope
        self.max_features = max_features

    def get_params(self, deep=True):
        return {key: getattr(self, key) for key in ("birth_edges", "death_edges", "dimensions",
            "min_persistence", "normalize", "min_value", "max_value", "step", "fit_scope", "max_features")}

    def _clear_fit(self):
        for name in ("feature_names_in_output_", "_fitted_params", "_source_schema", "_fit_metadata",
                     "schema_", "birth_edges_", "death_edges_", "scale_units_", "n_features_out_",
                     "fit_metadata_", "n_samples_fit_"):
            if hasattr(self, name):
                delattr(self, name)

    def set_params(self, **params):
        unknown = set(params) - set(self.get_params())
        if unknown:
            raise ValueError(f"unknown parameter {sorted(unknown)[0]!r}")
        for key, value in params.items():
            setattr(self, key, value)
        self._clear_fit()
        return self

    def fit(self, X, y=None):
        self._clear_fit()
        if self.fit_scope not in ("training", "descriptive"):
            raise ValueError("fit_scope must be 'training' or explicitly 'descriptive'")
        results = list(X)
        if not results:
            raise ValueError("fit requires at least one persistence result")
        degrees = validate_result(results[0], self.dimensions)
        reference_schema = source_schema(results[0])
        for result in results[1:]:
            validate_result(result, degrees)
            check_schema(result, reference_schema)
        inferred_maximum = collection_maximum(results, degrees)
        birth_axes, death_axes, learned = resolve_axes(degrees, birth_edges=self.birth_edges,
            death_edges=self.death_edges, min_value=self.min_value, max_value=self.max_value,
            step=self.step, default_min=default_minimum(results[0]), inferred_max=inferred_maximum,
            max_features=self.max_features)
        histogram_options = {"birth_edges": compact_edges(birth_axes), "death_edges": compact_edges(death_axes),
                  "dimensions": degrees, "min_persistence": self.min_persistence,
                  "normalize": bool(self.normalize), "max_features": self.max_features}
        reference_features = histogram_features(results[0], **histogram_options)
        fit_metadata = {"fit_scope": self.fit_scope, "learned_from_data": learned,
            "fit_sample_count": len(results), "range_policy": "collection_global_finite_max" if learned else "explicit",
            "fitted_finite_max": inferred_maximum if learned else None,
            "default_minimum": default_minimum(results[0]),
            "finite_max_sources": ("finite_births", "finite_deaths", "finite_filtration_end") if learned else (),
            "bin_ranges": deepcopy(reference_features.metadata["bin_ranges"])}
        self._fitted_params = histogram_options
        self._source_schema = deepcopy(reference_schema)
        self._fit_metadata = deepcopy(fit_metadata)
        self.feature_names_in_output_ = reference_features.names
        self.n_features_out_ = len(reference_features.names)
        self.n_samples_fit_ = len(results)
        self.scale_units_ = deepcopy(results[0].metadata.get("scale_units"))
        self.birth_edges_ = compact_edges(birth_axes)
        self.death_edges_ = compact_edges(death_axes)
        self.schema_ = deepcopy(reference_features.metadata["schema"])
        self.fit_metadata_ = deepcopy(fit_metadata)
        return self

    def transform_results(self, X):
        """Return per-sample records with all channels and fitted provenance."""
        if not hasattr(self, "_fitted_params"):
            raise RuntimeError("fit the bin schema before transform")
        results = list(X)
        for result in results:
            validate_result(result, self._fitted_params["dimensions"])
            check_schema(result, self._source_schema)
        records = []
        for result in results:
            feature = histogram_features(result, **self._fitted_params)
            feature.metadata.update(deepcopy(self._fit_metadata))
            records.append(feature)
        return tuple(records)

    def transform(self, X, *, output="flat"):
        """Return sample-by-feature vectors or opt-in per-sample Hq finite grids.

        output='grids' returns finite grids only. transform_results retains
        essential and overflow channels too. Ragged grids are never padded.
        """
        if output not in ("flat", "grids"):
            raise ValueError("output must be 'flat' or 'grids'")
        records = self.transform_results(X)
        if output == "grids":
            return tuple(record.grids for record in records)
        return np.asarray([record.values for record in records], dtype=float).reshape(len(records), self.n_features_out_)

    def fit_transform(self, X, y=None, *, output="flat"):
        results = list(X)
        return self.fit(results, y).transform(results, output=output)

    def get_feature_names_out(self, input_features=None):
        if not hasattr(self, "feature_names_in_output_"):
            raise RuntimeError("fit before requesting feature names")
        return np.asarray(self.feature_names_in_output_, dtype=object)
