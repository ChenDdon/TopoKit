"""Optional ML over already-computed feature matrices, never raw topology."""
from collections.abc import Mapping
import numbers
import numpy as np


def _task(task):
    if not isinstance(task, str) or task not in {"regression", "classification"}:
        raise ValueError("task must be regression or classification")
    return task


def make_estimator(*, task="regression", random_state=0, **options):
    """An unfitted gradient-boosted tree; import scikit-learn only on request.

    No labels, split policy, tuning, or training is inferred. Callers may instead
    supply any fit/predict-compatible estimator to fit_features.
    """
    _task(task)
    try:
        from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
    except ImportError as error:
        raise ImportError("Install topokit[ml] to use the optional gradient-boosted tree") from error
    factory = GradientBoostingRegressor if task == "regression" else GradientBoostingClassifier
    return factory(random_state=random_state, **options)


def _same_schema_value(left, right):
    """Compare nested schemas without NumPy's ambiguous dictionary equality."""
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (isinstance(left, Mapping) and isinstance(right, Mapping)
                and set(left) == set(right)
                and all(_same_schema_value(left[key], right[key]) for key in left))
    if isinstance(left, np.ndarray):
        left = left.tolist()
    if isinstance(right, np.ndarray):
        right = right.tolist()
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return (isinstance(left, (tuple, list)) and isinstance(right, (tuple, list))
                and len(left) == len(right)
                and all(_same_schema_value(left_item, right_item) for left_item, right_item in zip(left, right)))
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def feature_matrix(features):
    """Stack same-schema finite records without changing or imputing values."""
    items = tuple(features)
    if not items:
        raise ValueError("at least one feature record is required")
    rows, column_schemas = [], []
    for item in items:
        if not all(hasattr(item, attribute) for attribute in ("names", "values", "metadata")):
            raise TypeError("feature records must provide names, values, and metadata")
        if isinstance(item.names, (str, bytes)):
            raise ValueError("feature names must be an ordered sequence of column names")
        names = tuple(item.names)
        if (not names or any(not isinstance(name, str) or not name for name in names)
                or len(set(names)) != len(names)):
            raise ValueError("feature columns need nonempty, unique string names")
        if not isinstance(item.metadata, Mapping):
            raise TypeError("feature metadata must be a mapping")
        if np.iscomplexobj(item.values):
            raise ValueError("features must be real finite vectors")
        feature_row = np.asarray(item.values, dtype=float)
        if feature_row.ndim != 1 or feature_row.shape != (len(names),) or not np.all(np.isfinite(feature_row)):
            raise ValueError("features must be finite equal-length vectors; handle undefined values explicitly")
        if column_schemas and names != column_schemas[0][0]:
            raise ValueError("feature column schemas differ")
        column_schemas.append((names, item.metadata))
        rows.append(feature_row)
    # Matching labels alone do not prove bins, units or spectral scope match.
    schema_keys = ("representation", "scale_units", "dimensions", "dimension", "birth_edges", "death_edges",
                   "normalized", "min_persistence", "operator_kind", "scale", "start", "end",
                   "complete_spectrum", "positive_only", "statistics",
                   "statistic_scopes", "spectral_variance_definition",
                   "spectral_moment_definition", "spectral_moment_orders",
                   "laplacian_energy_definition", "laplacian_energy_available",
                   "absolute_tolerance", "relative_tolerance", "zero_count_semantics",
                   "essential_policy", "overflow_policy", "entropy_definition",
                   "empty_operator_policy", "empty_summary_policy",
                   "coefficient_field", "field", "definition_id", "topology_kind",
                   "route", "coordinate_units", "filtration_start", "filtration_end",
                   "partial_scope", "entropy_policy",
                   "complex_type", "object_type", "construction", "connection_support",
                   "cutoff_distance", "cutoff_policy", "graph_expansion",
                   "filtration_parameters", "coupling", "progression", "filtration_a", "filtration_b",
                   "factor_filtration_ranges", "factor_scale_units",
                   "schema_version", "feature_definition_id", "schema")
    reference_metadata = column_schemas[0][1]
    for _, metadata in column_schemas[1:]:
        for key in schema_keys:
            if ((key in reference_metadata) != (key in metadata)
                    or not _same_schema_value(metadata.get(key), reference_metadata.get(key))):
                raise ValueError(f"feature schema metadata differs: {key}")
        # In new spectral records the resolved threshold depends on each sample;
        # atol/rtol configuration is the schema. Legacy records had only tol.
        if "absolute_tolerance" not in reference_metadata:
            if (("tolerance" in reference_metadata) != ("tolerance" in metadata)
                    or not _same_schema_value(metadata.get("tolerance"), reference_metadata.get("tolerance"))):
                raise ValueError("feature schema metadata differs: tolerance")
        # Full-spectrum size legitimately changes with the sample. Partial
        # summaries must instead refer to the same requested number of modes.
        if _same_schema_value(reference_metadata.get("complete_spectrum"), False):
            if ("supplied_eigenvalues" not in reference_metadata or "supplied_eigenvalues" not in metadata
                    or not _same_schema_value(metadata["supplied_eigenvalues"], reference_metadata["supplied_eigenvalues"])):
                raise ValueError("partial feature schemas need the same supplied_eigenvalues count")
    if _same_schema_value(reference_metadata.get("complete_spectrum"), False) and "supplied_eigenvalues" not in reference_metadata:
        raise ValueError("partial feature schemas must state supplied_eigenvalues")
    return np.vstack(rows)


def fit_features(X_train, y_train, *, estimator=None, task="regression", random_state=0, **options):
    """Fit on the caller's training split only; custom estimators are supported."""
    _task(task)
    if np.iscomplexobj(X_train) or np.iscomplexobj(y_train):
        raise ValueError("training inputs and targets must be real, not complex")
    training_values, targets = np.array(X_train, dtype=float, copy=True), np.array(y_train, copy=True)
    raw_targets = np.asarray(y_train, dtype=object)
    if training_values.ndim != 2 or not training_values.shape[0] or not training_values.shape[1] or not np.all(np.isfinite(training_values)):
        raise ValueError("X_train must be a finite, nonempty sample-by-feature matrix")
    if targets.ndim != 1 or len(targets) != len(training_values):
        raise ValueError("y_train must contain one target per training sample")
    if task == "regression":
        try:
            targets = np.asarray(targets, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("regression targets must be finite numbers") from error
        if not np.all(np.isfinite(targets)):
            raise ValueError("regression targets must be finite numbers")
    else:
        for target in raw_targets:
            # Inspect the original scalars before NumPy can coerce a mixed
            # string/numeric list and turn a numeric NaN into the string 'nan'.
            try:
                missing = bool(target != target)
            except (TypeError, ValueError) as error:
                raise ValueError("classification targets must be nonmissing scalar labels") from error
            if (target is None or missing or np.iscomplexobj(target)
                    or (isinstance(target, numbers.Number) and not np.isfinite(target))):
                raise ValueError("classification targets must not be missing or nonfinite")
    if estimator is not None and options:
        raise ValueError("configure a custom estimator directly; options are for the default estimator")
    model = make_estimator(task=task, random_state=random_state, **options) if estimator is None else estimator
    if not callable(getattr(model, "fit", None)) or not callable(getattr(model, "predict", None)):
        raise TypeError("estimator must implement fit and predict")
    model.fit(training_values, targets)
    return model
