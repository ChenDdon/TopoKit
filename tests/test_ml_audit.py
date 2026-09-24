"""ML operates only on explicit training vectors and compatible schemas."""
import builtins
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from topokit.postprocessing.spectra import VectorResult
from topokit.workflows.ml import feature_matrix, fit_features, make_estimator


def record(values=(1.0, 2.0), names=("f0", "f1"), **metadata):
    return VectorResult(np.asarray(values), names, metadata)


class Recorder:
    def __init__(self):
        self.fit_calls = 0

    def fit(self, X, y):
        self.fit_calls += 1
        self.X = X.copy()
        self.y = y.copy()
        return self

    def predict(self, X):
        return np.repeat(self.y[0], len(X))


def no_sklearn(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "sklearn" or name.startswith("sklearn."):
            raise ImportError("simulated optional dependency absence")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


def test_feature_matrix_returns_only_supplied_same_schema_vectors():
    rows = [record((1, 2), representation="custom", scale_units="squared_radius"),
            record((3, 4), representation="custom", scale_units="squared_radius")]
    result = feature_matrix(iter(rows))
    np.testing.assert_array_equal(result, [[1, 2], [3, 4]])
    result[0, 0] = 99
    assert rows[0].values[0] == 1


@pytest.mark.parametrize("rows", [[], [record((), ())], [record((1, 2), ("a", "a"))],
    [record((1, 2), "ab")], [record((1,), ("a", "b"))],
    [record([[1, 2]], ("a", "b"))], [record((np.nan, 1))],
    [record((np.inf, 1))], [record((1j, 2))], [object()]])
def test_feature_matrix_rejects_empty_malformed_and_nonfinite(rows):
    with pytest.raises((ValueError, TypeError)):
        feature_matrix(rows)


def test_feature_matrix_preserves_column_order():
    with pytest.raises(ValueError, match="column schemas"):
        feature_matrix([record(), record(names=("f1", "f0"))])


@pytest.mark.parametrize("key,left,right", [
    ("scale_units", "squared_radius", "distance"),
    ("birth_edges", [0, 1, 2], [0, 2, 4]),
    ("death_edges", [0, 1], [0, 2]),
    ("dimensions", (0, 1), (1, 2)),
    ("essential_policy", "separate", "clip"),
    ("entropy_policy", {"base": "natural", "zero": 0}, {"base": 2, "zero": 0}),
    ("operator_kind", "ordinary", "persistent"),
    ("scale", 0.5, 1.0), ("start", 0.0, 0.5), ("end", 1.0, 2.0),
    ("definition_id", "simplicial-v1", "interaction-v1"),
    ("coefficient_field", "GF(2)", "R"),
    ("route", "simplicial", "interaction"),
    ("filtration_start", 0.0, 1.0), ("filtration_end", 1.0, 2.0),
    ("coordinate_units", "arbitrary", "nanometers"),
])
def test_feature_matrix_rejects_semantic_schema_differences(key, left, right):
    with pytest.raises(ValueError, match=key):
        feature_matrix([record(**{key: left}), record(**{key: right})])


def test_nested_schema_arrays_compare_by_values_not_object_identity():
    a = record(schema={"channels": [{"edges": np.array([0, 1, 2])}]})
    b = record(schema={"channels": ({"edges": [0, 1, 2]},)})
    assert feature_matrix([a, b]).shape == (2, 2)
    bad = record(schema={"channels": [{"edges": [0, 1, 3]}]})
    with pytest.raises(ValueError, match="schema"):
        feature_matrix([a, bad])


def test_partial_scope_requires_same_mode_count_not_full_operator_size():
    with pytest.raises(ValueError, match="supplied_eigenvalues"):
        feature_matrix([record(complete_spectrum=False, supplied_eigenvalues=5),
                        record(complete_spectrum=False, supplied_eigenvalues=10)])
    with pytest.raises(ValueError, match="supplied_eigenvalues"):
        feature_matrix([record(complete_spectrum=np.bool_(False))])
    assert feature_matrix([record(complete_spectrum=False, supplied_eigenvalues=5),
                           record(complete_spectrum=False, supplied_eigenvalues=5)]).shape == (2, 2)
    assert feature_matrix([record(complete_spectrum=True, supplied_eigenvalues=5),
                           record(complete_spectrum=True, supplied_eigenvalues=10)]).shape == (2, 2)


def test_custom_estimator_needs_no_sklearn_and_sees_exact_training_split(monkeypatch):
    no_sklearn(monkeypatch)
    whole = np.arange(24).reshape(8, 3)
    labels = np.arange(8)
    train = [0, 3, 6]
    recorder = Recorder()
    result = fit_features(whole[train], labels[train], estimator=recorder)
    assert result is recorder and recorder.fit_calls == 1
    np.testing.assert_array_equal(recorder.X, whole[train])
    np.testing.assert_array_equal(recorder.y, labels[train])
    assert len(recorder.X) == 3


def test_custom_classifier_accepts_string_labels_without_inference(monkeypatch):
    no_sklearn(monkeypatch)
    model = fit_features([[0], [1]], ["low", "high"], task="classification", estimator=Recorder())
    np.testing.assert_array_equal(model.y, ["low", "high"])
    with pytest.raises(ValueError, match="regression targets"):
        fit_features([[0], [1]], ["low", "high"], estimator=Recorder())


@pytest.mark.parametrize("X,y", [([], []), ([[], []], [0, 1]),
    ([1, 2], [0, 1]), ([[np.nan], [1]], [0, 1]), ([[np.inf], [1]], [0, 1]),
    ([[1j], [1]], [0, 1]), ([[0], [1]], [0]), ([[0], [1]], [[0], [1]]),
    ([[0], [1]], [0, np.nan]), ([[0], [1]], [0, np.inf]),
    ([[0], [1]], [0, None]), ([[0], [1]], [0, 1j])])
def test_regression_validates_data_before_custom_fit(X, y):
    recorder = Recorder()
    with pytest.raises((ValueError, TypeError)):
        fit_features(X, y, estimator=recorder)
    assert recorder.fit_calls == 0


@pytest.mark.parametrize("labels", [["a", None], ["a", np.nan], ["a", np.inf],
                                   [0, float("nan")], ["a", 1j]])
def test_classifier_rejects_missing_nonfinite_labels_before_string_coercion(labels):
    recorder = Recorder()
    with pytest.raises(ValueError):
        fit_features([[0], [1]], labels, estimator=recorder, task="classification")
    assert recorder.fit_calls == 0


def test_invalid_custom_contract_and_task_are_rejected():
    with pytest.raises(TypeError, match="fit and predict"):
        fit_features([[0], [1]], [0, 1], estimator=object())
    with pytest.raises(ValueError, match="configure a custom"):
        fit_features([[0], [1]], [0, 1], estimator=Recorder(), n_estimators=2)
    recorder = Recorder()
    with pytest.raises(ValueError, match="task"):
        fit_features([[0], [1]], [0, 1], estimator=recorder, task="infer")
    assert recorder.fit_calls == 0


def test_missing_default_dependency_is_actionable(monkeypatch):
    no_sklearn(monkeypatch)
    with pytest.raises(ImportError, match=r"topokit\[ml\]"):
        make_estimator(task="regression")


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_default_is_unfitted_gradient_boosted_tree_and_fits_only_on_request(task):
    pytest.importorskip("sklearn")
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    model = make_estimator(task=task, random_state=7, n_estimators=3, max_depth=1)
    assert isinstance(model, GradientBoostingRegressor if task == "regression" else GradientBoostingClassifier)
    assert not hasattr(model, "estimators_")
    X = np.arange(12, dtype=float).reshape(6, 2)
    y = np.arange(6, dtype=float) if task == "regression" else np.array([0, 1, 0, 1, 0, 1])
    fitted = fit_features(X, y, task=task, random_state=7, n_estimators=3, max_depth=1)
    assert fitted.predict(X).shape == (6,)


def test_importing_ml_does_not_import_sklearn():
    source = Path(__file__).resolve().parents[1] / "src"
    program = "import sys; import topokit.workflows.ml; assert not any(k == 'sklearn' or k.startswith('sklearn.') for k in sys.modules)"
    result = subprocess.run([sys.executable, "-B", "-c", program],
                            env={**os.environ, "PYTHONPATH": str(source), "PYTHONDONTWRITEBYTECODE": "1"},
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
