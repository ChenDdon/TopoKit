"""Regression-checker safeguards: reject semantic changes and mixed imports."""
import importlib.util
from pathlib import Path
import sys
import textwrap

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "topokit_readability_check", ROOT / "examples" / "pressure_test" / "check_readability.py")
CHECK = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CHECK
SPEC.loader.exec_module(CHECK)


def structural(before, after):
    return CHECK.structural_file(textwrap.dedent(before), textwrap.dedent(after))


def test_local_closure_and_comprehension_renames_keep_binding_identity():
    before = '''
    def transform(sequence):
        """Original docstring."""
        offset = 1
        def shift():
            return offset
        values = [item + offset for item in sequence]
        return values, shift()
    '''
    after = '''
    def transform(sequence):
        """A clearer docstring."""
        delta = 1
        def shift():
            return delta
        shifted = [entry + delta for entry in sequence]
        return shifted, shift()
    '''
    result = structural(before, after)
    assert result["passed"]
    assert result["changed_name_occurrences"] == 7
    assert {(item["before"], item["after"]) for item in result["renames"]} == {
        ("offset", "delta"), ("values", "shifted"), ("item", "entry")}


@pytest.mark.parametrize("before,after", [
    ("def f(x): return x", "def f(y): return y"),
    ("def f(x): return x.value", "def f(x): return x.other"),
    ("def f(x): return {'value': x}", "def f(x): return {'other': x}"),
    ("def f(x): return x + 1", "def f(x): return x + 2"),
    ("def f(x): return abs(x)", "def f(x): return round(x)"),
    ("class Result: value = 1", "class Result: other = 1"),
    ("VERSION = 1", "RELEASE = 1"),
    ("from math import sqrt\ndef f(x): return sqrt(x)",
     "from math import exp\ndef f(x): return exp(x)"),
    ("def f():\n a=1\n b=2\n return a,b", "def f():\n b=2\n a=1\n return a,b"),
])
def test_nonlocal_api_literal_and_statement_changes_are_rejected(before, after):
    assert not structural(before, after)["passed"]


@pytest.mark.parametrize("before,after", [
    ("def f():\n value=1\n return value", "def f():\n result=1\n return value"),
    ("def f():\n a=1\n b=2\n return a+b", "def f():\n value=1\n value=2\n return value+value"),
    ("def f():\n a=1\n def g():\n  b=2\n  return a+b\n return g()",
     "def f():\n b=1\n def g():\n  b=2\n  return b+b\n return g()"),
])
def test_partial_rename_merge_and_closure_capture_are_rejected(before, after):
    assert not structural(before, after)["passed"]


def test_docstrings_and_comments_may_change_but_other_string_expressions_may_not():
    assert structural('"""Old."""\ndef f():\n """Old."""\n return 1',
                      '"""New."""\n# explanation\ndef f():\n """New."""\n return 1')["passed"]
    assert not structural('def f():\n x="old"\n return x',
                          'def f():\n x="new"\n return x')["passed"]


def test_ast_audit_tracks_all_files_and_rejects_file_set_changes(tmp_path):
    baseline, current = tmp_path / "before", tmp_path / "after"
    baseline.mkdir()
    current.mkdir()
    (baseline / "one.py").write_text("def f():\n x=1\n return x\n")
    (current / "one.py").write_text("def f():\n value=1\n return value\n")
    report = CHECK.structural_audit(baseline, current)
    assert report["passed"] and report["files_checked"] == 1
    assert report["identifier_occurrences_changed"] == 2
    (current / "extra.py").write_text("pass\n")
    report = CHECK.structural_audit(baseline, current)
    assert not report["passed"] and report["added_files"] == ["extra.py"]


def test_numerical_comparison_reports_bitwise_and_tolerance_separately():
    baseline = CHECK.encode(np.array([0.0, 1.0]))
    close = CHECK.encode(np.array([-0.0, 1.0 + 1e-12]))
    exact = CHECK.compare_payload(baseline, close)
    tolerant = CHECK.compare_payload(baseline, close, allow_array_tolerance=True)
    assert not exact["passed"]
    assert tolerant["passed"] and not tolerant["exact_payload_equal"]
    assert not tolerant["arrays"][0]["bitwise_equal"]
    assert tolerant["arrays"][0]["allclose"]
    far = CHECK.encode(np.array([0.0, 1.0 + 1e-5]))
    assert not CHECK.compare_payload(baseline, far, allow_array_tolerance=True)["passed"]
    assert not CHECK.compare_payload(0.0, -0.0)["passed"]


def test_exact_birth_records_and_dtype_or_shape_changes_are_not_hidden():
    assert not CHECK.compare_payload([0.0, 1.0], [0.0, 1.0 + 1e-12], allow_array_tolerance=True)["passed"]
    original = CHECK.encode(np.array([1.0, 2.0]))
    for changed in (np.array([1, 2]), np.array([[1.0, 2.0]])):
        assert not CHECK.compare_payload(original, CHECK.encode(changed), allow_array_tolerance=True)["passed"]


def test_corrupted_array_values_cannot_pass_using_stale_hash():
    original = CHECK.encode(np.array([1.0]))
    corrupt = dict(original, values=[2.0])
    report = CHECK.compare_payload(original, corrupt, allow_array_tolerance=True)
    assert not report["passed"]
    assert not report["arrays"][0]["payload_hashes_valid"]


def test_portable_payload_preserves_ids_flags_nonfinite_and_array_hashes(tmp_path):
    payload = CHECK.encode({("sample", 4): (True, float("inf"), np.array([float("nan"), -0.0]))})
    path = tmp_path / "payload.json.gz"
    CHECK.write_compressed(path, payload)
    assert CHECK.read_compressed(path) == payload
    assert CHECK.compare_payload(payload, CHECK.read_compressed(path))["passed"]
    scalar = CHECK.encode(np.asarray(3.0))
    assert scalar["shape"] == []
    assert CHECK.compare_payload(scalar, scalar)["passed"]


def test_worker_import_isolation_and_all_three_family_contracts(tmp_path, monkeypatch):
    # An inherited PYTHONPATH must never select a different revision in a worker.
    poisoned = tmp_path / "poison"
    (poisoned / "topokit").mkdir(parents=True)
    (poisoned / "topokit" / "__init__.py").write_text("raise RuntimeError('wrong revision imported')\n")
    monkeypatch.setenv("PYTHONPATH", str(poisoned))
    output = tmp_path / "worker.json.gz"
    payload = CHECK.run_worker(ROOT / "src", output)
    assert len(payload["cases"]) == 10
    assert {case["kind"] for case in payload["cases"].values()} == {"simplicial", "hyperdigraph", "interaction"}
    assert all(check["passed"] for check in payload["internal_checks"])
    assert len(payload["internal_checks"]) == 110
    assert payload["environment"]["OPENBLAS_NUM_THREADS"] == "1"
    assert payload["environment"]["OMP_NUM_THREADS"] == "1"
    assert all(Path(path).is_relative_to(ROOT / "src") for path in payload["loaded_topokit_modules"].values())
    for case in payload["cases"].values():
        assert len(case["homology"]) == 2
        assert len(case["spectra"]) == 12
        assert all(spectrum["fields"]["complete"] for spectrum in case["spectra"].values())
        assert all(spectrum["fields"]["eigenvectors"] is None for spectrum in case["spectra"].values())
