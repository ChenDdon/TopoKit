"""Small correctness and reporting checks for the separate compact pressure route."""
import sys

import pytest

if sys.platform == "win32":
    pytest.skip("pressure workers require POSIX resource", allow_module_level=True)

import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np


PRESSURE_DIR = Path(__file__).resolve().parents[1] / "examples" / "pressure_test"


@pytest.fixture
def fast_worker(monkeypatch):
    monkeypatch.syspath_prepend(str(PRESSURE_DIR))
    spec = importlib.util.spec_from_file_location("pressure_fast_worker_under_test", PRESSURE_DIR / "fast_worker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_case(**updates):
    case = {"case_id": "compact-test", "family": "hyperdigraph", "construction": "complete",
        "operation": "persistence", "points": 8, "degree": 1, "weights": "distinct",
        "seed": 20260905, "execution_route": "compact_hyperdigraph", "limits": {}}
    return {**case, **updates}


@pytest.mark.parametrize("construction", ["delaunay", "complete"])
@pytest.mark.parametrize("degree", [0, 1])
def test_public_compact_route_matches_small_canonical_and_exports_exact_values(
        fast_worker, tmp_path, construction, degree):
    report = fast_worker.run_case(make_case(construction=construction, degree=degree), tmp_path / "checkpoint.json")
    assert report["status"] == "success", report.get("exception")
    assert report["execution_route"] == "compact_hyperdigraph"
    assert json.loads((tmp_path / "checkpoint.json").read_text()) == report
    assert report["execution"]["actual_backend"] == (
        "score_dag_union_find_early_stop" if degree == 0 else "modified_dlw_implicit_two_family")
    assert report["applied_limits"]["construction"] == report["applied_limits"]["analysis"] == {}
    assert "construction_and_analysis" in report["timings_seconds"]
    assert "analysis" not in report["timings_seconds"] and "construction" not in report["timings_seconds"]
    counts = report["construction_summary"]
    assert counts["directed_edge_count"] == counts["support_edge_count"] > 0
    assert counts["cell_count"] == 8 + counts["directed_edge_count"]
    assert counts["materialized_hyperedges_by_degree"]["2"] == 0
    if construction == "complete":
        assert counts["directed_edge_count"] == 28
        if degree == 1:
            assert counts["represented_paths_by_degree"]["2"] == 56
    elif degree == 1:
        assert counts["represented_paths_by_degree"]["2"] is None
    comparison = report["canonical_equivalence"]
    assert comparison["within_tolerance"] and comparison["same_interval_counts_by_degree"]
    assert comparison["scope"] == "entire case"
    assert comparison["same_infinite_endpoint_mask"]
    summary = report["result_summary"]
    artifact = summary["artifact"]
    with np.load(tmp_path / artifact["filename"], allow_pickle=False) as saved:
        triples = sorted(zip(saved["degree"], saved["birth"], saved["death"]))
        digest_data = [[int(q), float(birth).hex(), float(death).hex()] for q, birth, death in triples]
        assert fast_worker.worker._digest(digest_data) == summary["intervals_sha256"]
        assert len(triples) == summary["interval_count"]
        np.testing.assert_array_equal(saved["at_initial_stage"], saved["birth"] == 0.0)
    assert report["resources"]["peak_rss_bytes"] > 0


@pytest.mark.parametrize("construction", ["delaunay", "complete"])
def test_nonzero_h1_interval_with_tied_edge_births_matches_canonical(fast_worker, construction):
    from topokit.data import PointCloud
    from topokit.builders import hyperdigraph
    from topokit import core

    case = fast_worker.validate_case(make_case(points=4, construction=construction))
    cloud = PointCloud(np.asarray([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]]),
        ids=("a", "b", "c", "d"), weights=np.arange(4, dtype=float))
    compact = fast_worker.common_result(fast_worker.compute_compact(case, cloud))
    bonds, _ = fast_worker.worker.prepare_support(case, (cloud,))
    canonical = core.persistence(hyperdigraph.from_points(cloud, max_dimension=1, bonds=bonds), max_dimension=1)
    h1 = [item for item in compact.intervals if item.dimension == 1]
    assert len(h1) == 1 and h1[0].birth == 1.0 and h1[0].death == np.sqrt(2.)
    assert fast_worker.compare_intervals(compact, canonical)["exact_endpoint_match"]


def test_larger_case_probe_is_explicitly_bounded_and_not_claimed_as_full_equivalence(fast_worker, tmp_path):
    report = fast_worker.run_case(make_case(points=12), tmp_path / "checkpoint.json")
    assert report["status"] == "success", report.get("exception")
    comparison = report["canonical_equivalence"]
    assert comparison["points"] == fast_worker.EQUIVALENCE_POINTS < report["case"]["points"]
    assert "does not verify all large-case intervals" in comparison["scope"]


@pytest.mark.parametrize("change", [
    {"weights": "equal"}, {"family": "simplicial", "construction": "rips"},
    {"operation": "laplacian"}, {"operation": "persistent_laplacian"}, {"degree": 2},
    {"limits": {"max_chain_bytes": 1}}, {"execution_route": "canonical"},
    {"filtration_cutoff": 0.5}, {"points": 1}, {"points": 65_536},
    {"overlap": "half"}, {"spectrum": "partial"},
])
def test_ineligible_cases_fail_durably_before_computation(fast_worker, tmp_path, monkeypatch, change):
    def forbidden(*args):
        pytest.fail("ineligible compact case must not invoke scientific workflow")
    monkeypatch.setattr(fast_worker, "compute_compact", forbidden)
    report = fast_worker.run_case(make_case(**change), tmp_path / "checkpoint.json")
    assert report["status"] == "error" and report["stage"] == "initialization"
    assert report["exception"]["class"] == "ValueError"
    assert json.loads((tmp_path / "checkpoint.json").read_text()) == report


def test_one_ulp_difference_is_reported_and_not_rounded(fast_worker):
    from topokit.results import PersistenceInterval, PersistenceResult

    exact = PersistenceResult((PersistenceInterval(1, .25, .5),))
    changed_value = float(np.nextafter(.5, 1.))
    changed = PersistenceResult((PersistenceInterval(1, .25, changed_value),))
    result = fast_worker.compare_intervals(changed, exact)
    assert not result["exact_endpoint_match"] and result["within_tolerance"]
    assert result["status"] == "within_tolerance" and result["differing_finite_endpoints"] == 1
    assert result["max_absolute_endpoint_difference"] == changed_value - .5
    assert result["endpoint_difference_examples"][0]["compact_hex"] == changed_value.hex()
    assert result["endpoint_difference_examples"][0]["canonical_hex"] == (.5).hex()
    assert changed.intervals[0].death == changed_value


@pytest.mark.parametrize("intervals", [[], [(1, .25, .7)], [(1, .25, np.inf)], [(0, .25, .5)]])
def test_equivalence_rejects_count_degree_infinity_and_real_endpoint_mismatches(fast_worker, intervals):
    from topokit.results import PersistenceInterval, PersistenceResult

    exact = PersistenceResult((PersistenceInterval(1, .25, .5),))
    changed = PersistenceResult(tuple(PersistenceInterval(*item) for item in intervals))
    comparison = fast_worker.compare_intervals(changed, exact)
    assert not comparison["within_tolerance"] and comparison["status"] == "mismatch"


@pytest.mark.parametrize("exception_name,expected_status", [("MemoryError", "memory_error"),
    ("ResourceLimitError", "resource_limit"), ("RuntimeError", "error")])
def test_combined_stage_failure_retains_timing_and_explicit_status(
        fast_worker, tmp_path, monkeypatch, exception_name, expected_status):
    from topokit.exceptions import ResourceLimitError

    error_type = {"MemoryError": MemoryError, "ResourceLimitError": ResourceLimitError,
                  "RuntimeError": RuntimeError}[exception_name]
    def fail(*args):
        raise error_type("synthetic compact failure")
    monkeypatch.setattr(fast_worker, "compute_compact", fail)
    report = fast_worker.run_case(make_case(), tmp_path / "checkpoint.json")
    assert report["status"] == expected_status
    assert report["exception"]["stage"] == "construction_and_analysis"
    assert report["timings_seconds"]["construction_and_analysis"] >= 0
    assert json.loads((tmp_path / "checkpoint.json").read_text()) == report


def test_failed_scientific_probe_preserves_the_completed_result_artifact(fast_worker, tmp_path, monkeypatch):
    monkeypatch.setattr(fast_worker, "canonical_equivalence", lambda *args: {
        "within_tolerance": False, "status": "mismatch"})
    report = fast_worker.run_case(make_case(), tmp_path / "checkpoint.json")
    assert report["status"] == "numerical_failure"
    assert report["canonical_equivalence"]["status"] == "mismatch"
    assert (tmp_path / report["result_summary"]["artifact"]["filename"]).is_file()


def test_cli_malformed_input_is_a_checkpointed_failure(fast_worker, tmp_path):
    case_path = tmp_path / "case.json"
    output = tmp_path / "checkpoint.json"
    case_path.write_text("not JSON")
    completed = subprocess.run([sys.executable, "-B", str(PRESSURE_DIR / "fast_worker.py"),
        "--case", str(case_path), "--output", str(output)], capture_output=True, text=True, timeout=10)
    assert completed.returncode == 1
    assert json.loads(completed.stdout)["status"] == "error"
    assert json.loads(output.read_text())["exception"]["class"] == "JSONDecodeError"
