"""Small scientific and failure-path checks for the isolated pressure worker."""
import sys

import pytest

if sys.platform == "win32":
    pytest.skip("pressure workers require POSIX resource", allow_module_level=True)

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import warnings

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("pressure_worker", ROOT / "examples/pressure_test/worker.py")
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


def make_case(family="simplicial", construction="rips", operation="persistence", **updates):
    case = {"case_id": f"{family}-{construction}-{operation}", "family": family,
            "construction": construction, "operation": operation, "points": 5,
            "ambient_dimension": 3, "seed": 20260905, "degree": 1,
            "weights": "distinct", "overlap": "full", "spectrum": "full", "k": 2}
    case.update(updates)
    return case


@pytest.mark.parametrize("family,construction", [
    ("simplicial", "alpha"), ("simplicial", "rips"),
    ("hyperdigraph", "delaunay"), ("hyperdigraph", "complete"),
    ("interaction", "alpha"), ("interaction", "rips"),
])
@pytest.mark.parametrize("operation", ["persistence", "laplacian", "persistent_laplacian"])
def test_public_routes_operations_and_observation_metadata(tmp_path, family, construction, operation):
    case = make_case(family, construction, operation)
    report = worker.run_case(case, tmp_path / "result.json")
    assert report["status"] == "success", report.get("exception")
    assert json.loads((tmp_path / "result.json").read_text()) == report
    assert report["resources"]["peak_rss_bytes"] > 0
    assert report["resources"]["cpu_seconds"] >= 0
    assert set(report["timings_seconds"]) == {
        "initialization", "generation", "support_preparation", "construction",
        "observation_selection", "analysis", "summarization", "export"}
    assert all(value >= 0 for value in report["timings_seconds"].values())
    assert len(report["input_hash"]) == len(report["source_hash"]) == 64
    counts = report["construction_summary"]["cells_by_degree"]
    assert set(counts) == {"0", "1", "2"}
    observations = report["observations"]
    assert 0 < observations["start"] <= observations["end"]
    assert observations["source_degree_cell_count"] <= observations["target_degree_cell_count"]
    summary = report["result_summary"]
    artifact = summary["artifact"]
    assert len(artifact["sha256"]) == 64
    with np.load(tmp_path / artifact["filename"], allow_pickle=False) as saved:
        if operation == "persistence":
            assert saved["degree"].size == summary["interval_count"]
            assert saved["birth"].shape == saved["death"].shape == saved["at_initial_stage"].shape
            canonical = [[int(q), float(birth).hex(), float(death).hex()]
                         for q, birth, death in sorted(zip(saved["degree"], saved["birth"], saved["death"]))]
            assert worker._digest(canonical) == summary["intervals_sha256"]
        else:
            assert saved["eigenvalues"].size == summary["size"]
            assert worker._digest([float(value).hex() for value in saved["eigenvalues"]]) == summary["eigenvalues_sha256"]
    if operation == "persistence":
        assert summary["field"] == "GF(2)"
        assert len(summary["intervals_sha256"]) == 64
        assert summary["checks"]["valid_intervals"]
    else:
        assert summary["complete"]
        assert summary["size"] == summary["basis_size"]
        assert summary["checks"]["finite"] and summary["checks"]["psd_within_tolerance"]
        assert len(summary["eigenvalues_sha256"]) == 64
        if operation == "laplacian":
            assert summary["scale"] == observations["end"]
            assert summary["start"] is None and summary["end"] is None
        else:
            assert summary["start"] == observations["start"]
            assert summary["end"] == observations["end"]
            assert summary["scale"] is None


@pytest.mark.parametrize("family,construction", [
    ("simplicial", "rips"), ("hyperdigraph", "complete"), ("interaction", "rips")])
@pytest.mark.parametrize("degree", [0, 2])
def test_h0_h2_and_q_plus_one_cells(tmp_path, family, construction, degree):
    case = make_case(family, construction, "persistent_laplacian", degree=degree)
    report = worker.run_case(case, tmp_path / "result.json")
    assert report["status"] == "success", report.get("exception")
    assert report["construction_summary"]["cells_by_degree"][str(degree + 1)] > 0
    assert report["result_summary"]["degree"] == degree


def test_reproducible_independent_interaction_inputs_and_overlap():
    case = worker.validate_case(make_case("interaction", "rips", overlap="half", points=6))
    clouds = worker.generate_inputs(case)
    repeated = worker.generate_inputs(case)
    rng = np.random.default_rng(case["seed"])
    np.testing.assert_array_equal(clouds[0].points, rng.random((6, 3)))
    np.testing.assert_array_equal(clouds[1].points, rng.random((6, 3)))
    assert not np.array_equal(clouds[0].points, clouds[1].points)
    bonds, pairs = worker.prepare_support(case, clouds)
    assert pairs == tuple(zip(clouds[0].ids[:3], clouds[1].ids[:3]))
    assert worker.input_hash(clouds, bonds, pairs) == worker.input_hash(repeated, bonds, pairs)
    full = {**case, "overlap": "full"}
    _, all_pairs = worker.prepare_support(full, clouds)
    assert len(all_pairs) == 6
    assert worker.input_hash(clouds, None, pairs) != worker.input_hash(clouds, None, all_pairs)


@pytest.mark.parametrize("weights,multiplier", [("distinct", 1), ("equal", 2)])
def test_complete_support_and_weight_orientation(tmp_path, weights, multiplier):
    case = make_case("hyperdigraph", "complete", degree=0, weights=weights)
    report = worker.run_case(case, tmp_path / "result.json")
    assert report["status"] == "success", report.get("exception")
    assert report["input_summary"]["supplied_bond_count"] == 10
    assert report["construction_summary"]["support_edge_count"] == 10
    assert report["construction_summary"]["directed_edge_count"] == 10 * multiplier


@pytest.mark.parametrize("family,construction", [
    ("simplicial", "rips"), ("hyperdigraph", "complete"), ("interaction", "rips")])
def test_partial_spectrum_is_explicit(tmp_path, family, construction):
    case = make_case(family, construction, "laplacian", spectrum="partial", k=2)
    report = worker.run_case(case, tmp_path / "result.json")
    assert report["status"] == "success", report.get("exception")
    summary = report["result_summary"]
    assert summary["size"] == 2
    assert summary["basis_size"] > 2
    assert not summary["complete"] and summary["nullity"] is None


def test_no_positive_birth_fallback_is_declared(tmp_path):
    case = make_case(points=1, degree=0, operation="persistent_laplacian")
    report = worker.run_case(case, tmp_path / "result.json")
    assert report["status"] == "success", report.get("exception")
    assert report["observations"]["fallback_used"]
    assert report["observations"]["start"] == report["observations"]["end"] == 0
    assert report["result_summary"]["eigenvalues_sha256"]


@pytest.mark.parametrize("family,construction,limit", [
    ("simplicial", "rips", "max_simplices"),
    ("hyperdigraph", "complete", "max_hyperedges"),
    ("interaction", "rips", "max_cells"),
])
def test_guards_remain_resource_limits_with_finished_checkpoints(tmp_path, family, construction, limit):
    report = worker.run_case(make_case(family, construction, limits={limit: 1}), tmp_path / "result.json")
    assert report["status"] == "resource_limit"
    assert report["exception"]["stage"] == "construction"
    assert "support_preparation" in report["completed_stages"]
    assert "construction" in report["timings_seconds"]
    assert "analysis" not in report["timings_seconds"]
    assert "traceback_path" not in report["exception"]


def test_memory_error_retains_running_stage_and_warning(tmp_path, monkeypatch):
    output = tmp_path / "result.json"

    def fail_construction(*args):
        checkpoint = json.loads(output.read_text())
        assert checkpoint["status"] == "running" and checkpoint["stage"] == "construction"
        assert "support_preparation" in checkpoint["completed_stages"]
        warnings.warn("synthetic allocation warning", RuntimeWarning)
        raise MemoryError("synthetic allocation failure")

    monkeypatch.setattr(worker, "build_topology", fail_construction)
    report = worker.run_case(make_case(), output)
    assert report["status"] == "memory_error"
    assert report["exception"]["class"] == "MemoryError"
    assert report["exception"]["stage"] == "construction"
    assert report["warnings"][-1]["message"] == "synthetic allocation warning"
    assert report["warnings"][-1]["stage"] == "construction"
    assert not list(tmp_path.glob(".*.json.*"))


def test_true_errors_write_traceback_and_summarization_rejects_nan(tmp_path):
    report = worker.run_case(make_case(construction="invented"), tmp_path / "error.json")
    assert report["status"] == "error"
    assert Path(report["exception"]["traceback_path"]).is_file()
    from topokit.results import SpectrumResult
    with pytest.raises(ArithmeticError):
        worker.summarize_result(make_case(operation="laplacian"),
                                SpectrumResult(1, np.array([np.nan]), basis=("x",)))


@pytest.mark.parametrize("operation", ["persistence", "laplacian", "persistent_laplacian"])
def test_failed_scientific_checks_have_distinct_status(tmp_path, monkeypatch, operation):
    from topokit.results import PersistenceInterval, PersistenceResult, SpectrumResult
    invalid = (PersistenceResult((PersistenceInterval(0, np.nan, np.inf),)) if operation == "persistence"
               else SpectrumResult(1, np.array([-1.0]), basis=("x",)))
    monkeypatch.setattr(worker, "analyze", lambda *args: invalid)
    report = worker.run_case(make_case(operation=operation), tmp_path / "invalid.json")
    assert report["status"] == "numerical_failure"
    assert report["exception"]["class"] == "ScientificCheckError"
    assert Path(report["exception"]["traceback_path"]).is_file()
    assert not (tmp_path / "invalid.npz").exists()


def test_construction_counts_survive_observation_failure(tmp_path, monkeypatch):
    output = tmp_path / "result.json"

    def fail_observation(*args):
        checkpoint = json.loads(output.read_text())
        assert checkpoint["stage"] == "observation_selection"
        assert checkpoint["construction_summary"]["cell_count"] > 0
        assert "construction" in checkpoint["completed_stages"]
        raise MemoryError("synthetic observation failure")

    monkeypatch.setattr(worker, "select_observations", fail_observation)
    report = worker.run_case(make_case(), output)
    assert report["status"] == "memory_error"
    assert report["construction_summary"]["cells_by_degree"]["0"] == 5
    assert "available" in report["environment"]["runtime_threadpools"]


def test_cli_status_and_lazy_import(tmp_path):
    case_path, output = tmp_path / "case.json", tmp_path / "result.json"
    case_path.write_text(json.dumps(make_case(points=3, degree=0)))
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src"),
                   "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    process = subprocess.run([sys.executable, str(ROOT / "examples/pressure_test/worker.py"),
        "--case", str(case_path), "--output", str(output)], env=environment,
        text=True, capture_output=True, timeout=30)
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)["status"] == "success"
    loader = (
        "import importlib.util,sys; "
        f"s=importlib.util.spec_from_file_location('w',{str(ROOT / 'examples/pressure_test/worker.py')!r}); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        "assert 'topokit' not in sys.modules and 'numpy' not in sys.modules and 'scipy' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", loader], env=environment, check=True, timeout=10)
