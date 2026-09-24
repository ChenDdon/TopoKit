"""Local aggregation tests with small evidence fixtures; no numerical work."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


PRESSURE = Path(__file__).resolve().parents[1] / "examples/pressure_test"


@pytest.fixture
def aggregate(monkeypatch):
    monkeypatch.syspath_prepend(str(PRESSURE))
    spec = importlib.util.spec_from_file_location("aggregate_aws_under_test", PRESSURE / "aggregate_aws.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def case(name="case", **updates):
    return {"case_id": name, "profile": "hyperdigraph-complete-distinct", "family": "hyperdigraph",
        "construction": "complete", "operation": "persistence", "degree": 1, "points": 8,
        "weights": "distinct", "overlap": "full", "spectrum": "full", "k": 8,
        "seed": 20260905, "repetition": 0, "execution_route": "canonical", "limits": {"max_hyperedges": 10000}, **updates}


def manifest(parent, suite, specifications):
    run = parent / suite
    run.mkdir(parents=True, exist_ok=True)
    (run / "manifest.json").write_text(json.dumps({"planned_cases": len(specifications), "status": "complete",
        "signature": {"cases": specifications, "source_sha256": {"src/topokit/a.py": "a" * 64}},
        "machine": {"hostname": "fixture-local-host", "platform": "fixtureOS", "python": "3.12"}}))
    return run


def record(run, specification, *, status="success", wall=1.25, cpu=0.9, source="a" * 64):
    directory = run / "cases" / specification["case_id"]
    directory.mkdir(parents=True, exist_ok=True)
    artifact = directory / "checkpoint.npz"
    artifact.write_bytes(b"fixture numerical artifact; content is validated by worker tests")
    worker = {"case": specification, "source_hash": source, "input_hash": "b" * 64,
        "execution_route": specification["execution_route"], "stage": "complete" if status == "success" else "analysis",
        "completed_stages": ["construction", "analysis"] if status == "success" else ["construction"],
        "timings_seconds": {"construction": .1, "analysis": .2},
        "construction_summary": {"support_edge_count": 28, "directed_edge_count": 28,
            "cell_count": 92, "cells_by_degree": {"0": 8, "1": 28, "2": 56}},
        "diagnostics": {"analysis": {"diagnostics": {"low_dimensional_backend": "fixture_backend"}}},
        "result_summary": {"complete": True, "interval_count": 8, "artifact": {
            "filename": "checkpoint.npz", "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}},
        "resources": {"elapsed_seconds": .75, "cpu_seconds": .6, "peak_rss_bytes": 16 * 1024**2}}
    value = {"case_id": specification["case_id"], "case": specification, "worker": worker,
        "status": status, "last_stage": worker["stage"], "wall_seconds": wall,
        "cpu_seconds": cpu, "peak_rss_bytes": 20 * 1024**2, "wall_limit_seconds": 300,
        "rss_limit_bytes": 1024**3, "address_limit_bytes": 2 * 1024**3}
    (directory / "record.json").write_text(json.dumps(value))
    return value, directory


def test_missing_suites_remain_unrun_and_stdlib_cli_writes_three_reports(aggregate, tmp_path):
    data = aggregate.collect(tmp_path)
    assert data["integrity"] == "pass" and not data["cases"]
    assert len(data["suites"]) == 7
    assert all(suite["state"] == "unrun" and suite["planned_count"] is None for suite in data["suites"])
    process = subprocess.run([sys.executable, "-S", str(PRESSURE / "aggregate_aws.py"), str(tmp_path)],
                             capture_output=True, text=True, timeout=10)
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)["terminal_count"] == 0
    assert {path.name for path in tmp_path.iterdir()} == {"OVERVIEW.md", "combined.csv", "combined.json"}
    overview = (tmp_path / "OVERVIEW.md").read_text()
    assert "unrun" in overview and "0 / unknown" in overview
    assert "Unrun suites above remain untested" in overview


def test_largest_success_max_attempt_and_unstarted_are_distinct(aggregate, tmp_path):
    specifications = [case("small", points=8), case("large-timeout", points=64), case("unstarted", points=1024)]
    run = manifest(tmp_path, "pressure_q1", specifications)
    record(run, specifications[0])
    failed, directory = record(run, specifications[1], status="timeout", wall=300.03)
    failed["worker"]["exception"] = {"class": "ExampleError", "message": "detail | line\nnext"}
    (directory / "record.json").write_text(json.dumps(failed))
    data = aggregate.collect(tmp_path)
    assert data["terminal_count"] == data["attempted_count"] == 2
    group = next(group for group in data["largest_tested_groups"] if group["largest_successful_points"] is not None)
    assert group["largest_successful_points"] == 8 and group["max_tried_points"] == 64
    assert group["max_tried_case_keys"] == ["pressure_q1/large-timeout"]
    unstarted = next(row for row in data["cases"] if row["case_id"] == "unstarted")
    assert unstarted["status"] == "not_started" and not unstarted["attempted"] and unstarted["wall_seconds"] is None
    assert next(suite for suite in data["suites"] if suite["suite"] == "pressure_q1")["state"] == "incomplete"
    overview = aggregate.overview(data)
    assert "300.030 (censored)" in overview
    assert "ExampleError: detail \\| line<br>next" in overview
    assert "not capacity estimates" in overview


def test_same_ids_across_suites_are_individual_trials_not_duplicates(aggregate, tmp_path):
    specification = case()
    for suite, wall in (("smoke", 1.25), ("pressure_q1", 3.75)):
        run = manifest(tmp_path, suite, [specification])
        record(run, specification, wall=wall)
    data = aggregate.collect(tmp_path)
    assert data["integrity"] == "pass"
    assert len(data["largest_tested_groups"]) == 1
    group = data["largest_tested_groups"][0]
    assert group["largest_successful_case_keys"] == ["smoke/case", "pressure_q1/case"]
    overview = aggregate.overview(data)
    assert "1.250–3.750" in overview and "fixture-local-host / fixtureOS" in overview
    assert "fixture-local-host; source aaaaaaaaaaaa" in overview
    assert "not averages or speedup comparisons" in overview
    assert data["cases"][0]["requested_limits"] == {"max_hyperedges": 10000}
    assert data["cases"][0]["wall_limit_seconds"] == 300


def test_grouping_separates_route_full_partial_weights_overlap_and_degree(aggregate, tmp_path):
    specifications = [case("base", operation="laplacian"), case("partial", operation="laplacian", spectrum="partial"),
        case("partial-k4", operation="laplacian", spectrum="partial", k=4),
        case("equal", operation="laplacian", weights="equal"), case("half", operation="laplacian", overlap="half"),
        case("higher", operation="laplacian", degree=2), case("compact", execution_route="compact_hyperdigraph")]
    run = manifest(tmp_path, "supplemental", specifications)
    for specification in specifications:
        record(run, specification)
    data = aggregate.collect(tmp_path)
    assert len(data["largest_tested_groups"]) == len(specifications)
    assert next(row for row in data["cases"] if row["case_id"] == "partial")["complete_spectrum"] is True
    overview = aggregate.overview(data)
    assert "partial k=8" in overview and "partial k=4" in overview
    assert "Higher-dimensional q2" in overview and "Low-dimensional q0/q1" in overview


@pytest.mark.parametrize("problem", ["missing_artifact", "mismatched_artifact", "missing_manifest", "duplicate_ids"])
def test_integrity_failure_exits_nonzero_and_excludes_success_claim(aggregate, tmp_path, problem):
    specification = case()
    run = manifest(tmp_path, "smoke", [specification])
    saved, directory = record(run, specification)
    if problem == "missing_artifact":
        (directory / "checkpoint.npz").unlink()
    elif problem == "mismatched_artifact":
        (directory / "checkpoint.npz").write_bytes(b"changed")
    elif problem == "missing_manifest":
        (run / "manifest.json").unlink()
    else:
        (run / "results.jsonl").write_text(json.dumps(saved) + "\n" + json.dumps(saved) + "\n")
    data = aggregate.collect(tmp_path)
    assert data["integrity"] == "fail" and data["issues"]
    assert all(group["largest_successful_points"] is None for group in data["largest_tested_groups"])
    assert aggregate.main([str(tmp_path)]) == 1


def test_cross_suite_sources_are_reported_and_grouped_separately(aggregate, tmp_path):
    for suite, source in (("smoke", "a" * 64), ("compact_smoke", "b" * 64)):
        run = manifest(tmp_path, suite, [case()])
        record(run, case(), source=source)
    data = aggregate.collect(tmp_path)
    assert data["integrity"] == "fail" and len(data["largest_tested_groups"]) == 2
    assert any("across suites" in issue for issue in data["issues"])


def test_checkpoint_cpu_rss_counts_backend_and_compact_times_survive_exports(aggregate, tmp_path):
    specification = case("checkpoint", execution_route="compact_hyperdigraph", points=200)
    run = manifest(tmp_path, "compact_pressure", [specification])
    saved, directory = record(run, specification)
    worker = saved["worker"]
    worker["timings_seconds"] = {"construction_and_analysis": 5., "canonical_equivalence": .25}
    worker["construction_summary"].update(cell_count_semantics="packed vertices and edges only",
        represented_paths_by_degree={"0": 200, "1": 19900, "2": 1313400})
    worker["canonical_equivalence"] = {"status": "within_tolerance", "points": 8, "scope": "independent small prefix"}
    worker["stage"] = "canonical_equivalence"
    (directory / "record.json").unlink()
    (directory / "checkpoint.json").write_text(json.dumps(worker))
    data = aggregate.collect(tmp_path)
    row = data["cases"][0]
    assert row["status"] == "no_terminal_record" and row["attempted"] and not row["terminal"]
    assert row["wall_seconds"] == .75 and row["cpu_seconds"] == .6 and row["peak_rss_bytes"] == 16 * 1024**2
    assert row["analysis_seconds"] is None and row["construction_and_analysis_seconds"] == 5.
    assert row["backend"] == "low=fixture_backend" and row["probe_points"] == 8
    assert data["largest_tested_groups"][0]["max_tried_points"] == 200
    assert data["largest_tested_groups"][0]["largest_successful_points"] is None
    aggregate.write_reports(data, tmp_path / "output")
    decoded = json.loads((tmp_path / "output/combined.json").read_text())
    assert decoded["cases"][0]["represented_paths_by_degree"]["2"] == 1313400
    with (tmp_path / "output/combined.csv").open(newline="") as handle:
        csv_row = next(csv.DictReader(handle))
    assert csv_row["analysis_seconds"] == "" and csv_row["construction_and_analysis_seconds"] == "5.0"
    assert json.loads(csv_row["represented_paths_by_degree"])["2"] == 1313400
    assert "0.750 (checkpoint)" in (tmp_path / "output/OVERVIEW.md").read_text()


def test_interaction_points_and_factor_edges_are_preserved(aggregate, tmp_path):
    specification = case(family="interaction", construction="rips", profile="interaction-rips", points=16, overlap="half")
    run = manifest(tmp_path, "pressure_q1", [specification])
    saved, directory = record(run, specification)
    saved["worker"]["input_summary"] = {"points_per_cloud": [16, 16]}
    saved["worker"]["construction_summary"]["factor_support_edge_counts"] = [120, 120]
    (directory / "record.json").write_text(json.dumps(saved))
    row = aggregate.collect(tmp_path)["cases"][0]
    assert row["points_per_cloud"] == [16, 16] and row["factor_support_edge_counts"] == [120, 120]
