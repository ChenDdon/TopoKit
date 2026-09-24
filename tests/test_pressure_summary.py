"""Evidence-only report tests; no pressure workers or numerical libraries run."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples/pressure_test/summarize.py"
SPEC = importlib.util.spec_from_file_location("pressure_summary", SCRIPT)
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def case(case_id="case-a", **updates):
    value = {"case_id": case_id, "profile": "interaction-rips", "family": "interaction",
             "construction": "rips", "operation": "laplacian", "points": 6, "degree": 1,
             "spectrum": "full", "overlap": "half", "weights": "equal", "repetition": 0, "k": 8}
    value.update(updates)
    return value


def write_manifest(run, cases):
    run.mkdir(parents=True, exist_ok=True)
    manifest = {"planned_cases": len(cases), "status": "paused", "created_utc": "2026-09-05",
        "machine": {"hostname": "local-mac", "platform": "macOS-arm64", "python": "3.12",
                    "logical_cpus": 8, "thread_environment": {"OPENBLAS_NUM_THREADS": "1"}},
        "signature": {"cases": cases, "source_sha256": {"src/topokit/__init__.py": "a" * 64}}}
    (run / "manifest.json").write_text(json.dumps(manifest))


def write_record(run, specification, *, status="success", source="a" * 64, complete=True, folder=None):
    directory = run / "cases" / (folder or specification["case_id"])
    directory.mkdir(parents=True, exist_ok=True)
    artifact = directory / "checkpoint.npz"
    # A tiny ZIP fixture is sufficient: this reader verifies existence/hash,
    # while worker tests separately verify the numerical NPZ array contents.
    with zipfile.ZipFile(artifact, "w") as archive:
        archive.writestr("fixture", b"saved numerical fixture")
    worker = {"source_hash": source, "stage": "complete" if status == "success" else "analysis",
        "completed_stages": ["generation", "support_preparation", "construction", "analysis"] if status == "success" else ["generation", "construction"],
        "timings_seconds": {"construction": 0.02, "analysis": 0.03},
        "input_summary": {"points_per_cloud": [6, 6]},
        "construction_summary": {"factor_support_edge_counts": [12, 13], "cell_count": 95},
        "environment": {"numpy": "1.26", "scipy": "1.13", "runtime_threadpools": {
            "available": True, "pools": [{"internal_api": "openblas", "version": "0.3", "num_threads": 1}]}},
        "result_summary": {"basis_size": 19, "size": 19 if complete else 8, "complete": complete,
            "artifact": {"filename": "checkpoint.npz", "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}}}
    record = {"case_id": specification["case_id"], "case": specification, "status": status,
        "last_stage": worker["stage"], "wall_seconds": 0.25, "wall_limit_seconds": 0.2,
        "peak_rss_bytes": 20 * 1024**2, "exit_code": 0 if status == "success" else -9,
        "worker": worker}
    (directory / "record.json").write_text(json.dumps(record))
    return record, directory


def test_report_host_scope_counts_full_partial_and_artifacts(tmp_path):
    run = tmp_path / "local_smoke"
    specs = [case(), case("case-b", spectrum="partial")]
    write_manifest(run, specs)
    write_record(run, specs[0])
    write_record(run, specs[1], complete=False)
    report, audit = summary.summarize_run(run)
    assert not audit["issues"]
    assert audit["artifact_checks"] == {"case-a": "pass", "case-b": "pass"}
    assert "local-mac" in report and "macOS-arm64" in report and str(run.resolve()) in report
    assert "AWS" not in report
    assert "2 / 2 planned" in report
    assert "full → complete" in report and "partial k=8 → partial" in report
    assert "6 / 6" in report and "12 / 13" in report and "95" in report and "19" in report
    assert "one trial" in report and "openblas 0.3: 1 threads" in report


@pytest.mark.parametrize("problem", ["missing", "mismatch", "absent_metadata", "unsafe_name"])
def test_bad_artifact_is_reported_and_cli_returns_nonzero(tmp_path, problem):
    specs = [case()]
    write_manifest(tmp_path, specs)
    record, directory = write_record(tmp_path, specs[0])
    if problem == "missing":
        (directory / "checkpoint.npz").unlink()
    elif problem == "mismatch":
        (directory / "checkpoint.npz").write_bytes(b"changed")
    elif problem == "absent_metadata":
        record["worker"]["result_summary"].pop("artifact")
    else:
        record["worker"]["result_summary"]["artifact"]["filename"] = "../outside.npz"
    (directory / "record.json").write_text(json.dumps(record))
    report, audit = summary.summarize_run(tmp_path)
    assert audit["artifact_checks"]["case-a"] == "fail"
    assert "Artifact status: **fail**" in report
    assert summary.main([str(tmp_path), "--output", str(tmp_path / "report.md")]) == 1


def test_timeout_and_unfinished_records_are_not_completion_times(tmp_path):
    specs = [case(), case("case-b")]
    write_manifest(tmp_path, specs)
    record, directory = write_record(tmp_path, specs[0], status="timeout")
    record["worker"]["exception"] = {"class": "ExampleError", "message": "failure | detail\nsecond line"}
    (directory / "record.json").write_text(json.dumps(record))
    report, audit = summary.summarize_run(tmp_path)
    assert not audit["issues"]
    assert "1 / 2 planned" in report and "no_terminal_record" in report
    assert "0.250 (censored)" in report
    assert "not a completed runtime" in report
    assert "0.030 (partial)" in report
    assert "ExampleError: failure \\| detail<br>second line" in report
    assert "analysis" in report and "not started" in report


def test_repetitions_are_labeled_individually_and_complete_partial_request(tmp_path):
    specs = [case("trial-0", spectrum="partial", repetition=0), case("trial-1", spectrum="partial", repetition=1)]
    write_manifest(tmp_path, specs)
    for specification in specs:
        write_record(tmp_path, specification, complete=True)
    report, audit = summary.summarize_run(tmp_path)
    assert not audit["issues"]
    assert "r=0; 2 planned trials" in report and "r=1; 2 planned trials" in report
    assert "partial k=8 → complete" in report
    assert "times are not averages" in report


def test_duplicate_ids_and_source_hashes_fail_integrity(tmp_path):
    specs = [case(), case("case-b")]
    write_manifest(tmp_path, specs)
    first, _ = write_record(tmp_path, specs[0])
    write_record(tmp_path, specs[1], source="b" * 64)
    (tmp_path / "results.jsonl").write_text(json.dumps(first) + "\n" + json.dumps(first) + "\n")
    report, audit = summary.summarize_run(tmp_path)
    assert any("Duplicate case IDs" in issue for issue in audit["issues"])
    assert any("Inconsistent worker package source hashes" in issue for issue in audit["issues"])
    assert "Integrity checks: **fail**" in report


def test_jsonl_fallback_and_missing_success_source_hash(tmp_path):
    specs = [case()]
    write_manifest(tmp_path, specs)
    record, directory = write_record(tmp_path, specs[0], source=None)
    (directory / "record.json").unlink()
    (tmp_path / "results.jsonl").write_text(json.dumps(record) + "\n")
    _, audit = summary.summarize_run(tmp_path)
    assert audit["record_count"] == 1
    assert audit["artifact_checks"]["case-a"] == "pass"
    assert any("lacks a package source hash" in issue for issue in audit["issues"])


def test_cli_stdlib_only(tmp_path):
    write_manifest(tmp_path, [])
    process = subprocess.run([sys.executable, "-S", str(SCRIPT), str(tmp_path)],
                             text=True, capture_output=True, timeout=10)
    assert process.returncode == 0, process.stderr
    assert (tmp_path / "REPORT.md").is_file()
    assert json.loads(process.stdout)["integrity"] == "pass"


@pytest.mark.parametrize("degrees,expected", [([0, 1], "low"), ([2], "higher"), ([0, 2], "mixed")])
def test_actual_case_dimension_groups(tmp_path, degrees, expected):
    specs = [case(f"dimension-{degree}", degree=degree) for degree in degrees]
    write_manifest(tmp_path, specs)
    for specification in specs:
        write_record(tmp_path, specification)
    report, audit = summary.summarize_run(tmp_path)
    assert not audit["issues"]
    assert audit["requested_dimension_group"] == expected
    assert audit["requested_degrees"] == degrees
    assert "Requested dimension group:" in report
    assert ("## Low-dimensional (q0/q1) cases" in report) == any(degree in (0, 1) for degree in degrees)
    assert ("## Higher-dimensional (q2) cases" in report) == (2 in degrees)
    assert "### interaction-rips" in report
    assert "do not assert a speedup" in report


def test_backend_labels_follow_operation_and_degree():
    record = {"case": case(family="hyperdigraph", operation="persistence", degree=1),
              "worker": {"diagnostics": {"analysis": {"diagnostics": {
                  "low_dimensional_backend": "recorded_low_algorithm",
                  "higher_dimensional_backend": "recorded_high_algorithm"}}}}}
    assert summary._algorithm(record) == "low=recorded_low_algorithm"
    record["case"]["degree"] = 2
    assert summary._algorithm(record) == "low=recorded_low_algorithm; higher=recorded_high_algorithm"
    record["case"]["operation"] = "laplacian"
    # A persistence backend copied into spectral metadata must not be presented
    # as the operator's actual algorithm.
    assert summary._algorithm(record) == "unavailable"
    record["worker"]["diagnostics"]["analysis"]["laplacian_diagnostics"] = {"backend": "recorded_spectral_operator"}
    assert summary._algorithm(record) == "analysis=recorded_spectral_operator"


def test_reduction_and_construction_labels_are_recorded_without_inference():
    record = {"case": case(operation="persistence", degree=2), "worker": {"diagnostics": {
        "construction": {"construction_diagnostics": {"join_backend": "recorded_join"}},
        "analysis": {"reduction_diagnostics": {"reduction_backends": ["trivial", "int", "sparse"]}}}}}
    assert summary._algorithm(record) == "build=recorded_join; reduction[d0=trivial, d1=int, d2=sparse]"
    record["worker"]["diagnostics"]["analysis"] = {"diagnostics": {
        "algorithm": "recorded_clearing", "column_backend": "recorded_columns"}}
    assert "analysis=recorded_clearing; columns=recorded_columns" in summary._algorithm(record)
    assert summary._algorithm({"case": case(construction="alpha")}) == "unavailable"


def test_actual_and_requested_execution_routes_remain_distinct(tmp_path):
    specs = [case("canonical"), case("compact", execution_route="compact_hyperdigraph")]
    write_manifest(tmp_path, specs)
    first, first_directory = write_record(tmp_path, specs[0])
    second, second_directory = write_record(tmp_path, specs[1])
    first["worker"]["execution_route"] = "canonical"
    second["worker"]["execution_route"] = "compact_hyperdigraph"
    for record, directory in ((first, first_directory), (second, second_directory)):
        (directory / "record.json").write_text(json.dumps(record))
    report, audit = summary.summarize_run(tmp_path)
    assert not audit["issues"]
    assert "Execution route" in report and "Recorded algorithm/backend" in report
    assert "| canonical | canonical |" in report
    assert "| compact | compact_hyperdigraph |" in report
    assert summary._execution_route({"case": specs[1]}) == "requested compact_hyperdigraph; execution unrecorded"
    assert summary._execution_route({"case": specs[0]}) == "unrecorded"


def test_compact_combined_time_counts_and_bounded_probe_are_reported_separately(tmp_path):
    specs = [case("canonical", family="hyperdigraph", operation="persistence"),
             case("compact", family="hyperdigraph", operation="persistence", points=100,
                  execution_route="compact_hyperdigraph")]
    write_manifest(tmp_path, specs)
    canonical, canonical_directory = write_record(tmp_path, specs[0])
    canonical["worker"]["execution_route"] = "canonical"
    (canonical_directory / "record.json").write_text(json.dumps(canonical))
    record, directory = write_record(tmp_path, specs[1])
    worker = record["worker"]
    worker["execution_route"] = "compact_hyperdigraph"
    worker["timings_seconds"] = {"construction_and_analysis": 2.25, "canonical_equivalence": 0.125}
    worker["completed_stages"] = ["construction_and_analysis", "canonical_equivalence"]
    worker["construction_summary"] = {"cell_count": 5050, "support_edge_count": 4950,
        "directed_edge_count": 4950,
        "cell_count_semantics": "packed vertices and edges only; implicit two-paths excluded",
        "represented_paths_by_degree": {"0": 100, "1": 4950, "2": 161700},
        "represented_two_path_count_method": "n choose 3 for complete strict-score DAG"}
    worker["canonical_equivalence"] = {"status": "within_tolerance", "points": 8,
        "scope": "independent small prefix; does not verify all large-case intervals",
        "rtol": 1e-12, "atol": 1e-14}
    (directory / "record.json").write_text(json.dumps(record))
    report, audit = summary.summarize_run(tmp_path)
    assert not audit["issues"]
    assert "Combined build + analysis s | Probe s" in report
    assert "| 0.020 | 0.030 | — | — |" in report
    assert "| — | — | 2.250 | 0.125 |" in report
    assert "it is not an isolated analysis time" in report
    assert "These counts describe different stored representations" in report
    assert "it is not packed-array storage size" in report
    assert "packed vertices and edges only; implicit two-paths excluded | 161,700 |" in report
    assert "| within_tolerance | 8 | independent small prefix; does not verify all large-case intervals | rtol=1e-12, atol=1e-14 |" in report
    assert "does not establish full-case equivalence or a speedup" in report


def test_compact_failure_does_not_invent_probe_or_implicit_counts(tmp_path):
    specification = case("compact-timeout", family="hyperdigraph", operation="persistence",
                         execution_route="compact_hyperdigraph")
    write_manifest(tmp_path, [specification])
    record, directory = write_record(tmp_path, specification, status="timeout")
    worker = record["worker"]
    worker["execution_route"] = "compact_hyperdigraph"
    worker["timings_seconds"] = {"construction_and_analysis": 0.2}
    worker["completed_stages"] = ["generation"]
    worker["stage"] = record["last_stage"] = "construction_and_analysis"
    worker["construction_summary"] = {}
    (directory / "record.json").write_text(json.dumps(record))
    report, audit = summary.summarize_run(tmp_path)
    assert not audit["issues"]
    assert "| — | — | 0.200 (partial) | — |" in report
    assert "| compact-timeout | unavailable | — | unavailable | unavailable | — | unavailable | unavailable |" in report
    assert "0.250 (censored)" in report


def test_existing_local_smoke_read_only_when_present():
    run = ROOT / "examples/pressure_test/results/local_smoke_2026-09-05"
    if not run.is_dir():
        pytest.skip("local smoke artifact is not supplied in this checkout")
    report, audit = summary.summarize_run(run)
    assert not audit["issues"]
    assert audit["record_count"] == audit["planned_count"] == 72
    assert all(status == "pass" for status in audit["artifact_checks"].values())
    assert "cdsmbp" in report and "macOS" in report
