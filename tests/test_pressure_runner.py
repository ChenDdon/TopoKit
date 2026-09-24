"""Exercise orchestration with tiny synthetic children, never TDA workloads."""
import sys

import pytest

if sys.platform == "win32":
    pytest.skip("pressure orchestration requires POSIX fcntl/resource", allow_module_level=True)

from collections import Counter, defaultdict
import csv
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import textwrap


PRESSURE_DIR = Path(__file__).resolve().parents[1] / "examples" / "pressure_test"


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(PRESSURE_DIR))
    spec = importlib.util.spec_from_file_location("pressure_runner_under_test", PRESSURE_DIR / "run_pressure.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tiny_case(name="synthetic"):
    return {"case_id": name, "family": "simplicial", "construction": "alpha",
            "weights": "equal", "overlap": "full", "operation": "laplacian",
            "degree": 0, "points": 2, "spectrum": "full", "repetition": 0}


def child_command(output, case, body):
    checkpoint = output / "cases" / case["case_id"] / "checkpoint.json"
    preamble = """
        import json, os, sys, time
        from pathlib import Path
        checkpoint = Path(sys.argv[1])
        def save(value):
            temporary = checkpoint.with_suffix('.json.tmp')
            with temporary.open('w') as handle:
                json.dump(value, handle)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(checkpoint)
    """
    return [sys.executable, "-B", "-c", textwrap.dedent(preamble) + "\n" + textwrap.dedent(body), str(checkpoint)]


def run_synthetic(runner, output, case, body, **limits):
    options = {"wall_seconds": 3., "rss_gib": 1., "address_gib": 4., "sample_seconds": .01}
    options.update(limits)
    return runner.run_case(case, output, command=child_command(output, case, body), **options)


def assert_durable_record(output, case, record):
    path = output / "cases" / case["case_id"] / "record.json"
    assert json.loads(path.read_text()) == record


def test_success_captures_actual_child_cpu_rss_and_both_logs(runner, tmp_path):
    case = tiny_case()
    record = run_synthetic(runner, tmp_path, case, """
        memory = bytearray(8 * 1024 * 1024)
        deadline = time.process_time() + .06
        value = 1
        while time.process_time() < deadline:
            value = (value * 17 + 3) % 1000003
        print('synthetic stdout evidence', flush=True)
        print('synthetic stderr evidence', file=sys.stderr, flush=True)
        save({'status': 'success', 'stage': 'complete', 'result_summary': {'basis_size': 2, 'complete': True}})
    """)
    assert record["status"] == "success" and record["exit_code"] == 0
    assert record["cpu_seconds"] >= .03
    assert record["cpu_seconds"] == pytest.approx(record["cpu_user_seconds"] + record["cpu_system_seconds"])
    # Detect both missing RSS and Linux-kilobytes/macOS-bytes unit mistakes.
    assert 1024**2 < record["peak_rss_bytes"] < 1024**3
    directory = tmp_path / "cases" / case["case_id"]
    assert "synthetic stdout evidence" in (directory / "stdout.log").read_text()
    assert "synthetic stderr evidence" in (directory / "stderr.log").read_text()
    assert_durable_record(tmp_path, case, record)
    with pytest.raises(FileExistsError):
        run_synthetic(runner, tmp_path, case, "raise AssertionError('must not run twice')")


def test_true_error_and_nonzero_success_are_not_reported_as_success(runner, tmp_path):
    error_case = tiny_case("true-error")
    error_record = run_synthetic(runner, tmp_path, error_case, """
        save({'status': 'error', 'stage': 'analysis',
              'exception': {'class': 'RuntimeError', 'message': 'synthetic failure'}})
        raise RuntimeError('synthetic failure')
    """)
    assert error_record["status"] == "error"
    assert error_record["exit_code"] != 0
    assert "RuntimeError: synthetic failure" in (tmp_path / "cases/true-error/stderr.log").read_text()
    assert_durable_record(tmp_path, error_case, error_record)
    false_case = tiny_case("false-success")
    false_record = run_synthetic(runner, tmp_path, false_case, """
        save({'status': 'success', 'stage': 'complete'})
        sys.exit(3)
    """)
    assert false_record["status"] == "worker_failed"
    assert false_record["exit_code"] == 3


def test_hard_timeout_preserves_last_atomic_checkpoint(runner, tmp_path):
    case = tiny_case("timeout")
    record = run_synthetic(runner, tmp_path, case, """
        save({'status': 'running', 'stage': 'analysis', 'timings_seconds': {'construction': .012}})
        checkpoint.with_suffix('.json.tmp').write_text('{"status": "success"')
        print('checkpoint committed before timeout', flush=True)
        time.sleep(30)
    """, wall_seconds=.30)
    assert record["status"] == "timeout"
    assert record["exit_code"] < 0
    assert .25 <= record["wall_seconds"] < 3.
    assert record["last_stage"] == "analysis"
    assert record["worker"]["timings_seconds"]["construction"] == .012
    assert record["worker"]["status"] == "running"
    assert record["cpu_seconds"] > 0 and record["peak_rss_bytes"] > 0
    assert_durable_record(tmp_path, case, record)


def test_rss_ceiling_kills_child_and_retains_partial_stage(runner, tmp_path, monkeypatch):
    case = tiny_case("rss-stop")
    checkpoint = tmp_path / "cases" / case["case_id"] / "checkpoint.json"
    monkeypatch.setattr(runner, "process_rss_bytes", lambda pid: 32 * 1024**2 if checkpoint.exists() else 0)
    record = run_synthetic(runner, tmp_path, case, """
        save({'status': 'running', 'stage': 'construction'})
        time.sleep(30)
    """, rss_gib=.02)
    assert record["status"] == "rss_limit"
    assert record["exit_code"] < 0
    assert record["sampled_peak_rss_bytes"] > record["rss_limit_bytes"]
    assert record["last_stage"] == "construction"
    assert_durable_record(tmp_path, case, record)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="requires Linux /proc RSS")
def test_linux_proc_rss_ceiling_uses_real_resident_allocation(runner, tmp_path):
    case = tiny_case("linux-rss")
    record = run_synthetic(runner, tmp_path, case, """
        save({'status': 'running', 'stage': 'allocation'})
        memory = bytearray(48 * 1024 * 1024)
        time.sleep(30)
    """, rss_gib=32 / 1024)
    assert record["status"] == "rss_limit"
    assert record["sampled_peak_rss_bytes"] > 32 * 1024**2
    assert record["peak_rss_bytes"] > 32 * 1024**2


@pytest.mark.parametrize("name,value", [(name, value) for name in
    ("wall_seconds", "rss_gib", "address_gib", "sample_seconds") for value in (0., -1., float("nan"), float("inf"))])
def test_invalid_limits_fail_before_launch(runner, tmp_path, monkeypatch, name, value):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid limits must not launch a child")
    monkeypatch.setattr(runner.subprocess, "Popen", forbidden)
    with pytest.raises(ValueError):
        run_synthetic(runner, tmp_path, tiny_case(), "pass", **{name: value})


def test_address_limit_cannot_be_below_rss_limit(runner, tmp_path):
    with pytest.raises(ValueError):
        run_synthetic(runner, tmp_path, tiny_case(), "pass", rss_gib=2., address_gib=1.)


def test_atomic_json_never_replaces_good_evidence_with_partial_invalid_json(runner, tmp_path):
    path = tmp_path / "record.json"
    runner.atomic_json(path, {"status": "success", "proof": 7})
    with pytest.raises(ValueError):
        runner.atomic_json(path, {"status": float("nan")})
    assert json.loads(path.read_text()) == {"status": "success", "proof": 7}


@pytest.mark.parametrize("payload", [[{"status": "success"}], "success", 42])
def test_nonobject_checkpoint_is_a_durable_worker_failure(runner, tmp_path, payload):
    case = tiny_case()
    record = run_synthetic(runner, tmp_path, case, f"save({payload!r})")
    assert record["status"] == "worker_failed"
    assert_durable_record(tmp_path, case, record)


def test_stale_success_checkpoint_is_not_reused_for_a_new_attempt(runner, tmp_path):
    case = tiny_case()
    directory = tmp_path / "cases" / case["case_id"]
    directory.mkdir(parents=True)
    runner.atomic_json(directory / "checkpoint.json", {"status": "success", "stage": "complete", "old_attempt": True})
    (directory / "stdout.log").write_text("old stdout evidence")
    (directory / "stderr.log").write_text("old stderr evidence")
    record = run_synthetic(runner, tmp_path, case, "print('new attempt produced no checkpoint', flush=True)")
    assert record["status"] == "worker_failed"
    assert not record["worker"].get("old_attempt")
    previous_checkpoint, = directory.glob("previous-*-checkpoint.json")
    assert json.loads(previous_checkpoint.read_text())["old_attempt"] is True
    previous_stdout, = directory.glob("previous-*-stdout.log")
    previous_stderr, = directory.glob("previous-*-stderr.log")
    assert previous_stdout.read_text() == "old stdout evidence"
    assert previous_stderr.read_text() == "old stderr evidence"


def test_missing_executable_produces_durable_startup_failure(runner, tmp_path):
    case = tiny_case()
    record = runner.run_case(case, tmp_path, command=[str(tmp_path / "nonexistent-program")])
    assert record["status"] == "controller_error"
    assert record["controller_exception"]["class"] == "FileNotFoundError"
    assert record["last_stage"] == "startup"
    row = runner.flat_record(record)
    assert row["error_type"] == "FileNotFoundError"
    assert "nonexistent-program" in row["error_message"]
    assert_durable_record(tmp_path, case, record)


def test_monitor_interrupt_reaps_its_child_before_propagating(runner, tmp_path, monkeypatch):
    launched = []
    original_popen = subprocess.Popen
    def capture(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        launched.append(child)
        return child
    def interrupt(pid):
        raise KeyboardInterrupt("synthetic monitor interruption")
    monkeypatch.setattr(runner.subprocess, "Popen", capture)
    monkeypatch.setattr(runner, "process_rss_bytes", interrupt)
    leaked = False
    try:
        with pytest.raises(KeyboardInterrupt):
            run_synthetic(runner, tmp_path, tiny_case(), "time.sleep(30)")
        leaked = bool(launched and launched[0].poll() is None)
    finally:
        # Protect the test machine even when the runner's cleanup is broken.
        for child in launched:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=3)
    assert not leaked, "runner left its process group alive after monitor interruption"
    record = json.loads((tmp_path / "cases/synthetic/record.json").read_text())
    assert record["status"] == "interrupted"
    assert record["controller_exception"]["class"] == "KeyboardInterrupt"
    assert record["exit_code"] < 0


def test_monitor_error_is_durable_and_reaps_its_child(runner, tmp_path, monkeypatch):
    def fail(pid):
        raise OSError("synthetic RSS read failure")
    monkeypatch.setattr(runner, "process_rss_bytes", fail)
    case = tiny_case()
    record = run_synthetic(runner, tmp_path, case, "time.sleep(30)")
    assert record["status"] == "controller_error"
    assert record["controller_exception"]["class"] == "OSError"
    assert record["exit_code"] < 0
    assert_durable_record(tmp_path, case, record)


def test_active_case_lock_rejects_duplicate_worker_launch(runner, tmp_path, monkeypatch):
    case = tiny_case()
    directory = tmp_path / "cases" / case["case_id"]
    directory.mkdir(parents=True)
    def forbidden(*args, **kwargs):
        pytest.fail("an active case must not launch another worker")
    monkeypatch.setattr(runner.subprocess, "Popen", forbidden)
    with (directory / ".case.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="active runner or worker"):
            run_synthetic(runner, tmp_path, case, "pass")
    assert not (directory / "record.json").exists()


def test_csv_uses_worker_stage_and_exception_schema_without_losing_evidence(runner, tmp_path):
    record = {"case_id": "csv", "case": tiny_case("csv"), "status": "error",
              "worker": {"timings_seconds": {"generation": .1, "support_preparation": .2,
                         "construction": .3, "analysis": .4},
                         "construction_summary": {"support_edge_count": 4, "directed_edge_count": 8,
                            "cell_count": 18, "cells_by_degree": {"0": 2, "1": 8, "2": 8}},
                         "observations": {"start": .5, "end": .8},
                         "result_summary": {"basis_size": 12, "complete": False},
                         "exception": {"class": "MemoryError", "message": "allocation failed"}}}
    runner.write_aggregate(tmp_path, [record])
    with (tmp_path / "results.csv").open(newline="") as handle:
        row, = csv.DictReader(handle)
    assert float(row["support_seconds"]) == .2
    assert row["error_type"] == "MemoryError"
    assert row["error_message"] == "allocation failed"
    assert row["complete_spectrum"] == "False"
    assert int(row["matrix_size"]) == 12
    assert json.loads(row["cells_by_degree"]) == {"0": 2, "1": 8, "2": 8}
    assert json.loads((tmp_path / "results.jsonl").read_text()) == record


def test_pressure_matrix_covers_all_required_scientific_axes(runner):
    cases = runner.build_cases(suite="pressure")
    expected_profiles = {"simplicial-alpha", "simplicial-rips", "interaction-alpha", "interaction-rips",
        "hyperdigraph-delaunay-distinct", "hyperdigraph-delaunay-equal",
        "hyperdigraph-complete-distinct", "hyperdigraph-complete-equal"}
    operations = {"persistence", "laplacian", "persistent_laplacian"}
    assert len(cases) == 8 * 3 * 3 * 3
    assert len({case["case_id"] for case in cases}) == len(cases)
    sizes = defaultdict(set)
    for case in cases:
        sizes[case["profile"], case["degree"], case["operation"]].add(case["points"])
        assert case["spectrum"] == "full" and case["filtration_cutoff"] is None
        if case["family"] == "hyperdigraph":
            assert case["construction"] in {"delaunay", "complete"}
            assert "ordered distinct-vertex sequences" in case["construction_rule"]
            assert "consecutive-distance" in case["construction_rule"]
        else:
            assert case["construction"] in {"alpha", "rips"}
    assert set(sizes) == {(profile, q, operation) for profile in expected_profiles for q in range(3) for operation in operations}
    assert all(len(values) == 3 and min(values) > 0 for values in sizes.values())
    smoke = runner.build_cases(suite="smoke")
    assert len(smoke) == 72 and {case["points"] for case in smoke} == {6}


def test_supplements_cannot_masquerade_as_full_main_cases(runner):
    main = runner.build_cases(suite="pressure")
    supplements = runner.build_cases(suite="supplemental")
    assert len(supplements) == 54
    assert not ({case["case_id"] for case in main} & {case["case_id"] for case in supplements})
    for case in supplements:
        assert case["suite"] == "supplemental"
        assert case["spectrum"] == ("full" if case["operation"] == "persistence" else "partial")
        assert case["overlap"] == ("half" if case["family"] == "interaction" else "full")
        if case["operation"] == "persistence":
            assert case["family"] == "interaction"
    repeated = runner.build_cases(suite="pressure", repeats=2)
    assert len(repeated) == len(main) * 2
    assert Counter(case["repetition"] for case in repeated) == {0: len(main), 1: len(main)}
    repeated[0]["limits"]["max_cells"] = 1
    assert repeated[1]["limits"]["max_cells"] > 1
    assert runner.build_cases(suite="pressure")[0]["limits"]["max_cells"] > 1


def test_compact_matrix_is_separate_eligible_and_has_no_ignored_guards(runner):
    from cases import build_compact_cases
    cases = build_compact_cases()
    assert len(cases) == 12
    assert len(build_compact_cases(smoke=True)) == 4
    assert {case["points"] for case in build_compact_cases(smoke=True)} == {8}
    assert not {case["case_id"] for case in cases} & {case["case_id"] for case in runner.build_cases()}
    assert {(case["construction"], case["degree"]) for case in cases} == {
        (support, degree) for support in ("complete", "delaunay") for degree in (0, 1)}
    for case in cases:
        assert case["execution_route"] == "compact_hyperdigraph"
        assert case["weights"] == "distinct" and case["operation"] == "persistence"
        assert case["limits"] == {} and case["points"] <= 65_535


def test_unknown_route_is_rejected_before_launch(runner, tmp_path):
    with pytest.raises(ValueError, match="execution_route"):
        runner.run_case({**tiny_case(), "execution_route": "unknown"}, tmp_path)
    assert not list(tmp_path.iterdir())


def test_combined_timing_is_never_reported_as_analysis_only(runner):
    record = {"case": tiny_case(), "worker": {"execution_route": "compact_hyperdigraph",
        "timings_seconds": {"construction_and_analysis": 12., "canonical_equivalence": .1}}}
    row = runner.flat_record(record)
    assert row["execution_route"] == "compact_hyperdigraph"
    assert row["analysis_seconds"] is None and row["construction_seconds"] is None
    assert row["construction_and_analysis_seconds"] == 12. and row["equivalence_probe_seconds"] == .1


@pytest.mark.parametrize("route", ["canonical", "compact_hyperdigraph"])
def test_actual_worker_dispatch_and_durable_result(runner, tmp_path, route):
    from cases import build_compact_cases
    case = (build_compact_cases(smoke=True)[0] if route == "compact_hyperdigraph" else
            runner.build_cases(suite="smoke")[0])
    record = runner.run_case(case, tmp_path, wall_seconds=30., rss_gib=1., address_gib=4.)
    assert record["status"] == "success", record
    assert record["worker"]["execution_route"] == route
    assert_durable_record(tmp_path, case, record)


@pytest.fixture
def suite_seam(runner, tmp_path, monkeypatch):
    cases = [tiny_case("first"), tiny_case("second")]
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps(cases))
    output = tmp_path / "suite"
    launched = []
    monkeypatch.setattr(runner, "source_manifest", lambda: {"source.py": "stable-hash"})
    monkeypatch.setattr(runner, "machine_environment", lambda: {"test_machine": True})
    def fake(case, output, **limits):
        launched.append(case["case_id"])
        directory = Path(output) / "cases" / case["case_id"]
        directory.mkdir(parents=True, exist_ok=True)
        record = {"case_id": case["case_id"], "case": case, "status": "success",
                  "wall_seconds": .01, "peak_rss_bytes": 1024, "last_stage": "complete", "worker": {}}
        runner.atomic_json(directory / "record.json", record)
        return record
    monkeypatch.setattr(runner, "run_case", fake)
    return ["--output", str(output), "--matrix", str(matrix)], output, matrix, launched


def test_resume_skips_completed_cases_and_rebuilds_complete_aggregates(runner, suite_seam):
    args, output, matrix, launched = suite_seam
    assert runner.main(args + ["--max-cases", "1"]) == 0
    assert launched == ["first"]
    assert runner.read_json(output / "manifest.json")["status"] == "paused"
    assert runner.main(args + ["--resume"]) == 0
    assert launched == ["first", "second"]
    assert runner.main(args + ["--resume"]) == 0
    assert launched == ["first", "second"]
    assert len((output / "results.jsonl").read_text().splitlines()) == 2
    assert runner.read_json(output / "manifest.json")["status"] == "complete"


@pytest.mark.parametrize("mismatch", ["source", "limits", "matrix"])
def test_resume_rejects_changed_run_identity(runner, suite_seam, monkeypatch, mismatch):
    args, output, matrix, launched = suite_seam
    runner.main(args + ["--max-cases", "1"])
    if mismatch == "source":
        monkeypatch.setattr(runner, "source_manifest", lambda: {"source.py": "changed-hash"})
    elif mismatch == "limits":
        args += ["--wall-seconds", "7"]
    else:
        cases = json.loads(matrix.read_text())
        cases[0]["points"] += 1
        matrix.write_text(json.dumps(cases))
    with pytest.raises(SystemExit) as error:
        runner.main(args + ["--resume"])
    assert error.value.code == 2
    assert launched == ["first"]


def test_active_suite_lock_prevents_duplicate_case_launches(runner, suite_seam):
    args, output, matrix, launched = suite_seam
    output.mkdir()
    with (output / ".runner.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(SystemExit) as error:
            runner.main(args)
    assert error.value.code == 2
    assert launched == []
