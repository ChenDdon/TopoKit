"""Isolated pressure runner with hard wall limits and durable raw evidence.

Run from the checkout or an uploaded AWS source bundle. Each case runs in a
fresh process group. Linux wait4 captures CPU and RSS even after forced stops;
live RSS sampling supplies an independent memory ceiling. No process outside
the case's newly created group is stopped. Existing results can be resumed
only with an identical matrix, source manifest, limits, and interpreter.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import time

from cases import build_cases, PROFILES

ROOT = Path(__file__).resolve().parents[2]
THREAD_ENVIRONMENT = {name: "1" for name in (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}
CSV_FIELDS = (
    "case_id", "family", "construction", "weights", "overlap", "operation", "degree", "execution_route",
    "points", "spectrum", "repetition", "status", "last_stage", "wall_seconds",
    "wall_limit_seconds", "cpu_seconds", "peak_rss_mib", "sampled_peak_rss_mib",
    "initialization_seconds", "generation_seconds", "support_seconds", "construction_seconds",
    "observation_seconds", "analysis_seconds", "construction_and_analysis_seconds", "equivalence_probe_seconds",
    "summarization_seconds", "export_seconds",
    "support_edge_count", "directed_edge_count", "cell_count", "cells_by_degree",
    "factor_cells_by_degree", "factor_support_edge_counts", "source_scale", "target_scale", "matrix_size",
    "complete_spectrum", "interval_count", "error_type", "error_message",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def source_manifest(root=ROOT):
    paths = sorted((root / "src").rglob("*.py"))
    paths += [root / "examples" / "pressure_test" / name
              for name in ("cases.py", "run_pressure.py", "worker.py", "fast_worker.py")]
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def process_rss_bytes(pid):
    """Linux current resident size, not virtual memory or cumulative RSS."""
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, ProcessLookupError):
        pass
    return 0


def machine_environment():
    environment = {"python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "hostname": platform.node(),
        "logical_cpus": os.cpu_count(), "thread_environment": THREAD_ENVIRONMENT,
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None}
    for name, path in (("cpuinfo", "/proc/cpuinfo"), ("meminfo", "/proc/meminfo")):
        if Path(path).exists():
            environment[name] = Path(path).read_text()
    return environment


def _limit_address_space(limit_bytes):
    resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))


def run_case(case, output, *, wall_seconds=600., rss_gib=16., address_gib=22.,
             sample_seconds=.05, command=None):
    """Run one child. ``command`` is a test seam for exercising hard stops."""
    if not all(math.isfinite(value) and value > 0 for value in
               (wall_seconds, rss_gib, address_gib, sample_seconds)):
        raise ValueError("wall, RSS, address-space, and sample limits must be positive and finite")
    if address_gib < rss_gib:
        raise ValueError("address-space limit must be at least the RSS limit")
    route = case.get("execution_route", "canonical")
    if route not in {"canonical", "compact_hyperdigraph"}:
        raise ValueError("unknown execution_route")
    output = Path(output).resolve()
    case_dir = output / "cases" / case["case_id"]
    case_dir.mkdir(parents=True, exist_ok=True)
    with (case_dir / ".case.lock").open("a+") as case_lock:
        try:
            fcntl.flock(case_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("this case already has an active runner or worker") from error
        return _run_locked_case(case, case_dir, case_lock, wall_seconds=wall_seconds,
            rss_gib=rss_gib, address_gib=address_gib, sample_seconds=sample_seconds, command=command)


def _run_locked_case(case, case_dir, case_lock, *, wall_seconds, rss_gib,
                     address_gib, sample_seconds, command):
    worker_filename = "fast_worker.py" if case.get("execution_route") == "compact_hyperdigraph" else "worker.py"
    case_path, checkpoint_path = case_dir / "case.json", case_dir / "checkpoint.json"
    if (case_dir / "record.json").exists():
        raise FileExistsError("case already has a terminal record; use resume at the suite level")
    # Preserve abandoned evidence while preventing an earlier success checkpoint
    # from being attributed to a fresh worker that fails before writing anything.
    attempt_id = str(time.time_ns())
    for name in ("checkpoint.json", "stdout.log", "stderr.log"):
        previous = case_dir / name
        if previous.exists():
            previous.rename(case_dir / f"previous-{attempt_id}-{name}")
    atomic_json(case_path, case)
    environment = {**os.environ, **THREAD_ENVIRONMENT, "PYTHONHASHSEED": "0",
                   "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(ROOT / "src")}
    worker_command = command or [sys.executable, "-B", str(Path(__file__).with_name(worker_filename)),
                                 "--case", str(case_path), "--output", str(checkpoint_path)]
    started_utc = utc_now()
    sampled_peak_rss = 0
    stop_reason = None
    process = usage = None
    controller_error = None
    with (case_dir / "stdout.log").open("wb") as stdout, (case_dir / "stderr.log").open("wb") as stderr:
        started = time.monotonic()
        try:
            process = subprocess.Popen(worker_command, cwd=ROOT, env=environment, stdout=stdout,
                stderr=stderr, start_new_session=True, pass_fds=(case_lock.fileno(),),
                preexec_fn=(lambda: _limit_address_space(int(address_gib * 1024**3)))
                    if sys.platform.startswith("linux") else None)
            atomic_json(case_dir / "process.json", {"pid": process.pid, "attempt_id": attempt_id,
                        "started_utc": started_utc})
            while True:
                elapsed = time.monotonic() - started
                # A result first observed after the deadline is conservatively
                # censored; no extra grace period is added to computation time.
                if elapsed >= wall_seconds:
                    stop_reason = "timeout"
                    break
                child_pid, wait_status, usage = os.wait4(process.pid, os.WNOHANG)
                if child_pid:
                    process.returncode = os.waitstatus_to_exitcode(wait_status)
                    break
                sampled_peak_rss = max(sampled_peak_rss, process_rss_bytes(process.pid))
                if sampled_peak_rss > rss_gib * 1024**3:
                    stop_reason = "rss_limit"
                    break
                time.sleep(min(sample_seconds, max(.001, wall_seconds - elapsed)))
        except BaseException as error:
            controller_error = error
            stop_reason = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "controller_error"
        finally:
            if process is not None and process.returncode is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                _, wait_status, usage = os.wait4(process.pid, 0)
                process.returncode = os.waitstatus_to_exitcode(wait_status)
        elapsed = time.monotonic() - started
    checkpoint = read_json(checkpoint_path)
    if not isinstance(checkpoint, dict):
        checkpoint = {}
    worker_status = checkpoint.get("status")
    status = stop_reason or (worker_status if worker_status in
        {"success", "resource_limit", "memory_error", "error", "numerical_failure"}
        else "worker_failed")
    if status == "success" and process.returncode != 0:
        status = "worker_failed"
    peak_rss_bytes = (int(usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024))
                      if usage is not None else 0)
    record = {"case_id": case["case_id"], "case": case, "status": status,
        "attempt_id": attempt_id, "started_utc": started_utc, "finished_utc": utc_now(),
        "wall_seconds": elapsed, "wall_limit_seconds": wall_seconds,
        "wall_overshoot_seconds": max(0., elapsed - wall_seconds) if stop_reason == "timeout" else 0.,
        "cpu_user_seconds": usage.ru_utime if usage is not None else 0.,
        "cpu_system_seconds": usage.ru_stime if usage is not None else 0.,
        "cpu_seconds": usage.ru_utime + usage.ru_stime if usage is not None else 0.,
        "peak_rss_bytes": peak_rss_bytes, "sampled_peak_rss_bytes": sampled_peak_rss,
        "rss_limit_bytes": int(rss_gib * 1024**3), "address_limit_bytes": int(address_gib * 1024**3),
        "rss_sample_interval_seconds": sample_seconds,
        "exit_code": process.returncode if process is not None else None,
        "last_stage": checkpoint.get("stage", "startup"), "worker": checkpoint,
        "measurement_semantics": "fresh child; wall includes startup, generation, construction, analysis, checks and checkpoints; wait4 CPU and peak RSS include the entire child lifetime",
    }
    if controller_error is not None:
        record["controller_exception"] = {"class": type(controller_error).__name__, "message": str(controller_error)}
    atomic_json(case_dir / "record.json", record)
    if isinstance(controller_error, (KeyboardInterrupt, SystemExit)):
        raise controller_error
    return record


def flat_record(record):
    case, worker = record["case"], record.get("worker", {})
    timings = worker.get("timings_seconds", {})
    construction = worker.get("construction_summary", {})
    observation = worker.get("observations", {})
    result = worker.get("result_summary", {})
    error = worker.get("exception") or record.get("controller_exception") or {}
    row = {field: case.get(field) for field in CSV_FIELDS}
    row.update({field: record.get(field) for field in
                ("case_id", "status", "last_stage", "wall_seconds", "wall_limit_seconds", "cpu_seconds")})
    row.update(execution_route=worker.get("execution_route"),
        peak_rss_mib=record.get("peak_rss_bytes", 0) / 1024**2,
        sampled_peak_rss_mib=record.get("sampled_peak_rss_bytes", 0) / 1024**2,
        initialization_seconds=timings.get("initialization"),
        generation_seconds=timings.get("generation"), support_seconds=timings.get("support_preparation"),
        construction_seconds=timings.get("construction"), observation_seconds=timings.get("observation_selection"),
        analysis_seconds=timings.get("analysis"),
        construction_and_analysis_seconds=timings.get("construction_and_analysis"),
        equivalence_probe_seconds=timings.get("canonical_equivalence"),
        summarization_seconds=timings.get("summarization"),
        export_seconds=timings.get("export"),
        support_edge_count=construction.get("support_edge_count"),
        directed_edge_count=construction.get("directed_edge_count"),
        cell_count=construction.get("cell_count"),
        cells_by_degree=json.dumps(construction.get("cells_by_degree")),
        factor_cells_by_degree=json.dumps(construction.get("factor_cells_by_degree")),
        factor_support_edge_counts=json.dumps(construction.get("factor_support_edge_counts")),
        source_scale=observation.get("start"), target_scale=observation.get("end"),
        matrix_size=result.get("basis_size"), complete_spectrum=result.get("complete"),
        interval_count=result.get("interval_count"), error_type=error.get("class"),
        error_message=error.get("message"))
    return row


def write_aggregate(output, records):
    """Regenerate aggregate files from durable individual terminal records."""
    output = Path(output)
    temporary = output / "results.jsonl.tmp"
    with temporary.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(output / "results.jsonl")
    temporary = output / "results.csv.tmp"
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(flat_record(record) for record in records)
    temporary.replace(output / "results.csv")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=("smoke", "pressure", "supplemental"), default="pressure")
    parser.add_argument("--profiles", nargs="+", choices=tuple(PROFILES))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--wall-seconds", type=float, default=600.)
    parser.add_argument("--rss-gib", type=float, default=16.)
    parser.add_argument("--address-gib", type=float, default=22.)
    parser.add_argument("--sample-seconds", type=float, default=.05)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--matrix", type=Path, help="Use an explicit saved JSON case list")
    args = parser.parse_args(argv)
    if args.max_cases is not None and args.max_cases < 1:
        parser.error("max-cases must be positive")
    cases = (json.loads(args.matrix.read_text()) if args.matrix else
             build_cases(suite=args.suite, profiles=args.profiles, repeats=args.repeats))
    if not cases or len({case["case_id"] for case in cases}) != len(cases):
        parser.error("matrix needs unique nonempty case IDs")
    if any(not case["case_id"] or set(case["case_id"]) - set("abcdefghijklmnopqrstuvwxyz0123456789-_")
           for case in cases):
        parser.error("case IDs must be safe lowercase filenames")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    # The descriptor stays alive until main returns; a second runner cannot
    # execute the same output directory while the first still owns its lock.
    run_lock = (args.output / ".runner.lock").open("a+")
    try:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("another runner is active in this output directory")
    signature = {"cases": cases, "source_sha256": source_manifest(), "python": sys.version,
        "executable": sys.executable, "wall_seconds": args.wall_seconds,
        "rss_gib": args.rss_gib, "address_gib": args.address_gib,
        "sample_seconds": args.sample_seconds, "thread_environment": THREAD_ENVIRONMENT}
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        if not args.resume or read_json(manifest_path).get("signature") != signature:
            parser.error("existing run requires --resume and an identical source/matrix/environment/limits")
        manifest = read_json(manifest_path)
    else:
        manifest = {"created_utc": utc_now(), "signature": signature,
            "machine": machine_environment(), "status": "running", "planned_cases": len(cases)}
        atomic_json(manifest_path, manifest)
    records = []
    newly_run = 0
    for case in cases:
        record_path = args.output / "cases" / case["case_id"] / "record.json"
        existing = read_json(record_path)
        if existing is not None:
            if existing.get("case") != case:
                raise ValueError("saved case differs from the declared matrix")
            records.append(existing)
            continue
        if args.max_cases is not None and newly_run >= args.max_cases:
            break
        print(f"START {len(records)+1}/{len(cases)} {case['case_id']}", flush=True)
        record = run_case(case, args.output, wall_seconds=args.wall_seconds, rss_gib=args.rss_gib,
                          address_gib=args.address_gib, sample_seconds=args.sample_seconds)
        records.append(record)
        newly_run += 1
        write_aggregate(args.output, records)
        manifest.update(updated_utc=utc_now(), completed_cases=len(records),
                        status_counts={status: sum(item["status"] == status for item in records)
                                       for status in sorted({item["status"] for item in records})})
        atomic_json(manifest_path, manifest)
        print(f"DONE {record['status']} wall={record['wall_seconds']:.3f}s "
              f"RSS={record['peak_rss_bytes']/1024**2:.1f}MiB stage={record['last_stage']}", flush=True)
    write_aggregate(args.output, records)
    manifest.update(status="complete" if len(records) == len(cases) else "paused",
                    completed_cases=len(records), updated_utc=utc_now())
    atomic_json(manifest_path, manifest)
    return 1 if any(record["status"] in {"error", "worker_failed", "numerical_failure", "controller_error", "interrupted"}
                    for record in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
