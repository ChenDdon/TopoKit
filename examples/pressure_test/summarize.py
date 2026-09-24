"""Summarize existing pressure-run evidence and verify saved NPZ hashes.

This standard-library-only reader never executes a pressure case or contacts a
remote machine. Durable per-case records take precedence over aggregate JSONL.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys


def _read_json(path, issues):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        issues.append(f"Cannot read {path}: {error}")
        return {}
    if not isinstance(value, dict):
        issues.append(f"Expected a JSON object in {path}")
        return {}
    return value


def _duplicates(values):
    return sorted(str(value) for value, count in Counter(values).items() if count > 1)


def _case_directory(run, case_id):
    if not isinstance(case_id, str) or not case_id or Path(case_id).name != case_id or case_id in {".", ".."} or "\\" in case_id:
        return None
    directory = (run / "cases" / case_id).resolve()
    return directory if directory.is_relative_to(run) else None


def load_run(run_directory):
    run = Path(run_directory).resolve()
    if not run.is_dir():
        raise ValueError(f"Run directory does not exist: {run}")
    issues = []
    manifest = _read_json(run / "manifest.json", issues)
    planned = manifest.get("signature", {}).get("cases", [])
    if not isinstance(planned, list) or any(not isinstance(case, dict) for case in planned):
        issues.append("Manifest signature.cases must be a list of case objects")
        planned = []
    duplicate_plans = _duplicates(case.get("case_id") for case in planned)
    if duplicate_plans:
        issues.append(f"Duplicate planned case IDs: {', '.join(duplicate_plans)}")
    records = []
    record_paths = sorted((run / "cases").glob("*/record.json"))
    for path in record_paths:
        record = _read_json(path, issues)
        if record:
            if record.get("case_id") != path.parent.name:
                issues.append(f"Record case ID does not match its directory: {path}")
            records.append(record)
    aggregate = []
    aggregate_path = run / "results.jsonl"
    if aggregate_path.is_file():
        try:
            for line_number, line in enumerate(aggregate_path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"line {line_number} is not a JSON object")
                aggregate.append(value)
        except (OSError, ValueError) as error:
            issues.append(f"Cannot read complete aggregate {aggregate_path}: {error}")
        duplicate_aggregate = _duplicates(record.get("case_id") for record in aggregate)
        if duplicate_aggregate:
            issues.append(f"Duplicate case IDs in results.jsonl: {', '.join(duplicate_aggregate)}")
    if not record_paths:
        records = aggregate
    duplicates = _duplicates(record.get("case_id") for record in records)
    if duplicates:
        issues.append(f"Duplicate terminal case IDs: {', '.join(duplicates)}")
    for record in records:
        case_id = record.get("case_id")
        if _case_directory(run, case_id) is None:
            issues.append(f"Unsafe or missing record case ID: {case_id!r}")
        case = record.get("case", {})
        if case.get("case_id") != case_id:
            issues.append(f"Outer and inner case IDs disagree for {case_id!r}")
    source_hashes = sorted({record.get("worker", {}).get("source_hash") for record in records
                            if record.get("worker", {}).get("source_hash")})
    if len(source_hashes) > 1:
        issues.append(f"Inconsistent worker package source hashes across records: {', '.join(source_hashes)}")
    for record in records:
        if record.get("status") == "success" and not record.get("worker", {}).get("source_hash"):
            issues.append(f"Successful case {record.get('case_id')} lacks a package source hash")
    return run, manifest, planned, records, source_hashes, issues


def verify_artifacts(run, records, issues):
    checks = {}
    for record in records:
        if record.get("status") != "success":
            continue
        case_id = record.get("case_id")
        artifact = record.get("worker", {}).get("result_summary", {}).get("artifact", {})
        directory = _case_directory(run, case_id)
        filename, expected = artifact.get("filename"), artifact.get("sha256")
        reason = None
        if (directory is None or not isinstance(filename, str) or Path(filename).name != filename
                or "\\" in filename or Path(filename).suffix.lower() != ".npz"):
            reason = "missing or invalid relative NPZ basename"
        elif not isinstance(expected, str) or len(expected) != 64:
            reason = "missing or invalid recorded SHA256"
        else:
            path = (directory / filename).resolve()
            if not path.is_relative_to(directory):
                reason = "NPZ path resolves outside its case directory"
            elif not path.is_file():
                reason = f"NPZ file is missing: {filename}"
            else:
                try:
                    digest = hashlib.sha256()
                    with path.open("rb") as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(block)
                    if digest.hexdigest() != expected.lower():
                        reason = f"SHA256 mismatch: {filename}"
                except OSError as error:
                    reason = f"Cannot read NPZ: {error}"
        checks[case_id] = "fail" if reason else "pass"
        if reason:
            issues.append(f"Artifact verification failed for {case_id}: {reason}")
    return checks


def _escape(value):
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", "<br>")


def _number(value, digits=3):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "—"
    return f"{value:,.{digits}f}"


def _count(value):
    return f"{value:,}" if isinstance(value, int) and not isinstance(value, bool) else "—"


def _table(headers, rows):
    lines = ["| " + " | ".join(map(_escape, headers)) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(map(_escape, row)) + " |" for row in rows)
    return "\n".join(lines)


def _profile(case):
    return case.get("profile") or "-".join(str(case.get(name, "unknown")) for name in
        (("family", "construction", "weights") if case.get("family") == "hyperdigraph" else ("family", "construction")))


def _trial_key(case):
    return json.dumps({key: value for key, value in case.items() if key not in {"case_id", "repetition"}}, sort_keys=True)


def _dimension_group(case):
    degree = case.get("degree")
    if isinstance(degree, bool):
        return "unknown"
    if degree in (0, 1):
        return "low"
    return "higher" if degree == 2 else "unknown"


def _execution_route(record):
    worker = record.get("worker", {})
    route = worker.get("execution_route", record.get("execution_route"))
    if isinstance(route, str) and route:
        return route
    requested = record.get("case", {}).get("execution_route")
    return f"requested {requested}; execution unrecorded" if requested else "unrecorded"


def _algorithm(record):
    """Describe only named strategies captured by the relevant completed work."""
    worker = record.get("worker", {})
    diagnostics = worker.get("diagnostics", {})
    if not isinstance(diagnostics, dict):
        return "unavailable"
    case = record.get("case", {})
    construction = diagnostics.get("construction", {})
    analysis = diagnostics.get("analysis", {})
    labels = []

    def named(container, key, label):
        value = container.get(key) if isinstance(container, dict) else None
        if isinstance(value, str) and value and value not in {"not_requested", "not_applicable"}:
            text = f"{label}={value}"
            if text not in labels:
                labels.append(text)

    if isinstance(construction, dict):
        for container in (construction, construction.get("construction_diagnostics", {}), construction.get("diagnostics", {})):
            for key in ("algorithm", "backend", "join_backend", "support_backend"):
                named(container, key, "build")
    if isinstance(analysis, dict):
        if case.get("operation") == "persistence":
            containers = (analysis, analysis.get("diagnostics", {}), analysis.get("reduction_diagnostics", {}))
        else:
            containers = (analysis, analysis.get("laplacian_diagnostics", {}), analysis.get("diagnostics", {}))
        for container in containers:
            if not isinstance(container, dict):
                continue
            named(container, "algorithm", "analysis")
            named(container, "backend", "analysis")
            if case.get("operation") == "persistence":
                named(container, "low_dimensional_backend", "low")
                if _dimension_group(case) == "higher":
                    named(container, "higher_dimensional_backend", "higher")
                named(container, "column_backend", "columns")
                reductions = container.get("reduction_backends")
                if isinstance(reductions, (list, tuple)):
                    values = [f"d{degree}={value}" for degree, value in enumerate(reductions)
                              if isinstance(value, str)]
                    if values:
                        labels.append("reduction[" + ", ".join(values) + "]")
    return "; ".join(labels) if labels else "unavailable"


def _requested_spectrum(case, result):
    if case.get("operation") == "persistence":
        return "not applicable"
    request = "full" if case.get("spectrum", "full") == "full" else f"partial k={case.get('k', 'unknown')}"
    returned = {True: "complete", False: "partial"}.get(result.get("complete"), "unavailable")
    return f"{request} → {returned}"


def _stage_time(worker, stage):
    value = worker.get("timings_seconds", {}).get(stage)
    if value is None:
        return "—"
    suffix = " (partial)" if stage not in worker.get("completed_stages", []) else ""
    return _number(value) + suffix


def _failure_detail(record):
    worker = record.get("worker", {})
    error = worker.get("exception") or record.get("controller_exception") or {}
    details = []
    if error:
        details.append(f"{error.get('class', 'Exception')}: {error.get('message', '')}")
    if record.get("status") == "timeout":
        details.append(f"Wall limit {_number(record.get('wall_limit_seconds'))} s; observed elapsed "
                       f"{_number(record.get('wall_seconds'))} s is censored, not a completed runtime.")
    elif record.get("status") == "rss_limit":
        details.append(f"Stopped at RSS ceiling {_number(record.get('rss_limit_bytes', 0) / 1024**2)} MiB.")
    elif record.get("status") == "no_terminal_record":
        details.append("No terminal record; a saved checkpoint does not establish whether a process is still running.")
    if not details:
        details.append(f"No exception message recorded; exit code {record.get('exit_code', 'unavailable')}.")
    return " ".join(details)


def summarize_run(run_directory):
    run, manifest, planned, records, hashes, issues = load_run(run_directory)
    artifact_checks = verify_artifacts(run, records, issues)
    machine = manifest.get("machine", {})
    signature = manifest.get("signature", {})
    declared_count = manifest.get("planned_cases")
    planned_count = len(planned) if planned else declared_count
    if planned and declared_count is not None and declared_count != len(planned):
        issues.append(f"Manifest planned_cases={declared_count} disagrees with its {len(planned)} case specifications")
    statuses = Counter(record.get("status", "unknown") for record in records)
    seen = {record.get("case_id") for record in records}
    missing = [case for case in planned if case.get("case_id") not in seen]
    rows = list(records)
    for case in missing:
        directory = _case_directory(run, case.get("case_id"))
        checkpoint_path = directory / "checkpoint.json" if directory else None
        worker = _read_json(checkpoint_path, issues) if checkpoint_path and checkpoint_path.is_file() else {}
        rows.append({"case_id": case.get("case_id"), "case": case, "status": "no_terminal_record",
                     "last_stage": worker.get("stage", "not started"), "worker": worker})
    requested_cases = planned or [record.get("case", {}) for record in records]
    dimension_groups = {_dimension_group(case) for case in requested_cases}
    requested_group = next(iter(dimension_groups)) if len(dimension_groups) == 1 else "mixed" if dimension_groups else "unknown"
    requested_degrees = sorted({case["degree"] for case in requested_cases
                               if isinstance(case.get("degree"), int) and not isinstance(case["degree"], bool)})
    group_names = {"low": "low-dimensional (q0/q1)", "higher": "higher-dimensional (q2)",
                   "mixed": "mixed", "unknown": "unknown"}
    trial_counts = Counter(_trial_key(case) for case in requested_cases)
    workers = [record.get("worker", {}) for record in records]
    versions = {}
    for name in ("numpy", "scipy"):
        versions[name] = sorted({str(worker.get("environment", {}).get(name)) for worker in workers
                                 if worker.get("environment", {}).get(name) is not None})
    actual_pools = set()
    for worker in workers:
        pool_record = worker.get("environment", {}).get("runtime_threadpools", {})
        for pool in pool_record.get("pools") or []:
            if isinstance(pool, dict) and "num_threads" in pool:
                actual_pools.add(f"{pool.get('internal_api', 'unknown')} {pool.get('version', '')}: {pool['num_threads']} threads")
    manifest_hash = hashlib.sha256(json.dumps(signature.get("source_sha256", {}), sort_keys=True).encode()).hexdigest()
    lines = ["# TopoKit pressure-run results", "", f"Run directory: `{run}`", "",
             f"Recorded host: **{_escape(machine.get('hostname', 'unavailable'))}**. "
             f"Platform: {_escape(machine.get('platform', 'unavailable'))}.", "",
             f"Python: {_escape(machine.get('python', signature.get('python', 'unavailable')))}. "
             f"NumPy: {', '.join(versions['numpy']) or 'unavailable'}. SciPy: {', '.join(versions['scipy']) or 'unavailable'}.", "",
             f"Created: {_escape(manifest.get('created_utc', 'unavailable'))}. "
             f"Run status: {_escape(manifest.get('status', 'unavailable'))}. "
             f"Recorded logical CPUs: {_escape(machine.get('logical_cpus', 'unavailable'))}.", "",
             f"Requested dimension group: **{group_names[requested_group]}**. "
             "Requested degrees: " + (", ".join(f"q{degree}" for degree in requested_degrees) or "unavailable") + ".", "",
             f"Thread environment: `{json.dumps(machine.get('thread_environment', signature.get('thread_environment', {})), sort_keys=True)}`.", "",
             "Observed numerical thread pools: " + ("; ".join(sorted(actual_pools)) or "unavailable") + ".", "",
             f"Terminal case records: **{len(records)} / {planned_count if planned_count is not None else 'unknown'} planned**. "
             f"Successful analyses: **{statuses.get('success', 0)}**. Cases without a terminal record: **{len(missing)}**.", "",
             _table(("Recorded status", "Count"), sorted(statuses.items()) or [("none", 0)]), "",
             "One row represents one recorded trial; times are not averages. Repetition indices are shown as saved. "
             "Wall time includes child startup, generation, support preparation, construction, analysis, validation, export, and checkpoint overhead. "
             "Build and analysis are the separately recorded worker stages. Combined build + analysis is the compact public workflow's single recorded stage, including support construction; it is not an isolated analysis time. "
             "Probe time records the separate canonical equivalence check. Timeout wall times are censored; failed-stage times marked partial are time spent before failure. "
             "Missing values are unavailable, not zero.", "",
             "Interaction points and support edges are listed **per factor A / B**. Hyperdigraph support edges are undirected; directed edges are listed separately. "
             "Canonical cell counts are explicit native generators; compact counts include packed vertices and edges and exclude implicit two-paths. These counts describe different stored representations. "
             "The spectral basis dimension can differ for embedded hyperdigraph chains. Peak RSS is the recorded process peak, including imports, export, and any equivalence probe; it is not packed-array storage size. "
             "A requested partial spectrum may return a complete spectrum when the requested k covers the entire basis. "
             "Full and requested-partial cases remain separate rows.", "",
             "Dimension groups reflect the requested q and do not assert a speedup. Algorithm/backend names are copied from recorded construction or analysis diagnostics; "
             "unavailable means a backend name was not captured. Execution route distinguishes recorded canonical and compact_hyperdigraph runs; a case-only route is labeled requested.", "",
             "## Evidence integrity", "",
             f"Artifact status: **{'fail' if any(value == 'fail' for value in artifact_checks.values()) else 'pass'}**. "
             f"Verified successful NPZ files: {sum(value == 'pass' for value in artifact_checks.values())} / {statuses.get('success', 0)}.", "",
             "Worker package source hashes: " + (", ".join(f"`{value}`" for value in hashes) or "unavailable") + ".", "",
             f"Manifest source-map digest (distinct from the worker package hash): `{manifest_hash}`.", ""]
    if issues:
        lines.extend(["Integrity checks: **fail**.", ""] + [f"- {_escape(issue)}" for issue in issues] + [""])
    else:
        lines.extend(["Integrity checks: **pass**; no duplicate case IDs or differing available worker source hashes were found.", ""])
    groups = defaultdict(list)
    for record in rows:
        case = record.get("case", {})
        groups[(_dimension_group(case), _profile(case))].append(record)
    group_order = {"low": 0, "higher": 1, "unknown": 2}
    previous_group = None
    for (dimension_group, profile), group in sorted(groups.items(), key=lambda item: (group_order[item[0][0]], item[0][1])):
        if dimension_group != previous_group:
            lines.extend([f"## {group_names[dimension_group].capitalize()} cases", ""])
            previous_group = dimension_group
        lines.extend([f"### {_escape(profile)}", ""])
        table_rows = []
        for record in sorted(group, key=lambda item: str(item.get("case_id"))):
            case, worker = record.get("case", {}), record.get("worker", {})
            construction, result = worker.get("construction_summary", {}), worker.get("result_summary", {})
            if case.get("family") == "interaction":
                counts = worker.get("input_summary", {}).get("points_per_cloud", [case.get("points"), case.get("points")])
                points = " / ".join(_count(value) for value in counts)
                support = " / ".join(_count(value) for value in construction.get("factor_support_edge_counts", [])) or "—"
                variant = f"{case.get('overlap', 'unknown')} overlap"
            else:
                points = _count(case.get("points"))
                support = _count(construction.get("support_edge_count"))
                variant = str(case.get("weights", "unknown")) if case.get("family") == "hyperdigraph" else "—"
            repetitions = trial_counts.get(_trial_key(case), 1)
            trial = f"r={case.get('repetition', 'unavailable')}; " + ("one trial" if repetitions == 1 else f"{repetitions} planned trials")
            wall = _number(record.get("wall_seconds"))
            if record.get("status") == "timeout":
                wall += " (censored)"
            elif record.get("status") == "no_terminal_record":
                wall = _number(worker.get("resources", {}).get("elapsed_seconds")) + " (checkpoint)"
            rss = record.get("peak_rss_bytes", worker.get("resources", {}).get("peak_rss_bytes"))
            table_rows.append((record.get("case_id", "unknown"), _execution_route(record), _algorithm(record), points, variant, _requested_spectrum(case, result), trial,
                support, _count(construction.get("directed_edge_count")), _count(construction.get("cell_count")),
                _count(result.get("basis_size")), wall, _stage_time(worker, "construction"), _stage_time(worker, "analysis"),
                _stage_time(worker, "construction_and_analysis"), _stage_time(worker, "canonical_equivalence"),
                _number(rss / 1024**2 if isinstance(rss, (float, int)) else None), record.get("status", "unknown"),
                artifact_checks.get(record.get("case_id"), "not applicable")))
        lines.extend([_table(("Case", "Execution route", "Recorded algorithm/backend", "Points", "Variant", "Spectrum request → result", "Trial", "Support edges A / B",
            "Directed edges", "Total cells", "Basis", "Wall s", "Build s", "Analysis s", "Combined build + analysis s", "Probe s", "Peak RSS MiB", "Status", "NPZ"), table_rows), ""])
    compact_records = [record for record in rows if _execution_route(record) == "compact_hyperdigraph"]
    if compact_records:
        lines.extend(["## Compact workflow evidence", "",
            "The compact workflow uses an implicit representation for H0/H1. Implicit two-path counts below are copied from recorded metadata; unavailable counts are not estimated. "
            "A bounded canonical probe checks the recorded smaller input, not every interval of a larger pressure case. "
            "Its status distinguishes exact endpoints from agreement within the recorded tolerance; the exported pressure-case values are not rounded. "
            "For a larger pressure case, a passing bounded probe does not establish full-case equivalence or a speedup over the canonical route.", ""])
        probe_rows = []
        for record in sorted(compact_records, key=lambda item: str(item.get("case_id"))):
            worker = record.get("worker", {})
            construction = worker.get("construction_summary", {})
            probe = worker.get("canonical_equivalence", {})
            tolerance = (f"rtol={probe.get('rtol')}, atol={probe.get('atol')}"
                         if "rtol" in probe and "atol" in probe else "unavailable")
            probe_rows.append((record.get("case_id", "unknown"),
                construction.get("cell_count_semantics", "unavailable"),
                _count(construction.get("represented_paths_by_degree", {}).get("2")),
                construction.get("represented_two_path_count_method", "unavailable"),
                probe.get("status", "unavailable"), _count(probe.get("points")),
                probe.get("scope", "unavailable"), tolerance))
        lines.extend([_table(("Case", "Stored cell count semantics", "Implicit two-paths", "Count method", "Probe status", "Probe points", "Probe scope", "Endpoint tolerance"), probe_rows), ""])
    failures = [record for record in rows if record.get("status") != "success"]
    lines.extend(["## Failures and unfinished cases", ""])
    if failures:
        lines.append(_table(("Case", "Status", "Last stage", "Details"), [
            (record.get("case_id"), record.get("status"), record.get("last_stage", record.get("worker", {}).get("stage", "unavailable")),
             _failure_detail(record)) for record in failures]))
    else:
        lines.append("No failures or unfinished planned cases are present in these records.")
    lines.append("")
    audit = {"record_count": len(records), "planned_count": planned_count, "status_counts": dict(statuses),
             "artifact_checks": artifact_checks, "source_hashes": hashes, "issues": issues,
             "requested_dimension_group": requested_group, "requested_degrees": requested_degrees}
    return "\n".join(lines), audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    try:
        report, audit = summarize_run(arguments.run_directory)
        output = arguments.output or arguments.run_directory / "REPORT.md"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
    except (OSError, ValueError) as error:
        print(f"Summary failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"report": str(output.resolve()), "records": audit["record_count"],
                      "integrity": "fail" if audit["issues"] else "pass"}))
    return 1 if audit["issues"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
