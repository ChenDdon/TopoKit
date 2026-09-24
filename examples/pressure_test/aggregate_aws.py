"""Combine downloaded pressure-suite evidence using only the standard library.

This script reads local files, verifies successful NPZ hashes, and writes an
overview plus complete CSV/JSON records. It never runs benchmarks or contacts
AWS. A directory name is not evidence that the recorded host was on AWS.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile

import summarize


SUITES = ("smoke", "compact_smoke", "compact_pressure", "pressure_q0", "pressure_q1", "pressure_q2", "supplemental")
GROUP_FIELDS = ("profile", "family", "construction", "operation", "degree", "execution_route",
                "requested_execution_route", "weights", "overlap", "spectrum", "k",
                "hostname", "platform", "source_hash")
CSV_FIELDS = ("suite", "case_id", "case_key", "profile", "family", "construction", "operation", "degree",
    "execution_route", "requested_execution_route", "weights", "overlap", "spectrum", "k", "points",
    "points_per_cloud", "seed", "repetition", "attempted", "terminal", "status", "last_stage",
    "wall_seconds", "wall_time_semantics", "cpu_seconds", "peak_rss_bytes", "sampled_peak_rss_bytes",
    "wall_limit_seconds", "rss_limit_bytes", "address_limit_bytes", "requested_limits", "applied_limits",
    "construction_seconds", "analysis_seconds", "construction_and_analysis_seconds", "equivalence_probe_seconds",
    "timings_seconds", "completed_stages", "support_edge_count", "directed_edge_count", "cell_count",
    "cell_count_semantics", "cells_by_degree", "factor_support_edge_counts", "factor_cells_by_degree",
    "represented_paths_by_degree", "basis_size", "complete_spectrum", "interval_count", "backend",
    "probe_status", "probe_points", "probe_scope", "artifact_status", "suite_integrity", "artifact_filename", "artifact_sha256",
    "source_hash", "input_hash", "hostname", "platform", "python", "numpy", "scipy",
    "error_class", "error_message", "record_path")


def _write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.",
                                     encoding="utf-8", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(text)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _flatten(suite, run, manifest, record, artifact_status, *, terminal=True):
    case, worker = record.get("case", {}), record.get("worker", {})
    resources = worker.get("resources", {})
    construction, result = worker.get("construction_summary", {}), worker.get("result_summary", {})
    timings, probe = worker.get("timings_seconds", {}), worker.get("canonical_equivalence", {})
    machine, environment = manifest.get("machine", {}), worker.get("environment", {})
    error = worker.get("exception") or record.get("controller_exception") or {}
    case_id = record.get("case_id")
    attempted = terminal or bool(worker)
    points = case.get("points")
    points_per_cloud = worker.get("input_summary", {}).get("points_per_cloud")
    if points_per_cloud is None and points is not None:
        points_per_cloud = [points] * (2 if case.get("family") == "interaction" else 1)
    directory = summarize._case_directory(run, case_id)
    record_path = directory / ("record.json" if terminal else "checkpoint.json") if directory else None
    row = {field: case.get(field) for field in CSV_FIELDS}
    row.update(suite=suite, case_id=case_id, case_key=f"{suite}/{case_id}",
        profile=summarize._profile(case), execution_route=summarize._execution_route(record),
        requested_execution_route=case.get("execution_route"),
        points_per_cloud=points_per_cloud, attempted=attempted, terminal=terminal,
        status=record.get("status", "unknown"), last_stage=record.get("last_stage", worker.get("stage", "not started")),
        wall_seconds=record.get("wall_seconds") if terminal else resources.get("elapsed_seconds"),
        wall_time_semantics=("censored at wall limit" if record.get("status") == "timeout" else
                             "observed child lifetime" if terminal else "checkpoint elapsed; no terminal runtime" if attempted else "not started; no runtime"),
        cpu_seconds=record.get("cpu_seconds", resources.get("cpu_seconds")),
        peak_rss_bytes=record.get("peak_rss_bytes", resources.get("peak_rss_bytes")),
        sampled_peak_rss_bytes=record.get("sampled_peak_rss_bytes"),
        wall_limit_seconds=record.get("wall_limit_seconds"), rss_limit_bytes=record.get("rss_limit_bytes"),
        address_limit_bytes=record.get("address_limit_bytes"), requested_limits=case.get("limits"),
        applied_limits=worker.get("applied_limits"),
        construction_seconds=timings.get("construction"), analysis_seconds=timings.get("analysis"),
        construction_and_analysis_seconds=timings.get("construction_and_analysis"),
        equivalence_probe_seconds=timings.get("canonical_equivalence"), timings_seconds=timings,
        completed_stages=worker.get("completed_stages", []),
        support_edge_count=construction.get("support_edge_count"), directed_edge_count=construction.get("directed_edge_count"),
        cell_count=construction.get("cell_count"), cell_count_semantics=construction.get("cell_count_semantics"),
        cells_by_degree=construction.get("cells_by_degree"), factor_support_edge_counts=construction.get("factor_support_edge_counts"),
        factor_cells_by_degree=construction.get("factor_cells_by_degree"),
        represented_paths_by_degree=construction.get("represented_paths_by_degree"),
        basis_size=result.get("basis_size"), complete_spectrum=result.get("complete"), interval_count=result.get("interval_count"),
        backend=summarize._algorithm(record), probe_status=probe.get("status"), probe_points=probe.get("points"),
        probe_scope=probe.get("scope"), artifact_status=artifact_status,
        artifact_filename=result.get("artifact", {}).get("filename"), artifact_sha256=result.get("artifact", {}).get("sha256"),
        source_hash=worker.get("source_hash"), input_hash=worker.get("input_hash"),
        hostname=machine.get("hostname"), platform=machine.get("platform", environment.get("platform")),
        python=machine.get("python", environment.get("python")), numpy=environment.get("numpy"), scipy=environment.get("scipy"),
        error_class=error.get("class"), error_message=error.get("message"),
        record_path=str(record_path) if record_path and record_path.is_file() else None)
    # k is scientifically relevant only for a requested partial spectrum.
    if case.get("spectrum", "full") != "partial" or case.get("operation") == "persistence":
        row["k"] = None
    return row


def _largest(rows):
    eligible = [row for row in rows if isinstance(row.get("points"), int)
                and not isinstance(row["points"], bool) and row["points"] >= 0]
    largest = max((row["points"] for row in eligible), default=None)
    return largest, [row["case_key"] for row in eligible if row["points"] == largest]


def collect(parent_directory):
    parent = Path(parent_directory).resolve()
    if not parent.is_dir():
        raise ValueError(f"Run parent directory does not exist: {parent}")
    suites, rows, issues = [], [], []
    for name in SUITES:
        run = parent / name
        if not run.is_dir():
            suites.append({"suite": name, "directory": str(run), "state": "unrun",
                "planned_count": None, "terminal_count": 0, "attempted_count": 0,
                "status_counts": {}, "artifact_checks": {}, "issues": [], "machine": {}})
            continue
        run, manifest, planned, records, hashes, suite_issues = summarize.load_run(run)
        checks = summarize.verify_artifacts(run, records, suite_issues)
        declared = manifest.get("planned_cases")
        planned_count = len(planned) if "cases" in manifest.get("signature", {}) else declared
        if declared is not None and planned_count is not None and declared != planned_count:
            suite_issues.append(f"Manifest planned_cases={declared} differs from its {planned_count} case specifications")
        seen = {record.get("case_id") for record in records}
        suite_rows = [_flatten(name, run, manifest, record, checks.get(record.get("case_id"), "not applicable"))
                      for record in records]
        for case in planned:
            if case.get("case_id") in seen:
                continue
            directory = summarize._case_directory(run, case.get("case_id"))
            checkpoint = directory / "checkpoint.json" if directory else None
            worker = summarize._read_json(checkpoint, suite_issues) if checkpoint and checkpoint.is_file() else {}
            record = {"case_id": case.get("case_id"), "case": case, "worker": worker,
                "status": "no_terminal_record" if worker else "not_started", "last_stage": worker.get("stage", "not started")}
            suite_rows.append(_flatten(name, run, manifest, record, "not applicable", terminal=False))
        state = ("integrity_failure" if suite_issues else "complete" if planned_count is not None
                 and len(records) == planned_count and all(row["terminal"] for row in suite_rows) else "incomplete")
        suites.append({"suite": name, "directory": str(run), "state": state,
            "recorded_run_status": manifest.get("status"), "planned_count": planned_count,
            "terminal_count": len(records), "attempted_count": sum(row["attempted"] for row in suite_rows),
            "status_counts": dict(Counter(row["status"] for row in suite_rows)),
            "artifact_checks": checks, "issues": list(suite_issues), "source_hashes": hashes,
            "machine": manifest.get("machine", {}),
            "source_manifest_digest": hashlib.sha256(json.dumps(manifest.get("signature", {}).get("source_sha256", {}), sort_keys=True).encode()).hexdigest()})
        issues.extend(f"{name}: {issue}" for issue in suite_issues)
        for row in suite_rows:
            row["suite_integrity"] = "fail" if suite_issues else "pass"
        rows.extend(suite_rows)
    source_hashes = sorted({row["source_hash"] for row in rows if row.get("source_hash")})
    if len(source_hashes) > 1:
        issues.append("Different worker package source hashes occur across suites; groups remain separate by source hash")
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field) for field in GROUP_FIELDS)].append(row)
    largest_groups = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        maximum, maximum_keys = _largest([row for row in group if row["attempted"]])
        success, success_keys = _largest([row for row in group if row["status"] == "success"
            and row["artifact_status"] == "pass" and row["suite_integrity"] == "pass"])
        largest_groups.append({**dict(zip(GROUP_FIELDS, key)), "max_tried_points": maximum,
            "max_tried_case_keys": maximum_keys, "largest_successful_points": success,
            "largest_successful_case_keys": success_keys,
            "status_counts": dict(Counter(row["status"] for row in group)), "case_count": len(group)})
    return {"schema_version": 1, "parent_directory": str(parent), "suites": suites, "cases": rows,
        "largest_tested_groups": largest_groups, "source_hashes": source_hashes, "issues": issues,
        "integrity": "fail" if issues else "pass", "status_counts": dict(Counter(row["status"] for row in rows)),
        "terminal_count": sum(row["terminal"] for row in rows), "attempted_count": sum(row["attempted"] for row in rows)}


def _range(values, divisor=1):
    values = [value / divisor for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    if not values:
        return "—"
    low, high = min(values), max(values)
    return summarize._number(low) if low == high else f"{summarize._number(low)}–{summarize._number(high)}"


def overview(data):
    table, number, count = summarize._table, summarize._number, summarize._count
    rows_by_key = {row["case_key"]: row for row in data["cases"]}
    lines = ["# TopoKit combined pressure results", "", f"Local evidence directory: `{data['parent_directory']}`.", "",
        f"Evidence integrity: **{data['integrity']}**. Terminal cases: **{data['terminal_count']}**; attempted cases: **{data['attempted_count']}**.", "",
        "Host and platform are copied from each suite manifest; directory names do not establish AWS execution. Missing suites are unrun, with unknown planned counts.", "",
        table(("Suite", "State", "Recorded host / platform", "Terminal / planned", "Successful", "Status counts"), [
            (suite["suite"], suite["state"], f"{suite['machine'].get('hostname', 'unavailable')} / {suite['machine'].get('platform', 'unavailable')}",
             f"{suite['terminal_count']} / {suite['planned_count'] if suite['planned_count'] is not None else 'unknown'}",
             suite["status_counts"].get("success", 0), json.dumps(suite["status_counts"], sort_keys=True)) for suite in data["suites"]]), "",
        "All case records, stage times, CPU/RSS, counts, backends, and artifact checks are retained in [combined.csv](combined.csv) and [combined.json](combined.json). "
        "Case identity includes the suite, so repeated case IDs across suites remain separate trials.", "",
        "## Largest tested successes", "",
        "Each row is a recorded profile, operation, degree, execution route, weight/overlap variant, and spectrum request. "
        "Hosts and source hashes also remain separate. Points are per interaction factor. "
        "Largest success means an observed successful case with verified NPZ evidence and no suite integrity error; max tried includes failed attempts and excludes cases that never started. "
        "These are tested sizes under the recorded limits, not capacity estimates or guarantees at smaller sizes. "
        "Wall/CPU/RSS ranges retain all individual successful trials at that size; they are not averages or speedup comparisons.", "",
        "Canonical construction and analysis times remain separate in the exports. Compact construction + analysis is one public workflow stage. "
        "Wall time and process peak RSS include startup, exports, and any bounded canonical probe. Compact packed vertex/edge counts exclude implicit two-paths. "
        "A bounded prefix probe does not validate all intervals of a larger compact case. Requested partial spectra stay separate even when they return a complete spectrum.", ""]
    for dimension, label in (("low", "Low-dimensional q0/q1"), ("higher", "Higher-dimensional q2"), ("unknown", "Other requested degrees")):
        groups = [group for group in data["largest_tested_groups"] if summarize._dimension_group(group) == dimension]
        if not groups:
            continue
        lines.extend([f"### {label}", ""])
        output = []
        for group in groups:
            successes = [rows_by_key[key] for key in group["largest_successful_case_keys"]]
            spectrum = group.get("spectrum") or "unavailable"
            if spectrum == "partial":
                spectrum += f" k={group.get('k')}"
            variant = f"{group.get('weights')}; {group.get('overlap')} overlap; {spectrum}"
            host_source = f"{group.get('hostname') or 'unavailable'}; source {(group.get('source_hash') or 'unavailable')[:12]}"
            output.append((group["profile"], f"{group['operation']} q{group['degree']}", group["execution_route"], variant, host_source,
                count(group["largest_successful_points"]), count(group["max_tried_points"]),
                len(successes), _range([row["wall_seconds"] for row in successes]),
                _range([row["cpu_seconds"] for row in successes]), _range([row["peak_rss_bytes"] for row in successes], 1024**2),
                ", ".join(group["largest_successful_case_keys"]) or "none",
                ", ".join(group["max_tried_case_keys"]) or "none"))
        lines.extend([table(("Profile", "Operation", "Route", "Variant / request", "Host / source prefix", "Largest successful N", "Max tried N", "Successful trials at N",
            "Wall s", "CPU s", "Peak RSS MiB", "Successful cases", "Max-tried cases"), output), ""])
    failures = [row for row in data["cases"] if row["status"] != "success"]
    lines.extend(["## Failures and unfinished cases", ""])
    if failures:
        lines.append(table(("Case", "Status", "Last stage", "Observed wall s", "Details"), [
            (row["case_key"], row["status"], row["last_stage"],
             number(row["wall_seconds"]) + (" (censored)" if row["status"] == "timeout" else " (checkpoint)" if not row["terminal"] and row["attempted"] else ""),
             f"{row['error_class']}: {row['error_message']}" if row["error_class"] else row["wall_time_semantics"])
            for row in failures]))
    else:
        lines.append("No failures among the available case records. Unrun suites above remain untested.")
    lines.extend(["", "## Integrity", ""])
    if data["issues"]:
        lines.extend(f"- {summarize._escape(issue)}" for issue in data["issues"])
    else:
        lines.append("Successful NPZ artifacts passed existence and SHA256 checks; available suite records passed the source-hash and duplicate-ID checks.")
    return "\n".join(lines) + "\n"


def write_reports(data, output_directory):
    output = Path(output_directory).resolve()
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for row in data["cases"]:
        writer.writerow({key: json.dumps(value, sort_keys=True, allow_nan=False)
                         if isinstance(value, (list, dict)) else value for key, value in row.items()})
    _write_text(output / "combined.csv", stream.getvalue())
    _write_text(output / "combined.json", json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    _write_text(output / "OVERVIEW.md", overview(data))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output-directory", type=Path)
    arguments = parser.parse_args(argv)
    data = collect(arguments.run_directory)
    output = write_reports(data, arguments.output_directory or arguments.run_directory)
    print(json.dumps({"output_directory": str(output), "terminal_count": data["terminal_count"],
        "status_counts": data["status_counts"], "integrity": data["integrity"]}))
    return 1 if data["issues"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
