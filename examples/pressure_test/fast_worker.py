"""Measure the existing public compact score-directed H0/H1 workflow separately.

This route applies only to distinct scores and never materializes the directed
two-path collection. It is not the canonical builder/core route and does not
provide an H2 or Laplacian algorithm. The parent owns process resource limits.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import platform
import sys
import traceback
import warnings

import worker


PUBLIC_API = "topokit.workflows.hyperdigraph.compute_persistence_from_point_cloud"
EQUIVALENCE_POINTS = 8
EQUIVALENCE_RTOL = 1e-12
EQUIVALENCE_ATOL = 1e-14


def validate_case(case):
    if not isinstance(case, dict):
        raise TypeError("a case must be a JSON object")
    if case.get("limits"):
        raise ValueError("compact public workflow exposes no canonical resource-guard arguments; use limits={} and parent process limits")
    checked = worker.validate_case({"seed": 20260905, **case})
    checked["limits"] = {}
    if checked.get("execution_route") != "compact_hyperdigraph":
        raise ValueError("execution_route must be compact_hyperdigraph")
    if checked["family"] != "hyperdigraph" or checked["weights"] != "distinct":
        raise ValueError("compact workflow requires hyperdigraphs with distinct direction weights")
    if checked["operation"] != "persistence" or checked["degree"] not in {0, 1}:
        raise ValueError("compact workflow supports persistence H0/H1 only")
    if checked["overlap"] != "full" or checked["spectrum"] != "full":
        raise ValueError("compact persistence cases must use full overlap and full result labels")
    if checked.get("filtration_cutoff") is not None:
        raise ValueError("this compact suite requires unrestricted filtration")
    if not 2 <= checked["points"] <= 65_535:
        raise ValueError("compact packed storage supports 2 through 65,535 vertices")
    return checked


def compute_compact(case, cloud):
    from topokit.workflows.hyperdigraph import compute_persistence_from_point_cloud

    return compute_persistence_from_point_cloud(
        cloud.points, cloud.weights, max_dimension=case["degree"],
        connection_support=case["construction"], max_distance=None,
        include_diagonal=False, omega_backend="auto", reduction_backend="native_h1",
        storage_backend="numpy")


def common_result(result):
    """Preserve every returned float and its order in the common export schema."""
    from topokit.results import PersistenceInterval, PersistenceResult

    return PersistenceResult(tuple(PersistenceInterval(
        item.dimension, item.birth, item.death, item.birth == 0.0)
        for item in result.intervals), result.diagnostics.coefficient_field,
        result.max_dimension, {"diagnostics": asdict(result.diagnostics),
                               "execution_route": "compact_hyperdigraph"})


def compare_intervals(compact, canonical):
    """Distinguish bitwise equality from separately declared float tolerance."""
    left = sorted((item.dimension, float(item.birth), float(item.death)) for item in compact.intervals)
    right = sorted((item.dimension, float(item.birth), float(item.death)) for item in canonical.intervals)
    same_dimensions = [item[0] for item in left] == [item[0] for item in right]
    exact_match = [[q, birth.hex(), death.hex()] for q, birth, death in left] == [
        [q, birth.hex(), death.hex()] for q, birth, death in right]
    comparison = {"compact_interval_count": len(left), "canonical_interval_count": len(right),
        "same_interval_counts_by_degree": same_dimensions, "exact_endpoint_match": exact_match,
        "rtol": EQUIVALENCE_RTOL, "atol": EQUIVALENCE_ATOL,
        "differing_finite_endpoints": 0, "max_absolute_endpoint_difference": 0.0,
        "max_relative_endpoint_difference": 0.0, "endpoint_difference_examples": [],
        "same_infinite_endpoint_mask": same_dimensions, "within_tolerance": same_dimensions}
    if same_dimensions:
        for index, (actual, expected) in enumerate(zip(left, right)):
            for column, label in ((1, "birth"), (2, "death")):
                actual_value, expected_value = actual[column], expected[column]
                if math.isinf(actual_value) or math.isinf(expected_value):
                    if actual_value != expected_value:
                        comparison["same_infinite_endpoint_mask"] = False
                        comparison["within_tolerance"] = False
                    continue
                difference = abs(actual_value - expected_value)
                if difference:
                    comparison["differing_finite_endpoints"] += 1
                    comparison["max_absolute_endpoint_difference"] = max(
                        comparison["max_absolute_endpoint_difference"], difference)
                    relative = difference / max(abs(actual_value), abs(expected_value))
                    comparison["max_relative_endpoint_difference"] = max(
                        comparison["max_relative_endpoint_difference"], relative)
                    if len(comparison["endpoint_difference_examples"]) < 8:
                        comparison["endpoint_difference_examples"].append({
                            "sorted_interval_index": index, "degree": actual[0], "endpoint": label,
                            "compact_hex": actual_value.hex(), "canonical_hex": expected_value.hex(),
                            "absolute_difference": difference})
                if not math.isclose(actual_value, expected_value,
                                    rel_tol=EQUIVALENCE_RTOL, abs_tol=EQUIVALENCE_ATOL):
                    comparison["within_tolerance"] = False
    comparison["status"] = ("exact" if comparison["exact_endpoint_match"] else
                            "within_tolerance" if comparison["within_tolerance"] else "mismatch")
    return comparison


def canonical_equivalence(case, result):
    """Check the same generated geometry, bounded independently of pressure N."""
    from topokit.builders import hyperdigraph
    from topokit import core

    small_case = {**case, "points": min(case["points"], EQUIVALENCE_POINTS)}
    clouds = worker.generate_inputs(small_case)
    bonds, _ = worker.prepare_support(small_case, clouds)
    topology = hyperdigraph.from_points(clouds[0], max_dimension=case["degree"],
        bonds=bonds, max_hyperedges=worker.DEFAULT_LIMITS["max_hyperedges"])
    canonical = core.persistence(topology, max_dimension=case["degree"], field=2,
        include_diagonal=False, max_chain_bytes=worker.DEFAULT_LIMITS["max_chain_bytes"])
    compact = result if small_case["points"] == case["points"] else common_result(compute_compact(small_case, clouds[0]))
    comparison = compare_intervals(compact, canonical)
    comparison.update({"points": small_case["points"], "seed": case["seed"],
        "scope": "entire case" if small_case["points"] == case["points"] else "independent small prefix; does not verify all large-case intervals",
        "canonical_api": "topokit.builders.hyperdigraph.from_points + topokit.core.persistence",
        "canonical_limits": {"max_hyperedges": worker.DEFAULT_LIMITS["max_hyperedges"],
                             "max_chain_bytes": worker.DEFAULT_LIMITS["max_chain_bytes"]},
        "canonical_interval_sha256": worker.summarize_result(small_case, canonical)["intervals_sha256"],
        "compact_interval_sha256": worker.summarize_result(small_case, compact)["intervals_sha256"],
        "numerical_note": "same coordinates and support policy; NumPy distance evaluation order can change final float bits; no intervals rounded"})
    return comparison


def construction_summary(case, diagnostics):
    vertices, edges = diagnostics.vertex_count, diagnostics.edge_count
    represented = {"0": vertices, "1": edges}
    if case["degree"] == 1:
        represented["2"] = math.comb(vertices, 3) if case["construction"] == "complete" else None
    return {"support_edge_count": diagnostics.support_edge_count, "directed_edge_count": edges,
        "cells_by_degree": {"0": vertices, "1": edges}, "cell_count": vertices + edges,
        "cell_count_semantics": "packed vertices and edges only; implicit two-paths excluded",
        "materialized_hyperedges_by_degree": {"0": 0, "1": 0, "2": 0},
        "materialized_storage": "packed vertex/edge numeric arrays; no native Hyperedge objects or explicit two-path list",
        "represented_paths_by_degree": represented,
        "represented_two_path_count_method": ("not needed for H0" if case["degree"] == 0 else
            "n choose 3 for complete strict-score DAG" if case["construction"] == "complete" else
            "unknown: public compact diagnostics do not expose adjacency; support is not rebuilt to count paths"),
        "boundary_generators_processed": diagnostics.boundary_generator_count,
        "coordinate_units": "unit_cube", "built_through_degree": 1,
        "implicit_through_degree": case["degree"] + 1}


def run_case(case, output, *, input_error=None):
    checkpoint = worker.CaseCheckpoint(case, output)
    report = checkpoint.report
    report["execution_route"] = "compact_hyperdigraph"
    checkpoint.write()
    resource_limit_type = ()
    with warnings.catch_warnings(record=True) as captured:
        checkpoint.captured_warnings = captured
        warnings.simplefilter("always")
        try:
            with checkpoint.stage("initialization"):
                if input_error is not None:
                    raise input_error
                case = validate_case(case)
                report["case"] = case
                import numpy as np
                import scipy
                import topokit
                from topokit.exceptions import ResourceLimitError
                resource_limit_type = ResourceLimitError
                report["source_hash"] = worker.source_hash(Path(topokit.__file__).resolve().parent)
                report["environment"] = {"python": sys.version, "platform": platform.platform(),
                    "numpy": np.__version__, "scipy": scipy.__version__,
                    "topokit_source": str(Path(topokit.__file__).resolve()),
                    "threads": {name: os.environ.get(name) for name in
                        ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
                    "runtime_threadpools": worker.runtime_threadpools()}
                report["applied_limits"] = {"construction": {}, "analysis": {},
                    "intrinsic_api_caps": {"max_vertices": 65_535, "max_edges": 2_147_483_647},
                    "semantics": "public compact API has no allocation/reduction resource-guard arguments; canonical limits are rejected; wall/RSS/address limits belong to parent runner"}
                report["execution"] = {"route": "compact_hyperdigraph", "public_api": PUBLIC_API,
                    "options": {"max_dimension": case["degree"], "connection_support": case["construction"],
                        "max_distance": None, "include_diagonal": False, "omega_backend": "auto",
                        "reduction_backend": "native_h1", "storage_backend": "numpy"},
                    "timing_semantics": "construction_and_analysis includes support geometry, packed edge construction/sort and persistence; public API does not expose separate timings",
                    "support_semantics": "Delaunay 1-skeleton or complete pair support; directed distinct-vertex paths with consecutive-distance births; neither an alpha nor a Rips complex"}
            with checkpoint.stage("generation"):
                clouds = worker.generate_inputs(case)
                coordinate_hash = worker.input_hash(clouds, None, None)
                report["input_hash"] = worker._digest({"coordinate_weights_ids_hash": coordinate_hash,
                    "connection_support": case["construction"], "cutoff": None})
                report["input_summary"] = {"points_per_cloud": [case["points"]], "ambient_dimension": 3,
                    "coordinate_rule": "default_rng(seed).random((n,3))", "coordinate_units": "unit_cube",
                    "weight_rule": "arange(n)", "weight_role": "direction_only",
                    "coordinate_weights_ids_hash": coordinate_hash,
                    "input_hash_semantics": "digest of exact coordinate/weight/ID hash plus symbolic support policy; complete bonds not materialized for hashing"}
            with checkpoint.stage("construction_and_analysis"):
                native_result = compute_compact(case, clouds[0])
            with checkpoint.stage("summarization"):
                result = common_result(native_result)
                report["construction_summary"] = construction_summary(case, native_result.diagnostics)
                report["diagnostics"] = {"analysis": worker._compact(asdict(native_result.diagnostics))}
                report["execution"]["actual_backend"] = native_result.diagnostics.low_dimensional_backend
                report["result_summary"] = worker.summarize_result(case, result)
                report["environment"]["runtime_threadpools"] = worker.runtime_threadpools()
            with checkpoint.stage("export"):
                report["result_summary"]["artifact"] = worker.export_result(case, result, checkpoint.output)
            with checkpoint.stage("canonical_equivalence"):
                report["canonical_equivalence"] = canonical_equivalence(case, result)
                if not report["canonical_equivalence"]["within_tolerance"]:
                    raise worker.ScientificCheckError("compact intervals fail the small canonical equivalence check")
            report["status"] = "success"
            report["stage"] = "complete"
        except Exception as error:
            report["status"] = ("resource_limit" if isinstance(error, resource_limit_type) else
                "memory_error" if isinstance(error, MemoryError) else
                "numerical_failure" if isinstance(error, worker.ScientificCheckError) else "error")
            report["exception"] = {"class": type(error).__name__, "module": type(error).__module__,
                "message": str(error), "stage": report["stage"]}
            if report["status"] in {"error", "numerical_failure"}:
                log = checkpoint.output.with_suffix(".traceback.log")
                log.write_text(traceback.format_exc(), encoding="utf-8")
                report["exception"]["traceback_path"] = str(log)
        checkpoint.write()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    input_error = None
    try:
        case = json.loads(arguments.case.read_text(encoding="utf-8"))
    except Exception as error:
        input_error = error
        case = {"case_id": arguments.case.stem}
    report = run_case(case, arguments.output, input_error=input_error)
    print(json.dumps({"case_id": report["case_id"], "status": report["status"],
        "stage": report["stage"], "output": str(arguments.output.resolve())}), flush=True)
    return 1 if report["status"] in {"error", "numerical_failure"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
