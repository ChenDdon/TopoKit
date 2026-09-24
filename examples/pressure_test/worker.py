"""Run one explicit scientific pressure case with recoverable stage checkpoints.

This standalone example imports NumPy, SciPy, and TopoKit only when a case runs.
The parent runner owns process timeouts and process isolation. This worker never
changes topology, reduces a requested degree, or relaxes a resource guard.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
from itertools import combinations
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import tempfile
from time import perf_counter
import traceback
import warnings


DEFAULT_LIMITS = {
    "max_cells": 1_000_000,
    "max_simplices": 1_000_000,
    "max_hyperedges": 500_000,
    "max_chain_bytes": 536_870_912,
    "max_dense_entries": 4_000_000,
    "max_dense_bytes": 268_435_456,
    "max_sparse_entries": 10_000_000,
}
SPECTRAL_TOLERANCE = 1e-10


class ScientificCheckError(ArithmeticError):
    """A completed calculation returned scientifically invalid numerical data."""


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _positive_integer(value, name, *, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def validate_case(case):
    if not isinstance(case, dict):
        raise TypeError("a case must be a JSON object")
    checked = dict(case)
    if not isinstance(checked.get("case_id"), str) or not checked["case_id"]:
        raise ValueError("case_id must be a nonempty string")
    allowed = {
        "family": {"simplicial", "hyperdigraph", "interaction"},
        "operation": {"persistence", "laplacian", "persistent_laplacian"},
        "weights": {"distinct", "equal"},
        "overlap": {"full", "half"},
        "spectrum": {"full", "partial"},
    }
    for name, default in (("weights", "distinct"), ("overlap", "full"), ("spectrum", "full")):
        checked.setdefault(name, default)
    for name, values in allowed.items():
        if checked.get(name) not in values:
            raise ValueError(f"{name} must be one of {sorted(values)}")
    constructions = {"delaunay", "complete"} if checked["family"] == "hyperdigraph" else {"alpha", "rips"}
    if checked.get("construction") not in constructions:
        raise ValueError(f"{checked['family']} construction must be one of {sorted(constructions)}")
    checked["points"] = _positive_integer(checked.get("points"), "points")
    checked["degree"] = _positive_integer(checked.get("degree"), "degree", minimum=0)
    if checked["degree"] > 2:
        raise ValueError("pressure cases currently cover degrees 0, 1, and 2")
    checked["ambient_dimension"] = checked.get("ambient_dimension", 3)
    if checked["ambient_dimension"] != 3 or isinstance(checked["ambient_dimension"], bool):
        raise ValueError("ambient_dimension must be 3 for this unit-cube suite")
    checked["seed"] = _positive_integer(checked.get("seed", 0), "seed", minimum=0)
    checked["k"] = _positive_integer(checked.get("k", 8), "k")
    requested_limits = checked.get("limits", {})
    if not isinstance(requested_limits, dict):
        raise TypeError("limits must be a JSON object")
    unknown_limits = set(requested_limits) - set(DEFAULT_LIMITS)
    if unknown_limits:
        raise ValueError(f"unknown limits: {sorted(unknown_limits)}")
    checked["limits"] = {**DEFAULT_LIMITS, **requested_limits}
    for name, value in checked["limits"].items():
        _positive_integer(value, name)
    _canonical_json(checked)
    return checked


def source_hash(package_directory):
    """Hash the actually imported package, including baseline PYTHONPATH runs."""
    digest = hashlib.sha256()
    root = Path(package_directory)
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _compact(value, depth=0):
    if value is None or isinstance(value, (bool, int, str)):
        return value if not isinstance(value, str) else value[:2000]
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if depth >= 5:
        return {"omitted_type": type(value).__name__}
    if isinstance(value, dict):
        items = list(value.items())
        result = {str(key): _compact(item, depth + 1) for key, item in items[:64]}
        if len(items) > 64:
            result["omitted_items"] = len(items) - 64
        return result
    if isinstance(value, (list, tuple)):
        result = [_compact(item, depth + 1) for item in value[:32]]
        if len(value) > 32:
            result.append({"omitted_items": len(value) - 32})
        return result
    return {"omitted_type": type(value).__name__}


def compact_diagnostics(metadata):
    names = (
        "definition_id", "coefficient_field", "geometry_versions", "core_version",
        "construction_diagnostics", "reduction_diagnostics", "laplacian_diagnostics",
        "diagnostics", "chain_storage_bound_bytes", "estimated_solver_workspace_bytes",
        "backend", "operator_kind", "basis_kind", "tol", "tolerance",
        "eigenvalue_zero_tolerance", "rank_rtol", "constraint_rank", "omega_backend",
    )
    return {name: _compact(metadata[name]) for name in names if name in metadata}


def runtime_threadpools():
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        return {"available": False, "pools": None}
    return {"available": True, "pools": _compact(threadpool_info())}


class CaseCheckpoint:
    def __init__(self, case, output):
        self.output = Path(output).resolve()
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.started = perf_counter()
        self.captured_warnings = []
        self.warning_cursor = 0
        self.report = {
            "schema_version": 1, "case_id": case.get("case_id") if isinstance(case, dict) else None,
            "case": _compact(case), "status": "running", "stage": "initialization",
            "completed_stages": [], "timings_seconds": {}, "warnings": [],
            "source_hash": None, "input_hash": None,
        }

    def write(self):
        usage = resource.getrusage(resource.RUSAGE_SELF)
        self.report["resources"] = {
            "elapsed_seconds": perf_counter() - self.started,
            "cpu_user_seconds": usage.ru_utime, "cpu_system_seconds": usage.ru_stime,
            "cpu_seconds": usage.ru_utime + usage.ru_stime,
            "peak_rss_bytes": int(usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024),
        }
        for item in self.captured_warnings[self.warning_cursor:]:
            self.report["warnings"].append({
                "category": item.category.__name__, "message": str(item.message),
                "filename": item.filename, "lineno": item.lineno, "stage": self.report["stage"],
            })
        self.warning_cursor = len(self.captured_warnings)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{self.output.name}.", dir=self.output.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(self.report, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
            os.replace(temporary, self.output)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @contextmanager
    def stage(self, name):
        self.report["stage"] = name
        self.write()
        started = perf_counter()
        try:
            yield
        except BaseException:
            self.report["timings_seconds"][name] = perf_counter() - started
            self.write()
            raise
        else:
            self.report["timings_seconds"][name] = perf_counter() - started
            self.report["completed_stages"].append(name)
            self.write()


def generate_inputs(case):
    import numpy as np
    from topokit.data import PointCloud

    rng = np.random.default_rng(case["seed"])
    count = case["points"]
    weights = np.arange(count, dtype=float) if case["weights"] == "distinct" else np.ones(count)
    clouds = []
    for factor in range(2 if case["family"] == "interaction" else 1):
        coordinates = rng.random((count, 3))
        ids = tuple(f"f{factor}:p{index}" for index in range(count))
        clouds.append(PointCloud(coordinates, ids=ids, weights=weights,
                                 metadata={"coordinate_units": "unit_cube"}))
    return tuple(clouds)


def prepare_support(case, clouds):
    bonds = None
    overlap = None
    if case["family"] == "hyperdigraph" and case["construction"] == "complete":
        bonds = tuple(combinations(clouds[0].ids, 2))
    if case["family"] == "interaction":
        count = case["points"] if case["overlap"] == "full" else case["points"] // 2
        overlap = tuple(zip(clouds[0].ids[:count], clouds[1].ids[:count]))
    return bonds, overlap


def input_hash(clouds, bonds, overlap):
    digest = hashlib.sha256()
    for cloud in clouds:
        digest.update(_canonical_json({"ids": cloud.ids, "units": "unit_cube", "shape": cloud.points.shape}).encode())
        digest.update(cloud.points.astype("<f8", copy=False).tobytes())
        digest.update(cloud.weights.astype("<f8", copy=False).tobytes())
    digest.update(_canonical_json({"bonds": bonds, "overlap_vertices": overlap}).encode())
    return digest.hexdigest()


def build_topology(case, clouds, bonds, overlap):
    from topokit.builders import hyperdigraph, interaction, simplicial

    degree, limits = case["degree"], case["limits"]
    if case["family"] == "simplicial":
        return simplicial.from_points(clouds[0], max_dimension=degree,
            complex_type=case["construction"], max_simplices=limits["max_simplices"])
    if case["family"] == "hyperdigraph":
        return hyperdigraph.from_points(clouds[0], max_dimension=degree,
            bonds=bonds, max_hyperedges=limits["max_hyperedges"])
    return interaction.from_points(clouds[0], clouds[1], overlap_vertices=overlap,
        max_dimension=degree, factor_options={"complex_type": case["construction"]},
        max_cells=limits["max_cells"], max_simplices=limits["max_simplices"])


def births_by_degree(topology, maximum_degree):
    """Inspect supplied native grades without making geometric subobjects."""
    grades = [[] for _ in range(maximum_degree + 1)]
    if topology.kind == "simplicial":
        for simplex, birth in topology.native.get_filtration():
            grades[len(simplex) - 1].append(float(birth))
    elif topology.kind == "hyperdigraph":
        for degree in range(maximum_degree + 1):
            grades[degree].extend(float(item.birth) for item in topology.native.weighted_hyperedges(degree))
    else:
        for degree, layer in enumerate(topology.native.degrees):
            grades[degree].extend(layer.births)
    return grades


def select_observations(topology, degree):
    grades = births_by_degree(topology, degree + 1)
    if any(not math.isfinite(birth) for layer in grades for birth in layer):
        raise ScientificCheckError("constructed cell births must be finite")
    positive = sorted({birth for layer in grades for birth in layer if birth > 0.0})
    if positive:
        start_index = int(0.50 * (len(positive) - 1))
        end_index = int(0.80 * (len(positive) - 1))
        start, end = positive[start_index], positive[end_index]
    else:
        start_index = end_index = None
        start = end = 0.0
    source = {str(q): sum(birth <= start for birth in layer) for q, layer in enumerate(grades)}
    target = {str(q): sum(birth <= end for birth in layer) for q, layer in enumerate(grades)}
    observations = {
        "start": start, "end": end, "scale_units": topology.metadata.get("scale_units"),
        "selection_method": "floor(fraction * (positive_unique_count - 1)); fractions 0.50, 0.80",
        "positive_critical_count": len(positive), "start_index": start_index, "end_index": end_index,
        "fallback_used": not bool(positive), "fallback_reason": None if positive else "no positive finite critical births",
        "source_active_cells_by_degree": source, "target_active_cells_by_degree": target,
        "source_active_cell_count": sum(source.values()), "target_active_cell_count": sum(target.values()),
        "source_degree_cell_count": source[str(degree)], "target_degree_cell_count": target[str(degree)],
        "active_count_semantics": "native generators; hyperdigraph counts are not embedded-chain dimensions",
    }
    return construction_summary(topology, degree), observations


def construction_summary(topology, degree):
    """Retain counts before observation work, without materializing new cells."""
    if topology.kind == "simplicial":
        counts = {str(q): 0 for q in range(degree + 2)}
        for simplex, _ in topology.native.get_filtration():
            counts[str(len(simplex) - 1)] += 1
    elif topology.kind == "hyperdigraph":
        counts = {str(q): len(topology.native.weighted_hyperedges(q)) for q in range(degree + 2)}
    else:
        counts = {str(q): len(layer.keys) for q, layer in enumerate(topology.native.degrees)}
    construction = {"cells_by_degree": counts, "cell_count": sum(counts.values()),
                    "built_through_degree": degree + 1, "coordinate_units": "unit_cube"}
    if topology.kind == "interaction":
        factor_counts = [{str(q): len(factor.simplices_of_dimension(q)) for q in range(degree + 2)}
                         for factor in topology.native.factors]
        construction.update({"factor_cells_by_degree": factor_counts,
            "factor_simplex_counts": [factor.number_of_simplices for factor in topology.native.factors],
            "factor_support_edge_counts": [item.get("1", 0) for item in factor_counts],
            "overlap_count": topology.metadata["overlap_count"]})
    elif topology.kind == "hyperdigraph":
        construction["support_edge_count"] = topology.metadata.get(
            "retained_connection_count", topology.metadata.get("support_edge_count"))
        construction["directed_edge_count"] = topology.metadata["directed_edge_count"]
    else:
        construction["support_edge_count"] = counts.get("1", 0)
    return construction


def analyze(case, topology, observations):
    from topokit import core

    limits = case["limits"]
    if case["operation"] == "persistence":
        options = {"max_dimension": case["degree"], "field": 2}
        if case["family"] == "hyperdigraph":
            options["max_chain_bytes"] = limits["max_chain_bytes"]
        return core.persistence(topology, **options)
    options = {"dimension": case["degree"], "return_eigenvectors": False,
               "return_matrix": False, "max_dense_entries": limits["max_dense_entries"],
               "tol": SPECTRAL_TOLERANCE}
    if case["spectrum"] == "partial":
        options["k"] = case["k"]
    if case["family"] == "hyperdigraph":
        options["max_sparse_entries"] = limits["max_sparse_entries"]
    elif case["family"] == "interaction":
        options["max_dense_bytes"] = limits["max_dense_bytes"]
    if case["operation"] == "laplacian":
        return core.laplacian(topology, scale=observations["end"], **options)
    return core.persistent_laplacian(topology, start=observations["start"],
                                     end=observations["end"], **options)


def applied_limits(case):
    construction_names = {
        "simplicial": ("max_simplices",), "hyperdigraph": ("max_hyperedges",),
        "interaction": ("max_cells", "max_simplices"),
    }[case["family"]]
    if case["operation"] == "persistence":
        analysis_names = ("max_chain_bytes",) if case["family"] == "hyperdigraph" else ()
    else:
        analysis_names = ("max_dense_entries",)
        if case["family"] == "hyperdigraph":
            analysis_names += ("max_sparse_entries",)
        elif case["family"] == "interaction":
            analysis_names += ("max_dense_bytes",)
    return {"construction": {name: case["limits"][name] for name in construction_names},
            "analysis": {name: case["limits"][name] for name in analysis_names},
            "semantics": "only guards supported by the selected public APIs; process limits belong to the runner"}


def summarize_result(case, result):
    if case["operation"] == "persistence":
        triples = [(int(item.dimension), float(item.birth), float(item.death)) for item in result.intervals]
        valid = all(0 <= q <= case["degree"] and math.isfinite(birth)
                    and (math.isfinite(death) or death == math.inf) and birth <= death
                    for q, birth, death in triples)
        if not valid:
            raise ScientificCheckError("invalid persistence interval degree or finite-birth/death ordering")
        canonical = [[q, birth.hex(), death.hex()] for q, birth, death in sorted(triples)]
        return {"kind": "persistence", "field": result.field, "interval_count": len(triples),
            "intervals_by_degree": {str(q): {"total": sum(d == q for d, _, _ in triples),
                "finite": sum(d == q and math.isfinite(death) for d, _, death in triples),
                "infinite": sum(d == q and death == math.inf for d, _, death in triples)}
                for q in range(case["degree"] + 1)},
            "intervals_sha256": _digest(canonical),
            "digest_encoding": "canonical JSON of numerically sorted [degree, birth.float.hex(), death.float.hex()]",
            "analysis_domain": "all actual filtration events; selected observation scales do not restrict persistence",
            "checks": {"valid_intervals": True}}
    import numpy as np

    values = np.asarray(result.eigenvalues, dtype=float)
    finite = bool(np.all(np.isfinite(values)))
    magnitude = float(np.max(np.abs(values))) if values.size else 0.0
    tolerance = max(SPECTRAL_TOLERANCE, SPECTRAL_TOLERANCE * magnitude)
    sorted_values = bool(np.all(np.diff(values) >= -tolerance))
    psd = bool(np.all(values >= -tolerance))
    basis_size = len(result.basis)
    size_consistent = values.ndim == 1 and values.size <= basis_size and (not result.complete or values.size == basis_size)
    partial_nullity = result.complete or result.nullity is None
    if not (finite and sorted_values and psd and size_consistent and partial_nullity):
        raise ScientificCheckError("spectrum failed finite, sorted, PSD, basis-size, or partial-nullity checks")
    residual = next((result.metadata[name] for name in
        ("eigenpair_residual_max", "eigen_residual_max", "residual_max") if name in result.metadata), None)
    return {"kind": result.kind, "degree": result.dimension, "size": int(values.size),
        "basis_size": basis_size, "complete": bool(result.complete), "nullity": result.nullity,
        "minimum": float(values.min()) if values.size else None,
        "maximum": float(values.max()) if values.size else None,
        "eigenpair_residual_max": _compact(residual),
        "eigenvalues_sha256": _digest([float(value).hex() for value in values]),
        "digest_encoding": "canonical JSON of returned float.hex() eigenvalues in returned order",
        "scale": result.scale, "start": result.start, "end": result.end,
        "checks": {"finite": finite, "sorted": sorted_values, "psd_within_tolerance": psd,
                   "tolerance": tolerance, "basis_size_consistent": bool(size_consistent),
                   "partial_nullity_unspecified": bool(partial_nullity)}}


def export_result(case, result, output):
    """Save the exact returned numerical records, retaining their original order."""
    import numpy as np

    artifact = Path(output).with_suffix(".npz")
    if case["operation"] == "persistence":
        arrays = {
            "degree": np.asarray([item.dimension for item in result.intervals], dtype=np.int64),
            "birth": np.asarray([item.birth for item in result.intervals], dtype=np.float64),
            "death": np.asarray([item.death for item in result.intervals], dtype=np.float64),
        }
        if all(hasattr(item, "at_initial_stage") for item in result.intervals):
            arrays["at_initial_stage"] = np.asarray(
                [item.at_initial_stage for item in result.intervals], dtype=bool)
    else:
        arrays = {"eigenvalues": np.asarray(result.eigenvalues)}
    descriptor, temporary = tempfile.mkstemp(prefix=f".{artifact.stem}.", suffix=".npz", dir=artifact.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, artifact)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    digest = hashlib.sha256()
    with artifact.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"filename": artifact.name, "sha256": digest.hexdigest(), "format": "npz",
            "arrays": list(arrays), "ordering": "exact public-result order"}


def run_case(case, output, *, input_error=None):
    checkpoint = CaseCheckpoint(case, output)
    report = checkpoint.report
    report["execution_route"] = "canonical"
    resource_limit_type = ()
    checkpoint.write()
    with warnings.catch_warnings(record=True) as captured:
        checkpoint.captured_warnings = captured
        warnings.simplefilter("always")
        try:
            with checkpoint.stage("initialization"):
                if input_error is not None:
                    raise input_error
                case = validate_case(case)
                report["case"] = case
                report["applied_limits"] = applied_limits(case)
                import numpy as np
                import scipy
                import topokit
                from topokit.exceptions import ResourceLimitError
                resource_limit_type = ResourceLimitError
                report["source_hash"] = source_hash(Path(topokit.__file__).resolve().parent)
                report["environment"] = {"python": sys.version, "platform": platform.platform(),
                    "numpy": np.__version__, "scipy": scipy.__version__,
                    "topokit_source": str(Path(topokit.__file__).resolve()),
                    "threads": {name: os.environ.get(name) for name in
                        ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}}
                report["environment"]["runtime_threadpools"] = runtime_threadpools()
            with checkpoint.stage("generation"):
                clouds = generate_inputs(case)
                report["input_summary"] = {"points_per_cloud": [len(cloud) for cloud in clouds],
                    "ambient_dimension": 3, "coordinate_units": "unit_cube",
                    "coordinate_rule": "default_rng(seed).random((n,3)); independent sequential draws per factor",
                    "weight_rule": "arange(n)" if case["weights"] == "distinct" else "ones(n)",
                    "weight_role": "direction_only" if case["family"] == "hyperdigraph" else "retained_attributes_only"}
                report["input_hash"] = input_hash(clouds, None, None)
                report["input_hash_scope"] = "generated clouds before support preparation"
            with checkpoint.stage("support_preparation"):
                bonds, overlap = prepare_support(case, clouds)
                report["input_summary"].update({"supplied_bond_count": None if bonds is None else len(bonds),
                    "overlap_count": None if overlap is None else len(overlap),
                    "support_rule": ("all unordered stable-ID pairs" if bonds is not None else
                        "Delaunay support" if case["family"] == "hyperdigraph" else case["construction"]),
                    "direction_rule": ("strict weight order gives a DAG" if case["weights"] == "distinct"
                        else "equal weights retain both directions") if case["family"] == "hyperdigraph" else None,
                    "sequence_rule": "distinct vertices, consecutive allowed edges" if case["family"] == "hyperdigraph" else None})
                report["input_hash"] = input_hash(clouds, bonds, overlap)
                report["input_hash_scope"] = "generated clouds and explicit support/overlap"
            with checkpoint.stage("construction"):
                topology = build_topology(case, clouds, bonds, overlap)
                report["construction_summary"] = construction_summary(topology, case["degree"])
                report["diagnostics"] = {"construction": compact_diagnostics(topology.metadata)}
                report["environment"]["runtime_threadpools"] = runtime_threadpools()
            with checkpoint.stage("observation_selection"):
                construction, observations = select_observations(topology, case["degree"])
                report["construction_summary"] = construction
                report["observations"] = observations
            with checkpoint.stage("analysis"):
                result = analyze(case, topology, observations)
            with checkpoint.stage("summarization"):
                report["environment"]["runtime_threadpools"] = runtime_threadpools()
                report["result_summary"] = summarize_result(case, result)
                report["diagnostics"]["analysis"] = compact_diagnostics(result.metadata)
            with checkpoint.stage("export"):
                report["result_summary"]["artifact"] = export_result(case, result, checkpoint.output)
            report["status"] = "success"
            report["stage"] = "complete"
        except Exception as error:
            report["status"] = ("resource_limit" if isinstance(error, resource_limit_type)
                                else "memory_error" if isinstance(error, MemoryError)
                                else "numerical_failure" if isinstance(error, ScientificCheckError) else "error")
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
        # Preserve invalid-input failures in the same checkpoint format.
        input_error = error
        case = {"case_id": arguments.case.stem}
    report = run_case(case, arguments.output, input_error=input_error)
    print(json.dumps({"case_id": report["case_id"], "status": report["status"],
                      "stage": report["stage"], "output": str(arguments.output.resolve())}), flush=True)
    return 1 if report["status"] in {"error", "numerical_failure"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
