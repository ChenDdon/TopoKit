"""Reproducible structural and numerical checks for a naming-only refactor.

This is bounded regression evidence, not a proof of general scientific accuracy.
Both revisions run in separate Python processes using the same interpreter and
numerical libraries. No package is imported in the controller process.

Example:
    python examples/pressure_test/check_readability.py \
        --baseline-src /path/to/baseline/src --current-src ./src \
        --output examples/output/readability/report.json

The report links complete gzip-compressed JSON payloads for both revisions.
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from itertools import combinations

ARRAY_ATOL = 1e-10
ARRAY_RTOL = 1e-9
IGNORED_METADATA_KEYS = frozenset({
    "geometry_versions", "elapsed_seconds", "duration_seconds", "wall_seconds", "timing_seconds",
})


def json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def source_hashes(source_root):
    source_root = Path(source_root)
    return {str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(source_root.rglob("*.py"))}


def strip_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.body and isinstance(node.body[0], ast.Expr):
                value = node.body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    node.body.pop(0)
    return tree


@dataclasses.dataclass
class Scope:
    number: int
    kind: str
    parent: object = None
    bound: set = dataclasses.field(default_factory=set)
    protected: set = dataclasses.field(default_factory=set)
    globals: set = dataclasses.field(default_factory=set)
    nonlocals: set = dataclasses.field(default_factory=set)


class ScopeIndex(ast.NodeVisitor):
    """Resolve local bindings, including closures and comprehension scopes."""

    def __init__(self, tree):
        self.scopes = [Scope(0, "module")]
        self.current = self.scopes[0]
        self.owners = {}
        self.visit(tree)

    def enter(self, kind):
        parent = self.current
        self.current = Scope(len(self.scopes), kind, parent)
        self.scopes.append(self.current)
        return parent

    def visit_Name(self, node):
        self.owners[id(node)] = self.current
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.current.bound.add(node.id)

    def protect(self, name):
        self.current.bound.add(name)
        self.current.protected.add(name)

    def visit_Import(self, node):
        for alias in node.names:
            self.protect(alias.asname or alias.name.split(".")[0])

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.protect(alias.asname or alias.name)

    def visit_Global(self, node):
        self.current.globals.update(node.names)

    def visit_Nonlocal(self, node):
        self.current.nonlocals.update(node.names)

    def visit_ExceptHandler(self, node):
        if node.name:
            self.protect(node.name)
        self.generic_visit(node)

    def arguments_outside(self, arguments):
        for default in [*arguments.defaults, *arguments.kw_defaults]:
            if default is not None:
                self.visit(default)
        for argument in [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs,
                         arguments.vararg, arguments.kwarg]:
            if argument is not None and argument.annotation is not None:
                self.visit(argument.annotation)

    def arguments_inside(self, arguments):
        for argument in [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs,
                         arguments.vararg, arguments.kwarg]:
            if argument is not None:
                self.protect(argument.arg)

    def visit_FunctionDef(self, node):
        self.protect(node.name)
        for decorator in node.decorator_list:
            self.visit(decorator)
        self.arguments_outside(node.args)
        if node.returns is not None:
            self.visit(node.returns)
        parent = self.enter("function")
        self.arguments_inside(node.args)
        for statement in node.body:
            self.visit(statement)
        self.current = parent

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        self.arguments_outside(node.args)
        parent = self.enter("lambda")
        self.arguments_inside(node.args)
        self.visit(node.body)
        self.current = parent

    def visit_ClassDef(self, node):
        self.protect(node.name)
        for expression in [*node.decorator_list, *node.bases, *node.keywords]:
            self.visit(expression)
        parent = self.enter("class")
        for statement in node.body:
            self.visit(statement)
        self.current = parent

    def visit_comprehension_expression(self, node):
        # Python evaluates the first iterable in the surrounding scope.
        self.visit(node.generators[0].iter)
        parent = self.enter("comprehension")
        for index, generator in enumerate(node.generators):
            if index:
                self.visit(generator.iter)
            self.visit(generator.target)
            for condition in generator.ifs:
                self.visit(condition)
        if isinstance(node, ast.DictComp):
            self.visit(node.key)
            self.visit(node.value)
        else:
            self.visit(node.elt)
        self.current = parent

    visit_ListComp = visit_comprehension_expression
    visit_SetComp = visit_comprehension_expression
    visit_DictComp = visit_comprehension_expression
    visit_GeneratorExp = visit_comprehension_expression

    def visit_NamedExpr(self, node):
        self.visit(node.value)
        owner = self.current
        while owner.kind == "comprehension":
            owner = owner.parent
        if isinstance(node.target, ast.Name):
            self.owners[id(node.target)] = owner
            owner.bound.add(node.target.id)
        else:
            self.visit(node.target)

    def binding(self, node):
        name = node.id
        scope = self.owners[id(node)]
        if name in scope.globals:
            return self.scopes[0]
        skip_local = name in scope.nonlocals
        first = True
        while scope is not None:
            if not (first and skip_local) and name in scope.bound:
                return scope
            first = False
            scope = scope.parent
            # Enclosing class namespaces are not lexical closures for methods.
            while scope is not None and scope.kind == "class":
                scope = scope.parent
        return self.scopes[0]


def structural_file(before_text, after_text):
    before = strip_docstrings(ast.parse(before_text))
    after = strip_docstrings(ast.parse(after_text))
    before_names = [node for node in ast.walk(before) if isinstance(node, ast.Name)]
    after_names = [node for node in ast.walk(after) if isinstance(node, ast.Name)]
    # All other AST fields remain exact: args, attributes, strings, keys and order.
    before_ids, after_ids = [node.id for node in before_names], [node.id for node in after_names]
    for node in before_names + after_names:
        node.id = "__identifier__"
    same_shape = ast.dump(before, include_attributes=False) == ast.dump(after, include_attributes=False)
    for node, name in zip(before_names, before_ids):
        node.id = name
    for node, name in zip(after_names, after_ids):
        node.id = name
    if not same_shape:
        return {"passed": False, "reason": "Non-Name AST change (other than docstrings/comments)."}
    before_scope, after_scope = ScopeIndex(before), ScopeIndex(after)
    forward, inverse = {}, {}
    changed = 0
    changes = set()
    for old, new in zip(before_names, after_names):
        old_binding, new_binding = before_scope.binding(old), after_scope.binding(new)
        if old_binding.number != new_binding.number:
            return {"passed": False, "reason": f"Binding scope changed: {old.id} -> {new.id}."}
        key, reverse_key = (old_binding.number, old.id), (new_binding.number, new.id)
        if key in forward and forward[key] != new.id:
            return {"passed": False, "reason": f"Inconsistent rename of {old.id} in scope {old_binding.number}."}
        if reverse_key in inverse and inverse[reverse_key] != old.id:
            return {"passed": False, "reason": f"Multiple names merged into {new.id} in scope {new_binding.number}."}
        forward[key], inverse[reverse_key] = new.id, old.id
        if old.id != new.id:
            if (old_binding.kind in {"module", "class"} or old.id in old_binding.protected
                    or new.id in new_binding.protected):
                return {"passed": False, "reason": f"Protected binding changed: {old.id} -> {new.id}."}
            changed += 1
            changes.add((old_binding.number, old.id, new.id))
    return {"passed": True, "changed_name_occurrences": changed,
            "renames": [{"scope": scope, "before": old, "after": new}
                        for scope, old, new in sorted(changes)]}


def structural_audit(baseline_src, current_src):
    before_hashes, after_hashes = source_hashes(baseline_src), source_hashes(current_src)
    before_files, after_files = set(before_hashes), set(after_hashes)
    records = []
    for relative in sorted(before_files & after_files):
        result = structural_file((Path(baseline_src) / relative).read_text(),
                                 (Path(current_src) / relative).read_text())
        records.append({"file": relative, "baseline_sha256": before_hashes[relative],
                        "current_sha256": after_hashes[relative], **result})
    added, removed = sorted(after_files - before_files), sorted(before_files - after_files)
    return {"passed": not added and not removed and all(record["passed"] for record in records),
            "files_checked": len(records),
            "files_changed": sum(record["baseline_sha256"] != record["current_sha256"] for record in records),
            "identifier_occurrences_changed": sum(record.get("changed_name_occurrences", 0) for record in records),
            "added_files": added, "removed_files": removed, "files": records,
            "scope": "Python source only; comments/docstrings ignored; all other AST fields exact except consistently renamed local Name.id bindings."}


def encode(value):
    """Portable numerical payload with byte hashes and complete array values."""
    import numpy as np
    from scipy import sparse

    if isinstance(value, np.generic):
        return encode(value.item())
    if sparse.issparse(value):
        # Cases are deliberately small; compare the operator, independent of CSR layout.
        return encode(value.toarray())
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        return {"__array__": True, "dtype": array.dtype.str, "shape": list(array.shape),
                "sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(), "values": encode(array.tolist())}
    if isinstance(value, float) and not math.isfinite(value):
        return {"__float__": "nan" if math.isnan(value) else "inf" if value > 0 else "-inf"}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if dataclasses.is_dataclass(value):
        return {"__dataclass__": type(value).__name__,
                "fields": {field.name: encode(getattr(value, field.name)) for field in dataclasses.fields(value)}}
    if isinstance(value, tuple):
        return {"__tuple__": [encode(item) for item in value]}
    if isinstance(value, list):
        return [encode(item) for item in value]
    if isinstance(value, dict):
        items = [[encode(key), encode(item)] for key, item in value.items()
                 if not isinstance(key, str) or key not in IGNORED_METADATA_KEYS]
        return {"__mapping__": sorted(items, key=lambda item: json_bytes(item[0]))}
    raise TypeError(f"Unsupported regression payload value: {type(value).__name__}")


def decode_numbers(value):
    if isinstance(value, list):
        return [decode_numbers(item) for item in value]
    if isinstance(value, dict) and "__float__" in value:
        return float(value["__float__"])
    return value


def compare_payload(before, after, *, allow_array_tolerance=False):
    """Exact records by default; optionally report tolerant numerical arrays."""
    import numpy as np

    mismatches, array_checks = [], []

    def visit(left, right, path):
        if type(left) is not type(right):
            mismatches.append({"path": path, "reason": "different value types"})
            return
        if isinstance(left, dict) and left.get("__array__") is True:
            compatible = left["shape"] == right.get("shape") and left["dtype"] == right.get("dtype")
            bitwise = compatible and left["sha256"] == right.get("sha256")
            close, maximum_error, valid_hashes = False, None, False
            if compatible:
                a = np.asarray(decode_numbers(left["values"]), dtype=left["dtype"]).reshape(left["shape"])
                b = np.asarray(decode_numbers(right["values"]), dtype=right["dtype"]).reshape(right["shape"])
                valid_hashes = (hashlib.sha256(a.tobytes()).hexdigest() == left["sha256"]
                                and hashlib.sha256(b.tobytes()).hexdigest() == right["sha256"])
                bitwise = bitwise and valid_hashes
                close = bool(np.allclose(a, b, rtol=ARRAY_RTOL, atol=ARRAY_ATOL, equal_nan=True))
                finite = np.isfinite(a) & np.isfinite(b)
                maximum_error = float(np.max(np.abs(a[finite] - b[finite]))) if np.any(finite) else 0.0
            passed = valid_hashes and (bitwise or (allow_array_tolerance and close))
            array_checks.append({"path": path, "passed": passed, "bitwise_equal": bitwise,
                                 "allclose": close, "payload_hashes_valid": valid_hashes,
                                 "max_absolute_error": maximum_error,
                                 "baseline_sha256": left["sha256"], "current_sha256": right.get("sha256"),
                                 "shape": left["shape"], "dtype": left["dtype"]})
            if not passed:
                mismatches.append({"path": path, "reason": "numerical array changed"})
            return
        if isinstance(left, dict):
            if left.keys() != right.keys():
                mismatches.append({"path": path, "reason": "mapping keys changed"})
                return
            for key in left:
                visit(left[key], right[key], path + "/" + key)
        elif isinstance(left, list):
            if len(left) != len(right):
                mismatches.append({"path": path, "reason": "sequence length changed"})
                return
            for index, (a, b) in enumerate(zip(left, right)):
                visit(a, b, path + f"/{index}")
        elif left != right or (isinstance(left, float) and left.hex() != right.hex()):
            mismatches.append({"path": path, "reason": "exact scalar changed", "before": left, "after": right})

    visit(before, after, "")
    return {"passed": not mismatches, "exact_payload_equal": json_bytes(before) == json_bytes(after),
            "baseline_sha256": digest(before), "current_sha256": digest(after),
            "array_tolerance_allowed": allow_array_tolerance, "atol": ARRAY_ATOL, "rtol": ARRAY_RTOL,
            "arrays": array_checks, "mismatches": mismatches}


def cell_record(obj):
    if obj.kind == "simplicial":
        return {"filtration": obj.native.get_filtration()}
    if obj.kind == "hyperdigraph":
        return {"vertices": obj.native.vertices, "weighted_hyperedges": obj.native.weighted_hyperedges()}
    chain = obj.native
    return {"factors": tuple({"simplices": factor.simplices, "births": factor.births,
                               "face_ids": factor.face_ids} for factor in chain.factors),
            "degrees": tuple({"keys": degree.keys, "births": degree.births,
                               "boundaries_GF2": tuple(chain.boundary_rows(q, i) for i in range(len(degree.keys)))}
                              for q, degree in enumerate(chain.degrees))}


def event_values(obj):
    if obj.kind == "simplicial":
        return sorted({birth for _, birth in obj.native.get_filtration()})
    if obj.kind == "hyperdigraph":
        return list(obj.native.thresholds())
    return sorted({birth for degree in obj.native.degrees for birth in degree.births})


def scenarios():
    import numpy as np
    from topokit.data import PointCloud
    from topokit.builders import simplicial, interaction, hyperdigraph

    rng = np.random.default_rng(20260905)
    points = rng.normal(size=(6, 3))
    ids_a = tuple(f"a{i}" for i in range(6))
    ids_b = tuple(f"b{i}" for i in range(6))
    cloud_a = PointCloud(points, ids=ids_a, weights=np.ones(6), metadata={"coordinate_units": "fixture_unit"})
    cloud_b = PointCloud(points + .07 * rng.normal(size=(6, 3)), ids=ids_b,
                         weights=np.ones(6), metadata={"coordinate_units": "fixture_unit"})
    for kind in ("alpha", "rips"):
        yield f"simplicial_{kind}", simplicial.from_points(cloud_a, complex_type=kind, max_dimension=2)
    for kind in ("alpha", "rips"):
        for overlap in (6, 3):
            yield f"interaction_{kind}_{'full' if overlap == 6 else 'half'}_overlap", interaction.from_points(
                cloud_a, cloud_b, max_dimension=2, overlap_vertices=list(zip(ids_a[:overlap], ids_b[:overlap])),
                factor_options={"complex_type": kind})
    for support in ("delaunay", "complete"):
        for weights in ("equal", "distinct"):
            yield f"hyperdigraph_{support}_{weights}", hyperdigraph.from_points(
                cloud_a, max_dimension=2, weights=np.ones(6) if weights == "equal" else np.arange(6, dtype=float),
                bonds=None if support == "delaunay" else list(combinations(ids_a, 2)))


def collect_worker(source_root):
    import numpy as np
    import scipy
    import topokit
    from topokit.core import simplicial, interaction, hyperdigraph

    apis = {"simplicial": simplicial, "interaction": interaction, "hyperdigraph": hyperdigraph}

    imported = Path(topokit.__file__).resolve()
    if not imported.is_relative_to(Path(source_root).resolve()):
        raise RuntimeError(f"Imported {imported}, outside requested source root {source_root}")
    records, internal_checks = {}, []
    for name, obj in scenarios():
        api = apis[obj.kind]
        positive_events = [value for value in event_values(obj) if value > 0 and math.isfinite(value)]
        start = positive_events[len(positive_events) // 2] if positive_events else 0.0
        end = positive_events[-1] if positive_events else start
        record = {"kind": obj.kind, "cells": encode(cell_record(obj)), "metadata": encode(obj.metadata),
                  "cloud": encode(obj.cloud), "scales": [start, end], "homology": {}, "spectra": {}}
        options = {"include_zero": True} if obj.kind == "simplicial" else {"include_diagonal": True}
        bars = api.persistence(obj, max_dimension=2, **options)
        record["persistence"] = encode(bars)
        for stage, scale in (("source", start), ("target", end)):
            result = api.homology(obj, max_dimension=2, scale=scale)
            record["homology"][stage] = encode(result)
            internal_checks.append({"scenario": name, "check": f"barcode_vs_homology_{stage}",
                                    "homology_betti": result.betti_numbers, "barcode_betti": bars.betti_at(scale),
                                    "passed": result.betti_numbers == bars.betti_at(scale)})
            for q in range(3):
                spectrum = api.laplacian(obj, dimension=q, scale=scale, return_matrix=True)
                record["spectra"][f"ordinary_{stage}_q{q}"] = encode(spectrum)
        for q in range(3):
            persistent = api.persistent_laplacian(obj, dimension=q, start=start, end=end, return_matrix=True)
            equal_stage = api.persistent_laplacian(obj, dimension=q, start=start, end=start, return_matrix=True)
            record["spectra"][f"persistent_q{q}"] = encode(persistent)
            record["spectra"][f"persistent_equal_q{q}"] = encode(equal_stage)
            ordinary_fields = record["spectra"][f"ordinary_source_q{q}"]["fields"]
            equal_fields = record["spectra"][f"persistent_equal_q{q}"]["fields"]
            for field in ("matrix", "eigenvalues", "basis"):
                result = compare_payload(ordinary_fields[field], equal_fields[field], allow_array_tolerance=True)
                internal_checks.append({"scenario": name, "check": f"equal_stage_q{q}_{field}", **result})
        records[name] = record
    loaded_modules = {name: str(Path(module.__file__).resolve()) for name, module in sys.modules.items()
                      if (name == "topokit" or name.startswith("topokit.")) and getattr(module, "__file__", None)}
    if any(not Path(path).is_relative_to(Path(source_root).resolve()) for path in loaded_modules.values()):
        raise RuntimeError("A topokit submodule was imported outside the requested source root")
    return {"schema_version": 1, "source_root": str(Path(source_root).resolve()), "imported_package": str(imported),
            "loaded_topokit_modules": loaded_modules,
            "source_hashes": source_hashes(source_root), "environment": {
                "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS")},
            "seed": 20260905, "point_count": 6, "cases": records, "internal_checks": internal_checks}


def write_compressed(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # mtime=0 makes the archive itself reproducible.
    path.write_bytes(gzip.compress(json_bytes(payload), mtime=0))


def read_compressed(path):
    return json.loads(gzip.decompress(Path(path).read_bytes()))


def run_worker(source_root, output, timeout=240):
    environment = dict(os.environ)
    environment.update({"PYTHONPATH": str(Path(source_root).resolve()), "OPENBLAS_NUM_THREADS": "1",
                        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
                        "NUMEXPR_NUM_THREADS": "1", "PYTHONHASHSEED": "0"})
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(Path(source_root).resolve()),
               "--output", str(Path(output).resolve())]
    with tempfile.TemporaryDirectory(prefix="topokit-readability-worker-") as cwd:
        completed = subprocess.run(command, cwd=cwd, env=environment, capture_output=True, text=True, timeout=timeout)
    if completed.returncode:
        raise RuntimeError(f"Worker for {source_root} exited {completed.returncode}:\n{completed.stderr[-12000:]}")
    return read_compressed(output)


def compare_runs(before, after):
    checks = []
    if before["cases"].keys() != after["cases"].keys():
        return [{"check": "scenario_set", "passed": False}]
    for scenario, baseline in before["cases"].items():
        current = after["cases"][scenario]
        for field in ("kind", "cells", "metadata", "cloud", "scales", "persistence", "homology"):
            checks.append({"scenario": scenario, "check": field,
                           **compare_payload(baseline[field], current[field])})
        if baseline["spectra"].keys() != current["spectra"].keys():
            checks.append({"scenario": scenario, "check": "spectrum_set", "passed": False})
            continue
        for spectrum, record in baseline["spectra"].items():
            checks.append({"scenario": scenario, "check": spectrum,
                           **compare_payload(record, current["spectra"][spectrum], allow_array_tolerance=True)})
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-src", type=Path)
    parser.add_argument("--current-src", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker is not None:
        write_compressed(args.output, collect_worker(args.worker))
        return 0
    if args.baseline_src is None or args.current_src is None:
        parser.error("--baseline-src and --current-src are required")
    for root in (args.baseline_src, args.current_src):
        if not (root / "topokit" / "__init__.py").is_file():
            parser.error(f"source root must contain topokit/__init__.py: {root}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    baseline_path = args.output.with_name(args.output.stem + "_baseline.json.gz")
    current_path = args.output.with_name(args.output.stem + "_current.json.gz")
    before_hashes, current_hashes = source_hashes(args.baseline_src), source_hashes(args.current_src)
    try:
        baseline = run_worker(args.baseline_src, baseline_path)
        current = run_worker(args.current_src, current_path)
    except (RuntimeError, subprocess.TimeoutExpired) as error:
        failure = {"schema_version": 1, "passed": False, "failure": str(error),
                   "scope": "Worker execution failed; no numerical equivalence claim is made.",
                   "baseline_source": str(args.baseline_src.resolve()),
                   "current_source": str(args.current_src.resolve())}
        args.output.write_text(json.dumps(failure, indent=2) + "\n")
        print(json.dumps({"passed": False, "report": str(args.output.resolve())}))
        return 1
    structural = structural_audit(args.baseline_src, args.current_src)
    stable = (before_hashes == baseline["source_hashes"] == source_hashes(args.baseline_src)
              and current_hashes == current["source_hashes"] == source_hashes(args.current_src))
    comparisons = compare_runs(baseline, current)
    internal_passed = all(check["passed"] for run in (baseline, current) for check in run["internal_checks"])
    passed = structural["passed"] and stable and internal_passed and all(check["passed"] for check in comparisons)
    report = {"schema_version": 1, "passed": passed, "sources_stable_during_run": stable,
              "scope": "Bounded seeded n=6 regression evidence; not a proof of general accuracy or performance.",
              "scenario_count": len(baseline["cases"]), "comparison_count": len(comparisons),
              "ignored_metadata_keys": sorted(IGNORED_METADATA_KEYS), "array_atol": ARRAY_ATOL, "array_rtol": ARRAY_RTOL,
              "structural_audit": structural, "comparisons": comparisons,
              "internal_checks": {"baseline": baseline["internal_checks"], "current": current["internal_checks"]},
              "payloads": {"baseline": {"path": str(baseline_path.resolve()),
                                        "archive_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
                                        "uncompressed_json_sha256": digest(baseline)},
                           "current": {"path": str(current_path.resolve()),
                                       "archive_sha256": hashlib.sha256(current_path.read_bytes()).hexdigest(),
                                       "uncompressed_json_sha256": digest(current)}}}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": passed, "scenario_count": report["scenario_count"],
                      "comparison_count": len(comparisons), "structural_passed": structural["passed"],
                      "sources_stable_during_run": stable, "report": str(args.output.resolve())}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
