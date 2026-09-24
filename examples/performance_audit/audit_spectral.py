"""Bounded, reproducible performance probes; never changes installed kernels.

Run from the checkout: python examples/performance_audit/audit_spectral.py
Temporary monkeypatches below only demonstrate call-scoped reuse, and are
restored even on failure. They are not production cache implementations.
"""
from __future__ import annotations

import cProfile
from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
import platform
import pstats
import statistics
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np
import scipy
from scipy.sparse.linalg import LinearOperator
from topokit import builders, core
from topokit.builders import interaction as interaction_builder
from topokit.core import interaction as interaction_api
from topokit.core._interaction.laplacian import InteractionLaplacianEngine
from topokit.core._simplicial.algebra import boundary_matrix
from topokit.core._simplicial.complex import SimplexTree
from topokit.core._simplicial.laplacian import laplacian_matrix
from topokit.core._spectral import solve_symmetric
from topokit.exceptions import ResourceLimitError
from topokit.workflows import analyze, laplacian_series
from examples.data_fixture import load_demo_cloud
from examples.demo import DEFAULT_SCALES


def measure(fn, repeat=5):
    fn()
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        value = fn()
        times.append(time.perf_counter() - start)
    return value, {"median_seconds": statistics.median(times), "runs_seconds": times}


def diagonal_probe(n):
    factors = {(i,): 0.0 for i in range(n)}
    factors.update({(i, i + 1): 1.0 for i in range(n - 1)})
    obj = interaction_builder.from_complexes(factors, factors, max_dimension=0)
    operator = InteractionLaplacianEngine(obj.native).laplacian(0, filtration=1.)
    baseline = lambda: solve_symmetric(operator.as_linear_operator()).values
    candidate = lambda: np.sort(operator._diagonal)
    a, ta = measure(baseline, 3)
    b, tb = measure(candidate)
    np.testing.assert_array_equal(a, b)
    # Independent incidence count for a pair of identical path factors.
    expected = np.full(n, 4.0)
    expected[[0, -1]] = 2.0
    np.testing.assert_array_equal(b, np.sort(expected))
    public = core.laplacian(obj, dimension=0, scale=1.)
    np.testing.assert_array_equal(public.eigenvalues, b)
    return {"n": n, "generic_solver": ta, "sort_diagonal": tb,
            "kernel_speedup": ta["median_seconds"] / tb["median_seconds"],
            "bitwise_equal": True, "one_dense_matrix_bytes": 8*n*n,
            "diagonal_bytes": operator._diagonal.nbytes}


def diagonal_limit_probe():
    n = 2100
    factors = {(i,): 0.0 for i in range(n)}
    obj = interaction_builder.from_complexes(factors, factors, max_dimension=0)
    error = None
    try:
        core.laplacian(obj, dimension=0, scale=0.)
    except ResourceLimitError as caught:
        error = str(caught)
    assert error is not None
    operator = InteractionLaplacianEngine(obj.native).laplacian(0, filtration=0.)
    values = np.sort(operator._diagonal)
    assert len(values) == n and np.count_nonzero(values) == 0
    return {"n": n, "public_error": error,
            "exact_full_spectrum_count": len(values), "spectrum_bytes": values.nbytes}


def sparse_product_probe(n=700):
    tree = SimplexTree([(0, i) for i in range(1, n + 1)])
    explicit, assembly = measure(lambda: laplacian_matrix(tree, 1))
    down = boundary_matrix(tree, 1).astype(float)
    up = boundary_matrix(tree, 2).astype(float)
    operator = LinearOperator((n, n), dtype=float,
        matvec=lambda x: down.T @ (down @ x) + up @ (up.T @ x))
    rng = np.random.default_rng(20260905)
    values = rng.standard_normal((n, 8))
    error = float(np.max(np.abs(explicit @ values - operator @ values)))
    np.testing.assert_allclose(explicit @ values, operator @ values, rtol=1e-12, atol=1e-10)
    storage = lambda x: x.data.nbytes + x.indices.nbytes + x.indptr.nbytes
    return {"star_edges": n, "laplacian_nnz": explicit.nnz,
            "boundaries_nnz": down.nnz + up.nnz,
            "laplacian_csr_bytes": storage(explicit),
            "boundaries_csr_bytes": storage(down) + storage(up),
            "assembly": assembly, "max_matvec_absolute_difference": error,
            "scope": "same real operator, floating-point summation order differs; no eigensolver speedup claimed"}


@contextmanager
def reusable_engine():
    real = InteractionLaplacianEngine
    cache = {}
    def factory(chain, **kwargs):
        key = (id(chain), tuple(sorted(kwargs.items())))
        if key not in cache:
            cache[key] = real(chain, **kwargs)
        return cache[key]
    with patch.object(interaction_api._native, "InteractionLaplacianEngine", factory):
        yield


def reuse_probe():
    cloud = load_demo_cloud()
    subset = cloud.subset(cloud.metadata["subset_ids"])
    obj = builders.from_points(subset, kind="interaction", cloud_b=cloud,
        overlap_pairs=[(label, label) for label in subset.ids], max_dimension=2)
    scales = tuple(float(x) for x in np.linspace(0, 5, 9))
    def baseline():
        return laplacian_series(obj, max_dimension=2, scales=scales)
    def candidate():
        with reusable_engine():
            return baseline()
    a, ta = measure(baseline, 3)
    b, tb = measure(candidate, 3)
    for scale in scales:
        for q in range(3):
            np.testing.assert_array_equal(a[scale][q].eigenvalues, b[scale][q].eigenvalues)
            assert a[scale][q].basis == b[scale][q].basis
            assert a[scale][q].nullity == b[scale][q].nullity
            assert a[scale][q].metadata == b[scale][q].metadata
    return {"factor_points": [len(subset), len(cloud)], "scales": len(scales), "degrees": 3,
            "baseline": ta, "call_scoped_engine": tb,
            "speedup": ta["median_seconds"] / tb["median_seconds"],
            "eigenvalues_bitwise_equal": True, "basis_nullity_metadata_equal": True}


def profile_routes():
    cloud = load_demo_cloud()
    answer = {}
    for route in ("simplicial", "hyperdigraph", "interaction"):
        def build():
            if route == "interaction":
                subset = cloud.subset(cloud.metadata["subset_ids"])
                return builders.from_points(subset, kind=route, cloud_b=cloud,
                    overlap_pairs=[(label, label) for label in subset.ids], max_dimension=2)
            return builders.from_points(cloud, kind=route, max_dimension=2)
        obj, build_time = measure(build, 3)
        result, analysis_time = measure(lambda: analyze(obj, max_dimension=2, scales=DEFAULT_SCALES[route]), 3)
        profiler = cProfile.Profile()
        profiler.runcall(analyze, obj, max_dimension=2, scales=DEFAULT_SCALES[route])
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(35)
        (Path(__file__).parent / f"spectral_profile_{route}.txt").write_text(stream.getvalue())
        answer[route] = {"build": build_time, "analysis": analysis_time,
            "interval_count": len(result.persistence.intervals),
            "matrix_sizes": {str(s): [len(v.basis) for v in snap["laplacians"].values()]
                             for s, snap in result.snapshots.items()}}
    return answer


def barcode_query_probe():
    cloud = load_demo_cloud()
    records = {}
    for route in ("simplicial", "hyperdigraph", "interaction"):
        if route == "interaction":
            subset = cloud.subset(cloud.metadata["subset_ids"])
            obj = builders.from_points(subset, kind=route, cloud_b=cloud,
                overlap_pairs=[(label, label) for label in subset.ids], max_dimension=2)
        else:
            obj = builders.from_points(cloud, kind=route, max_dimension=2)
        scales = DEFAULT_SCALES[route]
        def baseline():
            bars = core.persistence(obj, max_dimension=2)
            return bars, [core.homology(obj, max_dimension=2, scale=t).betti_numbers for t in scales]
        def candidate():
            bars = core.persistence(obj, max_dimension=2)
            return bars, [bars.betti_at(t) for t in scales]
        a, ta = measure(baseline, 3)
        b, tb = measure(candidate, 3)
        assert a[0].intervals == b[0].intervals and a[1] == b[1]
        records[route] = {"scales": scales, "baseline": ta, "barcode_queries": tb,
            "speedup": ta["median_seconds"] / tb["median_seconds"], "betti_exact_equal": True,
            "scope": "barcodes plus Betti tuples only; not a substitute for chain/basis diagnostics or representatives"}
    return records


if __name__ == "__main__":
    report = {"date": "2026-09-05", "python": platform.python_version(),
              "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform()}
    try:
        from threadpoolctl import threadpool_info
        report["threadpools"] = threadpool_info()
    except ImportError:
        report["threadpools"] = "threadpoolctl unavailable"
    report["source_sha256"] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((ROOT / "src").rglob("*.py"))}
    report["diagonal"] = [diagonal_probe(n) for n in (400, 1000)]
    report["diagonal_resource_limit"] = diagonal_limit_probe()
    report["matrix_free_simplicial"] = sparse_product_probe()
    report["engine_reuse"] = reuse_probe()
    report["demo_profiles"] = profile_routes()
    report["barcode_queries"] = barcode_query_probe()
    path = Path(__file__).with_name("spectral_results.json")
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "source_sha256"}, indent=2))
