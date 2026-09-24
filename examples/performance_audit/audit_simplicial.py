"""Read-only performance-audit prototypes; no production modules are modified.

Run from the checkout: python examples/performance_audit/audit_simplicial.py
Results are local observations, not general benchmark claims.
"""
from __future__ import annotations

import json
import inspect
import math
import os
import platform
import statistics
import sys
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from topokit.builders._simplicial import graph_complex, _expand_flag, alpha_complex, GeometryError
from topokit.core._simplicial.algebra import boundary_rank
from topokit.core._simplicial.complex import facets
from topokit.core._simplicial.homology import homology
from topokit.core._simplicial.persistence import persistent_homology


def measured(fn, repeats=5):
    fn()
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        result = fn()
        samples.append(perf_counter() - start)
    return result, {"median_seconds": statistics.median(samples), "samples_seconds": samples}


def compare(name, baseline, candidate, equivalence, repeats=5):
    before, after = baseline(), candidate()
    equivalence(before, after)
    samples = [[], []]
    functions = [baseline, candidate]
    for repeat in range(repeats):
        for index in ((0, 1) if repeat % 2 == 0 else (1, 0)):
            start = perf_counter()
            result = functions[index]()
            samples[index].append(perf_counter() - start)
    t_before, t_after = ({"median_seconds": statistics.median(x), "samples_seconds": x} for x in samples)
    record = {"name": name, "baseline": t_before, "candidate": t_after,
              "speedup": t_before["median_seconds"] / t_after["median_seconds"],
              "equivalence": "passed"}
    print(json.dumps(record), flush=True)
    return record


def expand_trusted(tree, max_dimension):
    """Same traversal/arithmetic as _expand_flag; internal canonical keys only."""
    neighbors = {s[0]: set() for s in tree.simplices(0)}
    for u, v in tree.simplices(1):
        neighbors[u].add(v)
    nodes = tree._nodes

    def visit(prefix, candidates, value):
        if len(prefix) >= max_dimension + 1:
            return
        for vertex in sorted(candidates):
            simplex = prefix + (vertex,)
            birth = max(value, *(nodes[(u, vertex)].value for u in prefix))
            if len(simplex) > 2:
                tree._store(simplex, birth)
            visit(simplex, candidates.intersection(neighbors[vertex]), birth)

    if max_dimension > 1:
        for vertex in neighbors:
            visit((vertex,), neighbors[vertex], nodes[(vertex,)].value)
    return tree


def validate_trusted(tree):
    """Retains every face-closure and monotonicity check, avoids revalidation."""
    for simplex, value in tree.get_filtration():
        for face in facets(simplex):
            if face not in tree._nodes or tree._nodes[face].value > value:
                raise ValueError(f"Invalid face closure or filtration at {simplex}")
    return True


def rank_one_union_find(tree):
    """Rank of signed incidence d1 over ANY field; no numerical tolerance."""
    parent = {s[0]: s[0] for s in tree.simplices(0)}
    sizes = {v: 1 for v in parent}
    rank = 0

    def root(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    for u, v in tree.simplices(1):
        a, b = root(u), root(v)
        if a != b:
            if sizes[a] < sizes[b]:
                a, b = b, a
            parent[b] = a
            sizes[a] += sizes[b]
            rank += 1
    return rank


def assert_equal(a, b):
    assert a == b


def same_tree(a, b):
    assert a.get_filtration() == b.get_filtration()
    for field in (2, 3, 5):
        assert homology(a, max_dimension=2, field=field) == homology(b, max_dimension=2, field=field)
        assert persistent_homology(a, max_dimension=2, field=field, include_zero=True) == persistent_homology(b, max_dimension=2, field=field, include_zero=True)



def batched_alpha_births(geometry, coords, rank, tol, chunk_size=4096):
    """Same scalar centers, radii, tolerances and coface order; batch KD calls.

    Chunking bounds extra centers storage; all q+1 births still propagate to q.
    No LSQ batching, approximate neighbors, modified cutoff, or simplex pruning.
    """
    births = {s: math.inf for s in geometry}
    spatial = cKDTree(coords) if rank else None
    ordered = sorted(births, key=lambda s: (-len(s), s))
    for start in range(0, len(ordered), chunk_size):
        chunk = ordered[start:start + chunk_size]
        radii, centers, query_ids = {}, [], []
        for simplex in chunk:
            if len(simplex) == 1:
                continue
            vertices = coords[list(simplex)]
            edges = vertices[1:] - vertices[0]
            if len(simplex) == 2:
                if not np.any(edges[0]):
                    raise GeometryError("Numerically singular Delaunay simplex")
                offset = edges[0] * 0.5
            else:
                offset, _, simplex_rank, _ = np.linalg.lstsq(2 * edges, np.sum(edges * edges, axis=1), rcond=None)
                if simplex_rank != len(simplex) - 1:
                    raise GeometryError("Numerically singular Delaunay simplex")
            centers.append(vertices[0] + offset)
            radii[simplex] = float(offset @ offset)
            query_ids.append(simplex)
        nearest = (dict(zip(query_ids, (float(d) ** 2 for d in spatial.query(np.asarray(centers), eps=0, workers=1)[0])))
                   if centers else {})
        for simplex in chunk:
            if len(simplex) == 1:
                value = 0.0
            else:
                radius2 = radii[simplex]
                gabriel = nearest[simplex] >= radius2 - tol * max(1.0, radius2)
                value = min(births[simplex], radius2) if gabriel else births[simplex]
                if not np.isfinite(value):
                    raise GeometryError("Nonempty maximal circumsphere; geometry is numerically unreliable")
            births[simplex] = value
            for face in facets(simplex):
                births[face] = min(births[face], value)
    return births


def make_alpha_batched():
    """Read and wrap checked-in implementation; replace only birth-query block.

    Preserves all original preprocessing, affine projection, Qhull call,
    duplicate handling, closure creation, cutoff, and final validation.
    This is an auditable local experiment, not a production backend.
    """
    source = inspect.getsource(alpha_complex)
    start = source.index("    alpha_births = {s: math.inf for s in delaunay_complex}")
    end = source.index("    for simplex in sorted(alpha_births, key=lambda s: (len(s), s)):", start)
    source = (source[:start]
              + "    alpha_births = batched_alpha_births(delaunay_complex, affine_coordinates, affine_dimension, tol)\n\n"
              + source[end:])
    source = source.replace("def alpha_complex(", "def alpha_batched(", 1)
    namespace = dict(alpha_complex.__globals__)
    namespace["batched_alpha_births"] = batched_alpha_births
    exec(compile(source, "<audit_chunked_alpha>", "exec"), namespace)
    return namespace["alpha_batched"]


def main():
    records = []
    rng = np.random.default_rng(20260905)
    edges = [(u, v, float(rng.integers(-2, 8))) for u, v in combinations(range(27), 2)]
    vertices = {i: float(rng.integers(-3, 3)) for i in range(27)}
    graph = lambda: graph_complex(edges, vertices=vertices, max_dimension=1)
    records.append(compare("complete_27_vertices_flag_through_dimension_3_including_graph_setup",
                           lambda: _expand_flag(graph(), 3), lambda: expand_trusted(graph(), 3), same_tree))
    tree = expand_trusted(graph(), 3)
    records[-1]["num_simplices"] = len(tree)
    records.append(compare("validate_same_27_vertex_flag_all_checks_retained", tree.validate,
                           lambda: validate_trusted(tree), assert_equal))
    # Check failure preservation on a deliberately corrupt private fixture.
    edge = next(s for s in tree.simplices(1))
    old = tree._nodes[edge].value
    tree._nodes[edge].value = -100
    tree._ordered = None
    for fn in (tree.validate, lambda: validate_trusted(tree)):
        try:
            fn()
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid birth was not rejected")
    tree._nodes[edge].value = old
    tree._ordered = None

    for n in (2000, 10000):
        path = graph_complex(((i, i + 1) for i in range(n - 1)), vertices=range(n), max_dimension=1)
        records.append(compare(f"boundary_rank_d1_path_{n}_vertices_GF2",
                               lambda: boundary_rank(path, 1, 2), lambda: rank_one_union_find(path), assert_equal))
        assert rank_one_union_find(path) == boundary_rank(path, 1, 3)
        records[-1]["rank"] = n - 1

    points = rng.normal(size=(1000, 3))
    spatial = cKDTree(points)
    # The query kernel is isolated intentionally: no claim of end-to-end alpha speedup.
    queries = rng.normal(size=(12000, 3))
    records.append(compare("alpha_nearest_query_kernel_12000_centers_1000_points_only",
                           lambda: np.asarray([float(spatial.query(c)[0]) ** 2 for c in queries]),
                           lambda: np.asarray([float(d) ** 2 for d in spatial.query(queries, eps=0, workers=1)[0]]),
                           lambda a, b: np.testing.assert_array_equal(a, b)))

    alpha_batched = make_alpha_batched()
    alpha_points = rng.normal(size=(500, 3))
    records.append(compare("alpha_500_points_3d_full_builder_chunked_exact_queries",
                           lambda: alpha_complex(alpha_points, max_dimension=3),
                           lambda: alpha_batched(alpha_points, max_dimension=3),
                           lambda a, b: assert_equal((a.get_filtration(), a.metadata), (b.get_filtration(), b.metadata))))
    # Exact alpha comparisons include rank reduction, duplicate merge, ties,
    # coface-derived edge births, construction cutoff and skeleton projection.
    alpha_checks = 0
    alpha_cutoff_checks = 0
    cases = [np.zeros((0, 3)), np.zeros((1, 3)), np.asarray([[0., 0.], [1., 0.], [2., 0.]]),
             np.asarray([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]),
             np.asarray([[0., 0.], [2., 0.], [1., .1]])]
    for eps in (-1e-13, 0., 1e-13):
        cases.append(np.asarray([[0., 0.], [1., 0.], [1., 1. + eps], [0., 1.]]))
    cases.append(np.asarray([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]]))
    cases += [rng.normal(size=(20, dim)) for dim in (2, 3, 4) for _ in range(4)]
    cases += [np.asarray([[0., 0.], [1., 0.], [0., 0.]])]
    for case in cases:
        for q in (0, 1, 3):
            for cutoff in (math.inf, .2):
                kwargs = {"max_dimension": q, "max_scale": cutoff, "duplicates": "merge"}
                a, b = alpha_complex(case, **kwargs), alpha_batched(case, **kwargs)
                assert a.get_filtration() == b.get_filtration()
                assert a.metadata == b.metadata
                alpha_checks += 1
    for case in cases:
        full = alpha_complex(case, max_dimension=3, duplicates="merge")
        finite_births = sorted({value for _, value in full.get_filtration() if value > 0})
        for birth in finite_births[:3]:
            for cutoff in (np.nextafter(birth, 0.), birth, np.nextafter(birth, math.inf)):
                kwargs = {"max_dimension": 3, "max_scale": float(cutoff), "duplicates": "merge"}
                a, b = alpha_complex(case, **kwargs), alpha_batched(case, **kwargs)
                assert a.get_filtration() == b.get_filtration()
                assert a.metadata == b.metadata
                alpha_cutoff_checks += 1
    for case, kwargs in ((np.asarray([[0., 0.], [0., 0.]]), {}),
                         (np.asarray([[0., 0.], [1., 0.], [0., 1.]]), {"max_simplices": 2})):
        failures = []
        for fn in (alpha_complex, alpha_batched):
            try:
                fn(case, **kwargs)
            except Exception as exc:
                failures.append((type(exc), str(exc)))
        assert len(failures) == 2 and failures[0] == failures[1]
    records[-1]["additional_exact_alpha_cases"] = alpha_checks
    records[-1]["exact_and_nextafter_cutoff_checks"] = alpha_cutoff_checks
    records[-1]["num_simplices"] = len(alpha_complex(alpha_points, max_dimension=3))

    # Regression checks beyond the timed complete graph: arbitrary labels,
    # ties, disconnected components, negative vertex births, empty graph.
    for case in range(20):
        labels = [-11, -2, 3, 10, 18, 40, 61, 72]
        v = {i: float(rng.integers(-3, 3)) for i in labels}
        e = [(u, w, float(rng.integers(-3, 7))) for u, w in combinations(labels, 2) if rng.random() < .5]
        a = graph_complex(e, vertices=v, max_dimension=1)
        b = graph_complex(e, vertices=v, max_dimension=1)
        same_tree(_expand_flag(a, 4), expand_trusted(b, 4))
        for field in (2, 3, 5):
            assert boundary_rank(a, 1, field) == rank_one_union_find(a)
    same_tree(_expand_flag(graph_complex(), 3), expand_trusted(graph_complex(), 3))

    payload = {"environment": {"python": sys.version, "executable": sys.executable,
                               "platform": platform.platform(), "numpy": np.__version__, "scipy": scipy.__version__,
                               "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
                               "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS")},
               "seed": 20260905,
               "notes": ["Five interleaved timed repeats with alternating order after warmup/equivalence; shared desktop environment, no concurrent subprocesses launched by this script.",
                         "Prototype only; production source unmodified.",
                         "Flag check compares full exact filtration, GF(2/3/5) homology and interval provenance, including zero bars.",
                         "Nearest-query result is kernel-only, uses scalar float squaring to preserve baseline arithmetic; centers need batching inside full alpha algorithm.",
                         "20 extra seeded arbitrary-label flag cases and deliberate validation failure checked."],
               "results": records}
    output = Path(__file__).with_name("simplicial_benchmarks.json")
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
