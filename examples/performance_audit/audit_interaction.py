"""Read-only interaction performance audit; prototypes are not library changes.

Run from topokit: PYTHONPATH=src python examples/performance_audit/audit_interaction.py
All comparisons preserve cells, exact GF(2) arithmetic, filtration order and
diagonal intervals. Timings are illustrative local medians, not performance gates.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from itertools import combinations
import json
import os
from pathlib import Path
import platform
import random
from statistics import median
import sys
from time import perf_counter
import tracemalloc
from types import SimpleNamespace

import numpy as np
import scipy

from topokit.data import PointCloud
from topokit.builders.interaction import _overlap
from topokit.core._interaction import SimplicialComplexBuilder, build_interaction_chain_complex
from topokit.core._interaction.persistence import compute_persistence, _reduce_boundary, _ReducedBoundary
from topokit.core.interaction import persistence as public_persistence


def grid_factor(side, seed):
    rng = random.Random(seed)
    builder = SimplicialComplexBuilder()
    for vertex in range(side * side):
        builder.insert((vertex,), 0, with_faces=False)
    for x in range(side - 1):
        for y in range(side - 1):
            a, b = x * side + y, (x + 1) * side + y
            for triangle in ((a, a + 1, b + 1), (a, b, b + 1)):
                builder.insert(triangle, float(rng.randrange(1, 10)))
    return builder.freeze()


def complete_factor(vertices, seed):
    rng = random.Random(seed)
    builder = SimplicialComplexBuilder()
    for vertex in range(vertices):
        builder.insert((vertex,), 0, with_faces=False)
    for simplex in combinations(range(vertices), 4):
        builder.insert(simplex, float(rng.randrange(1, 10)))
    return builder.freeze()


def cleared_reductions(chain, backend="auto"):
    """Descending-degree clearing, with unchanged ordering in each degree."""
    reductions = [None] * len(chain.degrees)
    cleared = set()
    skipped = 0
    for degree in reversed(range(len(chain.degrees))):
        count = len(chain.degrees[degree].keys)
        if degree == 0 or count == 0:
            reductions[degree] = _ReducedBoundary(frozenset(range(count)), {}, "trivial", 0)
            continue
        selected = backend
        if selected == "auto":
            selected = "int" if len(chain.degrees[degree - 1].keys) <= 8192 else "sparse"
        pivots, deaths = {}, {}
        zero = set(cleared)
        additions = 0
        skipped += len(cleared)
        for column_index in range(count):
            if column_index in cleared:
                continue
            if selected == "int":
                column = 0
                for row in chain.boundary_rows(degree, column_index):
                    column ^= 1 << row
                while column:
                    pivot = column.bit_length() - 1
                    previous = pivots.get(pivot)
                    if previous is None:
                        pivots[pivot] = column
                        deaths[pivot] = column_index
                        break
                    column ^= previous
                    additions += 1
            else:
                column = set(chain.boundary_rows(degree, column_index))
                while column:
                    pivot = max(column)
                    previous = pivots.get(pivot)
                    if previous is None:
                        pivots[pivot] = frozenset(column)
                        deaths[pivot] = column_index
                        break
                    column.symmetric_difference_update(previous)
                    additions += 1
            if not column:
                zero.add(column_index)
        reductions[degree] = _ReducedBoundary(frozenset(zero), deaths, selected, additions)
        cleared = set(deaths)
    return tuple(reductions), skipped


def current_reductions(chain, backend="auto"):
    return tuple(_reduce_boundary(chain, q, backend) for q in range(len(chain.degrees)))


def cached_validate(chain):
    """Same GF(2), face admissibility and grade checks, rolling boundary cache."""
    previous = tuple(() for _ in chain.degrees[0].keys)
    for degree in range(1, len(chain.degrees)):
        target = chain.degrees[degree - 1]
        current = []
        for column_index, birth in enumerate(chain.degrees[degree].births):
            rows = chain.boundary_rows(degree, column_index, validate_missing=True)
            # The top-degree boundary is consumed immediately and never reused.
            if degree + 1 < len(chain.degrees):
                current.append(rows)
            twice = set()
            for row in rows:
                if target.births[row] > birth:
                    raise RuntimeError("an interaction boundary face is born after its source")
                if degree > 1:
                    twice.symmetric_difference_update(previous[row])
            if twice:
                raise RuntimeError("constructed interaction boundary does not square to zero")
        previous = tuple(current)


def timing(call, repeats=3):
    times = []
    for _ in range(repeats):
        start = perf_counter()
        call()
        times.append(perf_counter() - start)
    return {"median_seconds": median(times), "samples_seconds": times}


def peak(call):
    tracemalloc.start()
    call()
    _, high = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return high


def check_clear(chain, backend):
    standard = current_reductions(chain, backend)
    cleared, skipped = cleared_reductions(chain, backend)
    assert [(r.zero_columns, r.deaths) for r in standard] == [(r.zero_columns, r.deaths) for r in cleared]
    return {"exact_zero_columns_and_death_pairs": True, "skipped_columns": skipped,
            "standard_additions": sum(r.column_additions for r in standard),
            "cleared_additions": sum(r.column_additions for r in cleared)}


def random_checks():
    """Mixed overlap, disconnected factors, ties, and empty-factor tests."""
    checks = 0
    for seed in range(40):
        rng = random.Random(seed)
        factors = []
        for side in range(2):
            builder = SimplicialComplexBuilder()
            labels = range(side, 6 + side)
            for vertex in labels:
                builder.insert((vertex,), rng.choice((0.0, .1)))
            for size in (2, 3, 4):
                for simplex in combinations(labels, size):
                    if rng.random() < .18:
                        builder.insert(simplex, rng.choice((.2, .2, 1.7)))
            factors.append(builder.freeze())
        chain = build_interaction_chain_complex(factors, max_homology_dimension=3, validate=True)
        cached_validate(chain)
        for backend in ("int", "sparse", "auto"):
            check_clear(chain, backend)
            checks += 1
    empty = SimplicialComplexBuilder().freeze()
    factor = grid_factor(2, 1)
    for pair in ((empty, empty), (empty, factor), (factor, empty)):
        chain = build_interaction_chain_complex(pair, max_homology_dimension=2)
        for backend in ("int", "sparse", "auto"):
            check_clear(chain, backend)
            checks += 1
    return checks


def overlap_checks():
    results = []
    for count in (512, 2048, 4096):
        a = PointCloud(np.zeros((count, 3)))
        b = PointCloud(np.ones((count, 3)))
        pairs = tuple(zip(a.ids, b.ids))

        def cached():
            # Include construction of both dictionaries in the timing.
            av = SimpleNamespace(ids=a.ids, index=a.index)
            bv = SimpleNamespace(ids=b.ids, index=b.index)
            return _overlap(av, bv, pairs, False)

        assert _overlap(a, b, pairs, False) == cached()
        results.append({"points_per_factor": count, "overlap_pairs": count,
                        "exact_output_equal": True,
                        "current": timing(lambda: _overlap(a, b, pairs, False)),
                        "cached_index": timing(cached)})
    return results


def audit_case(name, factors):
    chain = build_interaction_chain_complex(factors, max_homology_dimension=2)
    chain.validate_boundary()
    cached_validate(chain)
    clipped = replace(chain, degrees=chain.degrees[:2], max_homology_dimension=0)
    # Compare all interval objects, including tied-grade birth/death indices.
    full = compute_persistence(chain, include_diagonal=True)
    h0 = compute_persistence(clipped, include_diagonal=True)
    assert tuple(i for i in full.intervals if i.dimension == 0) == h0.intervals
    result = {"name": name, "factor_simplex_counts": [f.number_of_simplices for f in factors],
              "cell_counts": [len(layer.keys) for layer in chain.degrees],
              "construction": timing(lambda: build_interaction_chain_complex(factors, max_homology_dimension=2)),
              "validation_current": timing(chain.validate_boundary),
              "validation_cached": timing(lambda: cached_validate(chain)),
              "validation_current_peak_python_bytes": peak(chain.validate_boundary),
              "validation_cached_peak_python_bytes": peak(lambda: cached_validate(chain)),
              "h0_current_public": timing(lambda: public_persistence(chain, max_dimension=0, include_diagonal=True)),
              "h0_restricted_public": timing(lambda: public_persistence(clipped, max_dimension=0, include_diagonal=True)),
              "h0_interval_objects_equal_including_diagonals": True,
              "backends": {}}
    for backend in ("auto", "int", "sparse"):
        result["backends"][backend] = {
            **check_clear(chain, backend),
            "standard": timing(lambda: current_reductions(chain, backend)),
            "cleared": timing(lambda: cleared_reductions(chain, backend)),
            "standard_peak_python_bytes": peak(lambda: current_reductions(chain, backend)),
        }
    print(json.dumps(result), flush=True)
    return result


def main():
    result = {"environment": {"python": sys.version, "platform": platform.platform(),
                              "numpy": np.__version__, "scipy": scipy.__version__,
                              "threads": {key: os.environ.get(key) for key in
                                          ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}},
              "randomized_and_empty_backend_checks": random_checks(),
              "overlap_index_checks": overlap_checks(),
              "cases": []}
    for name, factors in (
        ("triangulated_grid_12x12", (grid_factor(12, 1), grid_factor(12, 2))),
        ("triangulated_grid_24x24", (grid_factor(24, 1), grid_factor(24, 2))),
        ("complete_18_vertices_3_skeleton", (complete_factor(18, 1), complete_factor(18, 2))),
    ):
        result["cases"].append(audit_case(name, factors))
    path = Path(__file__).with_name("interaction_results.json")
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(str(path))


if __name__ == "__main__":
    main()
