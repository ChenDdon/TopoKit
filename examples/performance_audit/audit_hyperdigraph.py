"""Standalone efficiency audit; does not alter any production implementation.

Run: PYTHONPATH=src python examples/performance_audit/audit_hyperdigraph.py
Benchmarks compare exact interval multisets using identical stored edge births.
"""
from array import array
from collections import Counter
import cProfile
import gc
import io
import inspect
import itertools
import json
import math
import os
from pathlib import Path
import platform
import pstats
import random
import statistics
import sys
import time
from unittest.mock import patch

import numpy as np
import scipy
from topokit.builders import hyperdigraph as hyperdigraph_builder
from topokit.core import hyperdigraph as hyperdigraph_core
from topokit.core._hyperdigraph import (
    FilteredHyperdigraph, ScoreDirectedEdgeFiltration,
    build_filtered_chain_complex, compute_path_compatible_persistence,
    compute_persistence,
)
from topokit.core._hyperdigraph import chain_complex as chain_module

OUT = Path(__file__).resolve().parent


def signature(result):
    return Counter((x.dimension, x.birth, x.death) for x in result.intervals)


def timed(fn, repeats=3):
    fn()
    times = []
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    return result, statistics.median(times)


def compact_from_native(native):
    # This fixture uses vertex ID i and strict score i, all singleton births 0.
    # Reuse stored distances exactly: no geometry/roundoff/threshold changes.
    records = sorted((r.birth, (r.vertices[0] << 16) | r.vertices[1])
                     for r in native.weighted_hyperedges(1))
    return ScoreDirectedEdgeFiltration(
        len(native.vertices), array('I', (p for _, p in records)),
        array('d', (b for b, _ in records)), tuple(native.vertices),
        'audit_existing_edge_births', len(native.thresholds()), 'explicit')


def h0_only(native):
    # Exact restricted prototype: all vertices enter at 0; arbitrary directions.
    parents = list(range(len(native.vertices)))
    def find(x):
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x
    intervals = []
    for edge in native.weighted_hyperedges(1):
        left, right = map(find, edge.vertices)
        if left != right:
            parents[max(left, right)] = min(left, right)
            if edge.birth > 0:
                intervals.append((0, 0., edge.birth))
    intervals.extend((0, 0., math.inf) for i, p in enumerate(parents) if i == p)
    return Counter(intervals)


def retained_boundary_prototype(native, max_dimension):
    """Keep exact reduced boundary already tracked alongside every generator.

    The source transformation retains the existing checks, ordering, basis,
    and algebra verbatim. It adds an audit-only cache entry for the reduced
    boundary currently discarded by the generic compatible-basis function.
    Nothing is written to library source; patches are scoped to this call.
    """
    boundary_cache = {}
    original_apply = chain_module.apply_columns
    original_basis = chain_module._filtered_column_compatible_basis
    source = inspect.getsource(original_basis)
    old = 'generated.append((generator, entry, serial))'
    assert source.count(old) == 1
    source = source.replace(old, 'generated.append((generator, entry, serial, reduced))')
    marker = '    generators = tuple(item[0] for item in generated)'
    assert source.count(marker) == 1
    source = source.replace(marker, '''    for item in generated:
        ambient_boundary = 0
        for priority in iter_set_bits(item[3]):
            ambient_boundary ^= 1 << rows_by_priority[priority]
        _audit_boundary_cache[(id(prepared.columns), item[0])] = ambient_boundary
''' + marker)
    namespace = dict(vars(chain_module), _audit_boundary_cache=boundary_cache)
    exec(compile(source, '<audit retained exact boundaries>', 'exec'), namespace)
    new_basis = namespace['_filtered_column_compatible_basis']
    def cached_apply(selection, columns):
        key = (id(columns), selection)
        if key in boundary_cache:
            return boundary_cache[key]
        return original_apply(selection, columns)
    with patch.object(chain_module, '_filtered_column_compatible_basis', new_basis), \
         patch.object(chain_module, 'apply_columns', cached_apply):
        return build_filtered_chain_complex(native, max_dimension)


def main():
    report = {'environment': {'python': sys.version, 'numpy': np.__version__,
                             'scipy': scipy.__version__, 'platform': platform.platform(),
                             'OPENBLAS_NUM_THREADS': os.environ.get('OPENBLAS_NUM_THREADS'),
                             'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS')},
              'method': 'median of 3 warmed serial runs, same stored births, interval multiset equality',
              'h0': [], 'h1': [], 'h2': {}}
    for n in (1000, 4000, 8000):
        native = FilteredHyperdigraph(range(n),
                  (((i, (i + 1) % n), float(1 + i % 7)) for i in range(n)),
                  include_all_vertices=True)
        standard, seconds = timed(lambda: hyperdigraph_core.persistence(native, max_dimension=0))
        quick, quick_seconds = timed(lambda: h0_only(native))
        assert signature(standard) == quick
        chain = build_filtered_chain_complex(native, 0)
        basis_bytes = sum(sys.getsizeof(x) for degree in chain.omega_generators for x in degree)
        boundary_bytes = sum(sys.getsizeof(x) for degree in chain.boundary_columns for x in degree)
        report['h0'].append({'vertices': n, 'edges': n, 'public_seconds': seconds,
                             'direct_union_find_seconds': quick_seconds,
                             'speedup': seconds / quick_seconds, 'exact_intervals': True,
                             'omega_int_payload_bytes': basis_bytes,
                             'boundary_int_payload_bytes': boundary_bytes})
        print('H0', report['h0'][-1], flush=True)
        del chain
    rng = np.random.default_rng(20260905)
    for n, support in ((50, 'complete_supplied_bonds'), (120, 'delaunay')):
        points = rng.normal(size=(n, 3))
        bonds = [(i, j) for i in range(n) for j in range(i + 1, n)] if support.startswith('complete') else None
        obj = hyperdigraph_builder.from_points(points, max_dimension=1, weights=np.arange(n), bonds=bonds)
        compact = compact_from_native(obj.native)
        normal, normal_seconds = timed(lambda: hyperdigraph_core.persistence(obj, max_dimension=1))
        fast, fast_seconds = timed(lambda: compute_path_compatible_persistence(compact, 1))
        assert signature(normal) == signature(fast)
        diagonal_normal = hyperdigraph_core.persistence(obj, max_dimension=1, include_diagonal=True)
        diagonal_fast = compute_path_compatible_persistence(compact, 1, include_diagonal=True)
        assert signature(diagonal_normal) == signature(diagonal_fast)
        report['h1'].append({'vertices': n, 'support': support,
                             'hyperedge_counts': obj.metadata['hyperedge_counts'],
                             'public_seconds': normal_seconds, 'existing_compact_seconds': fast_seconds,
                             'speedup': normal_seconds / fast_seconds, 'exact_intervals': True,
                             'exact_diagonal_intervals': True,
                             'note': 'analysis only; existing geometry and explicit construction excluded from both timings'})
        print('H1', report['h1'][-1], flush=True)
    points = rng.normal(size=(24, 3))
    obj = hyperdigraph_builder.from_points(points, max_dimension=2)
    automatic, automatic_seconds = timed(lambda: compute_persistence(obj.native, 2))
    generic, generic_seconds = timed(lambda: compute_persistence(obj.native, 2, omega_backend='generic', reduction_backend='standard'))
    assert signature(automatic) == signature(generic)
    profiler = cProfile.Profile()
    profiler.runcall(compute_persistence, obj.native, 2)
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats('cumulative').print_stats(25)
    (OUT / 'hyperdigraph_profile.txt').write_text(stream.getvalue())
    chain = build_filtered_chain_complex(obj.native, 2)
    retained, retained_seconds = timed(lambda: retained_boundary_prototype(obj.native, 2))
    ordinary_chain, ordinary_chain_seconds = timed(lambda: build_filtered_chain_complex(obj.native, 2))
    # Stronger than barcode equality: every dataclass field must match exactly.
    assert retained == ordinary_chain
    random_state = random.Random(20260905)
    # Arbitrary explicitly supplied hyperedges, absent singleton/deletion faces,
    # independently tied/late births; this is not limited to geometric paths.
    for trial in range(60):
        records = [(edge, float(random_state.randrange(4)))
                   for dimension in range(4)
                   for edge in itertools.permutations(range(4), dimension + 1)
                   if random_state.random() < 0.35]
        arbitrary = FilteredHyperdigraph(range(4), records, include_all_vertices=False)
        assert retained_boundary_prototype(arbitrary, 2) == build_filtered_chain_complex(arbitrary, 2), trial
    report['h2'] = {'vertices': 24, 'weights': 'all equal, both edge orientations retained',
                    'hyperedge_counts': obj.metadata['hyperedge_counts'],
                    'omega_counts': chain.omega_dimensions,
                    'auto_seconds': automatic_seconds, 'generic_standard_seconds': generic_seconds,
                    'auto_speedup': generic_seconds / automatic_seconds,
                    'exact_intervals': True, 'omega_backends': chain.omega_backends,
                    'chain_seconds': ordinary_chain_seconds,
                    'retained_boundary_chain_seconds': retained_seconds,
                    'retained_boundary_chain_speedup': ordinary_chain_seconds / retained_seconds,
                    'retained_boundary_every_chain_field_exact': True,
                    'retained_boundary_arbitrary_hypergraph_equivalence_trials': 60}
    # Scientific counterexample: the two triples cancel their missing (0,3)
    # deletion face, killing H1. A directed clique expansion contains no triples.
    edges = [((0,1),1.), ((1,3),1.), ((0,2),1.), ((2,3),1.)]
    embedded = FilteredHyperdigraph(range(4), edges + [((0,1,3),2.), ((0,2,3),2.)], include_all_vertices=True)
    clique = FilteredHyperdigraph(range(4), edges, include_all_vertices=True)
    embedded_h1 = [x for x in signature(compute_persistence(embedded,1)) if x[0] == 1]
    clique_h1 = [x for x in signature(compute_persistence(clique,1)) if x[0] == 1]
    assert embedded_h1 == [(1, 1., 2.)] and clique_h1 == [(1, 1., math.inf)]
    report['clique_counterexample'] = {'embedded_h1': embedded_h1, 'clique_h1': clique_h1}
    # Portable JSON encodes infinities as strings, preserving exact meaning.
    def safe(x):
        if isinstance(x, float) and not math.isfinite(x): return str(x)
        if isinstance(x, (tuple, list)): return [safe(v) for v in x]
        if isinstance(x, dict): return {k: safe(v) for k, v in x.items()}
        return x
    (OUT / 'hyperdigraph_audit.json').write_text(json.dumps(safe(report), indent=2) + '\n')
    print(json.dumps(safe(report['h2']), indent=2), flush=True)


if __name__ == '__main__':
    main()
