"""Exact finite-field persistence with union-find H0 and clearing reduction."""

from dataclasses import dataclass
import math

import numpy as np

from ._validation import nonnegative_int, prime, real
from .algebra import boundary_column, reduce_column
from .complex import Simplex, as_complex


@dataclass(frozen=True)
class PersistenceInterval:
    dimension: int
    birth: float
    death: float
    birth_simplex: Simplex
    death_simplex: Simplex | None


@dataclass(frozen=True)
class PersistenceResult:
    intervals: tuple[PersistenceInterval, ...]
    field: int
    max_dimension: int
    diagnostics: dict

    def diagram(self, dimension):
        dimension = nonnegative_int(dimension, "dimension")
        return np.asarray([(i.birth, i.death) for i in self.intervals if i.dimension == dimension], dtype=float).reshape(-1, 2)

    def persistent_betti(self, dimension, source, target):
        dimension = nonnegative_int(dimension, "dimension")
        if dimension > self.max_dimension:
            raise ValueError("Dimension was not computed")
        source = real(source, "source")
        target = real(target, "target")
        if source > target:
            raise ValueError("source must be <= target")
        return sum(i.dimension == dimension and i.birth <= source and target < i.death for i in self.intervals)

    def betti_at(self, scale):
        return tuple(self.persistent_betti(q, scale, scale) for q in range(self.max_dimension + 1))


def _h0(vertices, edges, values):
    parents = {s[0]: s[0] for s in vertices}
    eldest_vertices = {s[0]: s[0] for s in vertices}
    component_sizes = {s[0]: 1 for s in vertices}

    def root(v):
        while parents[v] != v:
            parents[v] = parents[parents[v]]
            v = parents[v]
        return v

    intervals, cycle_edges = [], set()
    for edge in edges:
        root_a, root_b = (root(v) for v in edge)
        if root_a == root_b:
            cycle_edges.add(edge)
            continue
        # The oldest birth survives even when union-by-size chooses another root.
        surviving_vertex, dying_vertex = sorted(
            (eldest_vertices[root_a], eldest_vertices[root_b]), key=lambda v: (values[(v,)], v))
        intervals.append(PersistenceInterval(0, values[(dying_vertex,)], values[edge], (dying_vertex,), edge))
        if component_sizes[root_a] < component_sizes[root_b]:
            root_a, root_b = root_b, root_a
        parents[root_b] = root_a
        component_sizes[root_a] += component_sizes[root_b]
        eldest_vertices[root_a] = surviving_vertex
    for v in parents:
        if root(v) == v:
            birth_simplex = (eldest_vertices[v],)
            intervals.append(PersistenceInterval(0, values[birth_simplex], math.inf, birth_simplex, None))
    return intervals, cycle_edges


def persistent_homology(complex_, *, max_dimension=None, field=2,
                        include_zero=False, clearing=True):
    """Compute [birth,death) intervals; +inf denotes survival in this input.

    Process dimensions downward and columns in filtration order. Clearing
    skips positive columns paired in the preceding dimension. GF(2) uses XOR
    bitsets, other prime fields sparse modular columns. H0 uses union-find.
    Boundary columns are generated on demand; no dense global matrix exists.
    To capture Hq deaths, the supplied complex must include (q+1)-simplices.
    """
    complex_ = as_complex(complex_)
    field = prime(field)
    maximum_dimension = (max(complex_.dimension, 0) if max_dimension is None
                         else nonnegative_int(max_dimension, "max_dimension"))
    filtration_order = complex_.get_filtration()
    filtration_values = dict(filtration_order)
    simplices_by_dimension = {}
    for simplex, _ in filtration_order:
        simplices_by_dimension.setdefault(len(simplex) - 1, []).append(simplex)
    intervals, edge_cycles = _h0(
        simplices_by_dimension.get(0, []), simplices_by_dimension.get(1, []), filtration_values)
    positive_simplices = set(edge_cycles) if maximum_dimension >= 1 else set()
    # Map each paired birth simplex to its killing simplex; its column can clear.
    death_partners, clearing_candidates = {}, set()
    reduced_column_count = cleared_column_count = 0
    for q in range(min(maximum_dimension + 1, complex_.dimension), 1, -1):
        row_basis = simplices_by_dimension.get(q - 1, [])
        row_index = {s: i for i, s in enumerate(row_basis)}
        pivot_columns = {}
        for simplex in simplices_by_dimension.get(q, []):
            if clearing and simplex in clearing_candidates:
                positive_simplices.add(simplex)
                cleared_column_count += 1
                continue
            reduced_column_count += 1
            pivot_row = reduce_column(boundary_column(simplex, row_index, field), pivot_columns, field)
            if pivot_row is None:
                positive_simplices.add(simplex)
            else:
                birth_simplex = row_basis[pivot_row]
                death_partners[birth_simplex] = simplex
                clearing_candidates.add(birth_simplex)
    for birth_simplex in positive_simplices | set(death_partners):
        q = len(birth_simplex) - 1
        if q > maximum_dimension:
            continue
        death_simplex = death_partners.get(birth_simplex)
        intervals.append(PersistenceInterval(
            q, filtration_values[birth_simplex],
            filtration_values[death_simplex] if death_simplex else math.inf,
            birth_simplex, death_simplex))
    intervals = tuple(sorted((i for i in intervals if include_zero or i.death > i.birth),
                             key=lambda i: (i.dimension, i.birth, i.death, i.birth_simplex)))
    return PersistenceResult(intervals, field, maximum_dimension, {
        "algorithm": "union_find_h0+dimension_clearing",
        "column_backend": "xor_bitset" if field == 2 else "sparse_modular",
        "reduced_columns": reduced_column_count, "cleared_columns": cleared_column_count,
        "num_simplices": len(complex_), "clearing": bool(clearing),
    })
