"""Seeded cross-checks against independent interaction-homology oracles.

The reference routines in this file intentionally do not use the production
posting join, precomputed factor-face IDs, interaction boundary lookup, or
dimension-separated persistence reducer.  They are small Cartesian/tuple
implementations suitable only for correctness tests.
"""

from __future__ import annotations

from collections import Counter
import itertools
from math import inf
import random
import unittest

from topokit.core._interaction import (
    FilteredSimplicialComplex,
    build_interaction_chain_complex,
    compute_persistence,
)


InteractionKey = tuple[int, ...]


def _common_vertices(
    factors: tuple[FilteredSimplicialComplex, ...], key: InteractionKey
) -> set[int]:
    common = set(factors[0].simplex(key[0]))
    for factor, simplex_id in zip(factors[1:], key[1:]):
        common.intersection_update(factor.simplex(simplex_id))
    return common


def _brute_layers(
    factors: tuple[FilteredSimplicialComplex, ...], maximum_degree: int
) -> tuple[tuple[tuple[InteractionKey, float], ...], ...]:
    """Enumerate the defining Cartesian product without incidence indexes."""

    layers: list[list[tuple[InteractionKey, float]]] = [
        [] for _ in range(maximum_degree + 1)
    ]
    factor_ranges = tuple(range(factor.number_of_simplices) for factor in factors)
    for key in itertools.product(*factor_ranges):
        degree = sum(
            factor.dimension(simplex_id)
            for factor, simplex_id in zip(factors, key)
        )
        if degree > maximum_degree or not _common_vertices(factors, key):
            continue
        birth = max(
            factor.filtration(simplex_id)
            for factor, simplex_id in zip(factors, key)
        )
        layers[degree].append((key, birth))

    return tuple(
        tuple(sorted(layer, key=lambda item: (item[1], item[0])))
        for layer in layers
    )


def _tuple_face_boundary(
    factors: tuple[FilteredSimplicialComplex, ...], key: InteractionKey
) -> set[InteractionKey]:
    """Compute the GF(2) quotient boundary directly from simplex tuples."""

    ids_by_simplex = tuple(
        {simplex: simplex_id for simplex_id, simplex in enumerate(factor.simplices)}
        for factor in factors
    )
    boundary: set[InteractionKey] = set()
    mutable_key = list(key)
    for factor_index, (factor, simplex_id) in enumerate(zip(factors, key)):
        simplex = factor.simplex(simplex_id)
        if len(simplex) == 1:
            continue
        for omitted in range(len(simplex)):
            face = simplex[:omitted] + simplex[omitted + 1 :]
            face_id = ids_by_simplex[factor_index][face]
            mutable_key[factor_index] = face_id
            target = tuple(mutable_key)
            if _common_vertices(factors, target):
                if target in boundary:
                    boundary.remove(target)
                else:
                    boundary.add(target)
        mutable_key[factor_index] = simplex_id
    return boundary


def _global_reference_intervals(
    factors: tuple[FilteredSimplicialComplex, ...], max_homology_dimension: int
) -> Counter[tuple[int, float, float]]:
    """Run a global ordinary reduction on independently constructed columns."""

    layers = _brute_layers(factors, max_homology_dimension + 1)
    records = [
        (birth, degree, key)
        for degree, layer in enumerate(layers)
        for key, birth in layer
    ]
    # Dimension before key makes every equal-grade face precede its cofaces.
    records.sort(key=lambda item: (item[0], item[1], item[2]))
    global_index = {
        (degree, key): index
        for index, (_, degree, key) in enumerate(records)
    }

    pivot_columns: dict[int, int] = {}
    death_column_by_birth: dict[int, int] = {}
    zero_columns: set[int] = set()
    for column_index, (_, degree, key) in enumerate(records):
        column = 0
        if degree:
            for face_key in _tuple_face_boundary(factors, key):
                row_index = global_index[(degree - 1, face_key)]
                if row_index >= column_index:
                    raise AssertionError(
                        "the independent filtration did not put a face before its coface"
                    )
                column ^= 1 << row_index
        while column:
            pivot = column.bit_length() - 1
            previous = pivot_columns.get(pivot)
            if previous is None:
                pivot_columns[pivot] = column
                death_column_by_birth[pivot] = column_index
                break
            column ^= previous
        if not column:
            zero_columns.add(column_index)

    intervals: Counter[tuple[int, float, float]] = Counter()
    for birth_column in zero_columns:
        birth, degree, _ = records[birth_column]
        if degree > max_homology_dimension:
            continue
        death_column = death_column_by_birth.get(birth_column)
        death = records[death_column][0] if death_column is not None else inf
        intervals[(degree, birth, death)] += 1
    return intervals


def _random_filtered_complex(
    random_source: random.Random, *, force_triangle: bool
) -> FilteredSimplicialComplex:
    """Generate a small face-closed complex with many tied filtration values."""

    vertices = tuple(range(4))
    edges = {(0, 1)}
    edges.update(
        edge
        for edge in itertools.combinations(vertices, 2)
        if random_source.random() < 0.55
    )
    triangles: set[tuple[int, ...]] = set()
    if force_triangle:
        triangles.add((0, 1, 2))
    triangles.update(
        triangle
        for triangle in itertools.combinations(vertices, 3)
        if random_source.random() < 0.30
    )
    for triangle in triangles:
        edges.update(itertools.combinations(triangle, 2))

    birth_by_simplex: dict[tuple[int, ...], float] = {
        (vertex,): float(random_source.choice((0, 0, 1)))
        for vertex in vertices
    }
    for edge in sorted(edges):
        face_birth = max(birth_by_simplex[(vertex,)] for vertex in edge)
        birth_by_simplex[edge] = face_birth + random_source.choice((0.0, 0.0, 1.0))
    for triangle in sorted(triangles):
        face_birth = max(
            birth_by_simplex[edge]
            for edge in itertools.combinations(triangle, 2)
        )
        birth_by_simplex[triangle] = face_birth + random_source.choice(
            (0.0, 0.0, 1.0)
        )

    simplices: dict[int, list[tuple[int, ...]]] = {}
    filtrations: dict[int, list[float]] = {}
    for dimension in range(3):
        layer = [
            simplex
            for simplex in birth_by_simplex
            if len(simplex) - 1 == dimension
        ]
        if not layer:
            continue
        random_source.shuffle(layer)
        simplices[dimension] = layer
        filtrations[dimension] = [birth_by_simplex[simplex] for simplex in layer]
    return FilteredSimplicialComplex.from_dimension_dict(simplices, filtrations)


def _signature(result) -> Counter[tuple[int, float, float]]:
    return Counter(
        (interval.dimension, interval.birth, interval.death)
        for interval in result.intervals
    )


class SeededRandomizedCrossChecks(unittest.TestCase):
    def test_indexed_cells_boundaries_and_persistence_match_independent_oracles(
        self,
    ) -> None:
        random_source = random.Random(20260901)
        case_number = 0
        for factor_count, repetitions in ((2, 16), (3, 10)):
            for repetition in range(repetitions):
                case_number += 1
                factors = tuple(
                    _random_filtered_complex(
                        random_source,
                        force_triangle=(repetition + factor_index) % 3 == 0,
                    )
                    for factor_index in range(factor_count)
                )
                self.assertTrue(
                    any(
                        len(set(factor.births)) < len(factor.births)
                        for factor in factors
                    ),
                    "the randomized suite must exercise filtration ties",
                )

                chain = build_interaction_chain_complex(
                    factors,
                    max_homology_dimension=2,
                    validate=True,
                )
                brute_layers = _brute_layers(factors, chain.max_cell_degree)

                with self.subTest(
                    case=case_number,
                    factors=factor_count,
                    check="cells-and-births",
                ):
                    for degree, (actual_layer, expected_layer) in enumerate(
                        zip(chain.degrees, brute_layers)
                    ):
                        actual = dict(zip(actual_layer.keys, actual_layer.births))
                        expected = dict(expected_layer)
                        self.assertEqual(actual, expected, f"degree {degree}")

                with self.subTest(
                    case=case_number,
                    factors=factor_count,
                    check="quotient-boundaries",
                ):
                    for degree in range(1, len(chain.degrees)):
                        for cell_index, key in enumerate(chain.degrees[degree].keys):
                            actual = {
                                chain.degrees[degree - 1].keys[row]
                                for row in chain.boundary_rows(degree, cell_index)
                            }
                            self.assertEqual(
                                actual,
                                _tuple_face_boundary(factors, key),
                                f"degree {degree}, key {key}",
                            )

                reference = _global_reference_intervals(factors, 2)
                with self.subTest(
                    case=case_number,
                    factors=factor_count,
                    check="persistence-int",
                ):
                    self.assertEqual(
                        _signature(
                            compute_persistence(
                                chain, backend="int", include_diagonal=True
                            )
                        ),
                        reference,
                    )
                with self.subTest(
                    case=case_number,
                    factors=factor_count,
                    check="persistence-sparse",
                ):
                    self.assertEqual(
                        _signature(
                            compute_persistence(
                                chain, backend="sparse", include_diagonal=True
                            )
                        ),
                        reference,
                    )


if __name__ == "__main__":
    unittest.main()
