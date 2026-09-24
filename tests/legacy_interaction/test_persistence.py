from __future__ import annotations

import unittest
from math import inf

from topokit.core._interaction import (
    SimplicialComplexBuilder,
    build_interaction_chain_complex,
    compute_persistence,
)
from topokit.workflows.interaction import persistence_from_dimension_dicts


def filtered_edge():
    builder = SimplicialComplexBuilder()
    builder.insert((0,), 0.0)
    builder.insert((1,), 0.0)
    builder.insert((0, 1), 1.0, with_faces=False)
    return builder.freeze()


def triangle_boundary():
    builder = SimplicialComplexBuilder()
    for vertex in range(3):
        builder.insert((vertex,), 0.0, with_faces=False)
    for edge in ((0, 1), (0, 2), (1, 2)):
        builder.insert(edge, 0.0, with_faces=False)
    return builder.freeze()


def signature(result):
    intervals = result.intervals if hasattr(result, "intervals") else result
    return tuple(
        (interval.dimension, interval.birth, interval.death)
        for interval in intervals
    )


def global_reference_signature(chain):
    records = [
        (birth, dimension, local_index)
        for dimension, layer in enumerate(chain.degrees)
        for local_index, birth in enumerate(layer.births)
    ]
    records.sort()
    global_index = {
        (dimension, local_index): index
        for index, (_, dimension, local_index) in enumerate(records)
    }
    reduced_by_pivot = {}
    zero_columns = set()
    death_of = {}
    for column_index, (_, dimension, local_index) in enumerate(records):
        column = 0
        if dimension:
            for row in chain.boundary_rows(dimension, local_index):
                row_index = global_index[(dimension - 1, row)]
                if row_index >= column_index:
                    raise AssertionError("reference filtration is not face-compatible")
                column ^= 1 << row_index
        while column:
            pivot = column.bit_length() - 1
            previous = reduced_by_pivot.get(pivot)
            if previous is None:
                reduced_by_pivot[pivot] = column
                death_of[pivot] = column_index
                break
            column ^= previous
        if not column:
            zero_columns.add(column_index)

    output = []
    for birth_index in sorted(zero_columns):
        birth, dimension, _ = records[birth_index]
        if dimension > chain.max_homology_dimension:
            continue
        death_index = death_of.get(birth_index)
        death = records[death_index][0] if death_index is not None else inf
        output.append((dimension, birth, death))
    return tuple(sorted(output))


class InteractionPersistenceTests(unittest.TestCase):
    def test_filtered_edge_pair(self) -> None:
        factor = filtered_edge()
        chain = build_interaction_chain_complex(
            (factor, factor), max_homology_dimension=1, validate=True
        )
        result = compute_persistence(chain)
        self.assertEqual(
            signature(result),
            (
                (0, 0.0, 1.0),
                (0, 0.0, 1.0),
                (1, 1.0, inf),
            ),
        )

    def test_single_factor_reduces_to_ordinary_persistence(self) -> None:
        factor = filtered_edge()
        chain = build_interaction_chain_complex(
            (factor,), max_homology_dimension=1, validate=True
        )
        result = compute_persistence(chain)
        self.assertEqual(
            signature(result),
            ((0, 0.0, 1.0), (0, 0.0, inf)),
        )

    def test_sparse_and_integer_reducers_agree(self) -> None:
        factor = filtered_edge()
        chain = build_interaction_chain_complex(
            (factor, factor), max_homology_dimension=2, validate=True
        )
        self.assertEqual(
            signature(compute_persistence(chain, backend="int", include_diagonal=True)),
            signature(compute_persistence(chain, backend="sparse", include_diagonal=True)),
        )

    def test_minimum_persistence_filters_numerical_bars(self) -> None:
        builder = SimplicialComplexBuilder()
        builder.insert((0,), 0.0)
        builder.insert((1,), 0.0)
        builder.insert((0, 1), 1e-12, with_faces=False)
        factor = builder.freeze()
        chain = build_interaction_chain_complex(
            (factor, factor), max_homology_dimension=1, validate=True
        )
        self.assertEqual(
            signature(compute_persistence(chain)),
            (
                (0, 0.0, 1e-12),
                (0, 0.0, 1e-12),
                (1, 1e-12, inf),
            ),
        )
        self.assertEqual(
            signature(compute_persistence(chain, minimum_persistence=1e-10)),
            ((1, 1e-12, inf),),
        )

    def test_minimum_persistence_is_validated(self) -> None:
        factor = filtered_edge()
        chain = build_interaction_chain_complex(
            (factor, factor), max_homology_dimension=1
        )
        with self.assertRaises(TypeError):
            compute_persistence(chain, minimum_persistence=True)
        with self.assertRaises(ValueError):
            compute_persistence(chain, minimum_persistence=-1.0)

    def test_highest_interaction_cycle_is_reported(self) -> None:
        factor = triangle_boundary()
        chain = build_interaction_chain_complex(
            (factor, factor), max_homology_dimension=2, validate=True
        )
        result = compute_persistence(chain)
        self.assertEqual(signature(result.in_dimension(2)), ((2, 0.0, inf),))

    def test_dimension_reducer_matches_global_reference(self) -> None:
        first = SimplicialComplexBuilder()
        first.insert((0,), 0.0)
        first.insert((1,), 0.2)
        first.insert((2,), 0.4)
        first.insert((0, 1), 0.8, with_faces=False)
        first.insert((0, 2), 1.1, with_faces=False)
        first.insert((1, 2), 1.3, with_faces=False)
        first.insert((0, 1, 2), 1.8, with_faces=False)
        second = SimplicialComplexBuilder()
        second.insert((0,), 0.1)
        second.insert((1,), 0.1)
        second.insert((2,), 0.5)
        second.insert((0, 1), 0.9, with_faces=False)
        second.insert((0, 2), 1.0, with_faces=False)
        second.insert((1, 2), 1.4, with_faces=False)
        second.insert((0, 1, 2), 2.0, with_faces=False)
        chain = build_interaction_chain_complex(
            (first.freeze(), second.freeze()),
            max_homology_dimension=2,
            validate=True,
        )
        self.assertEqual(
            signature(compute_persistence(chain, include_diagonal=True)),
            global_reference_signature(chain),
        )

    def test_legacy_dictionary_adapter(self) -> None:
        result = persistence_from_dimension_dicts(
            [
                {0: [(0,), (1,)], 1: [(0, 1)]},
                {0: [(0,), (1,)], 1: [(0, 1)]},
            ],
            [
                {0: [0.0, 0.0], 1: [1.0]},
                {0: [0.0, 0.0], 1: [1.0]},
            ],
            max_dimension=1,
            validate=True,
        )
        self.assertEqual(
            signature(result),
            ((0, 0.0, 1.0), (0, 0.0, 1.0), (1, 1.0, inf)),
        )


if __name__ == "__main__":
    unittest.main()
