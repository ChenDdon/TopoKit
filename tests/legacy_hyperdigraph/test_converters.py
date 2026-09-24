from __future__ import annotations

import math
import unittest

from topokit.core._hyperdigraph import FilteredHyperdigraph, Hyperdigraph, compute_homology, compute_persistence
from topokit.builders._hyperdigraph_converters import bounded_filtration, filtered_hyperdigraph_from_distance_matrix, hyperdigraph_from_adjacency_matrix, hyperdigraph_from_dimension_dict
from topokit.workflows.hyperdigraph import compute_homology_from_adjacency_matrix, compute_persistence_from_distance_matrix


def _signature(result):
    return tuple(
        (interval.dimension, interval.birth, interval.death)
        for interval in result.intervals
    )


class ConverterTests(unittest.TestCase):
    def test_legacy_dummy_adjacency_matrix(self) -> None:
        adjacency = (
            (0, 1, 0, 0, 1, 0),
            (0, 0, 1, 0, 0, 0),
            (0, 0, 0, 1, 0, 0),
            (0, 0, 0, 0, 0, 0),
            (0, 0, 1, 0, 0, 0),
            (0, 0, 0, 0, 0, 0),
        )
        hyperdigraph = hyperdigraph_from_adjacency_matrix(
            adjacency, max_path_dimension=2
        )
        self.assertEqual(
            hyperdigraph.directed_hyperedges(1),
            ((0, 1), (0, 4), (1, 2), (2, 3), (4, 2)),
        )
        self.assertEqual(
            hyperdigraph.directed_hyperedges(2),
            ((0, 1, 2), (0, 4, 2), (1, 2, 3), (4, 2, 3)),
        )
        result = compute_homology_from_adjacency_matrix(adjacency, 1)
        self.assertEqual(result.betti_numbers, (2, 0))

    def test_legacy_dimension_dictionary_matches_laplacian_nullities(self) -> None:
        legacy = {
            0: [(1,), (2,), (3,), (4,), (5,)],
            1: [
                (1, 2),
                (2, 1),
                (1, 3),
                (3, 2),
                (4, 2),
                (4, 3),
                (5, 2),
            ],
            2: [(1, 4, 2), (1, 4, 3), (1, 5, 2), (3, 4, 2)],
        }
        hyperdigraph = hyperdigraph_from_dimension_dict(legacy)
        self.assertEqual(compute_homology(hyperdigraph, 1).betti_numbers, (1, 2))

    def test_minimum_and_maximum_adjacency_control_filtration(self) -> None:
        distances = (
            (0, 2, 5),
            (math.inf, 0, 3),
            (math.inf, math.inf, 0),
        )
        minimum = (
            (0, 0, 1),
            (0, 0, 0),
            (0, 0, 0),
        )
        maximum = (
            (0, 1, 1),
            (0, 0, 0),
            (0, 0, 0),
        )
        filtration = filtered_hyperdigraph_from_distance_matrix(
            distances,
            max_path_dimension=2,
            minimum_adjacency=minimum,
            maximum_adjacency=maximum,
        )
        self.assertEqual(filtration.birth_of((0, 2)), 0.0)
        self.assertEqual(filtration.birth_of((0, 1)), 2.0)
        self.assertIsNone(filtration.birth_of((1, 2)))
        self.assertEqual(filtration.directed_hyperedges(2), ())

    def test_explicit_minimum_can_start_above_vertices(self) -> None:
        distances = (
            (0, 4, math.inf),
            (math.inf, 0, 4),
            (math.inf, math.inf, 0),
        )
        minimum = Hyperdigraph(
            (0, 1, 2),
            ((0, 1), (1, 2), (0, 1, 2)),
            include_all_vertices=True,
        )
        filtration = filtered_hyperdigraph_from_distance_matrix(
            distances,
            max_path_dimension=2,
            minimum_hyperdigraph=minimum,
        )
        self.assertEqual(filtration.birth_of((0, 1)), 0.0)
        self.assertEqual(filtration.birth_of((1, 2)), 0.0)
        self.assertEqual(filtration.birth_of((0, 1, 2)), 0.0)

    def test_maximum_hyperdigraph_removes_unwanted_higher_paths(self) -> None:
        adjacency = (
            (0, 1, 1),
            (0, 0, 1),
            (0, 0, 0),
        )
        maximum = Hyperdigraph(
            (0, 1, 2),
            ((0, 1), (0, 2), (1, 2)),
            include_all_vertices=True,
        )
        bounded = hyperdigraph_from_adjacency_matrix(
            adjacency,
            max_path_dimension=2,
            maximum_hyperdigraph=maximum,
        )
        standard = hyperdigraph_from_adjacency_matrix(
            adjacency, max_path_dimension=2
        )
        self.assertEqual(bounded.directed_hyperedges(2), ())
        self.assertEqual(standard.directed_hyperedges(2), ((0, 1, 2),))
        self.assertEqual(compute_homology(bounded, 1).betti_numbers, (1, 1))
        self.assertEqual(compute_homology(standard, 1).betti_numbers, (1, 0))

    def test_bounded_existing_filtration(self) -> None:
        filtration = FilteredHyperdigraph(
            (0, 1, 2),
            (
                ((0,), 0),
                ((1,), 0),
                ((2,), 0),
                ((0, 1), 3),
                ((1, 2), 2),
                ((0, 2), 4),
                ((0, 1, 2), 5),
            ),
        )
        minimum = Hyperdigraph((0, 1, 2), ((0, 1),))
        maximum = Hyperdigraph(
            (0, 1, 2),
            ((0, 1), (1, 2), (0, 2)),
        )
        bounded = bounded_filtration(
            filtration,
            minimum_hyperdigraph=minimum,
            maximum_hyperdigraph=maximum,
        )
        self.assertEqual(bounded.birth_of((0, 1)), 0.0)
        self.assertIsNone(bounded.birth_of((0, 1, 2)))
        self.assertEqual(bounded.directed_hyperedges(0), ((0,), (1,), (2,)))

    def test_distance_persistence_fast_and_standard_agree(self) -> None:
        distances = (
            (0, 1, 3, math.inf),
            (math.inf, 0, 1, 3),
            (math.inf, math.inf, 0, 1),
            (math.inf, math.inf, math.inf, 0),
        )
        filtration = filtered_hyperdigraph_from_distance_matrix(
            distances, max_path_dimension=2
        )
        fast = compute_persistence(filtration, 1)
        standard = compute_persistence(
            filtration, 1, reduction_backend="standard"
        )
        direct = compute_persistence_from_distance_matrix(distances, 1)
        self.assertEqual(_signature(fast), _signature(standard))
        self.assertEqual(_signature(fast), _signature(direct))

    def test_inconsistent_minimum_and_maximum_are_rejected(self) -> None:
        adjacency = ((0, 0), (0, 0))
        minimum = ((0, 1), (0, 0))
        maximum = ((0, 0), (0, 0))
        with self.assertRaisesRegex(ValueError, "contained"):
            hyperdigraph_from_adjacency_matrix(
                adjacency,
                minimum_adjacency=minimum,
                maximum_adjacency=maximum,
            )


if __name__ == "__main__":
    unittest.main()
