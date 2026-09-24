from __future__ import annotations

import itertools
import math
import random
import unittest

from topokit.core._hyperdigraph import Hyperdigraph, build_chain_complex, compute_homology


def _boundary(chain: int, basis: tuple[tuple[int, ...], ...]) -> dict[tuple[int, ...], int]:
    result: dict[tuple[int, ...], int] = {}
    for column, hyperedge in enumerate(basis):
        if not ((chain >> column) & 1):
            continue
        if len(hyperedge) == 1:
            # The published complex uses d_0 = 0; there is no empty face.
            continue
        for removed in range(len(hyperedge)):
            face = hyperedge[:removed] + hyperedge[removed + 1 :]
            result[face] = result.get(face, 0) ^ 1
    return {face: value for face, value in result.items() if value}


def _bruteforce_betti(hyperdigraph: Hyperdigraph, max_dimension: int) -> tuple[int, ...]:
    bases = tuple(
        tuple(hyperdigraph.directed_hyperedges(dimension))
        for dimension in range(max_dimension + 2)
    )
    basis_sets = tuple(set(basis) for basis in bases)
    omega: list[tuple[int, ...]] = []
    boundaries: list[set[int]] = []

    for dimension, basis in enumerate(bases):
        omega_chains: list[int] = []
        boundary_outputs: set[int] = set()
        lower_index = (
            {edge: index for index, edge in enumerate(bases[dimension - 1])}
            if dimension
            else {}
        )
        for chain in range(1 << len(basis)):
            ambient = _boundary(chain, basis) if dimension else {}
            if dimension and any(face not in basis_sets[dimension - 1] for face in ambient):
                continue
            omega_chains.append(chain)
            output = 0
            for face in ambient:
                output ^= 1 << lower_index[face]
            boundary_outputs.add(output)
        omega.append(tuple(omega_chains))
        boundaries.append(boundary_outputs)

    result: list[int] = []
    for dimension in range(max_dimension + 1):
        cycles = sum(
            not _boundary(chain, bases[dimension])
            for chain in omega[dimension]
        )
        result.append(int(math.log2(cycles)) - int(math.log2(len(boundaries[dimension + 1]))))
    return tuple(result)


class HomologyTests(unittest.TestCase):
    def test_published_two_way_edge_without_zero_hyperedges(self) -> None:
        hyperdigraph = Hyperdigraph((0, 1), ((0, 1), (1, 0)))
        result = compute_homology(hyperdigraph, 1, representatives=True)
        self.assertEqual(result.betti_numbers, (0, 1))
        self.assertEqual(
            set(result.representatives[1][0]), {(0, 1), (1, 0)}
        )

    def test_unfilled_and_filled_directed_triangle(self) -> None:
        edges = ((0, 1), (1, 2), (0, 2))
        unfilled = Hyperdigraph((0, 1, 2), edges, include_all_vertices=True)
        filled = Hyperdigraph(
            (0, 1, 2), edges + ((0, 1, 2),), include_all_vertices=True
        )
        self.assertEqual(compute_homology(unfilled, 1).betti_numbers, (1, 1))
        self.assertEqual(compute_homology(filled, 1).betti_numbers, (1, 0))

    def test_missing_shortcut_pair_uses_grouping_backend(self) -> None:
        hyperdigraph = Hyperdigraph(
            (0, 1, 2, 3),
            (
                (0, 1),
                (1, 3),
                (0, 2),
                (2, 3),
                (0, 1, 3),
                (0, 2, 3),
            ),
            include_all_vertices=True,
        )
        result = compute_homology(hyperdigraph, 1)
        self.assertEqual(result.betti_numbers, (1, 0))
        self.assertEqual(
            result.diagnostics.dimensions[1].omega_backend, "face_closed_units"
        )
        chain = build_chain_complex(hyperdigraph, 1)
        self.assertEqual(chain.omega_backends[2], "single_missing_face_grouping")
        self.assertEqual(len(chain.omega_generators[2]), 1)

    def test_general_native_example_is_not_path_homology(self) -> None:
        hyperdigraph = Hyperdigraph(
            (1, 2, 3),
            ((1, 2), (1, 3), (3, 2), (1, 2, 3)),
        )
        result = compute_homology(hyperdigraph, 1)
        self.assertEqual(result.betti_numbers, (0, 1))
        chain = build_chain_complex(hyperdigraph, 1)
        self.assertEqual(chain.omega_backends[2], "single_missing_face_grouping")
        self.assertEqual(len(chain.omega_generators[2]), 0)

    def test_published_six_vertex_example(self) -> None:
        hyperdigraph = Hyperdigraph(
            (0, 1, 2, 3, 4, 5),
            (
                (1,), (2,), (3,), (4,), (5,),
                (0, 1), (1, 2), (2, 1), (2, 3), (2, 4), (2, 5),
                (3, 4), (4, 1),
                (0, 1, 2), (0, 5, 1), (2, 3, 4), (2, 4, 1),
            ),
        )
        self.assertEqual(compute_homology(hyperdigraph, 1).betti_numbers, (1, 1))

    def test_complete_ordered_three_vertex_h2(self) -> None:
        vertices = (0, 1, 2)
        edges = tuple((u, v) for u in vertices for v in vertices if u != v)
        triples = tuple(itertools.permutations(vertices, 3))
        hyperdigraph = Hyperdigraph(
            vertices, edges + triples, include_all_vertices=True
        )
        result = compute_homology(hyperdigraph, 2, representatives=True)
        self.assertEqual(result.betti_numbers, (1, 0, 2))
        self.assertEqual(len(result.representatives[2]), 2)

    def test_oriented_tetrahedron_boundary_and_filling(self) -> None:
        vertices = (0, 1, 2, 3)
        proper_faces = tuple(
            combination
            for length in range(1, 4)
            for combination in itertools.combinations(vertices, length)
        )
        sphere = Hyperdigraph(vertices, proper_faces)
        filled = Hyperdigraph(vertices, proper_faces + (vertices,))
        self.assertEqual(compute_homology(sphere, 2).betti_numbers, (1, 0, 1))
        self.assertEqual(compute_homology(filled, 3).betti_numbers, (1, 0, 0, 0))

    def test_seeded_random_small_cases_match_exhaustive_definition(self) -> None:
        rng = random.Random(20260831)
        vertices = (0, 1, 2, 3)
        candidates = {
            dimension: list(itertools.permutations(vertices, dimension + 1))
            for dimension in range(4)
        }
        for _ in range(60):
            selected: list[tuple[int, ...]] = []
            for dimension in range(4):
                count = rng.randrange(0, min(4, len(candidates[dimension])) + 1)
                selected.extend(rng.sample(candidates[dimension], count))
            hyperdigraph = Hyperdigraph(vertices, selected)
            expected = _bruteforce_betti(hyperdigraph, 2)
            actual = compute_homology(
                hyperdigraph, 2, omega_backend="generic"
            ).betti_numbers
            self.assertEqual(actual, expected)
            self.assertEqual(
                compute_homology(hyperdigraph, 2).betti_numbers,
                expected,
            )


if __name__ == "__main__":
    unittest.main()
