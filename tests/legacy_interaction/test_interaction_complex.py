from __future__ import annotations

import itertools
import unittest

from topokit.core._interaction import (
    SimplicialComplexBuilder,
    build_interaction_chain_complex,
)


def full_simplex(vertices: tuple[int, ...], birth: float = 0.0):
    builder = SimplicialComplexBuilder()
    builder.insert(vertices, birth)
    return builder.freeze()


def brute_force_keys(factors, degree):
    output = set()
    for key in itertools.product(*(range(f.number_of_simplices) for f in factors)):
        if sum(f.dimension(sid) for f, sid in zip(factors, key)) != degree:
            continue
        common = set(factors[0].simplex(key[0]))
        for factor, simplex_id in zip(factors[1:], key[1:]):
            common.intersection_update(factor.simplex(simplex_id))
        if common:
            output.add(key)
    return output


class InteractionConstructionTests(unittest.TestCase):
    def test_edge_tensor_edge_has_one_generator(self) -> None:
        factor = full_simplex((0, 1))
        chain = build_interaction_chain_complex(
            (factor, factor), max_homology_dimension=1, validate=True
        )
        self.assertEqual(chain.diagnostics.interaction_cell_counts, (2, 4, 1))

        edge = factor.simplex_id((0, 1))
        self.assertEqual(chain.degrees[2].keys, ((edge, edge),))
        boundary_keys = {
            chain.degrees[1].keys[row] for row in chain.boundary_rows(2, 0)
        }
        vertex0 = factor.simplex_id((0,))
        vertex1 = factor.simplex_id((1,))
        self.assertEqual(
            boundary_keys,
            {
                (vertex0, edge),
                (vertex1, edge),
                (edge, vertex0),
                (edge, vertex1),
            },
        )

    def test_empty_intersection_facets_are_quotient_zero(self) -> None:
        left = full_simplex((0, 1))
        right = full_simplex((1, 2))
        chain = build_interaction_chain_complex(
            (left, right), max_homology_dimension=1, validate=True
        )
        edge_key = (left.simplex_id((0, 1)), right.simplex_id((1, 2)))
        source = chain.degrees[2].index_of(edge_key)
        self.assertIsNotNone(source)
        boundary_keys = {
            chain.degrees[1].keys[row]
            for row in chain.boundary_rows(2, source, validate_missing=True)
        }
        self.assertEqual(
            boundary_keys,
            {
                (left.simplex_id((1,)), right.simplex_id((1, 2))),
                (left.simplex_id((0, 1)), right.simplex_id((1,))),
            },
        )

    def test_filled_triangle_counts_match_unique_tensor_basis(self) -> None:
        factor = full_simplex((0, 1, 2))
        chain = build_interaction_chain_complex(
            (factor, factor), max_homology_dimension=3, validate=True
        )
        self.assertEqual(
            chain.diagnostics.interaction_cell_counts,
            (3, 12, 15, 6, 1),
        )

    def test_posting_join_matches_cartesian_definition(self) -> None:
        first = full_simplex((0, 1, 2))
        second = full_simplex((1, 2, 3))
        third = full_simplex((0, 2, 3))
        factors = (first, second, third)
        chain = build_interaction_chain_complex(
            factors, max_homology_dimension=2, validate=True
        )
        for degree, layer in enumerate(chain.degrees):
            self.assertEqual(set(layer.keys), brute_force_keys(factors, degree))

    def test_pairwise_intersections_do_not_imply_three_factor_interaction(self) -> None:
        factors = (
            full_simplex((0, 1)),
            full_simplex((1, 2)),
            full_simplex((0, 2)),
        )
        chain = build_interaction_chain_complex(
            factors, max_homology_dimension=2, validate=True
        )
        facets = tuple(
            factor.simplex_id(vertices)
            for factor, vertices in zip(factors, ((0, 1), (1, 2), (0, 2)))
        )
        self.assertIsNone(chain.degrees[3].index_of(facets))
        self.assertNotIn(facets, brute_force_keys(factors, 3))

    def test_disjoint_factors_produce_the_zero_chain(self) -> None:
        factors = (full_simplex((0, 1)), full_simplex((2, 3)))
        chain = build_interaction_chain_complex(
            factors, max_homology_dimension=2, validate=True
        )
        self.assertEqual(chain.diagnostics.interaction_cell_counts, (0, 0, 0, 0))

    def test_signed_boundary_squares_to_zero_and_reduces_mod_two(self) -> None:
        triangle = full_simplex((0, 1, 2))
        for factors in ((triangle,), (triangle, triangle), (triangle,) * 3):
            chain = build_interaction_chain_complex(
                factors, max_homology_dimension=3, validate=True
            )
            chain.validate_signed_boundary()
            for degree, layer in enumerate(chain.degrees):
                for index in range(len(layer.keys)):
                    entries = chain.signed_boundary_entries(degree, index)
                    self.assertEqual(
                        tuple(row for row, value in entries if value % 2),
                        chain.boundary_rows(degree, index),
                    )

    def test_signed_edge_tensor_edge_koszul_sign(self) -> None:
        factor = full_simplex((0, 1))
        chain = build_interaction_chain_complex(
            (factor, factor), max_homology_dimension=1
        )
        edge = factor.simplex_id((0, 1))
        v0 = factor.simplex_id((0,))
        v1 = factor.simplex_id((1,))
        self.assertEqual(
            {chain.degrees[1].keys[row]: value
             for row, value in chain.signed_boundary_entries(2, 0)},
            {(v1, edge): 1, (v0, edge): -1, (edge, v1): -1, (edge, v0): 1},
        )

    def test_signed_boundary_rejects_invalid_indexes(self) -> None:
        chain = build_interaction_chain_complex(
            (full_simplex((0, 1)),), max_homology_dimension=0
        )
        for degree, index in ((-1, 0), (2, 0), (0, -1), (0, 2)):
            with self.assertRaises(IndexError):
                chain.signed_boundary_entries(degree, index)
        for degree, index in ((True, 0), (0, False), (0.0, 0), (0, 0.0)):
            with self.assertRaises(TypeError):
                chain.signed_boundary_entries(degree, index)


if __name__ == "__main__":
    unittest.main()
