from __future__ import annotations

import unittest

from topokit.core._interaction import (
    FilteredSimplicialComplex,
    SimplicialComplexBuilder,
)


class FactorModelTests(unittest.TestCase):
    def test_raw_indexed_construction_is_private(self) -> None:
        with self.assertRaisesRegex(TypeError, "SimplicialComplexBuilder"):
            FilteredSimplicialComplex()

    def test_builder_completes_faces_and_lowers_existing_filtration(self) -> None:
        builder = SimplicialComplexBuilder()
        builder.insert((0, 1, 2), 2.0)
        builder.insert((0,), 0.0)
        complex_ = builder.freeze()

        self.assertEqual(complex_.number_of_simplices, 7)
        self.assertEqual(complex_.filtration(complex_.simplex_id((0,))), 0.0)
        self.assertEqual(complex_.filtration(complex_.simplex_id((1,))), 2.0)
        triangle_id = complex_.simplex_id((2, 1, 0))
        self.assertIsNotNone(triangle_id)
        self.assertEqual(
            tuple(complex_.simplex(face) for face in complex_.face_ids[triangle_id]),
            ((1, 2), (0, 2), (0, 1)),
        )

    def test_dimension_dictionary_requires_closure_by_default(self) -> None:
        with self.assertRaisesRegex(ValueError, "not face-closed"):
            FilteredSimplicialComplex.from_dimension_dict(
                {1: [(0, 1)]}, {1: [1.0]}
            )

    def test_dimension_dictionary_rejects_late_face(self) -> None:
        with self.assertRaisesRegex(ValueError, "face is born after"):
            FilteredSimplicialComplex.from_dimension_dict(
                {0: [(0,), (1,)], 1: [(0, 1)]},
                {0: [2.0, 0.0], 1: [1.0]},
            )

    def test_postings_are_partitioned_by_dimension(self) -> None:
        builder = SimplicialComplexBuilder()
        builder.insert((0, 1, 2), 1.0)
        complex_ = builder.freeze()
        edge_ids = set(complex_.postings(1, 0))
        self.assertEqual(
            {complex_.simplex(simplex_id) for simplex_id in edge_ids},
            {(0, 1), (0, 2)},
        )
        self.assertEqual(
            {complex_.simplex(simplex_id) for simplex_id in complex_.postings(2, 0)},
            {(0, 1, 2)},
        )


if __name__ == "__main__":
    unittest.main()
