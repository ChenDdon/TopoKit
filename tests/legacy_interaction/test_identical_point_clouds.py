from __future__ import annotations

from collections import Counter
from itertools import combinations
from math import dist, inf, sqrt
import unittest

from topokit.core._interaction import (
    SimplicialComplexBuilder,
    build_interaction_chain_complex,
    compute_persistence,
)


def vietoris_rips_factor(points, maximum_dimension):
    builder = SimplicialComplexBuilder()
    for dimension in range(maximum_dimension + 1):
        for simplex in combinations(range(len(points)), dimension + 1):
            birth = 0.0 if dimension == 0 else max(
                dist(points[left], points[right])
                for left, right in combinations(simplex, 2)
            )
            builder.insert(simplex, birth, with_faces=False)
    return builder.freeze()


def barcode_signature(result):
    return Counter(
        (interval.dimension, interval.birth, interval.death)
        for interval in result.intervals
    )


class IdenticalPointCloudTests(unittest.TestCase):
    def test_known_fully_overlapping_vr_examples(self) -> None:
        fixtures = (
            (
                ((0.0, 0.0), (1.0, 0.0)),
                1,
                1,
                (2, 4, 1),
                Counter({(0, 0.0, 1.0): 2, (1, 1.0, inf): 1}),
            ),
            (
                ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
                2,
                2,
                (3, 12, 15, 6),
                Counter(
                    {
                        (0, 0.0, 1.0): 3,
                        (1, 1.0, sqrt(2.0)): 1,
                        (2, sqrt(2.0), inf): 1,
                    }
                ),
            ),
            (
                ((0.0, 0.0), (1.0, 0.0), (3.0, 0.0)),
                2,
                2,
                (3, 12, 15, 6),
                Counter(
                    {
                        (0, 0.0, 1.0): 2,
                        (0, 0.0, 2.0): 1,
                        (1, 1.0, 3.0): 1,
                        (2, 3.0, inf): 1,
                    }
                ),
            ),
            (
                ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
                3,
                3,
                (4, 24, 54, 56, 28),
                Counter(
                    {
                        (0, 0.0, 1.0): 4,
                        (1, 1.0, sqrt(2.0)): 1,
                        (2, 1.0, sqrt(2.0)): 1,
                        (3, sqrt(2.0), inf): 1,
                    }
                ),
            ),
        )

        for points, factor_dimension, homology_dimension, counts, intervals in fixtures:
            with self.subTest(points=points):
                factor = vietoris_rips_factor(points, factor_dimension)
                chain = build_interaction_chain_complex(
                    (factor, factor),
                    max_homology_dimension=homology_dimension,
                    validate=True,
                )
                self.assertEqual(chain.diagnostics.interaction_cell_counts, counts)
                self.assertEqual(
                    barcode_signature(compute_persistence(chain)), intervals
                )


if __name__ == "__main__":
    unittest.main()
