from __future__ import annotations

import unittest

from topokit.core._hyperdigraph import FilteredHyperdigraph, Hyperdigraph, WeightedHyperedge


class ModelTests(unittest.TestCase):
    def test_static_singletons_are_explicit_by_default(self) -> None:
        hyperdigraph = Hyperdigraph((0, 1), ((0, 1),))
        self.assertEqual(hyperdigraph.directed_hyperedges(0), ())
        self.assertEqual(hyperdigraph.directed_hyperedges(1), ((0, 1),))
        self.assertEqual(hyperdigraph.singleton_policy, "explicit_only")

    def test_graph_like_singleton_policy(self) -> None:
        hyperdigraph = Hyperdigraph(
            ("a", "b"), (("a", "b"),), include_all_vertices=True
        )
        self.assertEqual(
            hyperdigraph.directed_hyperedges(0), (("a",), ("b",))
        )
        self.assertEqual(hyperdigraph.synthesized_zero_hyperedges, ("a", "b"))

    def test_filtered_snapshot_and_births(self) -> None:
        filtration = FilteredHyperdigraph(
            (0, 1, 2),
            (
                WeightedHyperedge((0, 1), 2.0),
                ((1, 2), 3.0),
            ),
            include_all_vertices=True,
            vertex_birth=1.0,
        )
        self.assertEqual(filtration.thresholds(), (1.0, 2.0, 3.0))
        self.assertEqual(
            filtration.snapshot(2.0).directed_hyperedges(1), ((0, 1),)
        )
        self.assertEqual(filtration.birth_of((1, 2)), 3.0)

    def test_invalid_hyperedges_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot repeat"):
            Hyperdigraph((0, 1), ((0, 1, 0),))
        with self.assertRaisesRegex(ValueError, "undeclared"):
            Hyperdigraph((0, 1), ((0, 2),))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            Hyperdigraph((0, 1), ((0, 1), (0, 1)))
        with self.assertRaisesRegex(TypeError, "ordered"):
            Hyperdigraph({0, 1})


if __name__ == "__main__":
    unittest.main()
