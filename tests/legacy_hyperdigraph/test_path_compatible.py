from __future__ import annotations
from topokit.builders._hyperdigraph_compact import score_directed_filtration_from_points

from array import array
from collections import Counter
from itertools import combinations
import random
import unittest

from topokit.core._hyperdigraph import FilteredHyperdigraph, ScoreDirectedEdgeFiltration, compute_path_compatible_persistence, compute_persistence
from topokit.workflows.hyperdigraph import compute_persistence_from_point_cloud
from topokit.builders._hyperdigraph_converters import filtered_hyperdigraph_from_distance_matrix
from topokit.builders._hyperdigraph_compact import score_directed_distance_matrix


def _signature(result):
    return Counter(
        (interval.dimension, interval.birth, interval.death)
        for interval in result.intervals
        if interval.birth < interval.death
    )


class PathCompatiblePersistenceTests(unittest.TestCase):
    def _compact(self, vertex_count, edges):
        ordered = sorted(
            (float(weight), (source << 16) | target) for source, target, weight in edges
        )
        return ScoreDirectedEdgeFiltration(
            vertex_count,
            array("I", (pair for _, pair in ordered)),
            array("d", (weight for weight, _ in ordered)),
            tuple(range(vertex_count)),
            "unit_test",
            len({weight for weight, _ in ordered}) + 1,
        )

    def test_delaunay_is_default_and_complete_remains_available(self):
        points = ((0, 0), (1, 0), (0, 2), (3, 0))
        scores = (3, 0, 2, 1)
        filtration = score_directed_filtration_from_points(
            points,
            scores,
            storage_backend="python",
        )
        result = compute_path_compatible_persistence(filtration, 0)
        self.assertEqual(filtration.connection_support, "delaunay")
        self.assertEqual(filtration.edge_count, 5)
        self.assertEqual(result.diagnostics.connection_support, "delaunay")
        self.assertEqual(result.diagnostics.support_edge_count, 5)
        self.assertEqual(len(result.in_dimension(0)), 4)
        self.assertEqual(sum(i.is_infinite for i in result.in_dimension(0)), 1)
        self.assertLessEqual(result.diagnostics.reduction_column_count, 5)

        complete = score_directed_filtration_from_points(
            points,
            scores,
            connection_support="complete",
            storage_backend="python",
        )
        self.assertEqual(complete.connection_support, "complete")
        self.assertEqual(complete.edge_count, 6)

    def test_delaunay_support_edges_and_storage_backends(self):
        points = ((0.0, 0.0), (2.0, 0.0), (0.0, 1.0), (2.0, 2.0))
        scores = (0.0, 1.0, 2.0, 3.0)
        python = score_directed_filtration_from_points(
            points, scores, storage_backend="python"
        )
        numpy = score_directed_filtration_from_points(
            points, scores, storage_backend="numpy"
        )
        self.assertEqual(tuple(python.packed_pairs), tuple(numpy.packed_pairs))
        self.assertEqual(tuple(python.edge_weights), tuple(numpy.edge_weights))
        decoded = {
            (
                python.score_order[int(pair) >> 16],
                python.score_order[int(pair) & 0xFFFF],
            )
            for pair in python.packed_pairs
        }
        self.assertEqual(
            decoded,
            {(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)},
        )
        self.assertEqual(python.ambient_dimension, 2)
        self.assertEqual(python.intrinsic_dimension, 2)
        self.assertEqual(python.support_maximal_simplex_count, 2)

    def test_affine_rank_projection_and_duplicate_policy(self):
        collinear = score_directed_filtration_from_points(
            ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (3.0, 3.0, 3.0)),
            (2.0, 0.0, 1.0),
            storage_backend="python",
        )
        self.assertEqual(collinear.edge_count, 2)
        self.assertEqual(collinear.intrinsic_dimension, 1)
        self.assertTrue(collinear.projected_to_affine_hull)

        with self.assertRaisesRegex(ValueError, "duplicates"):
            score_directed_filtration_from_points(
                ((0.0, 0.0), (0.0, 0.0), (1.0, 0.0)),
                (0.0, 1.0, 2.0),
            )
        complete = score_directed_filtration_from_points(
            ((0.0, 0.0), (0.0, 0.0), (1.0, 0.0)),
            (0.0, 1.0, 2.0),
            connection_support="complete",
            storage_backend="python",
        )
        self.assertEqual(complete.edge_count, 3)

    def test_delaunay_fast_h0_h1_matches_normal_omega_method(self):
        points = ((0.0, 0.0), (2.0, 0.0), (0.0, 1.0), (2.0, 2.0))
        scores = (0.0, 1.0, 2.0, 3.0)
        compact = score_directed_filtration_from_points(points, scores)
        fast = compute_path_compatible_persistence(compact, 1)
        native = compute_persistence(
            filtered_hyperdigraph_from_distance_matrix(
                score_directed_distance_matrix(compact),
                max_path_dimension=2,
            ),
            1,
            omega_backend="generic",
            reduction_backend="standard",
        )
        self.assertEqual(_signature(fast), _signature(native))

        wrapped_normal = compute_persistence_from_point_cloud(
            points,
            scores,
            1,
            omega_backend="generic",
            reduction_backend="standard",
        )
        self.assertEqual(_signature(fast), _signature(wrapped_normal))
        self.assertEqual(
            wrapped_normal.diagnostics.low_dimensional_backend,
            "standard_filtered_chain_reduction",
        )

        complete_h0 = compute_persistence_from_point_cloud(
            points, scores, 0, connection_support="complete"
        )
        delaunay_h0 = compute_persistence_from_point_cloud(points, scores, 0)
        self.assertEqual(_signature(delaunay_h0), _signature(complete_h0))

    def test_seeded_delaunay_clouds_match_normal_method_and_h0_complete(self):
        for vertex_count in range(3, 8):
            for seed in range(6):
                random_source = random.Random(50_000 * vertex_count + seed)
                points = tuple(
                    (random_source.random(), random_source.random())
                    for _ in range(vertex_count)
                )
                scores = list(range(vertex_count))
                random_source.shuffle(scores)
                compact = score_directed_filtration_from_points(points, scores)
                fast = compute_path_compatible_persistence(compact, 1)
                native = compute_persistence(
                    filtered_hyperdigraph_from_distance_matrix(
                        score_directed_distance_matrix(compact),
                        max_path_dimension=2,
                    ),
                    1,
                    omega_backend="generic",
                    reduction_backend="standard",
                )
                self.assertEqual(_signature(fast), _signature(native))

                delaunay_h0 = compute_path_compatible_persistence(compact, 0)
                complete_h0 = compute_persistence_from_point_cloud(
                    points, scores, 0, connection_support="complete"
                )
                self.assertEqual(_signature(delaunay_h0), _signature(complete_h0))

    def test_point_cloud_higher_dimension_uses_normal_reduction(self):
        points = ((0.0, 0.0), (2.0, 0.0), (0.0, 1.0), (2.0, 2.0))
        scores = (0.0, 1.0, 2.0, 3.0)
        result = compute_persistence_from_point_cloud(points, scores, 2)
        self.assertEqual(result.max_dimension, 2)
        self.assertEqual(result.diagnostics.connection_support, "delaunay")
        self.assertEqual(
            result.diagnostics.higher_dimensional_backend,
            "ordinary_filtered_boundary_reduction",
        )

    def test_delaunay_changes_h1_but_not_h0_on_known_fixture(self):
        points = ((0.0, 0.0), (5.0, 0.0), (0.0, 1.0), (2.0, 3.0))
        scores = (0.0, 1.0, 3.0, 2.0)
        delaunay = compute_persistence_from_point_cloud(points, scores, 1)
        complete = compute_persistence_from_point_cloud(
            points, scores, 1, connection_support="complete"
        )
        self.assertEqual(
            Counter((item.birth, item.death) for item in delaunay.in_dimension(0)),
            Counter((item.birth, item.death) for item in complete.in_dimension(0)),
        )
        self.assertNotEqual(
            Counter((item.birth, item.death) for item in delaunay.in_dimension(1)),
            Counter((item.birth, item.death) for item in complete.in_dimension(1)),
        )

    def test_invalid_connection_support_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "connection_support"):
            score_directed_filtration_from_points(
                ((0.0,), (1.0,)),
                (0.0, 1.0),
                connection_support="unknown",  # type: ignore[arg-type]
            )

    def test_only_triangle_and_quadrangle_families_are_reported(self):
        filtration = self._compact(
            4,
            (
                (0, 1, 1),
                (1, 3, 2),
                (0, 2, 1),
                (2, 3, 3),
                (0, 3, 5),
            ),
        )
        result = compute_path_compatible_persistence(filtration, 1)
        counts = dict(result.diagnostics.boundary_generator_type_counts)
        self.assertEqual(set(counts), {"triangle", "quadrangle"})
        self.assertNotIn("bigon", counts)

    def test_seeded_score_dags_match_materialized_native_omega(self):
        for vertex_count in range(3, 8):
            for seed in range(20):
                random_source = random.Random(10_000 * vertex_count + seed)
                edge_births = {}
                edges = []
                for source, target in combinations(range(vertex_count), 2):
                    if random_source.random() < 0.75:
                        birth = float(random_source.randrange(1, 8))
                        edge_births[source, target] = birth
                        edges.append((source, target, birth))

                compact = self._compact(vertex_count, edges)
                implicit = compute_path_compatible_persistence(compact, 1)
                weighted = [
                    ((source, target), birth)
                    for (source, target), birth in edge_births.items()
                ]
                for source, middle, target in combinations(range(vertex_count), 3):
                    if (source, middle) in edge_births and (
                        middle,
                        target,
                    ) in edge_births:
                        weighted.append(
                            (
                                (source, middle, target),
                                max(
                                    edge_births[source, middle],
                                    edge_births[middle, target],
                                ),
                            )
                        )
                native = compute_persistence(
                    FilteredHyperdigraph(
                        range(vertex_count), weighted, include_all_vertices=True
                    ),
                    1,
                    omega_backend="generic",
                    reduction_backend="standard",
                )
                self.assertEqual(_signature(implicit), _signature(native))


if __name__ == "__main__":
    unittest.main()
