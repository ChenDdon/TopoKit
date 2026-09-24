from __future__ import annotations

import itertools
import math
import unittest

import numpy as np

from topokit.core._hyperdigraph import FilteredHyperdigraph, Hyperdigraph, compute_homology, compute_laplacian, compute_laplacian_filtration, compute_persistence, compute_persistent_laplacian
from topokit.workflows.hyperdigraph import compute_laplacian_filtration_from_distance_matrix, compute_laplacian_filtration_from_point_cloud, compute_laplacian_from_adjacency_matrix, compute_laplacian_from_dimension_dict, compute_persistent_laplacian_from_distance_matrix, compute_persistent_laplacian_from_point_cloud
from topokit.builders._hyperdigraph_converters import filtered_hyperdigraph_from_distance_matrix, hyperdigraph_from_adjacency_matrix


def _spectra(result):
    return tuple(dimension.eigenvalues for dimension in result.dimensions)


class StaticLaplacianTests(unittest.TestCase):
    def test_complete_ordered_three_vertex_published_spectra(self) -> None:
        vertices = (0, 1, 2)
        edges = tuple(
            (source, target)
            for source in vertices
            for target in vertices
            if source != target
        )
        triples = tuple(itertools.permutations(vertices, 3))
        hyperdigraph = Hyperdigraph(
            vertices, edges + triples, include_all_vertices=True
        )
        result = compute_laplacian(hyperdigraph, 2)
        expected = (
            (0.0, 6.0, 6.0),
            (1.0, 4.0, 4.0, 6.0, 6.0, 9.0),
            (0.0, 0.0, 1.0, 4.0, 4.0, 9.0),
        )
        for actual, wanted in zip(_spectra(result), expected):
            np.testing.assert_allclose(actual, wanted, atol=1.0e-9)
        self.assertEqual(result.betti_numbers, (1, 0, 2))
        self.assertEqual(
            result.betti_numbers,
            compute_homology(hyperdigraph, 2).betti_numbers,
        )

    def test_filled_and_unfilled_triangle_nullities(self) -> None:
        edges = ((0, 1), (1, 2), (0, 2))
        unfilled = Hyperdigraph((0, 1, 2), edges, include_all_vertices=True)
        filled = Hyperdigraph(
            (0, 1, 2), edges + ((0, 1, 2),), include_all_vertices=True
        )
        self.assertEqual(compute_laplacian(unfilled, 1).betti_numbers, (1, 1))
        self.assertEqual(compute_laplacian(filled, 1).betti_numbers, (1, 0))

    def test_published_missing_zero_hyperedges(self) -> None:
        hyperdigraph = Hyperdigraph((0, 1), ((0, 1), (1, 0)))
        result = compute_laplacian(hyperdigraph, 1)
        self.assertEqual(result.betti_numbers, (0, 1))
        self.assertEqual(result.in_dimension(0).omega_dimension, 0)
        self.assertEqual(result.in_dimension(1).omega_dimension, 1)

    def test_single_missing_face_structural_qr_matches_svd_with_signs(self) -> None:
        # The common absent face (0,2) has opposite boundary signs in the two
        # triples, so this also audits the real-oriented grouping formula.
        hyperdigraph = Hyperdigraph(
            (0, 1, 2, 3),
            (
                (1, 2),
                (0, 1),
                (3, 2),
                (3, 0),
                (0, 1, 2),
                (3, 0, 2),
            ),
            include_all_vertices=True,
        )
        structural = compute_laplacian(hyperdigraph, 1)
        reference = compute_laplacian(hyperdigraph, 1, omega_backend="svd")
        self.assertEqual(
            structural.diagnostics.chain_dimensions[2].omega_backend,
            "single_missing_face_structural_qr",
        )
        self.assertEqual(structural.diagnostics.chain_dimensions[2].omega_dimension, 1)
        for actual, wanted in zip(_spectra(structural), _spectra(reference)):
            np.testing.assert_allclose(actual, wanted, atol=1.0e-9)

    def test_multi_missing_face_uses_general_svd(self) -> None:
        hyperdigraph = Hyperdigraph(
            (0, 1, 2, 3),
            (
                (1, 2),
                (2, 3),
                (3, 1),
                (0, 1, 2),
                (0, 2, 3),
                (0, 3, 1),
            ),
            include_all_vertices=True,
        )
        result = compute_laplacian(hyperdigraph, 1)
        omega2 = result.diagnostics.chain_dimensions[2]
        self.assertEqual(omega2.max_missing_face_count, 2)
        self.assertEqual(omega2.omega_backend, "svd_constraint_nullspace")
        self.assertEqual(omega2.omega_dimension, 1)

    def test_higher_dimension_uses_normal_way(self) -> None:
        vertices = (0, 1, 2, 3)
        sphere = Hyperdigraph(
            vertices,
            tuple(
                face
                for length in range(1, 4)
                for face in itertools.combinations(vertices, length)
            ),
        )
        result = compute_laplacian(sphere, 2)
        self.assertEqual(result.betti_numbers, (1, 0, 1))
        self.assertEqual(
            result.diagnostics.higher_dimensional_backend,
            "face_closed_or_empty_normal_way",
        )

    def test_non_face_closed_higher_dimension_uses_svd(self) -> None:
        vertices = (0, 1, 2, 3)
        lower_faces = tuple(
            face
            for length in range(1, 4)
            for face in itertools.combinations(vertices, length)
            if face != (1, 2, 3)
        )
        hyperdigraph = Hyperdigraph(vertices, lower_faces + (vertices,))
        result = compute_laplacian(hyperdigraph, 2)
        self.assertEqual(
            result.diagnostics.chain_dimensions[3].omega_backend,
            "svd_constraint_nullspace",
        )
        self.assertEqual(
            result.diagnostics.higher_dimensional_backend,
            "svd_normal_way",
        )
        self.assertEqual(
            result.betti_numbers,
            compute_homology(hyperdigraph, 2).betti_numbers,
        )

    def test_optional_matrix_and_legacy_statistics(self) -> None:
        hyperdigraph = Hyperdigraph(
            (0, 1, 2),
            ((0, 1), (1, 2), (0, 2)),
            include_all_vertices=True,
        )
        result = compute_laplacian(hyperdigraph, 1, return_matrices=True)
        for dimension in result.dimensions:
            self.assertIsNotNone(dimension.matrix)
            matrix = np.asarray(dimension.matrix)
            np.testing.assert_allclose(matrix, matrix.T, atol=1.0e-12)
            self.assertAlmostEqual(dimension.trace, float(np.trace(matrix)))
            self.assertGreaterEqual(dimension.nonzero_standard_deviation, 0.0)


class PersistentLaplacianTests(unittest.TestCase):
    def setUp(self) -> None:
        self.filtration = FilteredHyperdigraph(
            (0, 1, 2),
            (
                ((0, 1), 1),
                ((1, 2), 1),
                ((0, 2), 2),
                ((0, 1, 2), 3),
            ),
            include_all_vertices=True,
        )

    def test_pairwise_nullities_track_directed_triangle_bar(self) -> None:
        alive = compute_persistent_laplacian(self.filtration, 2, 2, 1)
        dead = compute_persistent_laplacian(self.filtration, 2, 3, 1)
        self.assertEqual(alive.persistent_betti_numbers, (1, 1))
        self.assertEqual(dead.persistent_betti_numbers, (1, 0))
        self.assertIn("pairwise_restriction", dead.diagnostics.low_dimensional_backend)

    def test_pairwise_quadrangle_death_uses_structural_omega2(self) -> None:
        filtration = FilteredHyperdigraph(
            (0, 1, 2, 3),
            (
                ((0, 1), 1),
                ((1, 3), 1),
                ((0, 2), 1),
                ((2, 3), 1),
                ((0, 1, 3), 2),
                ((0, 2, 3), 2),
            ),
            include_all_vertices=True,
        )
        result = compute_persistent_laplacian(filtration, 1, 2, 1)
        self.assertEqual(result.persistent_betti_numbers, (1, 0))
        self.assertEqual(
            result.diagnostics.end_chain_dimensions[2].omega_backend,
            "single_missing_face_structural_qr",
        )

    def test_pairwise_general_circuit_death_uses_svd(self) -> None:
        filtration = FilteredHyperdigraph(
            (0, 1, 2, 3),
            (
                ((1, 2), 0),
                ((2, 3), 0),
                ((3, 1), 0),
                ((0, 1, 2), 1),
                ((0, 2, 3), 2),
                ((0, 3, 1), 3),
            ),
            include_all_vertices=True,
        )
        alive = compute_persistent_laplacian(filtration, 0, 2, 1)
        dead = compute_persistent_laplacian(filtration, 0, 3, 1)
        self.assertEqual(alive.in_dimension(1).nullity, 1)
        self.assertEqual(dead.in_dimension(1).nullity, 0)
        self.assertEqual(
            dead.diagnostics.end_chain_dimensions[2].omega_backend,
            "svd_constraint_nullspace",
        )

    def test_all_pair_nullities_match_barcode_on_torsion_free_fixture(self) -> None:
        barcode = compute_persistence(self.filtration, 1)
        thresholds = self.filtration.thresholds(2)
        for start_index, start in enumerate(thresholds):
            for end in thresholds[start_index:]:
                laplacian = compute_persistent_laplacian(self.filtration, start, end, 1)
                self.assertEqual(
                    laplacian.persistent_betti_numbers,
                    tuple(
                        barcode.persistent_betti(dimension, start, end)
                        for dimension in range(2)
                    ),
                )

    def test_diagonal_pair_equals_ordinary_snapshot(self) -> None:
        for threshold in self.filtration.thresholds(2):
            ordinary = compute_laplacian(
                self.filtration.snapshot(threshold),
                1,
                return_matrices=True,
            )
            persistent = compute_persistent_laplacian(
                self.filtration,
                threshold,
                threshold,
                1,
                return_matrices=True,
            )
            self.assertEqual(
                persistent.persistent_betti_numbers, ordinary.betti_numbers
            )
            for pair_dimension, ordinary_dimension in zip(
                persistent.dimensions, ordinary.dimensions
            ):
                np.testing.assert_allclose(
                    pair_dimension.matrix,
                    ordinary_dimension.matrix,
                    atol=1.0e-9,
                )

    def test_persistent_l0_restricts_end_edges_to_start_vertices(self) -> None:
        filtration = FilteredHyperdigraph(
            (0, 1),
            (
                ((0,), 0),
                ((1,), 0),
                ((0, 1), 1),
            ),
        )
        result = compute_persistent_laplacian(filtration, 0, 1, 0)
        self.assertEqual(result.persistent_betti_numbers, (1,))

    def test_snapshot_scan_is_distinct_from_pairwise_operator(self) -> None:
        scan = compute_laplacian_filtration(self.filtration, 1)
        self.assertEqual(scan.thresholds, (0.0, 1.0, 2.0, 3.0))
        self.assertEqual(scan.at(2.0).betti_numbers, (1, 1))
        self.assertEqual(scan.at(3.0).betti_numbers, (1, 0))
        with self.assertRaises(KeyError):
            scan.at(2.5)

    def test_pairwise_h2_sphere_death_uses_normal_way(self) -> None:
        vertices = (0, 1, 2, 3)
        weighted = [
            (face, 0)
            for length in range(1, 4)
            for face in itertools.combinations(vertices, length)
        ]
        weighted.append((vertices, 2))
        filtration = FilteredHyperdigraph(vertices, weighted)
        alive = compute_persistent_laplacian(filtration, 0, 0, 2)
        dead = compute_persistent_laplacian(filtration, 0, 2, 2)
        self.assertEqual(alive.in_dimension(2).nullity, 1)
        self.assertEqual(dead.in_dimension(2).nullity, 0)


class LaplacianAdapterTests(unittest.TestCase):
    def test_legacy_dimension_dictionary(self) -> None:
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
        result = compute_laplacian_from_dimension_dict(legacy, 1)
        self.assertEqual(result.betti_numbers, (1, 2))

    def test_adjacency_adapter_matches_explicit_conversion(self) -> None:
        adjacency = (
            (0, 1, 0, 0, 1, 0),
            (0, 0, 1, 0, 0, 0),
            (0, 0, 0, 1, 0, 0),
            (0, 0, 0, 0, 0, 0),
            (0, 0, 1, 0, 0, 0),
            (0, 0, 0, 0, 0, 0),
        )
        direct = compute_laplacian_from_adjacency_matrix(adjacency, 1)
        explicit = compute_laplacian(
            hyperdigraph_from_adjacency_matrix(adjacency, max_path_dimension=2),
            1,
        )
        self.assertEqual(direct.betti_numbers, (2, 0))
        for actual, wanted in zip(_spectra(direct), _spectra(explicit)):
            np.testing.assert_allclose(actual, wanted, atol=1.0e-9)

    def test_distance_pair_and_scan_adapters(self) -> None:
        distances = (
            (0, 1, 2),
            (math.inf, 0, 1),
            (math.inf, math.inf, 0),
        )
        filtration = filtered_hyperdigraph_from_distance_matrix(
            distances, max_path_dimension=2
        )
        direct_pair = compute_persistent_laplacian(filtration, 1, 2, 1)
        adapted_pair = compute_persistent_laplacian_from_distance_matrix(
            distances, 1, 2, 1
        )
        self.assertEqual(
            direct_pair.persistent_betti_numbers,
            adapted_pair.persistent_betti_numbers,
        )
        for actual, wanted in zip(_spectra(direct_pair), _spectra(adapted_pair)):
            np.testing.assert_allclose(actual, wanted, atol=1.0e-9)

        scan = compute_laplacian_filtration_from_distance_matrix(
            distances, 1, thresholds=(0, 1, 1.5, 2)
        )
        self.assertEqual(scan.thresholds, (0.0, 1.0, 1.5, 2.0))
        self.assertEqual(scan.at(1.0).betti_numbers, scan.at(1.5).betti_numbers)

    def test_score_directed_point_cloud_adapters(self) -> None:
        points = ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))
        scores = (0.0, 1.0, 2.0)
        scan = compute_laplacian_filtration_from_point_cloud(
            points,
            scores,
            1,
            storage_backend="python",
        )
        self.assertEqual(scan.thresholds, (0.0, 1.0))
        self.assertEqual(
            scan.snapshots[-1].result.diagnostics.connection_support,
            "delaunay",
        )
        self.assertEqual(
            scan.snapshots[-1].result.diagnostics.retained_edge_count,
            2,
        )
        pair = compute_persistent_laplacian_from_point_cloud(
            points,
            scores,
            1.0,
            2.0,
            1,
            storage_backend="python",
        )
        distances = (
            (0.0, 1.0, math.inf),
            (math.inf, 0.0, 1.0),
            (math.inf, math.inf, 0.0),
        )
        reference = compute_persistent_laplacian_from_distance_matrix(
            distances, 1.0, 2.0, 1
        )
        self.assertEqual(
            pair.persistent_betti_numbers,
            reference.persistent_betti_numbers,
        )
        for actual, wanted in zip(_spectra(pair), _spectra(reference)):
            np.testing.assert_allclose(actual, wanted, atol=1.0e-9)

        complete = compute_laplacian_filtration_from_point_cloud(
            points,
            scores,
            1,
            connection_support="complete",
            storage_backend="python",
        )
        self.assertEqual(complete.thresholds, (0.0, 1.0, 2.0))
        self.assertEqual(
            complete.snapshots[-1].result.diagnostics.connection_support,
            "complete",
        )

    def test_delaunay_laplacian_matches_explicit_supported_distances(self) -> None:
        points = ((0.0, 0.0), (2.0, 0.0), (0.0, 1.0), (2.0, 2.0))
        scores = (0.0, 1.0, 2.0, 3.0)
        root_five = math.sqrt(5.0)
        distances = (
            (0.0, 2.0, 1.0, math.inf),
            (math.inf, 0.0, root_five, 2.0),
            (math.inf, math.inf, 0.0, root_five),
            (math.inf, math.inf, math.inf, 0.0),
        )
        thresholds = (0.0, 1.0, 2.0, root_five)
        automatic = compute_laplacian_filtration_from_point_cloud(
            points, scores, 1, thresholds=thresholds
        )
        reference = compute_laplacian_filtration_from_distance_matrix(
            distances, 1, thresholds=thresholds
        )
        for actual_snapshot, reference_snapshot in zip(
            automatic.snapshots, reference.snapshots
        ):
            self.assertEqual(
                actual_snapshot.result.betti_numbers,
                reference_snapshot.result.betti_numbers,
            )
            for actual, wanted in zip(
                _spectra(actual_snapshot.result),
                _spectra(reference_snapshot.result),
            ):
                np.testing.assert_allclose(actual, wanted, atol=1.0e-9)

    def test_point_cloud_minimum_cannot_escape_delaunay_support(self) -> None:
        points = ((0.0, 0.0), (2.0, 0.0), (0.0, 1.0), (2.0, 2.0))
        scores = (0.0, 1.0, 2.0, 3.0)
        minimum = (
            (0, 0, 0, 1),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
        )
        with self.assertRaisesRegex(ValueError, "point-cloud connection support"):
            compute_laplacian_filtration_from_point_cloud(
                points, scores, 0, minimum_adjacency=minimum
            )

    def test_weighted_point_cloud_l2_row_order_regression(self) -> None:
        points = (
            (0.0, 0.0),
            (0.8, 0.1),
            (1.7, -0.2),
            (2.5, 0.4),
            (3.4, 0.0),
        )
        scores = (3.0, 0.0, 4.0, 1.0, 2.0)
        threshold = 2.566860073242985
        automatic = (
            compute_laplacian_filtration_from_point_cloud(
                points,
                scores,
                2,
                thresholds=(threshold,),
                connection_support="complete",
                omega_backend="auto",
                storage_backend="python",
            )
            .snapshots[0]
            .result
        )
        reference = (
            compute_laplacian_filtration_from_point_cloud(
                points,
                scores,
                2,
                thresholds=(threshold,),
                connection_support="complete",
                omega_backend="svd",
                storage_backend="python",
            )
            .snapshots[0]
            .result
        )
        expected_l2 = (2.0, 4.0, 4.0, 4.0, 5.0)
        np.testing.assert_allclose(
            automatic.in_dimension(2).eigenvalues,
            expected_l2,
            atol=1.0e-9,
        )
        np.testing.assert_allclose(
            automatic.in_dimension(2).eigenvalues,
            reference.in_dimension(2).eigenvalues,
            atol=1.0e-9,
        )


class LaplacianValidationTests(unittest.TestCase):
    def test_invalid_pair_and_options_are_rejected(self) -> None:
        filtration = FilteredHyperdigraph(
            (0, 1), (((0, 1), 1),), include_all_vertices=True
        )
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            compute_persistent_laplacian(filtration, 2, 1)
        with self.assertRaisesRegex(ValueError, "omega_backend"):
            compute_laplacian(
                filtration.snapshot(1), 0, omega_backend="unknown"  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(ValueError, "tolerance"):
            compute_laplacian(filtration.snapshot(1), 0, tolerance=0)


if __name__ == "__main__":
    unittest.main()
