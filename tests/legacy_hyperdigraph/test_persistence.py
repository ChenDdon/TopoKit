from __future__ import annotations

import itertools
import math
import random
import unittest

from topokit.core._hyperdigraph import FilteredHyperdigraph, build_chain_complex, build_filtered_chain_complex, compute_homology, compute_persistence
from topokit.core._hyperdigraph.gf2 import (
    apply_columns,
    iter_set_bits,
    nullspace_basis,
    rank,
    transpose_columns,
)


def _signature(result):
    return tuple(
        (interval.dimension, interval.birth, interval.death)
        for interval in result.intervals
    )


def _lift_to_final(vector, local_hyperedges, final_index):
    lifted = 0
    for local_index in iter_set_bits(vector):
        lifted ^= 1 << final_index[local_hyperedges[local_index]]
    return lifted


def _cycles_and_boundaries_in_final(snapshot, dimension, final_index):
    """Return Z_p and B_p in the final explicit F_p coordinate system."""

    chain = build_chain_complex(snapshot, 2, omega_backend="generic")
    local_hyperedges = chain.hyperedges[dimension]
    cycle_equations = transpose_columns(
        chain.boundary_columns[dimension],
        len(chain.omega_generators[dimension - 1]) if dimension else 0,
    )
    cycles_in_omega = nullspace_basis(
        cycle_equations, len(chain.omega_generators[dimension])
    )
    cycles = tuple(
        _lift_to_final(
            apply_columns(vector, chain.omega_generators[dimension]),
            local_hyperedges,
            final_index,
        )
        for vector in cycles_in_omega
    )
    boundaries = tuple(
        _lift_to_final(
            apply_columns(vector, chain.omega_generators[dimension]),
            local_hyperedges,
            final_index,
        )
        for vector in chain.boundary_columns[dimension + 1]
    )
    return cycles, boundaries


def _direct_omega_dimension(filtration, dimension, threshold):
    """Compute dim Omega_p(t) directly from forbidden deletion faces."""

    active = tuple(
        record
        for record in filtration.weighted_hyperedges(dimension)
        if record.birth <= threshold
    )
    forbidden_rows = {}
    for column, record in enumerate(active):
        for removed in range(len(record.vertices)):
            face = record.vertices[:removed] + record.vertices[removed + 1 :]
            face_birth = filtration.birth_of(face)
            if face_birth is None or face_birth > threshold:
                forbidden_rows[face] = forbidden_rows.get(face, 0) ^ (1 << column)
    return len(active) - rank(forbidden_rows.values())


class PersistenceTests(unittest.TestCase):
    def test_directed_triangle_interval(self) -> None:
        filtration = FilteredHyperdigraph(
            (0, 1, 2),
            (
                ((0, 1), 1),
                ((1, 2), 1),
                ((0, 2), 2),
                ((0, 1, 2), 3),
            ),
            include_all_vertices=True,
        )
        result = compute_persistence(filtration, 1)
        self.assertIn((1, 2.0, 3.0), _signature(result))
        self.assertEqual(
            result.diagnostics.low_dimensional_backend,
            "modified_dlw_native_omega2",
        )

    def test_native_quadrangle_interval_and_standard_crosscheck(self) -> None:
        filtration = FilteredHyperdigraph(
            (0, 1, 2, 3),
            (
                ((0, 1), 1), ((1, 3), 1),
                ((0, 2), 1), ((2, 3), 1),
                ((0, 1, 3), 2), ((0, 2, 3), 2),
            ),
            include_all_vertices=True,
        )
        fast = compute_persistence(filtration, 1)
        standard = compute_persistence(
            filtration, 1, reduction_backend="standard"
        )
        self.assertEqual(_signature(fast), _signature(standard))
        self.assertIn((1, 1.0, 2.0), _signature(fast))
        self.assertEqual(
            fast.diagnostics.omega_backends[2],
            "single_missing_face_grouping",
        )

    def test_general_late_faces_use_kernel_and_drop_diagonal(self) -> None:
        filtration = FilteredHyperdigraph(
            (0, 1, 2),
            (
                ((0, 1, 2), 0),
                ((0, 1), 1),
                ((1, 2), 2),
                ((0, 2), 3),
            ),
            include_all_vertices=True,
        )
        clean = compute_persistence(filtration, 2)
        raw = compute_persistence(filtration, 2, include_diagonal=True)
        self.assertEqual(
            clean.diagnostics.omega_backends[2],
            "general_circuit_column_reduction",
        )
        self.assertNotIn((1, 3.0, 3.0), _signature(clean))
        self.assertIn((1, 3.0, 3.0), _signature(raw))

    def test_three_hyperedge_circuit_is_required_and_detected(self) -> None:
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
        fast = compute_persistence(filtration, 1)
        standard = compute_persistence(
            filtration, 1, reduction_backend="standard"
        )
        self.assertEqual(_signature(fast), _signature(standard))
        self.assertIn((1, 0.0, 3.0), _signature(fast))
        self.assertEqual(
            fast.diagnostics.omega_backends[2],
            "general_circuit_column_reduction",
        )
        self.assertEqual(fast.diagnostics.omega2_max_late_faces_at_birth, 2)
        self.assertEqual(
            dict(fast.diagnostics.omega2_generator_type_counts),
            {"multi_hyperedge_circuit": 1},
        )

    def test_one_pass_compatible_basis_matches_every_direct_snapshot(self) -> None:
        rng = random.Random(9012026)
        vertices = tuple(range(4))
        all_edges = list(itertools.permutations(vertices, 2))
        all_triples = list(itertools.permutations(vertices, 3))
        for _ in range(40):
            weighted = [
                (edge, rng.randrange(4))
                for edge in rng.sample(all_edges, rng.randrange(7))
            ]
            weighted.extend(
                (triple, rng.randrange(4))
                for triple in rng.sample(all_triples, rng.randrange(7))
            )
            filtration = FilteredHyperdigraph(
                vertices, weighted, include_all_vertices=True
            )
            chain = build_filtered_chain_complex(
                filtration, 1, omega_backend="generic"
            )
            self.assertEqual(
                chain.omega_backends[2],
                "general_circuit_column_reduction",
            )
            for threshold in filtration.thresholds(2):
                active_generators = tuple(
                    generator
                    for generator, birth in zip(
                        chain.omega_generators[2], chain.omega_births[2]
                    )
                    if birth <= threshold
                )
                self.assertEqual(rank(active_generators), len(active_generators))
                self.assertEqual(
                    len(active_generators),
                    _direct_omega_dimension(filtration, 2, threshold),
                )

    def test_h2_sphere_dies_when_tetrahedron_appears(self) -> None:
        vertices = (0, 1, 2, 3)
        weighted = [
            (combination, 0)
            for length in range(1, 4)
            for combination in itertools.combinations(vertices, length)
        ]
        weighted.append((vertices, 2))
        filtration = FilteredHyperdigraph(vertices, weighted)
        result = compute_persistence(filtration, 2)
        self.assertIn((2, 0.0, 2.0), _signature(result))
        self.assertEqual(
            result.diagnostics.low_dimensional_backend,
            "modified_dlw_native_omega2",
        )
        self.assertEqual(
            result.diagnostics.higher_dimensional_backend,
            "ordinary_filtered_boundary_reduction",
        )
        self.assertEqual(
            _signature(result),
            _signature(
                compute_persistence(
                    filtration, 2, reduction_backend="standard"
                )
            ),
        )
        self.assertEqual(
            _signature(result),
            _signature(
                compute_persistence(
                    filtration, 2, reduction_backend="native_h1"
                )
            ),
        )

    def test_snapshot_betti_numbers_match_barcode(self) -> None:
        filtration = FilteredHyperdigraph(
            (0, 1, 2, 3),
            (
                ((0, 1), 1), ((1, 2), 1), ((2, 0), 2),
                ((0, 1, 2), 4),
                ((0, 3), 2), ((3, 1), 3), ((0, 3, 1), 5),
            ),
            include_all_vertices=True,
        )
        result = compute_persistence(
            filtration, 2, reduction_backend="standard"
        )
        for threshold in filtration.thresholds(3):
            static = compute_homology(filtration.snapshot(threshold), 2)
            self.assertEqual(
                tuple(result.betti_at(dimension, threshold) for dimension in range(3)),
                static.betti_numbers,
            )

    def test_seeded_face_compatible_fast_and_standard_agree(self) -> None:
        rng = random.Random(4102026)
        vertices = tuple(range(5))
        all_edges = list(itertools.permutations(vertices, 2))
        all_triples = list(itertools.permutations(vertices, 3))
        for _ in range(30):
            selected_edges = rng.sample(all_edges, rng.randrange(3, 10))
            edge_birth = {edge: float(rng.randrange(1, 5)) for edge in selected_edges}
            selected_triples = rng.sample(all_triples, rng.randrange(0, 8))
            weighted = [(edge, birth) for edge, birth in edge_birth.items()]
            weighted.extend(
                (triple, float(rng.randrange(1, 6)))
                for triple in selected_triples
            )
            filtration = FilteredHyperdigraph(
                vertices, weighted, include_all_vertices=True
            )
            fast = compute_persistence(filtration, 1)
            standard = compute_persistence(
                filtration, 1, reduction_backend="standard"
            )
            reference = compute_persistence(
                filtration,
                1,
                omega_backend="generic",
                reduction_backend="standard",
            )
            self.assertEqual(_signature(fast), _signature(standard))
            self.assertEqual(_signature(fast), _signature(reference))

    def test_seeded_persistent_ranks_match_direct_inclusion_maps(self) -> None:
        """Audit barcodes against dim im(H_p(s) -> H_p(t)) directly."""

        rng = random.Random(8312026)
        vertices = tuple(range(4))
        candidates = {
            dimension: list(itertools.permutations(vertices, dimension + 1))
            for dimension in range(1, 4)
        }
        for _ in range(10):
            weighted = []
            for dimension, maximum in ((1, 8), (2, 6), (3, 3)):
                count = rng.randrange(maximum + 1)
                weighted.extend(
                    (edge, rng.randrange(4))
                    for edge in rng.sample(candidates[dimension], count)
                )
            filtration = FilteredHyperdigraph(
                vertices, weighted, include_all_vertices=True
            )
            result = compute_persistence(
                filtration,
                2,
                omega_backend="generic",
                reduction_backend="standard",
            )
            hybrid = compute_persistence(
                filtration,
                2,
                omega_backend="generic",
                reduction_backend="auto",
            )
            self.assertEqual(_signature(hybrid), _signature(result))
            thresholds = filtration.thresholds(3)
            chains_at = {}
            for threshold in thresholds:
                snapshot = filtration.snapshot(threshold)
                chains_at[threshold] = {}
                for dimension in range(3):
                    final_index = {
                        edge: index
                        for index, edge in enumerate(
                            filtration.directed_hyperedges(dimension)
                        )
                    }
                    chains_at[threshold][dimension] = (
                        _cycles_and_boundaries_in_final(
                            snapshot, dimension, final_index
                        )
                    )

            for start_index, start in enumerate(thresholds):
                for end in thresholds[start_index:]:
                    for dimension in range(3):
                        cycles_at_start, _ = chains_at[start][dimension]
                        _, boundaries_at_end = chains_at[end][dimension]
                        direct_rank = rank(
                            (*cycles_at_start, *boundaries_at_end)
                        ) - rank(boundaries_at_end)
                        self.assertEqual(
                            result.persistent_betti(dimension, start, end),
                            direct_rank,
                        )

    def test_infinite_interval_helpers(self) -> None:
        filtration = FilteredHyperdigraph((0, 1), (), include_all_vertices=True)
        result = compute_persistence(filtration, 0)
        self.assertEqual(result.betti_at(0, 10), 2)
        self.assertEqual(result.persistent_betti(0, 0, 100), 2)
        self.assertTrue(all(math.isinf(interval.death) for interval in result.intervals))


if __name__ == "__main__":
    unittest.main()
