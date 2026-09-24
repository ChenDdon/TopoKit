"""Signed and real-valued Laplacian checks against independent references.

The numerical reference intentionally does not call the implementation's
boundary helpers, cached face indexes, or persistence reducer.  In particular,
it includes linear combinations of late chains whose unwanted boundary rows
cancel, with the inherited Euclidean metric on that subspace.
"""

from __future__ import annotations

import itertools
import math
import random
import unittest

from topokit.core._interaction import (
    SimplicialComplexBuilder,
    build_interaction_chain_complex,
)

try:
    import numpy as np
    import scipy
except ImportError:
    np = None
    scipy = None


def full_simplex(vertices, birth=0.0):
    builder = SimplicialComplexBuilder()
    builder.insert(vertices, birth)
    return builder.freeze()


def filtered_edge():
    builder = SimplicialComplexBuilder()
    builder.insert((0,), 0.0)
    builder.insert((1,), 0.0)
    builder.insert((0, 1), 1.0, with_faces=False)
    return builder.freeze()


def paper_paths():
    left = SimplicialComplexBuilder()
    right = SimplicialComplexBuilder()
    for edge in ((0, 1), (1, 2)):
        left.insert(edge, 0.0)
    for edge in ((1, 2), (2, 3)):
        right.insert(edge, 0.0)
    return left.freeze(), right.freeze()


def cancellation_fixture():
    """Neither new triangle alone has an entirely old degree-one boundary."""
    builder = SimplicialComplexBuilder()
    for vertex in range(4):
        builder.insert((vertex,), 0.0)
    for edge in ((0, 1), (0, 3)):
        builder.insert(edge, 0.0, with_faces=False)
    for edge in ((0, 2), (1, 2), (2, 3)):
        builder.insert(edge, 1.0, with_faces=False)
    for triangle in ((0, 1, 2), (0, 2, 3)):
        builder.insert(triangle, 1.0, with_faces=False)
    return builder.freeze(), full_simplex((0,))


def reference_keys(factors, degree, filtration):
    """Cartesian construction independent of the production posting join."""
    if degree < 0:
        return ()
    keys = []
    for key in itertools.product(*(range(len(f.simplices)) for f in factors)):
        simplices = tuple(f.simplices[sid] for f, sid in zip(factors, key))
        if sum(len(simplex) - 1 for simplex in simplices) != degree:
            continue
        if max(f.births[sid] for f, sid in zip(factors, key)) > filtration:
            continue
        if set.intersection(*(set(simplex) for simplex in simplices)):
            keys.append(key)
    return tuple(keys)


def reference_boundary(factors, degree, filtration):
    """Column-source integer tensor differential, then quotient by disjointness."""
    sources = reference_keys(factors, degree, filtration)
    targets = reference_keys(factors, degree - 1, filtration)
    target_index = {key: row for row, key in enumerate(targets)}
    simplex_ids = [
        {simplex: sid for sid, simplex in enumerate(factor.simplices)}
        for factor in factors
    ]
    matrix = np.zeros((len(targets), len(sources)))
    if degree == 0:
        return matrix, targets, sources
    for column, source in enumerate(sources):
        prefix_degree = 0
        for position, (factor, sid) in enumerate(zip(factors, source)):
            simplex = factor.simplices[sid]
            if len(simplex) > 1:
                for omitted in range(len(simplex)):
                    face = simplex[:omitted] + simplex[omitted + 1 :]
                    key = list(source)
                    key[position] = simplex_ids[position][face]
                    row = target_index.get(tuple(key))
                    if row is not None:
                        matrix[row, column] += (-1) ** (prefix_degree + omitted)
            prefix_degree += len(simplex) - 1
    return matrix, targets, sources


def reference_nullspace(matrix):
    if matrix.shape[1] == 0:
        return np.empty((0, 0))
    if matrix.shape[0] == 0:
        return np.eye(matrix.shape[1])
    _, singular_values, vh = np.linalg.svd(matrix, full_matrices=True)
    rank = int(np.count_nonzero(singular_values > 1e-10))
    return vh[rank:].T


def reference_rank(matrix):
    if not min(matrix.shape):
        return 0
    return int(np.count_nonzero(np.linalg.svd(matrix, compute_uv=False) > 1e-10))


def reference_persistent_laplacian(factors, degree, start, end):
    down, _, early_keys = reference_boundary(factors, degree, start)
    late_up, late_keys, _ = reference_boundary(factors, degree + 1, end)
    late_index = {key: row for row, key in enumerate(late_keys)}
    early_indices = [late_index[key] for key in early_keys]
    early_set = set(early_keys)
    late_indices = [row for row, key in enumerate(late_keys) if key not in early_set]
    permitted = late_up[early_indices, :]
    constraints = late_up[late_indices, :]
    up = permitted @ reference_nullspace(constraints)
    matrix = down.T @ down + up @ up.T

    # Rank(im H_q(start)->H_q(end)) = rank([embedded Z_q(start),
    # B_q(end)]) - rank(B_q(end)), independently of the Laplacian formula.
    old_cycles = reference_nullspace(down)
    embedded_cycles = np.zeros((len(late_keys), old_cycles.shape[1]))
    embedded_cycles[early_indices, :] = old_cycles
    persistent_betti = reference_rank(
        np.concatenate((embedded_cycles, late_up), axis=1)
    ) - reference_rank(late_up)
    return matrix, early_keys, persistent_betti


def random_factor(rng, vertex_count=3):
    builder = SimplicialComplexBuilder()
    vertex_births = {vertex: float(rng.randrange(2)) for vertex in range(vertex_count)}
    for vertex, birth in vertex_births.items():
        builder.insert((vertex,), birth)
    edge_births = {}
    for edge in itertools.combinations(range(vertex_count), 2):
        if rng.random() < 0.8:
            birth = max(*(vertex_births[v] for v in edge), float(rng.randrange(3)))
            builder.insert(edge, birth, with_faces=False)
            edge_births[edge] = birth
    for triangle in itertools.combinations(range(vertex_count), 3):
        faces = tuple(itertools.combinations(triangle, 2))
        if all(face in edge_births for face in faces) and rng.random() < 0.65:
            builder.insert(
                triangle, max(edge_births[face] for face in faces) + rng.randrange(2),
                with_faces=False,
            )
    return builder.freeze()


def rips_factor(points, global_ids):
    """Small exhaustive Vietoris--Rips input; no optional geometry library."""
    builder = SimplicialComplexBuilder()
    for size in range(1, len(points) + 1):
        for local in itertools.combinations(range(len(points)), size):
            birth = max(
                (math.dist(points[a], points[b]) for a, b in itertools.combinations(local, 2)),
                default=0.0,
            )
            builder.insert(tuple(global_ids[i] for i in local), birth, with_faces=False)
    return builder.freeze()


class SignedBoundaryTests(unittest.TestCase):
    def test_signed_tensor_boundary_squares_to_zero_one_two_three_factors(self):
        fixtures = (
            (full_simplex((0, 1, 2, 3)),),
            (full_simplex((0, 1, 2)), full_simplex((1, 2, 3))),
            (full_simplex((0, 1, 2)),) * 3,
        )
        for factors in fixtures:
            with self.subTest(factor_count=len(factors)):
                chain = build_interaction_chain_complex(
                    factors, max_homology_dimension=3, validate=True
                )
                chain.validate_signed_boundary()
                for degree in range(1, len(chain.degrees)):
                    for column in range(chain.number_of_cells(degree)):
                        entries = chain.signed_boundary_entries(degree, column)
                        self.assertEqual(
                            tuple(row for row, coefficient in entries if coefficient % 2),
                            chain.boundary_rows(degree, column),
                        )
                        twice = {}
                        for row, coefficient in entries:
                            for lower, second in chain.signed_boundary_entries(degree - 1, row):
                                twice[lower] = twice.get(lower, 0) + coefficient * second
                        self.assertTrue(all(value == 0 for value in twice.values()))

    def test_edge_tensor_edge_has_the_koszul_sign(self):
        factor = full_simplex((0, 1))
        chain = build_interaction_chain_complex((factor, factor), max_homology_dimension=1)
        edge = factor.simplex_id((0, 1))
        vertex0 = factor.simplex_id((0,))
        vertex1 = factor.simplex_id((1,))
        column = chain.degrees[2].index_of((edge, edge))
        actual = {
            chain.degrees[1].keys[row]: coefficient
            for row, coefficient in chain.signed_boundary_entries(2, column)
        }
        self.assertEqual(actual, {
            (vertex1, edge): 1,
            (vertex0, edge): -1,
            (edge, vertex1): -1,
            (edge, vertex0): 1,
        })


@unittest.skipUnless(np is not None and scipy is not None, "requires the optional laplacian extra")
class InteractionLaplacianTests(unittest.TestCase):
    def setUp(self):
        # Delay importing numerical APIs so the signed tests run without extras.
        from topokit.core._interaction import InteractionLaplacianEngine

        self.engine_class = InteractionLaplacianEngine

    def assert_reference(self, chain, dimension, start, end):
        result = self.engine_class(chain).persistent_laplacian(
            dimension, start=start, end=end
        )
        self.assertEqual(result.dimension, dimension)
        self.assertEqual(result.start, start)
        self.assertEqual(result.end, end)
        expected, keys, betti = reference_persistent_laplacian(
            chain.factors, dimension, start, end
        )
        self.assertEqual(set(result.basis_keys), set(keys))
        index = {key: row for row, key in enumerate(keys)}
        order = [index[key] for key in result.basis_keys]
        expected = expected[np.ix_(order, order)]
        actual = result.to_dense()
        np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=2e-9)
        np.testing.assert_allclose(actual, actual.T, atol=1e-12)
        spectrum = result.spectrum()
        self.assertTrue(spectrum.is_complete)
        self.assertEqual(spectrum.nullity, betti)
        self.assertEqual(spectrum.zero_eigenvalue_count, betti)
        if len(spectrum.eigenvalues):
            self.assertGreaterEqual(min(spectrum.eigenvalues), -2e-9)
        return result

    def test_paper_section_2_4_exact_matrices_and_spectra(self):
        chain = build_interaction_chain_complex(paper_paths(), max_homology_dimension=2)
        engine = self.engine_class(chain)
        np.testing.assert_array_equal(engine.laplacian(0).to_dense(), np.diag([3.0, 3.0]))
        expected_spectra = (
            [3.0, 3.0],
            [0.0, 3 - math.sqrt(3), 2.0, 3.0, 3.0, 3 + math.sqrt(3)],
            [3 - math.sqrt(3), 2.0, 3 + math.sqrt(3)],
        )
        for degree, expected in enumerate(expected_spectra):
            with self.subTest(degree=degree):
                np.testing.assert_allclose(engine.laplacian(degree).spectrum().eigenvalues, expected, atol=1e-10)
                self.assert_reference(chain, degree, math.inf, math.inf)

        # Compare the printed paper matrix in its own tensor basis order.
        factor1, factor2 = chain.factors
        printed_basis = (
            ((1,), (1, 2)), ((2,), (1, 2)), ((2,), (2, 3)),
            ((0, 1), (1,)), ((1, 2), (1,)), ((1, 2), (2,)),
        )
        keys = tuple((factor1.simplex_id(a), factor2.simplex_id(b)) for a, b in printed_basis)
        result = engine.laplacian(1)
        indices = [result.basis_keys.index(key) for key in keys]
        printed_matrix = np.array([
            [3, -1, 0, 0, 0, 1], [-1, 2, -1, 0, 1, 0],
            [0, -1, 2, 0, 0, 0], [0, 0, 0, 2, -1, 0],
            [0, 1, 0, -1, 2, -1], [1, 0, 0, 0, -1, 3],
        ])
        np.testing.assert_array_equal(result.to_dense()[np.ix_(indices, indices)], printed_matrix)

    def test_persistent_cancellation_and_inherited_metric(self):
        chain = build_interaction_chain_complex(cancellation_fixture(), max_homology_dimension=1)
        engine = self.engine_class(chain)
        result = engine.persistent_laplacian(1, start=0.0, end=1.0)
        np.testing.assert_allclose(result.to_dense(), [[1.5, 0.5], [0.5, 1.5]], atol=1e-12)
        np.testing.assert_allclose(result.spectrum().eigenvalues, [1.0, 2.0], atol=1e-12)
        # Keeping only individually supported columns would give the all-ones
        # lower term; using an unnormalized sum would give the wrong weight.
        np.testing.assert_array_equal(engine.laplacian(1, filtration=0.0).to_dense(), np.ones((2, 2)))
        self.assert_reference(chain, 1, 0.0, 1.0)

    def test_single_factor_ordinary_and_persistent_graph_laplacian(self):
        chain = build_interaction_chain_complex((filtered_edge(),), max_homology_dimension=1)
        engine = self.engine_class(chain)
        np.testing.assert_array_equal(engine.laplacian(0, filtration=0).to_dense(), np.zeros((2, 2)))
        expected = [[1.0, -1.0], [-1.0, 1.0]]
        np.testing.assert_array_equal(engine.laplacian(0, filtration=1).to_dense(), expected)
        np.testing.assert_array_equal(engine.persistent_laplacian(0, start=0, end=1).to_dense(), expected)
        self.assert_reference(chain, 0, 0.0, 1.0)

    def test_single_factor_l0_requires_cancellation_at_a_late_vertex(self):
        builder = SimplicialComplexBuilder()
        builder.insert((0,), 0.0)
        builder.insert((2,), 0.0)
        builder.insert((1,), 1.0)
        builder.insert((0, 1), 1.0, with_faces=False)
        builder.insert((1, 2), 1.0, with_faces=False)
        chain = build_interaction_chain_complex((builder.freeze(),), max_homology_dimension=0)
        result = self.engine_class(chain).persistent_laplacian(0, start=0.0, end=1.0)
        np.testing.assert_allclose(result.to_dense(), [[0.5, -0.5], [-0.5, 0.5]], atol=1e-12)
        self.assertEqual(result.spectrum().nullity, 1)
        self.assert_reference(chain, 0, 0.0, 1.0)

    def test_identical_cloud_filtrations(self):
        fixtures = (
            ((0.0, 0.0), (1.0, 0.0)),
            ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
            ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        )
        for points in fixtures:
            factor = rips_factor(points, tuple(range(len(points))))
            chain = build_interaction_chain_complex((factor, factor), max_homology_dimension=1)
            for degree in (0, 1):
                for start, end in ((0, 1), (1, 1), (1, math.sqrt(2))):
                    with self.subTest(points=len(points), degree=degree, start=start, end=end):
                        self.assert_reference(chain, degree, start, end)

    def test_paper_point_cloud_spectral_gaps(self):
        first = rips_factor(((0, 0), (1, 0), (1, 1)), (0, 1, 2))
        second = rips_factor(((1, 0), (1, 1), (2, 1)), (1, 2, 3))
        chain = build_interaction_chain_complex((first, second), max_homology_dimension=1)
        engine = self.engine_class(chain)
        for filtration, expected0, expected1 in (
            (0.0, 0.0, 0.0), (1.0, 3.0, 3 - math.sqrt(3)), (math.sqrt(2), 4.0, 2.0)
        ):
            with self.subTest(filtration=filtration):
                self.assertAlmostEqual(engine.laplacian(0, filtration=filtration).spectrum().spectral_gap, expected0)
                self.assertAlmostEqual(engine.laplacian(1, filtration=filtration).spectrum().spectral_gap, expected1)

    def test_full_overlap_higher_degrees_and_three_factors(self):
        triangle = full_simplex((0, 1, 2))
        for factor_count, highest_degree in ((2, 3), (3, 2)):
            chain = build_interaction_chain_complex(
                (triangle,) * factor_count, max_homology_dimension=highest_degree
            )
            for degree in range(highest_degree + 1):
                with self.subTest(factor_count=factor_count, degree=degree):
                    self.assert_reference(chain, degree, 0.0, 0.0)

    def test_persistent_l2_genuine_filtration_with_multiple_factors(self):
        # Face-closed partial tetrahedral factors and three triangular
        # factors both produce nontrivial coupled late-row constraints.
        # Keep these fixed and small: this is a correctness check, not a
        # randomized performance stress test.
        for factor_count, vertex_count, seed in ((2, 4, 3), (3, 3, 0)):
            with self.subTest(factor_count=factor_count, vertex_count=vertex_count):
                rng = random.Random(seed)
                factors = tuple(
                    random_factor(rng, vertex_count) for _ in range(factor_count)
                )
                chain = build_interaction_chain_complex(
                    factors, max_homology_dimension=2, validate=True
                )
                result = self.assert_reference(chain, 2, 1.0, 3.0)
                self.assertGreater(result.diagnostics.start_cell_count, 0)
                self.assertGreater(result.diagnostics.constraint_nnz, 0)
                self.assertGreater(result.diagnostics.correction_rank, 0)

    def test_random_two_three_factor_boundaries_and_persistent_real_ranks(self):
        for factor_count in (2, 3):
            for seed in range(8):
                rng = random.Random(991 * factor_count + seed)
                factors = tuple(random_factor(rng) for _ in range(factor_count))
                chain = build_interaction_chain_complex(factors, max_homology_dimension=1, validate=True)
                engine = self.engine_class(chain)
                for degree in (1, 2):
                    expected, targets, sources = reference_boundary(factors, degree, 2.0)
                    rows = {key: row for row, key in enumerate(targets)}
                    columns = {key: column for column, key in enumerate(sources)}
                    source_order = [columns[key] for key, birth in zip(chain.degrees[degree].keys, chain.degrees[degree].births) if birth <= 2.0]
                    target_order = [rows[key] for key, birth in zip(chain.degrees[degree - 1].keys, chain.degrees[degree - 1].births) if birth <= 2.0]
                    actual = engine.boundary_matrix(degree, filtration=2.0).toarray()
                    np.testing.assert_array_equal(actual, expected[np.ix_(target_order, source_order)])
                for degree in (0, 1):
                    for start, end in ((0.0, 1.0), (1.0, 2.0), (2.0, 2.0)):
                        with self.subTest(factors=factor_count, seed=seed, degree=degree, start=start, end=end):
                            self.assert_reference(chain, degree, start, end)
                            if start == end:
                                np.testing.assert_allclose(
                                    engine.laplacian(degree, filtration=start).to_dense(),
                                    engine.persistent_laplacian(degree, start=start, end=end).to_dense(),
                                    atol=1e-10,
                                )

    def test_no_shared_vertices_and_before_first_birth_are_empty(self):
        chain = build_interaction_chain_complex((full_simplex((0, 1)), full_simplex((2, 3))), max_homology_dimension=1)
        engine = self.engine_class(chain)
        for degree in (0, 1):
            result = engine.persistent_laplacian(degree, start=0.0, end=1.0)
            self.assertEqual(result.to_dense().shape, (0, 0))
            self.assertEqual(result.to_sparse().shape, (0, 0))
            self.assertEqual(result.spectrum().nullity, 0)
            self.assertEqual(result.spectrum().spectral_gap, 0.0)
        factor = full_simplex((0, 1), birth=1.0)
        chain = build_interaction_chain_complex((factor, factor), max_homology_dimension=1)
        self.assertEqual(self.engine_class(chain).laplacian(0, filtration=0).to_dense().shape, (0, 0))

    def test_matrix_operator_sparse_and_convenience_apis_agree(self):
        from topokit.core._interaction import (
            compute_interaction_laplacian,
            compute_persistent_interaction_laplacian,
            signed_boundary_matrix,
        )

        chain = build_interaction_chain_complex(cancellation_fixture(), max_homology_dimension=1)
        engine = self.engine_class(chain)
        result = compute_persistent_interaction_laplacian(chain, dimension=1, start=0.0, end=1.0)
        dense = result.to_dense()
        vector = np.array([0.5, -1.0])
        np.testing.assert_allclose(result.to_sparse().toarray(), dense)
        np.testing.assert_allclose(result.as_linear_operator() @ vector, dense @ vector)
        np.testing.assert_allclose(result.as_linear_operator() @ np.eye(2), dense)
        np.testing.assert_allclose(compute_interaction_laplacian(chain, dimension=1, filtration=1).to_dense(), engine.laplacian(1, filtration=1).to_dense())
        np.testing.assert_array_equal(signed_boundary_matrix(chain, 2, filtration=1).toarray(), engine.boundary_matrix(2, filtration=1).toarray())

    def test_diagonal_low_degree_partial_spectrum_is_labelled(self):
        factor = full_simplex(tuple(range(6)))
        chain = build_interaction_chain_complex((factor, factor), max_homology_dimension=0)
        result = self.engine_class(chain).laplacian(0)
        np.testing.assert_array_equal(result.to_dense(), np.diag([10.0] * 6))
        full = result.spectrum()
        self.assertTrue(full.is_complete)
        self.assertEqual(full.nullity, 0)
        self.assertEqual(full.spectral_gap, 10.0)
        partial = result.spectrum(k=2)
        self.assertFalse(partial.is_complete)
        self.assertIsNone(partial.nullity)
        np.testing.assert_allclose(partial.eigenvalues, [10.0, 10.0])

    def test_general_partial_spectrum_and_zero_operator(self):
        chain = build_interaction_chain_complex(paper_paths(), max_homology_dimension=1)
        partial = self.engine_class(chain).laplacian(1).spectrum(k=2)
        self.assertFalse(partial.is_complete)
        self.assertIsNone(partial.nullity)
        self.assertIsNone(partial.spectral_gap)
        self.assertEqual(partial.zero_eigenvalue_count, 1)
        np.testing.assert_allclose(partial.eigenvalues, [0.0, 3 - math.sqrt(3)], atol=1e-10)

        builder = SimplicialComplexBuilder()
        for vertex in range(3):
            builder.insert((vertex,), 0.0)
        chain = build_interaction_chain_complex((builder.freeze(),), max_homology_dimension=0)
        result = self.engine_class(chain).laplacian(0)
        partial = result.spectrum(k=1)
        self.assertEqual(partial.eigenvalues, (0.0,))
        self.assertIsNone(partial.nullity)
        self.assertIsNone(partial.spectral_gap)
        self.assertEqual(result.spectrum().nullity, 3)

        # Every old isolated vertex acquires a distinct new neighbour.  Each
        # late edge is constrained to coefficient zero, so this persistent
        # operator is also exactly zero despite nonzero late boundaries.
        builder = SimplicialComplexBuilder()
        for vertex in (0, 1):
            builder.insert((vertex,), 0.0)
        for vertex in (2, 3):
            builder.insert((vertex,), 1.0)
        for edge in ((0, 2), (1, 3)):
            builder.insert(edge, 1.0, with_faces=False)
        chain = build_interaction_chain_complex((builder.freeze(),), max_homology_dimension=0)
        result = self.engine_class(chain).persistent_laplacian(0, start=0.0, end=1.0)
        np.testing.assert_array_equal(result.to_dense(), np.zeros((2, 2)))
        self.assertEqual(result.spectrum(k=1).eigenvalues, (0.0,))
        self.assert_reference(chain, 0, 0.0, 1.0)

    def test_returned_matrix_mutation_does_not_change_cached_operator(self):
        chain = build_interaction_chain_complex(paper_paths(), max_homology_dimension=1)
        engine = self.engine_class(chain)
        expected_boundary = engine.boundary_matrix(1).toarray()
        boundary = engine.boundary_matrix(1)
        boundary.data[:] = 17.0
        np.testing.assert_array_equal(engine.boundary_matrix(1).toarray(), expected_boundary)
        for degree in (0, 1):
            result = engine.laplacian(degree)
            expected = result.to_dense()
            dense = result.to_dense()
            dense[:] = 17.0
            sparse = result.to_sparse()
            sparse.data[:] = 17.0
            np.testing.assert_allclose(result.to_dense(), expected)
            np.testing.assert_allclose(engine.laplacian(degree).to_dense(), expected)

    def test_dimension_and_filtration_inputs_are_checked(self):
        chain = build_interaction_chain_complex(paper_paths(), max_homology_dimension=1)
        engine = self.engine_class(chain)
        for dimension in (-1, 2):
            with self.assertRaises(ValueError):
                engine.laplacian(dimension)
        with self.assertRaises(TypeError):
            engine.laplacian(True)
        with self.assertRaises(ValueError):
            engine.persistent_laplacian(0, start=2, end=1)
        with self.assertRaises(ValueError):
            engine.laplacian(0, filtration=math.nan)
        with self.assertRaises(TypeError):
            engine.laplacian(0, filtration=True)
        for tolerance in (-1.0, 1.0, math.inf, math.nan):
            with self.assertRaises(ValueError):
                self.engine_class(chain, rank_rtol=tolerance)
        with self.assertRaises(TypeError):
            self.engine_class(chain, rank_rtol=True)
        result = engine.laplacian(0)
        with self.assertRaises(ValueError):
            result.spectrum(k=0)
        with self.assertRaises(TypeError):
            result.spectrum(k=True)
        for argument in ("zero_atol", "zero_rtol", "tol"):
            with self.assertRaises(ValueError):
                result.spectrum(**{argument: -1.0})

    def test_dense_budget_is_checked_before_materializing(self):
        chain = build_interaction_chain_complex(paper_paths(), max_homology_dimension=1)
        engine = self.engine_class(chain, max_dense_bytes=8)
        result = engine.laplacian(1)
        with self.assertRaises(MemoryError):
            result.to_dense()
        with self.assertRaises(MemoryError):
            result.spectrum()

    def test_empty_output_and_constraint_component_memory_guards(self):
        chain = build_interaction_chain_complex(
            (full_simplex((0,)), full_simplex((1,))), max_homology_dimension=1
        )
        empty = self.engine_class(chain, max_dense_bytes=1).laplacian(1)
        self.assertEqual(empty.to_dense().shape, (0, 0))
        self.assertEqual(empty.to_sparse().shape, (0, 0))
        self.assertEqual(empty.spectrum().eigenvalues, ())

        chain = build_interaction_chain_complex(cancellation_fixture(), max_homology_dimension=1)
        with self.assertRaises(MemoryError):
            self.engine_class(chain, max_dense_bytes=8).persistent_laplacian(1, start=0.0, end=1.0)

        # A one-column constraint component has two new rows.  With no
        # admissible upper combination it must project away completely.
        early_edge = full_simplex((0, 1))
        late_edge = filtered_edge()
        chain = build_interaction_chain_complex((early_edge, late_edge), max_homology_dimension=1)
        self.assert_reference(chain, 1, 0.0, 1.0)

        # Guard an actual multi-row, multi-column SVD allocation, not only
        # singleton projection or final square-matrix materialization.
        rng = random.Random(3)
        factors = tuple(random_factor(rng, 4) for _ in range(2))
        chain = build_interaction_chain_complex(factors, max_homology_dimension=2)
        with self.assertRaisesRegex(MemoryError, "constraint-component SVD"):
            self.engine_class(chain, max_dense_bytes=128).persistent_laplacian(
                2, start=1.0, end=3.0
            )


if __name__ == "__main__":
    unittest.main()
