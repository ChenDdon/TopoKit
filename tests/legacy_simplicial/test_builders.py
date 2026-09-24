from itertools import combinations

import numpy as np
import pytest

from topokit.core._simplicial import homology, persistent_homology
from topokit.builders._simplicial import alpha_complex, build_complex, rips_complex
from topokit.workflows.simplicial import analyze_points

SQUARE = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)


def test_square_alpha_and_rips_barcodes():
    alpha = build_complex(SQUARE)
    assert alpha.metadata["complex_type"] == "alpha"
    np.testing.assert_allclose(persistent_homology(alpha).diagram(1), [[0.25, 0.5]])
    rips = build_complex(SQUARE, complex_type="rips")
    np.testing.assert_allclose(persistent_homology(rips).diagram(1), [[1, np.sqrt(2)]])


def test_obtuse_triangle_non_gabriel_edge_and_truncation():
    points = [[0, 0], [2, 0], [.2, .2]]
    full = alpha_complex(points)
    assert full.filtration((0, 1)) == pytest.approx(1.64)
    assert full.filtration((0, 1, 2)) == pytest.approx(1.64)
    short = alpha_complex(points, max_dimension=1)
    assert short.filtration((0, 1)) == pytest.approx(1.64)
    cutoff = alpha_complex(points, max_scale=1.0)
    assert (0, 1) not in cutoff
    assert homology(cutoff).betti_numbers == (1, 0)


def test_alpha_affine_hull_and_scaling():
    for points in ([[0], [1], [3]], [[0, 0, 0], [1, 1, 1], [3, 3, 3]]):
        tree = alpha_complex(points)
        assert tree.dimension == 1
        assert (0, 2) not in tree
        assert tree.metadata["affine_dimension"] == 1
    embedded = np.column_stack((SQUARE, np.zeros(4)))
    first = alpha_complex(embedded)
    transformed = alpha_complex(embedded * 7 + [3, 8, 9])
    np.testing.assert_allclose(persistent_homology(transformed).diagram(1),
                               persistent_homology(first).diagram(1) * 49)


def test_alpha_duplicates_and_small_inputs():
    with pytest.raises(ValueError, match="Duplicate"):
        alpha_complex([[0, 0], [0, 0]], duplicates="error")
    merged = alpha_complex([[1, 0], [0, 0], [1, 0]], duplicates="merge")
    assert merged.metadata["original_to_vertex"] == (0, 1, 0)
    assert merged.simplices(0) == ((0,), (1,))
    assert alpha_complex(np.empty((0, 3))).dimension == -1
    assert alpha_complex([[2, 3]]).simplices() == ((0,),)
    assert alpha_complex([[0, 0], [2, 0]]).filtration((0, 1)) == 1
    tetrahedron = alpha_complex([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert tetrahedron.filtration((0, 1, 2, 3)) == pytest.approx(.75)


@pytest.mark.parametrize("seed", range(5))
def test_rips_matches_bruteforce(seed):
    points = np.random.default_rng(seed).normal(size=(9, 3))
    distance = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
    for cutoff in [.9, 2, np.inf]:
        tree = rips_complex(points, max_dimension=3, max_scale=cutoff)
        matrix_tree = rips_complex(distance_matrix=distance, max_dimension=3, max_scale=cutoff)
        expected = {}
        for size in range(1, 5):
            for simplex in combinations(range(len(points)), size):
                birth = max((distance[i, j] for i, j in combinations(simplex, 2)), default=0.)
                if birth <= cutoff:
                    expected[simplex] = birth
        assert set(tree) == set(expected) == set(matrix_tree)
        for s in expected:
            assert tree.filtration(s) == pytest.approx(expected[s])
        assert tree.validate()


def test_rips_infinite_missing_edges_and_zero_distance():
    matrix = [[0, 0, np.inf], [0, 0, 1], [np.inf, 1, 0]]
    tree = rips_complex(distance_matrix=matrix)
    assert (0, 2) not in tree
    assert tree.filtration((0, 1)) == 0
    assert len(rips_complex(SQUARE, max_dimension=0)) == 4


@pytest.mark.parametrize("dimension", [2, 3, 4])
def test_random_alpha_full_complex_contractible(dimension):
    points = np.random.default_rng(15).normal(size=(12, dimension))
    tree = alpha_complex(points)
    assert tree.validate()
    assert tree.dimension == dimension
    assert homology(tree).betti_numbers == (1,) + (0,) * dimension


def test_high_level_builds_death_dimension():
    result = analyze_points(SQUARE, scales=[.3, .6])
    assert result.complex.dimension == 2
    assert result.snapshots[.3].homology.betti_numbers == (1, 1)
    assert result.snapshots[.6].homology.betti_numbers == (1, 0)


@pytest.mark.parametrize("kwargs", [{"points": [[np.nan, 0]]}, {"points": [0, 1]},
                                    {"distance_matrix": [[0, 1], [2, 0]]},
                                    {"distance_matrix": [[0, -1], [-1, 0]]}])
def test_invalid_rips_inputs(kwargs):
    with pytest.raises(ValueError):
        rips_complex(**kwargs)
