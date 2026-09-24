from itertools import combinations

import numpy as np
import pytest

from topokit.core._simplicial import (
    ComplexityLimitError, SimplexTree, boundary_matrix, homology, laplacian,
    persistent_homology,
)
from topokit.builders._simplicial import graph_complex
from topokit.workflows.simplicial import analyze_complex


def test_trie_closure_lowering_and_order():
    tree = SimplexTree({(3, 2, 1): 2, (3, 2): 1})
    assert len(tree) == 7
    assert tree.find((3, 1))
    assert tree.filtration((2,)) == 1
    assert tree.filtration((1, 2, 3)) == 2
    assert tree.dimension == 2
    assert tree.validate()
    order = {s: i for i, (s, _) in enumerate(tree.get_filtration())}
    for s in tree:
        for face in combinations(s, len(s) - 1):
            if face:
                assert order[face] < order[s]
    assert set(tree.at(1)) == {(2,), (3,), (2, 3)}
    assert len(tree.skeleton(1)) == 6
    assert tree.cofaces((1, 2), 1) == ((1, 2, 3),)


def test_atomic_limit_and_validation():
    tree = SimplexTree([(0, 1)], max_simplices=4)
    before = tree.get_filtration()
    with pytest.raises(ComplexityLimitError):
        tree.insert((1, 2), -3)
    assert tree.get_filtration() == before
    for simplex in [(), (1, 1), (0, True), (1, 2.5)]:
        with pytest.raises((TypeError, ValueError)):
            tree.insert(simplex)
    for value in [float("nan"), float("inf"), True]:
        with pytest.raises((TypeError, ValueError)):
            tree.insert((3,), value)


@pytest.mark.parametrize("field", [None, 2, 3, 11])
def test_boundary_squared_zero(field):
    tree = SimplexTree([(0, 1, 2, 3, 4)])
    for q in range(1, 5):
        a = boundary_matrix(tree, q, field=field)
        b = boundary_matrix(tree, q + 1, field=field)
        product = (a @ b).toarray()
        if field:
            product %= field
        assert not np.any(product)


@pytest.mark.parametrize("field", [2, 3, 5, "real"])
def test_known_homology(field):
    cycle = graph_complex([(0, 1), (1, 2), (2, 0)], vertices=[9])
    assert homology(cycle, field=field).betti_numbers == (2, 1)
    disk = SimplexTree([(0, 1, 2)])
    assert homology(disk, field=field).betti_numbers == (1, 0, 0)
    sphere = SimplexTree(combinations(range(4), 3))
    assert homology(sphere, field=field).betti_numbers == (1, 0, 1)
    ball = SimplexTree([range(4)])
    assert homology(ball, field=field).betti_numbers == (1, 0, 0, 0)
    assert homology([], field=field, max_dimension=2).betti_numbers == (0, 0, 0)


def projective_plane():
    return SimplexTree([(0, 1, 2), (0, 1, 3), (0, 2, 4), (0, 3, 5), (0, 4, 5),
                        (1, 2, 5), (1, 3, 4), (1, 4, 5), (2, 3, 4), (2, 3, 5)])


def test_torsion_distinguishes_real_and_finite_field():
    rp2 = projective_plane()
    assert homology(rp2, field=2).betti_numbers == (1, 1, 1)
    assert homology(rp2, field=3).betti_numbers == (1, 0, 0)
    assert homology(rp2, field="real").betti_numbers == (1, 0, 0)
    assert laplacian(rp2, 1).betti_number == 0
    assert homology(rp2).euler_characteristic == 1
    assert persistent_homology(rp2, field=2).betti_at(0) == (1, 1, 1)
    assert persistent_homology(rp2, field=3).betti_at(0) == (1, 0, 0)


@pytest.mark.parametrize("field", [2, 3, "real"])
def test_periodic_triangulated_torus(field):
    vertex = lambda i, j: (i % 3) * 3 + j % 3
    facets = []
    for i in range(3):
        for j in range(3):
            a, b, c, d = vertex(i, j), vertex(i + 1, j), vertex(i, j + 1), vertex(i + 1, j + 1)
            facets.extend([(a, b, d), (a, c, d)])
    torus = SimplexTree(facets)
    assert homology(torus, field=field).betti_numbers == (1, 2, 1)
    assert laplacian(torus, 1).betti_number == 2


@pytest.mark.parametrize("dimension", [1, 2, 3, 4])
@pytest.mark.parametrize("field", [2, 3])
def test_higher_spheres_and_filling(dimension, field):
    vertices = tuple(range(dimension + 2))
    tree = SimplexTree({s: 0 for s in combinations(vertices, dimension + 1)})
    expected = (1,) + (0,) * (dimension - 1) + (1,)
    assert homology(tree, field=field).betti_numbers == expected
    tree.insert(vertices, 2)
    ph = persistent_homology(tree, field=field)
    np.testing.assert_array_equal(ph.diagram(dimension), [[0, 2]])


def test_graph_is_not_implicitly_filled_and_births():
    edges = [(0, 1, 1), (1, 2, 2), (2, 0, 3)]
    graph = graph_complex(edges, vertices={0: 2, 4: 7})
    assert graph.filtration((0, 1)) == 2
    assert graph.dimension == 1
    assert homology(graph).betti_numbers == (2, 1)
    flag = graph_complex(edges, flag=True)
    assert flag.filtration((0, 1, 2)) == 3
    assert homology(flag).betti_numbers == (1, 0, 0)
    with pytest.raises(ValueError):
        graph_complex([(0, 0)])


def test_combined_fixed_input_api():
    result = analyze_complex([(0, 1), (1, 2), (0, 2)])
    assert result.homology.betti_numbers == (1, 1)
    assert result.laplacians[1].betti_number == 1


@pytest.mark.parametrize("field", [0, 1, 4, 9, True, 2.5])
def test_invalid_field(field):
    with pytest.raises((TypeError, ValueError)):
        homology([(0,)], field=field)
