"""Compare native alpha values with independent constrained ball optimization."""

import numpy as np
import pytest
from scipy.optimize import minimize

from topokit.core._simplicial import ComplexityLimitError
from topokit.builders._simplicial import alpha_complex, rips_complex


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("backend", ["native", "gudhi_exact"])
def test_alpha_against_voronoi_quadratic_program(dimension, seed, backend):
    if backend == "gudhi_exact":
        pytest.importorskip("gudhi")
    points = np.random.default_rng(seed).normal(size=(7, dimension))
    tree = alpha_complex(points,backend=backend)
    for simplex in tree:
        if len(simplex) == 1:
            continue
        # Center is relative to the first vertex. Other simplex vertices must
        # lie on the sphere, and all remaining input points must lie outside.
        delta = points - points[simplex[0]]
        norms = np.sum(delta * delta, axis=1)
        equal = list(simplex[1:])
        outside = [i for i in range(len(points)) if i not in simplex]
        constraints = [{"type": "eq", "fun": lambda c: 2 * delta[equal] @ c - norms[equal],
                        "jac": lambda c: 2 * delta[equal]}]
        if outside:
            constraints.append({"type": "ineq", "fun": lambda c: norms[outside] - 2 * delta[outside] @ c,
                                "jac": lambda c: -2 * delta[outside]})
        initial = np.linalg.lstsq(2 * delta[equal], norms[equal], rcond=None)[0]
        result = minimize(lambda c: c @ c, initial, jac=lambda c: 2 * c,
                          method="SLSQP", constraints=constraints,
                          options={"ftol": 1e-11, "maxiter": 200})
        assert result.success, result.message
        assert tree.filtration(simplex) == pytest.approx(result.fun, rel=1e-8, abs=1e-9)


def test_rips_construction_limits():
    for cutoff in [np.inf, 2]:
        with pytest.raises(ComplexityLimitError):
            rips_complex(np.zeros((10, 2)), max_scale=cutoff, max_simplices=20)
    with pytest.raises(ComplexityLimitError):
        rips_complex(np.zeros((10, 2)), max_dimension=0, max_simplices=5)
