"""Analysis of explicit two-factor interaction chains.

This layer knows factor simplices, quotient boundaries, and scalar grades.
It does not read point clouds, construct geometric complexes, or import any
builder, reader, plotting, postprocessing, or workflow module.
"""
from dataclasses import asdict

from ..results import (Topology, HomologyResult, PersistenceInterval,
                       PersistenceResult, SpectrumResult)
from ..exceptions import ResourceLimitError
from .._validation import dimension as _dimension, validate_query
from ._spectral import solve_symmetric
from . import _interaction as _native


DEFINITION_ID = _native.DEFINITION_ID
FactorComplex = _native.FilteredSimplicialComplex
FactorComplexBuilder = _native.SimplicialComplexBuilder
InteractionChain = _native.FilteredInteractionChainComplex


def _limit(value, name):
    value = _dimension(value, name)
    if value == 0:
        raise ValueError(f"{name} must be positive")
    return value


def _as_topology(obj):
    if isinstance(obj, _native.FilteredInteractionChainComplex):
        maps = tuple({v: v for simplex in factor.simplices for v in simplex}
                     for factor in obj.factors)
        first_birth = min((birth for factor in obj.factors for birth in factor.births),
                          default=0.0)
        return Topology("interaction", obj, metadata={
            "definition_id": DEFINITION_ID, "route": "interaction",
            "factor_count": len(obj.factors), "factor_vertex_id_maps": maps,
            "filtration_start": first_birth, "start_policy": "native_grades",
            "coupling": "maximum_factor_birth", "filtration_parameters": 1,
            "scale_units": "user_scalar", "input_mode": "explicit_chain"})
    return obj


def _object(obj, maximum=None):
    if not isinstance(obj, Topology) or obj.kind != "interaction":
        raise TypeError("expected an interaction Topology or explicit InteractionChain")
    if not isinstance(obj.native, _native.FilteredInteractionChainComplex) or len(obj.native.factors) != 2:
        raise ValueError("public interaction objects must have exactly two factors")
    if maximum is not None and maximum > obj.native.max_homology_dimension:
        raise ValueError("requested dimension exceeds the constructed interaction chain; reconstruct with a larger max_dimension")
    return obj.native


def _gf2(field):
    if isinstance(field, bool) or field not in (2, "GF(2)", "F2"):
        raise ValueError("interaction homology and persistence currently support GF(2) only")


def _query(obj, scale):
    value = validate_query(obj.metadata, scale)
    if value is not None:
        return value
    return max((birth for layer in obj.native.degrees for birth in layer.births),
               default=obj.metadata.get("filtration_start", 0.0))


def persistence(obj, *, max_dimension=2, field=2, backend="auto",
                include_diagonal=False, minimum_persistence=0.0):
    """GF(2) intervals of the scalar maximum-coupled interaction filtration."""
    obj = _as_topology(obj)
    highest = _dimension(max_dimension)
    chain = _object(obj, highest)
    _gf2(field)
    native_result = _native.compute_persistence(chain, backend=backend,
        include_diagonal=include_diagonal, minimum_persistence=minimum_persistence)
    start = obj.metadata.get("filtration_start", 0.0)
    intervals = tuple(PersistenceInterval(item.dimension, item.birth, item.death,
                                         at_initial_stage=item.birth == start)
                      for item in native_result.intervals if item.dimension <= highest)
    initial_semantics = obj.metadata.get("initial_stage_semantics", (
        "birth may be left-truncated"
        if obj.metadata.get("start_policy") == "clamp_all_simplex_births"
        else "earliest native filtration stage"))
    return PersistenceResult(intervals, field="GF(2)", max_dimension=highest,
                             metadata={**obj.metadata, "coefficient_field": "GF(2)",
                                       "reduction_diagnostics": asdict(native_result.diagnostics),
                                       "initial_stage_semantics": initial_semantics})


def homology(obj, *, max_dimension=2, scale=None, field=2, backend="auto"):
    """GF(2) Betti numbers at one inclusive scalar threshold."""
    obj = _as_topology(obj)
    highest = _dimension(max_dimension)
    _object(obj, highest)
    time = _query(obj, scale)
    bars = persistence(obj, max_dimension=highest, field=field, backend=backend)
    betti = tuple(sum(item.dimension == q and item.birth <= time < item.death
                      for item in bars.intervals) for q in range(highest + 1))
    return HomologyResult(betti, "GF(2)", {**bars.metadata, "scale": time})


def _spectrum(obj, dimension, start, end, *, persistent, return_eigenvectors,
              k, return_matrix, max_dense_entries, max_dense_bytes, rank_rtol, tol):
    chain = _object(obj, dimension)
    entry_budget = _limit(max_dense_entries, "max_dense_entries")
    memory_budget = _limit(max_dense_bytes, "max_dense_bytes")
    if k is not None:
        k = _limit(k, "k")
    engine = _native.InteractionLaplacianEngine(chain, rank_rtol=rank_rtol,
                                               max_dense_bytes=memory_budget)
    try:
        operator = engine.persistent_laplacian(dimension, start=start, end=end)
        if return_matrix and operator.size ** 2 > entry_budget:
            raise ResourceLimitError("matrix export exceeds max_dense_entries; use matrix-free eigenpairs or raise the explicit budget")
        size = operator.size
        complete = k is None or k >= size
        if complete:
            estimated_solver_bytes = 64 * size * size
        else:
            ncv = min(size, max(2 * k + 1, 20))
            estimated_solver_bytes = 8 * (4 * size * ncv + 8 * size + 4 * ncv ** 2)
        estimated_solver_bytes += operator.diagnostics.correction_storage_bytes
        if estimated_solver_bytes > memory_budget:
            raise ResourceLimitError(
                f"eigensolver workspace estimate {estimated_solver_bytes:,} bytes exceeds "
                f"max_dense_bytes={memory_budget:,}; request fewer modes or raise the budget")
        matrix_free = operator.as_linear_operator()
        eigenpairs = solve_symmetric(matrix_free, k=k, return_eigenvectors=return_eigenvectors,
                                max_dense_entries=entry_budget, tol=tol)
        matrix = operator.to_sparse() if return_matrix else None
    except MemoryError as error:
        raise ResourceLimitError(str(error)) from error
    vertex_id_maps = obj.metadata["factor_vertex_id_maps"]
    basis = tuple(tuple(tuple(vertex_id_maps[i][v] for v in chain.factors[i].simplices[simplex_id])
                        for i, simplex_id in enumerate(key)) for key in operator.basis_keys)
    return SpectrumResult(
        dimension, eigenpairs.values, eigenpairs.vectors, matrix, basis=basis,
        scale=None if persistent else start, start=start if persistent else None,
        end=end if persistent else None, kind="persistent" if persistent else "ordinary",
        complete=eigenpairs.complete, nullity=eigenpairs.nullity,
        metadata={**obj.metadata, "coefficient_field": "R", "native_basis_keys": operator.basis_keys,
                  "basis_semantics": "ordered factor-simplex pair at source time",
                  "operator_dimension": size,
                  "source_chain_dimension": size,
                  "structural_absence": size == 0,
                  "estimated_solver_workspace_bytes": estimated_solver_bytes,
                  "eigenpair_residual_max": eigenpairs.residual_max,
                  "laplacian_diagnostics": asdict(operator.diagnostics), "eigenvalue_zero_tolerance": tol})


def laplacian(obj, dimension=0, *, scale=None, return_eigenvectors=False,
              k=None, return_matrix=False, max_dense_entries=4_000_000,
              max_dense_bytes=256 * 1024**2, rank_rtol=None, tol=1e-10):
    """Ordinary real interaction Laplacian and optional labelled eigenvectors."""
    obj = _as_topology(obj)
    degree = _dimension(dimension, "dimension")
    _object(obj, degree)
    time = _query(obj, scale)
    return _spectrum(obj, degree, time, time, persistent=False,
        return_eigenvectors=return_eigenvectors, k=k, return_matrix=return_matrix,
        max_dense_entries=max_dense_entries, max_dense_bytes=max_dense_bytes,
        rank_rtol=rank_rtol, tol=tol)


def persistent_laplacian(obj, dimension=0, *, start, end, return_eigenvectors=False,
                         k=None, return_matrix=False, max_dense_entries=4_000_000,
                         max_dense_bytes=256 * 1024**2, rank_rtol=None, tol=1e-10):
    """Genuine two-time real Laplacian in the ordered source-time basis."""
    obj = _as_topology(obj)
    degree = _dimension(dimension, "dimension")
    _object(obj, degree)
    if start is None or end is None:
        raise ValueError("start and end must be finite scalar thresholds")
    source, target = _query(obj, start), _query(obj, end)
    if source > target:
        raise ValueError("start must not exceed end")
    return _spectrum(obj, degree, source, target, persistent=True,
        return_eigenvectors=return_eigenvectors, k=k, return_matrix=return_matrix,
        max_dense_entries=max_dense_entries, max_dense_bytes=max_dense_bytes,
        rank_rtol=rank_rtol, tol=tol)


__all__ = ["FactorComplex", "FactorComplexBuilder", "InteractionChain",
           "homology", "persistence", "laplacian", "persistent_laplacian", "DEFINITION_ID"]
