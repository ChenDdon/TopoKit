"""Analysis-only simplicial homology, persistence and Hodge Laplacians.

Inputs are defined simplicial objects or Topology envelopes. This layer does
not read scientific files or construct geometry from coordinates. Ordinary
snapshot spectra and genuine two-scale persistent spectra remain distinct.
"""

import math
import warnings

from ..results import (Topology, HomologyResult, PersistenceInterval,
                      PersistenceResult, SpectrumResult)
from ._spectral import solve_symmetric
from .._validation import dimension as validate_dimension, validate_query
from . import _simplicial as _native
from ._simplicial.complex import as_complex


SimplicialComplex = _native.SimplexTree
ComplexityLimitError = _native.ComplexityLimitError

__all__ = ["SimplicialComplex", "ComplexityLimitError",
           "homology", "persistence", "laplacian",
           "persistent_laplacian"]


def _field_name(field):
    return "R" if field == "real" else f"GF({field})"


def _unpack(obj):
    if isinstance(obj, Topology):
        if obj.kind != "simplicial":
            raise ValueError("Expected a simplicial Topology; mathematical cores are not interchangeable")
        native, cloud = as_complex(obj.native), obj.cloud
        metadata = dict(native.metadata)
        metadata.update(obj.metadata)
    else:
        native, cloud = as_complex(obj), None
        metadata = dict(native.metadata)
    metadata.setdefault("route", "simplicial")
    metadata.setdefault("source", "explicit_simplicial_object")
    metadata.setdefault("filtration_start", min((v for _, v in native.get_filtration()), default=0.0))
    metadata.setdefault("filtration_end", math.inf)
    metadata.setdefault("scale_units", "user_defined")
    return native, cloud, metadata


def _label_simplex(simplex, cloud):
    return tuple(cloud.ids[index] for index in simplex) if cloud is not None else tuple(simplex)


def _basis(native, q, cloud):
    return tuple(_label_simplex(simplex, cloud) for simplex in native.simplices(q))


def _warn_coface_truncation(metadata, q):
    """Flag omitted geometric cofaces without redefining the supplied object."""
    if metadata.get("construction_kind") != "point_cloud":
        return
    required = min(q + 1, metadata["geometry_dimension_bound"])
    available = metadata["max_simplex_dimension"]
    if available < required:
        message = (f"Point-cloud construction is truncated at simplex dimension {available}; "
                   f"analysis through dimension {q} can omit H{q} deaths or the L{q} up-term. "
                   "Results describe the explicitly truncated complex, not the full geometric construction.")
        metadata["dimension_warnings"] = metadata.get("dimension_warnings", ()) + (message,)
        warnings.warn(message, UserWarning, stacklevel=3)


def _check_analysis_degree(metadata, q):
    # The constructor already reports a deliberately truncated initial request.
    # A later higher-degree request needs its own visible warning.
    if q > metadata.get("max_homology_dimension", q):
        _warn_coface_truncation(metadata, q)


def homology(obj, *, max_dimension=2, scale=None, field=2, tol=1e-10,
             max_dense_entries=4_000_000):
    """Ordinary homology of the supplied object or one filtration snapshot."""
    native, cloud, metadata = _unpack(obj)
    maximum_dimension = validate_dimension(max_dimension)
    _check_analysis_degree(metadata, maximum_dimension)
    scale = validate_query(metadata, scale)
    homology_result = _native.homology(native, max_dimension=maximum_dimension, scale=scale,
                              field=field, tol=tol, max_dense_entries=max_dense_entries)
    snapshot = native if scale is None else native.at(scale)
    metadata.update({"scale": scale, "max_dimension": maximum_dimension,
                     "chain_dimensions": homology_result.chain_dimensions,
                     "boundary_ranks": homology_result.boundary_ranks,
                     "euler_characteristic": homology_result.euler_characteristic,
                     "basis": tuple(_basis(snapshot, q, cloud) for q in range(maximum_dimension + 1)),
                     "tolerance": tol})
    return HomologyResult(homology_result.betti_numbers, _field_name(homology_result.field), metadata)


def persistence(obj, *, max_dimension=2, field=2, include_zero=False, clearing=True):
    """Prime-field intervals; infinity means survival to this input's cutoff."""
    native, cloud, metadata = _unpack(obj)
    maximum_dimension = validate_dimension(max_dimension)
    _check_analysis_degree(metadata, maximum_dimension)
    persistence_result = _native.persistent_homology(native, max_dimension=maximum_dimension, field=field,
                                         include_zero=include_zero, clearing=clearing)
    filtration_start = metadata["filtration_start"]
    intervals = tuple(PersistenceInterval(i.dimension, i.birth, i.death,
                                         i.birth == filtration_start) for i in persistence_result.intervals)
    metadata.update({"diagnostics": persistence_result.diagnostics,
                     "interval_simplex_provenance": tuple({
                         "birth_simplex": _label_simplex(i.birth_simplex, cloud),
                         "death_simplex": None if i.death_simplex is None else _label_simplex(i.death_simplex, cloud)
                     } for i in persistence_result.intervals)})
    return PersistenceResult(intervals, _field_name(persistence_result.field), maximum_dimension, metadata)


def _spectrum_result(native, cloud, metadata, matrix, q, *, scale=None,
                     start=None, end=None, kind="ordinary", k=None,
                     return_eigenvectors=False, return_matrix=False,
                     tol=1e-10, max_dense_entries=4_000_000):
    operator_dimension = int(matrix.shape[0])
    solution = solve_symmetric(matrix, k=k, return_eigenvectors=return_eigenvectors,
                               tol=tol, max_dense_entries=max_dense_entries)
    metadata.update({"coefficient_field": "R", "tolerance": tol,
                     "residual_max": solution.residual_max,
                     "operator_dimension": operator_dimension,
                     "source_chain_dimension": operator_dimension,
                     "structural_absence": operator_dimension == 0,
                     "basis_labels": "point_ids" if cloud is not None else "explicit_vertex_labels"})
    return SpectrumResult(q, solution.values, solution.vectors,
                          matrix if return_matrix else None, _basis(native, q, cloud),
                          scale, start, end, kind, solution.complete, solution.nullity,
                          metadata)


def laplacian(obj, dimension=0, *, scale=None, return_eigenvectors=False,
              k=None, return_matrix=False, tol=1e-10,
              max_dense_entries=4_000_000):
    """Real ordinary Hodge spectrum; eigenvectors and matrix export are opt-in.

    Partial spectra have ``complete=False`` and ``nullity=None``. They must
    not be interpreted as a complete count of homology classes.
    """
    native, cloud, metadata = _unpack(obj)
    q = validate_dimension(dimension, "dimension")
    _check_analysis_degree(metadata, q)
    scale = validate_query(metadata, scale)
    snapshot = native if scale is None else native.at(scale)
    matrix = _native.laplacian_matrix(snapshot, q, sparse_output=True,
                                      max_dense_entries=max_dense_entries)
    return _spectrum_result(snapshot, cloud, metadata, matrix, q, scale=scale,
                            k=k, return_eigenvectors=return_eigenvectors,
                            return_matrix=return_matrix, tol=tol,
                            max_dense_entries=max_dense_entries)


def persistent_laplacian(obj, dimension=0, *, start, end,
                         return_eigenvectors=False, k=None, return_matrix=False,
                         tol=1e-10, max_dense_entries=4_000_000):
    """Opt-in genuine two-scale operator, not successive snapshot spectra."""
    native, cloud, metadata = _unpack(obj)
    q = validate_dimension(dimension, "dimension")
    _check_analysis_degree(metadata, q)
    start, end = validate_query(metadata, start), validate_query(metadata, end)
    if start is None or end is None or start > end:
        raise ValueError("start and end must be finite scales with start <= end")
    source, target = native.at(start), native.at(end)
    matrix = _native.persistent_laplacian_matrix(source, target, q, tol=tol,
                                                 sparse_output=True,
                                                 max_dense_entries=max_dense_entries)
    return _spectrum_result(source, cloud, metadata, matrix, q, start=start, end=end,
                            kind="persistent", k=k,
                            return_eigenvectors=return_eigenvectors,
                            return_matrix=return_matrix, tol=tol,
                            max_dense_entries=max_dense_entries)
