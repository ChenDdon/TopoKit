"""Explicit legacy construct-and-compute compositions.

These helpers intentionally live above builders and core. The canonical
shared-result API separates builders.hyperdigraph from core.hyperdigraph.
"""
from __future__ import annotations
from collections.abc import Hashable, Iterable, Mapping
from numbers import Real
from typing import Any
from dataclasses import replace
from ..core._hyperdigraph.model import SequenceHyperdigraph
from ..core._hyperdigraph.results import (HomologyResult, PersistenceResult, LaplacianResult,
    PersistentLaplacianResult, LaplacianFiltrationResult, LaplacianDiagnostics, LaplacianSnapshot)
from ..core._hyperdigraph.chain_complex import compute_homology
from ..core._hyperdigraph.persistence import compute_persistence
from ..core._hyperdigraph.laplacian import (RealOmegaBackend, compute_laplacian,
    compute_laplacian_filtration, compute_persistent_laplacian, _validate_max_dimension)
from ..core._hyperdigraph.path_compatible import (ScoreDirectedEdgeFiltration,
    compute_path_compatible_persistence, _support_notes)
from ..builders._hyperdigraph_geometry import ConnectionSupport
from ..builders._hyperdigraph_converters import (hyperdigraph_from_adjacency_matrix,
    hyperdigraph_from_dimension_dict, filtered_hyperdigraph_from_distance_matrix,
    _validate_nonnegative_dimension)
from ..builders._hyperdigraph_compact import (score_directed_filtration_from_points,
    score_directed_distance_matrix, _score_directed_distance_matrix,
    _point_cloud_maximum_adjacency, _validate_minimum_hyperdigraph_point_support)
_RealValue = Real | float

def compute_homology_from_adjacency_matrix(
    adjacency_matrix: object,
    max_dimension: int = 1,
    *,
    vertices: Iterable[Hashable] | None = None,
    minimum_adjacency: object | None = None,
    maximum_adjacency: object | None = None,
    minimum_hyperdigraph: SequenceHyperdigraph | None = None,
    maximum_hyperdigraph: SequenceHyperdigraph | None = None,
    representatives: bool = False,
    omega_backend: str = "auto",
) -> HomologyResult:
    """Compute native homology directly from a binary adjacency matrix."""

    _validate_nonnegative_dimension(max_dimension, "max_dimension")
    hyperdigraph = hyperdigraph_from_adjacency_matrix(
        adjacency_matrix,
        vertices=vertices,
        max_path_dimension=max_dimension + 1,
        minimum_adjacency=minimum_adjacency,
        maximum_adjacency=maximum_adjacency,
        minimum_hyperdigraph=minimum_hyperdigraph,
        maximum_hyperdigraph=maximum_hyperdigraph,
        include_all_vertices=True,
    )
    from ..core._hyperdigraph.chain_complex import compute_homology

    return compute_homology(
        hyperdigraph,
        max_dimension,
        representatives=representatives,
        omega_backend=omega_backend,  # type: ignore[arg-type]
    )

def compute_persistence_from_distance_matrix(
    distance_matrix: object,
    max_dimension: int = 1,
    *,
    vertices: Iterable[Hashable] | None = None,
    filtration_start: _RealValue = 0.0,
    minimum_adjacency: object | None = None,
    maximum_adjacency: object | None = None,
    minimum_hyperdigraph: SequenceHyperdigraph | None = None,
    maximum_hyperdigraph: SequenceHyperdigraph | None = None,
    include_diagonal: bool = False,
    omega_backend: str = "auto",
    reduction_backend: str = "auto",
) -> PersistenceResult:
    """Compute persistence directly from a directed distance matrix."""

    _validate_nonnegative_dimension(max_dimension, "max_dimension")
    filtration = filtered_hyperdigraph_from_distance_matrix(
        distance_matrix,
        vertices=vertices,
        max_path_dimension=max_dimension + 1,
        filtration_start=filtration_start,
        minimum_adjacency=minimum_adjacency,
        maximum_adjacency=maximum_adjacency,
        minimum_hyperdigraph=minimum_hyperdigraph,
        maximum_hyperdigraph=maximum_hyperdigraph,
        include_all_vertices=True,
    )
    from ..core._hyperdigraph.persistence import compute_persistence

    return compute_persistence(
        filtration,
        max_dimension,
        include_diagonal=include_diagonal,
        omega_backend=omega_backend,  # type: ignore[arg-type]
        reduction_backend=reduction_backend,  # type: ignore[arg-type]
    )


def compute_persistence_from_point_cloud(
    points: Iterable[Iterable[Any]],
    point_weights: Iterable[Any],
    max_dimension: int = 1,
    *,
    max_distance: Any | None = None,
    connection_support: ConnectionSupport = "delaunay",
    include_diagonal: bool = False,
    omega_backend: str = "auto",
    reduction_backend: str = "auto",
    storage_backend: str = "auto",
) -> PersistenceResult:
    """Compute score-directed point-cloud persistent hyperdigraph homology.

    Dimensions zero and one use the compact two-boundary-family engine.
    Requests containing dimension two or higher materialize the corresponding
    simple directed paths and use the normal filtered-chain reduction.
    """

    if isinstance(max_dimension, bool) or not isinstance(max_dimension, int):
        raise TypeError("max_dimension must be a non-negative integer")
    if max_dimension < 0:
        raise ValueError("max_dimension must be non-negative")
    compact = score_directed_filtration_from_points(
        points,
        point_weights,
        max_distance=max_distance,
        connection_support=connection_support,
        storage_backend=storage_backend,
    )
    use_compact_low_dimensions = (
        max_dimension <= 1
        and omega_backend == "auto"
        and reduction_backend in {"auto", "native_h1"}
    )
    if use_compact_low_dimensions:
        return compute_path_compatible_persistence(
            compact,
            max_dimension,
            include_diagonal=include_diagonal,
        )

    from ..builders._hyperdigraph_converters import filtered_hyperdigraph_from_distance_matrix
    from ..core._hyperdigraph.persistence import compute_persistence

    native_filtration = filtered_hyperdigraph_from_distance_matrix(
        score_directed_distance_matrix(compact),
        max_path_dimension=max_dimension + 1,
        include_all_vertices=True,
    )
    result = compute_persistence(
        native_filtration,
        max_dimension,
        include_diagonal=include_diagonal,
        omega_backend=omega_backend,  # type: ignore[arg-type]
        reduction_backend=reduction_backend,  # type: ignore[arg-type]
    )
    materialization_note = (
        "Point-cloud H2 and higher used materialized simple paths and the normal persistence reduction."
        if max_dimension >= 2
        else "Explicit backend selection used materialized simple paths and the normal persistence API."
    )
    diagnostics = replace(
        result.diagnostics,
        notes=result.diagnostics.notes
        + _support_notes(compact)
        + (materialization_note,),
        vertex_count=compact.vertex_count,
        edge_count=compact.edge_count,
        connection_support=compact.connection_support,
        ambient_dimension=compact.ambient_dimension,
        intrinsic_dimension=compact.intrinsic_dimension,
        support_edge_count=compact.support_edge_count,
        support_maximal_simplex_count=compact.support_maximal_simplex_count,
        projected_to_affine_hull=compact.projected_to_affine_hull,
    )
    return replace(result, diagnostics=diagnostics)


def compute_laplacian_from_adjacency_matrix(
    adjacency_matrix: object,
    max_dimension: int = 1,
    *,
    vertices: Iterable[Hashable] | None = None,
    minimum_adjacency: object | None = None,
    maximum_adjacency: object | None = None,
    minimum_hyperdigraph: SequenceHyperdigraph | None = None,
    maximum_hyperdigraph: SequenceHyperdigraph | None = None,
    tolerance: Real | None = None,
    return_matrices: bool = False,
    omega_backend: RealOmegaBackend = "auto",
) -> LaplacianResult:
    """Compute ordinary Laplacians from a binary adjacency matrix."""

    _validate_max_dimension(max_dimension)
    from ..builders._hyperdigraph_converters import hyperdigraph_from_adjacency_matrix

    hyperdigraph = hyperdigraph_from_adjacency_matrix(
        adjacency_matrix,
        vertices=vertices,
        max_path_dimension=max_dimension + 1,
        minimum_adjacency=minimum_adjacency,
        maximum_adjacency=maximum_adjacency,
        minimum_hyperdigraph=minimum_hyperdigraph,
        maximum_hyperdigraph=maximum_hyperdigraph,
        include_all_vertices=True,
    )
    return compute_laplacian(
        hyperdigraph,
        max_dimension,
        tolerance=tolerance,
        return_matrices=return_matrices,
        omega_backend=omega_backend,
    )

def compute_laplacian_from_dimension_dict(
    hyperedges_by_dimension: Mapping[int, Iterable[Iterable[Hashable]]],
    max_dimension: int | None = None,
    *,
    vertices: Iterable[Hashable] | None = None,
    include_all_vertices: bool = False,
    tolerance: Real | None = None,
    return_matrices: bool = False,
    omega_backend: RealOmegaBackend = "auto",
) -> LaplacianResult:
    """Compute ordinary Laplacians from the legacy dimension dictionary."""

    from ..builders._hyperdigraph_converters import hyperdigraph_from_dimension_dict

    hyperdigraph = hyperdigraph_from_dimension_dict(
        hyperedges_by_dimension,
        vertices=vertices,
        include_all_vertices=include_all_vertices,
    )
    return compute_laplacian(
        hyperdigraph,
        max_dimension,
        tolerance=tolerance,
        return_matrices=return_matrices,
        omega_backend=omega_backend,
    )

def compute_persistent_laplacian_from_distance_matrix(
    distance_matrix: object,
    start: Real,
    end: Real,
    max_dimension: int = 1,
    *,
    vertices: Iterable[Hashable] | None = None,
    filtration_start: Real = 0.0,
    minimum_adjacency: object | None = None,
    maximum_adjacency: object | None = None,
    minimum_hyperdigraph: SequenceHyperdigraph | None = None,
    maximum_hyperdigraph: SequenceHyperdigraph | None = None,
    tolerance: Real | None = None,
    return_matrices: bool = False,
    omega_backend: RealOmegaBackend = "auto",
) -> PersistentLaplacianResult:
    """Compute a pairwise persistent Laplacian from directed distances."""

    _validate_max_dimension(max_dimension)
    from ..builders._hyperdigraph_converters import filtered_hyperdigraph_from_distance_matrix

    filtration = filtered_hyperdigraph_from_distance_matrix(
        distance_matrix,
        vertices=vertices,
        max_path_dimension=max_dimension + 1,
        filtration_start=filtration_start,
        minimum_adjacency=minimum_adjacency,
        maximum_adjacency=maximum_adjacency,
        minimum_hyperdigraph=minimum_hyperdigraph,
        maximum_hyperdigraph=maximum_hyperdigraph,
        include_all_vertices=True,
    )
    return compute_persistent_laplacian(
        filtration,
        start,
        end,
        max_dimension,
        tolerance=tolerance,
        return_matrices=return_matrices,
        omega_backend=omega_backend,
    )

def compute_laplacian_filtration_from_distance_matrix(
    distance_matrix: object,
    max_dimension: int = 1,
    *,
    vertices: Iterable[Hashable] | None = None,
    filtration_start: Real = 0.0,
    minimum_adjacency: object | None = None,
    maximum_adjacency: object | None = None,
    minimum_hyperdigraph: SequenceHyperdigraph | None = None,
    maximum_hyperdigraph: SequenceHyperdigraph | None = None,
    thresholds: Iterable[Real] | None = None,
    tolerance: Real | None = None,
    return_matrices: bool = False,
    omega_backend: RealOmegaBackend = "auto",
) -> LaplacianFiltrationResult:
    """Evaluate ordinary Laplacian spectra along a distance filtration."""

    _validate_max_dimension(max_dimension)
    from ..builders._hyperdigraph_converters import filtered_hyperdigraph_from_distance_matrix

    filtration = filtered_hyperdigraph_from_distance_matrix(
        distance_matrix,
        vertices=vertices,
        max_path_dimension=max_dimension + 1,
        filtration_start=filtration_start,
        minimum_adjacency=minimum_adjacency,
        maximum_adjacency=maximum_adjacency,
        minimum_hyperdigraph=minimum_hyperdigraph,
        maximum_hyperdigraph=maximum_hyperdigraph,
        include_all_vertices=True,
    )
    return compute_laplacian_filtration(
        filtration,
        max_dimension,
        thresholds=thresholds,
        tolerance=tolerance,
        return_matrices=return_matrices,
        omega_backend=omega_backend,
    )

def _point_cloud_laplacian_diagnostics(
    diagnostics: LaplacianDiagnostics,
    compact: ScoreDirectedEdgeFiltration,
) -> LaplacianDiagnostics:
    notes = (
        f"Point-cloud connection support: {compact.connection_support}.",
        f"{compact.edge_count} directed edges were retained from "
        f"{compact.support_edge_count} support edges.",
    )
    if compact.connection_support == "delaunay":
        notes += (
            (
                "Delaunay support was computed after projection to the intrinsic affine hull."
                if compact.projected_to_affine_hull
                else "Delaunay support used the ambient-coordinate affine dimension."
            ),
        )
    return replace(
        diagnostics,
        notes=diagnostics.notes + notes,
        connection_support=compact.connection_support,
        ambient_dimension=compact.ambient_dimension,
        intrinsic_dimension=compact.intrinsic_dimension,
        support_edge_count=compact.support_edge_count,
        retained_edge_count=compact.edge_count,
        support_maximal_simplex_count=compact.support_maximal_simplex_count,
        projected_to_affine_hull=compact.projected_to_affine_hull,
    )

def compute_persistent_laplacian_from_point_cloud(
    points: Iterable[Iterable[Real]],
    point_weights: Iterable[Real],
    start: Real,
    end: Real,
    max_dimension: int = 1,
    *,
    vertices: Iterable[Hashable] | None = None,
    max_distance: Real | None = None,
    connection_support: ConnectionSupport = "delaunay",
    filtration_start: Real = 0.0,
    minimum_adjacency: object | None = None,
    maximum_adjacency: object | None = None,
    minimum_hyperdigraph: SequenceHyperdigraph | None = None,
    maximum_hyperdigraph: SequenceHyperdigraph | None = None,
    tolerance: Real | None = None,
    return_matrices: bool = False,
    omega_backend: RealOmegaBackend = "auto",
    storage_backend: str = "auto",
) -> PersistentLaplacianResult:
    """Compute a score-directed point-cloud persistent Laplacian.

    Unlike the packed homology-only engine, this spectral adapter materializes
    the native simple paths needed through ``max_dimension + 1``.  A full
    ``L1`` spectrum is therefore intended for moderate systems.
    """

    distances, compact = _score_directed_distance_matrix(
        points,
        point_weights,
        max_distance,
        connection_support,
        storage_backend,
    )
    effective_minimum, effective_maximum = _point_cloud_maximum_adjacency(
        compact, minimum_adjacency, maximum_adjacency
    )
    point_labels = (
        tuple(vertices) if vertices is not None else tuple(range(len(distances)))
    )
    _validate_minimum_hyperdigraph_point_support(
        minimum_hyperdigraph, point_labels, effective_maximum
    )
    result = compute_persistent_laplacian_from_distance_matrix(
        distances,
        start,
        end,
        max_dimension,
        vertices=point_labels,
        filtration_start=filtration_start,
        minimum_adjacency=effective_minimum,
        maximum_adjacency=effective_maximum,
        minimum_hyperdigraph=minimum_hyperdigraph,
        maximum_hyperdigraph=maximum_hyperdigraph,
        tolerance=tolerance,
        return_matrices=return_matrices,
        omega_backend=omega_backend,
    )
    return replace(
        result,
        diagnostics=_point_cloud_laplacian_diagnostics(result.diagnostics, compact),
    )

def compute_laplacian_filtration_from_point_cloud(
    points: Iterable[Iterable[Real]],
    point_weights: Iterable[Real],
    max_dimension: int = 1,
    *,
    vertices: Iterable[Hashable] | None = None,
    max_distance: Real | None = None,
    connection_support: ConnectionSupport = "delaunay",
    filtration_start: Real = 0.0,
    minimum_adjacency: object | None = None,
    maximum_adjacency: object | None = None,
    minimum_hyperdigraph: SequenceHyperdigraph | None = None,
    maximum_hyperdigraph: SequenceHyperdigraph | None = None,
    thresholds: Iterable[Real] | None = None,
    tolerance: Real | None = None,
    return_matrices: bool = False,
    omega_backend: RealOmegaBackend = "auto",
    storage_backend: str = "auto",
) -> LaplacianFiltrationResult:
    """Evaluate score-directed point-cloud snapshot Laplacian spectra."""

    distances, compact = _score_directed_distance_matrix(
        points,
        point_weights,
        max_distance,
        connection_support,
        storage_backend,
    )
    effective_minimum, effective_maximum = _point_cloud_maximum_adjacency(
        compact, minimum_adjacency, maximum_adjacency
    )
    point_labels = (
        tuple(vertices) if vertices is not None else tuple(range(len(distances)))
    )
    _validate_minimum_hyperdigraph_point_support(
        minimum_hyperdigraph, point_labels, effective_maximum
    )
    spectral_scan = compute_laplacian_filtration_from_distance_matrix(
        distances,
        max_dimension,
        vertices=point_labels,
        filtration_start=filtration_start,
        minimum_adjacency=effective_minimum,
        maximum_adjacency=effective_maximum,
        minimum_hyperdigraph=minimum_hyperdigraph,
        maximum_hyperdigraph=maximum_hyperdigraph,
        thresholds=thresholds,
        tolerance=tolerance,
        return_matrices=return_matrices,
        omega_backend=omega_backend,
    )
    snapshots = tuple(
        LaplacianSnapshot(
            snapshot.threshold,
            replace(
                snapshot.result,
                diagnostics=_point_cloud_laplacian_diagnostics(
                    snapshot.result.diagnostics, compact
                ),
            ),
        )
        for snapshot in spectral_scan.snapshots
    )
    return replace(spectral_scan, snapshots=snapshots)


__all__ = ['compute_homology_from_adjacency_matrix', 'compute_persistence_from_distance_matrix', 'compute_persistence_from_point_cloud', 'compute_laplacian_from_adjacency_matrix', 'compute_laplacian_from_dimension_dict', 'compute_persistent_laplacian_from_distance_matrix', 'compute_laplacian_filtration_from_distance_matrix', 'compute_persistent_laplacian_from_point_cloud', 'compute_laplacian_filtration_from_point_cloud']
