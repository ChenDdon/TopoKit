"""Explicit-input interaction construction plus analysis workflows.

These reference workflows preserve native GF(2) interval results, including
arbitrary finite scalar grades. Builders never run the barcode reduction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set
from numbers import Real
from typing import Any

from ..builders.interaction import _cell_estimate
from .._validation import dimension
from ..exceptions import ResourceLimitError
from ..core._interaction.model import FilteredSimplicialComplex
from ..core._interaction.interaction import build_interaction_chain_complex
from ..core._interaction.persistence import ReductionBackend, compute_persistence
from ..core._interaction.results import PersistenceResult


def compute_interaction_persistence(
    factors: Sequence[FilteredSimplicialComplex],
    *,
    max_dimension: int,
    include_diagonal: bool = False,
    minimum_persistence: Real = 0.0,
    backend: ReductionBackend = "auto",
    validate: bool = False,
    max_cells: int = 1_000_000,
) -> PersistenceResult:
    """Construct an explicit two-factor chain, then reduce its filtration."""
    factors = tuple(factors)
    if len(factors) != 2:
        raise ValueError("public interaction workflows require exactly two factors")
    highest = dimension(max_dimension)
    budget = dimension(max_cells, "max_cells")
    if budget == 0:
        raise ValueError("max_cells must be positive")
    estimate = _cell_estimate(factors, highest)
    if sum(estimate) > budget:
        raise ResourceLimitError(
            f"Interaction construction upper bound is {sum(estimate):,} cells, "
            f"above max_cells={budget:,}; reduce the factors or explicitly raise the budget")
    chain = build_interaction_chain_complex(
        factors, max_homology_dimension=highest, validate=validate)
    return compute_persistence(chain, include_diagonal=include_diagonal,
                               minimum_persistence=minimum_persistence,
                               backend=backend)


def persistence_from_dimension_dicts(
    complexes: Sequence[Mapping[int, Sequence[Any]]],
    filtrations: Sequence[Mapping[int, Sequence[Real]]],
    *,
    max_dimension: int,
    complete_faces: bool = False,
    include_diagonal: bool = False,
    minimum_persistence: Real = 0.0,
    backend: ReductionBackend = "auto",
    validate: bool = False,
    max_cells: int = 1_000_000,
) -> PersistenceResult:
    """Compute PIH directly from the legacy ``{dimension: values}`` format."""

    if len(complexes) != 2 or len(filtrations) != 2:
        raise ValueError("public interaction workflows require exactly two factor dictionaries")
    factors = tuple(
        FilteredSimplicialComplex.from_dimension_dict(
            complex_, filtration, complete_faces=complete_faces
        )
        for complex_, filtration in zip(complexes, filtrations)
    )
    return compute_interaction_persistence(
        factors,
        max_dimension=max_dimension,
        include_diagonal=include_diagonal,
        minimum_persistence=minimum_persistence,
        backend=backend,
        validate=validate,
        max_cells=max_cells,
    )


__all__ = ["compute_interaction_persistence", "persistence_from_dimension_dicts"]


def homology_at_pair(obj, scale_a, scale_b=None, *, max_dimension=2,
                     max_cells=None, validate=True, **options):
    """Compute ordinary homology at independent inclusive factor thresholds."""
    from ..builders.interaction import snapshot_at_pair
    from ..core.interaction import homology

    snapshot = snapshot_at_pair(obj, scale_a, scale_b, max_dimension=max_dimension,
                                max_cells=max_cells, validate=validate)
    return homology(snapshot, max_dimension=max_dimension, scale=0.0, **options)


def laplacian_at_pair(obj, scale_a, scale_b=None, *, dimension=0,
                      max_cells=None, validate=True, **options):
    """Compute an ordinary interaction Laplacian at a factor-threshold pair.

    The result's ``scale`` is snapshot stage 0; ``factor_scale_pair`` metadata
    records the actual two thresholds. Eigenvectors retain stable-ID bases.
    """
    from ..builders.interaction import snapshot_at_pair
    from ..core.interaction import laplacian

    snapshot = snapshot_at_pair(obj, scale_a, scale_b, max_dimension=dimension,
                                max_cells=max_cells, validate=validate)
    return laplacian(snapshot, dimension=dimension, scale=0.0, **options)


def persistence_along_pairs(obj, filtration_a, filtration_b=None, *, progression=None,
                            max_dimension=2, max_cells=None, validate=True, **options):
    """Compute sampled, window-censored PH along one monotone paired path.

    This opt-in workflow does not replace exact no-schedule scalar PH. The
    second schedule defaults to the first schedule, not its factor births.
    """
    from ..builders.interaction import regrade_pair_filtration
    from ..core.interaction import persistence

    paired_filtration = regrade_pair_filtration(
        obj, filtration_a, filtration_b, progression=progression,
        max_dimension=max_dimension, max_cells=max_cells, validate=validate)
    return persistence(paired_filtration, max_dimension=max_dimension, **options)


def persistent_laplacian_between_pairs(obj, dimension=0, *, source_pair, target_pair,
                                       max_cells=None, validate=True, **options):
    """Compute the genuine source-pair→target-pair persistent Laplacian.

    The target must be componentwise at least the source, within each factor's
    constructed range. The native persistent boundary-space projection is
    used, not subtraction/comparison of ordinary spectra. Only endpoint pairs
    determine this inclusion map; the two scalar stages are 0 and 1. The
    returned matrix/eigenvectors use the SOURCE pair's interaction-cell basis.
    """
    from ..builders.interaction import regrade_pair_filtration
    from ..core.interaction import persistent_laplacian

    validated_pairs = []
    for name, pair in (("source_pair", source_pair), ("target_pair", target_pair)):
        if isinstance(pair, (str, bytes, Mapping, Set)):
            raise ValueError(f"{name} must contain exactly two factor thresholds")
        try:
            pair = tuple(pair)
        except TypeError as error:
            raise ValueError(f"{name} must contain exactly two factor thresholds") from error
        if len(pair) != 2:
            raise ValueError(f"{name} must contain exactly two factor thresholds")
        validated_pairs.append(pair)
    source_scales, target_scales = validated_pairs
    paired_filtration = regrade_pair_filtration(
        obj, (source_scales[0], target_scales[0]), (source_scales[1], target_scales[1]),
        max_dimension=dimension, max_cells=max_cells, validate=validate)
    paired_filtration.metadata.update({"observation_kind": "persistent_factor_pair_inclusion",
                          "source_factor_scales": paired_filtration.metadata["factor_thresholds"][0],
                          "target_factor_scales": paired_filtration.metadata["factor_thresholds"][1],
                          "persistent_pair_semantics": "source_basis_with_target_boundary_projection"})
    return persistent_laplacian(paired_filtration, dimension=dimension, start=0.0, end=1.0, **options)


__all__ += ["homology_at_pair", "laplacian_at_pair", "persistence_along_pairs",
            "persistent_laplacian_between_pairs"]
