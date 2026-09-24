"""Immutable public result objects."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf

from .model import Hyperedge


Chain = tuple[Hyperedge, ...]
RealMatrix = tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class DimensionDiagnostics:
    dimension: int
    hyperedge_count: int
    omega_dimension: int
    boundary_rank: int
    constraint_row_count: int
    constraint_rank: int
    omega_backend: str
    generator_type_counts: tuple[tuple[str, int], ...] = ()
    max_late_faces_at_birth: int = 0


@dataclass(frozen=True, slots=True)
class HomologyDiagnostics:
    definition_id: str
    coefficient_field: str
    dimensions: tuple[DimensionDiagnostics, ...]
    ignored_hyperedges_above: int


@dataclass(frozen=True, slots=True)
class HomologyResult:
    """Betti numbers and optional representative chains by dimension."""

    betti_numbers: tuple[int, ...]
    representatives: tuple[tuple[Chain, ...], ...] | None
    diagnostics: HomologyDiagnostics

    def betti(self, dimension: int) -> int:
        if dimension < 0:
            raise ValueError("dimension must be non-negative")
        return (
            self.betti_numbers[dimension] if dimension < len(self.betti_numbers) else 0
        )


@dataclass(frozen=True, slots=True)
class PersistenceInterval:
    dimension: int
    birth: float
    death: float = inf
    birth_index: int | None = None
    death_index: int | None = None
    kind: str = "ordinary"

    @property
    def is_infinite(self) -> bool:
        return self.death == inf

    @property
    def persistence(self) -> float:
        return self.death - self.birth


@dataclass(frozen=True, slots=True)
class PersistenceDiagnostics:
    definition_id: str
    coefficient_field: str
    critical_value_count: int
    chain_generator_counts: tuple[int, ...]
    omega_backends: tuple[str, ...]
    reduction_column_count: int
    finite_pair_count: int
    low_dimensional_backend: str
    omega2_generator_type_counts: tuple[tuple[str, int], ...] = ()
    omega2_max_late_faces_at_birth: int = 0
    higher_dimensional_backend: str = "not_requested"
    notes: tuple[str, ...] = ()
    vertex_count: int = 0
    edge_count: int = 0
    negative_edge_count: int = 0
    positive_edge_count: int = 0
    boundary_generator_count: int = 0
    boundary_rank: int = 0
    boundary_generator_type_counts: tuple[tuple[str, int], ...] = ()
    connection_support: str = "not_applicable"
    ambient_dimension: int | None = None
    intrinsic_dimension: int | None = None
    support_edge_count: int | None = None
    support_maximal_simplex_count: int | None = None
    projected_to_affine_hull: bool = False


@dataclass(frozen=True, slots=True)
class PersistenceResult:
    intervals: tuple[PersistenceInterval, ...]
    max_dimension: int
    diagnostics: PersistenceDiagnostics

    def in_dimension(self, dimension: int) -> tuple[PersistenceInterval, ...]:
        return tuple(
            interval for interval in self.intervals if interval.dimension == dimension
        )

    def betti_at(self, dimension: int, threshold: float) -> int:
        return sum(
            interval.birth <= threshold < interval.death
            for interval in self.intervals
            if interval.dimension == dimension
        )

    def persistent_betti(self, dimension: int, start: float, end: float) -> int:
        if start > end:
            raise ValueError("start must not exceed end")
        return sum(
            interval.birth <= start and interval.death > end
            for interval in self.intervals
            if interval.dimension == dimension
        )


@dataclass(frozen=True, slots=True)
class RealChainDimensionDiagnostics:
    """Construction details for one real embedded chain space."""

    dimension: int
    hyperedge_count: int
    omega_dimension: int
    constraint_row_count: int
    constraint_rank: int
    max_missing_face_count: int
    omega_backend: str


@dataclass(frozen=True, slots=True)
class LaplacianDiagnostics:
    """Numerical and backend metadata shared by Laplacian results."""

    definition_id: str
    coefficient_field: str
    tolerance: float
    chain_dimensions: tuple[RealChainDimensionDiagnostics, ...]
    low_dimensional_backend: str
    higher_dimensional_backend: str
    end_chain_dimensions: tuple[RealChainDimensionDiagnostics, ...] = ()
    ignored_hyperedges_above: int = 0
    notes: tuple[str, ...] = ()
    connection_support: str = "not_applicable"
    ambient_dimension: int | None = None
    intrinsic_dimension: int | None = None
    support_edge_count: int | None = None
    retained_edge_count: int | None = None
    support_maximal_simplex_count: int | None = None
    projected_to_affine_hull: bool = False


@dataclass(frozen=True, slots=True)
class LaplacianDimensionResult:
    """Spectrum and optional matrix for one Laplacian dimension."""

    dimension: int
    eigenvalues: tuple[float, ...]
    nullity: int
    omega_dimension: int
    down_rank: int
    up_rank: int
    up_domain_dimension: int
    smallest_positive_eigenvalue: float | None
    matrix: RealMatrix | None = None

    @property
    def nonzero_eigenvalues(self) -> tuple[float, ...]:
        return tuple(value for value in self.eigenvalues if value > 0.0)

    @property
    def maximum_eigenvalue(self) -> float:
        return max(self.eigenvalues, default=0.0)

    @property
    def trace(self) -> float:
        return sum(self.eigenvalues)

    @property
    def nonzero_mean(self) -> float:
        values = self.nonzero_eigenvalues
        return sum(values) / len(values) if values else 0.0

    @property
    def nonzero_variance(self) -> float:
        values = self.nonzero_eigenvalues
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        return sum((value - mean) ** 2 for value in values) / len(values)

    @property
    def nonzero_standard_deviation(self) -> float:
        return self.nonzero_variance**0.5


@dataclass(frozen=True, slots=True)
class LaplacianResult:
    """Ordinary hyperdigraph Laplacians at one filtration snapshot."""

    dimensions: tuple[LaplacianDimensionResult, ...]
    max_dimension: int
    diagnostics: LaplacianDiagnostics

    def in_dimension(self, dimension: int) -> LaplacianDimensionResult:
        if dimension < 0:
            raise ValueError("dimension must be non-negative")
        if dimension > self.max_dimension:
            raise IndexError("dimension was not computed")
        return self.dimensions[dimension]

    @property
    def betti_numbers(self) -> tuple[int, ...]:
        """Real Betti numbers, equal to the Laplacian nullities."""

        return tuple(result.nullity for result in self.dimensions)


@dataclass(frozen=True, slots=True)
class PersistentLaplacianResult:
    """The genuine persistent Laplacians for one pair ``start <= end``."""

    start: float
    end: float
    dimensions: tuple[LaplacianDimensionResult, ...]
    max_dimension: int
    diagnostics: LaplacianDiagnostics

    def in_dimension(self, dimension: int) -> LaplacianDimensionResult:
        if dimension < 0:
            raise ValueError("dimension must be non-negative")
        if dimension > self.max_dimension:
            raise IndexError("dimension was not computed")
        return self.dimensions[dimension]

    @property
    def persistent_betti_numbers(self) -> tuple[int, ...]:
        """Ranks of the persistent homology maps over the reals."""

        return tuple(result.nullity for result in self.dimensions)


@dataclass(frozen=True, slots=True)
class LaplacianSnapshot:
    threshold: float
    result: LaplacianResult


@dataclass(frozen=True, slots=True)
class LaplacianFiltrationResult:
    """Ordinary spectra evaluated at selected filtration thresholds."""

    snapshots: tuple[LaplacianSnapshot, ...]
    max_dimension: int

    @property
    def thresholds(self) -> tuple[float, ...]:
        return tuple(snapshot.threshold for snapshot in self.snapshots)

    def at(self, threshold: float) -> LaplacianResult:
        for snapshot in self.snapshots:
            if snapshot.threshold == threshold:
                return snapshot.result
        raise KeyError(f"threshold {threshold!r} was not evaluated")


__all__ = [
    "Chain",
    "DimensionDiagnostics",
    "HomologyDiagnostics",
    "HomologyResult",
    "PersistenceDiagnostics",
    "PersistenceInterval",
    "PersistenceResult",
    "LaplacianDiagnostics",
    "LaplacianDimensionResult",
    "LaplacianFiltrationResult",
    "LaplacianResult",
    "LaplacianSnapshot",
    "PersistentLaplacianResult",
    "RealChainDimensionDiagnostics",
    "RealMatrix",
]
