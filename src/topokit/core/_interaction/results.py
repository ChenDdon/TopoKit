"""Immutable public persistence results."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf, isfinite


@dataclass(frozen=True, slots=True)
class PersistenceInterval:
    dimension: int
    birth: float
    death: float = inf
    birth_index: int | None = None
    death_index: int | None = None

    def __post_init__(self) -> None:
        if self.dimension < 0:
            raise ValueError("an interval dimension must be non-negative")
        if not isfinite(self.birth):
            raise ValueError("an interval birth must be finite")
        if self.death != inf and not isfinite(self.death):
            raise ValueError("an interval death must be finite or positive infinity")
        if self.death < self.birth:
            raise ValueError("an interval cannot die before it is born")

    @property
    def is_infinite(self) -> bool:
        return self.death == inf

    @property
    def persistence(self) -> float:
        return self.death - self.birth


@dataclass(frozen=True, slots=True)
class PersistenceDiagnostics:
    coefficient_field: str
    interaction_cell_counts: tuple[int, ...]
    reduction_backends: tuple[str, ...]
    reduction_column_count: int
    column_addition_count: int
    finite_pair_count: int


@dataclass(frozen=True, slots=True)
class PersistenceResult:
    intervals: tuple[PersistenceInterval, ...]
    max_dimension: int
    diagnostics: PersistenceDiagnostics

    def in_dimension(self, dimension: int) -> tuple[PersistenceInterval, ...]:
        if dimension < 0:
            raise ValueError("dimension must be non-negative")
        return tuple(
            interval for interval in self.intervals if interval.dimension == dimension
        )

    def betti_at(self, dimension: int, threshold: float) -> int:
        return sum(
            interval.birth <= threshold < interval.death
            for interval in self.in_dimension(dimension)
        )

    def persistent_betti(self, dimension: int, start: float, end: float) -> int:
        if start > end:
            raise ValueError("start must not exceed end")
        return sum(
            interval.birth <= start and interval.death > end
            for interval in self.in_dimension(dimension)
        )


__all__ = [
    "PersistenceDiagnostics",
    "PersistenceInterval",
    "PersistenceResult",
]

