"""Common result envelopes; native chain objects remain mathematically distinct."""

from dataclasses import dataclass, field as dc_field
from typing import Any
import math
import numpy as np


@dataclass
class Topology:
    kind: str
    native: Any
    cloud: Any = None
    metadata: dict = dc_field(default_factory=dict)


@dataclass(frozen=True)
class HomologyResult:
    betti_numbers: tuple[int, ...]
    field: str = "GF(2)"
    metadata: dict = dc_field(default_factory=dict)


@dataclass(frozen=True)
class PersistenceInterval:
    dimension: int
    birth: float
    death: float
    at_initial_stage: bool = False

    @property
    def lifetime(self):
        return self.death - self.birth


@dataclass(frozen=True)
class PersistenceResult:
    intervals: tuple[PersistenceInterval, ...]
    field: str = "GF(2)"
    max_dimension: int = 2
    metadata: dict = dc_field(default_factory=dict)

    def as_array(self):
        return np.asarray([(interval.dimension, interval.birth, interval.death) for interval in self.intervals], dtype=float).reshape(-1, 3)

    def betti_at(self, scale):
        query_scale = float(scale)
        domain_start = self.metadata.get("filtration_start", -math.inf)
        domain_end = self.metadata.get("filtration_end", math.inf)
        if not math.isfinite(query_scale) or query_scale < domain_start or query_scale > domain_end:
            raise ValueError("scale must be finite and within the recorded filtration domain")
        return tuple(sum(interval.dimension == q and interval.birth <= query_scale < interval.death for interval in self.intervals)
                     for q in range(self.max_dimension + 1))


@dataclass(frozen=True)
class SpectrumResult:
    dimension: int
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray | None = None
    matrix: Any = None
    basis: tuple = ()
    scale: float | None = None
    start: float | None = None
    end: float | None = None
    kind: str = "ordinary"
    complete: bool = True
    nullity: int | None = None
    metadata: dict = dc_field(default_factory=dict)

    @property
    def betti_number(self):
        """Real spectral nullity, not an assertion about GF(2) homology."""
        return self.nullity


@dataclass(frozen=True)
class StationaryResult:
    """Homology, ordinary Laplacians, and summaries at one fixed scale.

    The topology remains the authoritative mathematical object. ``summaries``
    contains display-ready scalar values derived from the supplied complete
    spectra; it does not replace the spectra or finite-field homology result.
    """

    topology: Topology
    scale: float | None
    dimensions: tuple[int, ...]
    homology: HomologyResult
    spectra: dict[int, SpectrumResult]
    summaries: dict[int, dict[str, float]]
    metadata: dict = dc_field(default_factory=dict)

    def summary_rows(self, dimensions=None):
        """Return plain records suitable for a table or CSV writer.

        Missing requested dimensions are retained with NaN values. This is
        useful for layouts where, for example, a graph intentionally reports
        L0 while leaving the L1 row empty.
        """
        degrees = self.dimensions if dimensions is None else tuple(dimensions)
        names = tuple(self.metadata.get(
            "summary_statistics", ("min", "max", "mean", "zero_count", "energy")
        ))
        rows = []
        for degree in degrees:
            values = self.summaries.get(degree, {})
            rows.append({"dimension": degree, **{
                name: float(values.get(name, math.nan)) for name in names
            }})
        return tuple(rows)
