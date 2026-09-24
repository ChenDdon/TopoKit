"""Persistent native hyperdigraph homology over GF(2)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import inf
from typing import Literal

from .chain_complex import (
    FilteredChainComplex,
    OmegaBackend,
    build_filtered_chain_complex,
)
from .gf2 import XorBasis, iter_set_bits
from .model import DEFINITION_ID, FilteredSequenceHyperdigraph
from .results import (
    PersistenceDiagnostics,
    PersistenceInterval,
    PersistenceResult,
)


ReductionBackend = Literal["auto", "standard", "native_h1"]


@dataclass(frozen=True, slots=True)
class _GeneratorRecord:
    birth: float
    dimension: int
    local_index: int


@dataclass(frozen=True, slots=True)
class _LowDimensionalReduction:
    intervals: tuple[PersistenceInterval, ...]
    reduction_column_count: int
    finite_pair_count: int
    omega2_zero_columns: frozenset[int]


def _validate_options(
    max_dimension: int,
    include_diagonal: bool,
    reduction_backend: ReductionBackend,
) -> None:
    if isinstance(max_dimension, bool) or not isinstance(max_dimension, int):
        raise TypeError("max_dimension must be a non-negative integer")
    if max_dimension < 0:
        raise ValueError("max_dimension must be non-negative")
    if not isinstance(include_diagonal, bool):
        raise TypeError("include_diagonal must be a boolean")
    if reduction_backend not in {"auto", "standard", "native_h1"}:
        raise ValueError("reduction_backend must be 'auto', 'standard', or 'native_h1'")


def _filtered_records(
    chain: FilteredChainComplex,
) -> tuple[tuple[_GeneratorRecord, ...], dict[tuple[int, int], int]]:
    records = [
        _GeneratorRecord(birth, dimension, local_index)
        for dimension, births in enumerate(chain.omega_births)
        for local_index, birth in enumerate(births)
    ]
    records.sort(key=lambda item: (item.birth, item.dimension, item.local_index))
    index = {
        (record.dimension, record.local_index): global_index
        for global_index, record in enumerate(records)
    }
    return tuple(records), index


def _ordinary_intervals_for_dimensions(
    chain: FilteredChainComplex,
    dimensions: tuple[int, ...],
    *,
    known_zero_columns: dict[int, frozenset[int]] | None = None,
) -> tuple[tuple[PersistenceInterval, ...], int, int]:
    """Run ordinary persistence reduction only in requested dimensions.

    Boundary matrices of different source dimensions never interact during
    column reduction.  This lets the hybrid backend reuse the native reduction
    of ``d_2`` and start ordinary reduction at ``d_3`` for ``H_2`` and above.
    """

    requested_dimensions = tuple(sorted(set(dimensions)))
    if not requested_dimensions:
        return (), 0, 0
    if requested_dimensions[0] < 0 or requested_dimensions[-1] > chain.max_homology_dimension:
        raise ValueError("requested homology dimension is outside the chain")

    _, global_index = _filtered_records(chain)
    supplied_zero_columns = known_zero_columns or {}
    required_source_dimensions = {
        source_dimension
        for homology_dimension in requested_dimensions
        for source_dimension in (homology_dimension, homology_dimension + 1)
    }
    zero_columns: dict[int, frozenset[int]] = {}
    death_by_dimension: dict[int, dict[int, int]] = {}
    reduction_column_count = 0

    for source_dimension in sorted(required_source_dimensions):
        known_zeros = supplied_zero_columns.get(source_dimension)
        if known_zeros is not None:
            if any(
                index < 0
                or index >= len(chain.omega_generators[source_dimension])
                for index in known_zeros
            ):
                raise ValueError("known zero-column index is outside the chain basis")
            zero_columns[source_dimension] = known_zeros
            continue

        columns = chain.boundary_columns[source_dimension]
        reduction_column_count += len(columns)
        if source_dimension == 0:
            zero_columns[source_dimension] = frozenset(range(len(columns)))
            continue

        reduced_by_pivot: dict[int, int] = {}
        zero_column_indices: set[int] = set()
        deaths: dict[int, int] = {}
        for column_index, column in enumerate(columns):
            reduced = column
            while reduced:
                pivot = reduced.bit_length() - 1
                previous = reduced_by_pivot.get(pivot)
                if previous is None:
                    reduced_by_pivot[pivot] = reduced
                    deaths[pivot] = column_index
                    break
                reduced ^= previous
            if not reduced:
                zero_column_indices.add(column_index)
        zero_columns[source_dimension] = frozenset(zero_column_indices)
        death_by_dimension[source_dimension - 1] = deaths

    intervals: list[PersistenceInterval] = []
    finite_pairs = 0
    for dimension in requested_dimensions:
        births = zero_columns[dimension]
        deaths = death_by_dimension.get(dimension, {})
        if any(pivot not in births for pivot in deaths):
            raise RuntimeError(
                "persistence pivot did not correspond to a cycle birth"
            )
        for local_birth_index in sorted(births):
            local_death_index = deaths.get(local_birth_index)
            birth = chain.omega_births[dimension][local_birth_index]
            death = (
                chain.omega_births[dimension + 1][local_death_index]
                if local_death_index is not None
                else inf
            )
            global_birth_index = global_index[(dimension, local_birth_index)]
            global_death_index = (
                global_index[(dimension + 1, local_death_index)]
                if local_death_index is not None
                else None
            )
            finite_pairs += local_death_index is not None
            intervals.append(
                PersistenceInterval(
                    dimension=dimension,
                    birth=birth,
                    death=death,
                    birth_index=global_birth_index,
                    death_index=global_death_index,
                    kind="ordinary_boundary_reduction",
                )
            )
    return tuple(intervals), reduction_column_count, finite_pairs


def _standard_intervals(
    chain: FilteredChainComplex,
    max_dimension: int,
) -> tuple[tuple[PersistenceInterval, ...], int, int]:
    return _ordinary_intervals_for_dimensions(
        chain, tuple(range(max_dimension + 1))
    )


class _ElderDisjointSet:
    __slots__ = ("parent", "birth", "generator")

    def __init__(self, size: int) -> None:
        self.parent = [-1] * size
        self.birth = [inf] * size
        self.generator = [-1] * size

    def activate(self, vertex: int, birth: float, generator: int) -> None:
        if self.parent[vertex] != -1:
            raise RuntimeError("a zero-hyperedge vertex was activated twice")
        self.parent[vertex] = vertex
        self.birth[vertex] = birth
        self.generator[vertex] = generator

    def find(self, item: int) -> int:
        if self.parent[item] == -1:
            raise RuntimeError("an edge endpoint is not active")
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, first: int, second: int) -> int | None:
        root_first = self.find(first)
        root_second = self.find(second)
        if root_first == root_second:
            return None
        key_first = (self.birth[root_first], self.generator[root_first])
        key_second = (self.birth[root_second], self.generator[root_second])
        survivor, killed = (
            (root_first, root_second)
            if key_first <= key_second
            else (root_second, root_first)
        )
        self.parent[killed] = survivor
        return killed

    def roots(self) -> tuple[int, ...]:
        return tuple(
            vertex
            for vertex, parent in enumerate(self.parent)
            if parent == vertex
        )


def _native_h0_h1_intervals(
    hyperdigraph: FilteredSequenceHyperdigraph,
    chain: FilteredChainComplex,
    max_dimension: int,
) -> _LowDimensionalReduction:
    """Dey--Li--Wang-style reduction with a native Omega_2 oracle.

    This route is exact when every directed edge is born no earlier than its
    two singleton faces.  It retains union--find and fundamental-cycle
    coordinates, but obtains 1-boundaries from explicit native 2-hyperedges.
    """

    if not chain.face_compatible_one_skeleton:
        raise ValueError(
            "native_h1 requires every directed edge to be born no earlier "
            "than both singleton faces"
        )

    vertex_order = hyperdigraph.vertex_index
    zero_edges = chain.hyperedges[0]
    zero_births = chain.hyperedge_births[0]
    one_edges = chain.hyperedges[1]
    one_births = chain.hyperedge_births[1]
    one_index = {edge: index for index, edge in enumerate(one_edges)}
    unit_omega1_basis = all(
        generator == 1 << index
        for index, generator in enumerate(chain.omega_generators[1])
    )
    zero_local_by_vertex = {edge[0]: index for index, edge in enumerate(zero_edges)}

    disjoint_set = _ElderDisjointSet(len(hyperdigraph.vertices))
    h0_intervals: list[PersistenceInterval] = []
    positive_edges: list[int] = []
    positive_coordinate_by_edge = [-1] * len(one_edges)

    events = [
        (birth, 0, index) for index, birth in enumerate(zero_births)
    ] + [
        (birth, 1, index) for index, birth in enumerate(one_births)
    ]
    events.sort()

    for birth, kind, local_index in events:
        if kind == 0:
            vertex = zero_edges[local_index][0]
            disjoint_set.activate(vertex_order[vertex], birth, local_index)
            continue
        source, target = one_edges[local_index]
        killed = disjoint_set.union(vertex_order[source], vertex_order[target])
        if killed is None:
            coordinate = len(positive_edges)
            positive_edges.append(local_index)
            positive_coordinate_by_edge[local_index] = coordinate
        else:
            h0_intervals.append(
                PersistenceInterval(
                    dimension=0,
                    birth=disjoint_set.birth[killed],
                    death=birth,
                    birth_index=disjoint_set.generator[killed],
                    death_index=local_index,
                    kind="h0_union_find",
                )
            )

    for root in disjoint_set.roots():
        h0_intervals.append(
            PersistenceInterval(
                dimension=0,
                birth=disjoint_set.birth[root],
                death=inf,
                birth_index=disjoint_set.generator[root],
                death_index=None,
                kind="h0_union_find",
            )
        )

    if max_dimension == 0:
        return _LowDimensionalReduction(
            intervals=tuple(h0_intervals),
            reduction_column_count=len(events),
            finite_pair_count=len(h0_intervals) - len(disjoint_set.roots()),
            omega2_zero_columns=frozenset(),
        )

    boundary_basis = XorBasis()
    paired_positive_coordinates: set[int] = set()
    omega2_zero_columns: set[int] = set()
    h1_intervals: list[PersistenceInterval] = []

    for omega2_index, (generator, death) in enumerate(
        zip(chain.omega_generators[2], chain.omega_births[2])
    ):
        if unit_omega1_basis:
            boundary = chain.boundary_columns[2][omega2_index]
        else:
            ambient_boundary_faces: set[tuple[object, ...]] = set()
            for two_hyperedge_index in iter_set_bits(generator):
                two_hyperedge = chain.hyperedges[2][two_hyperedge_index]
                for removed in range(len(two_hyperedge)):
                    face = (
                        two_hyperedge[:removed]
                        + two_hyperedge[removed + 1 :]
                    )
                    if face in ambient_boundary_faces:
                        ambient_boundary_faces.remove(face)
                    else:
                        ambient_boundary_faces.add(face)

            boundary = 0
            for face in ambient_boundary_faces:
                edge_index = one_index.get(face)
                if edge_index is None:
                    raise RuntimeError(
                        "an Omega_2 generator retained a non-edge boundary face"
                    )
                if one_births[edge_index] > death:
                    raise RuntimeError(
                        "an Omega_2 generator retained an edge born after its entry"
                    )
                boundary ^= 1 << edge_index

        cycle_coordinates = 0
        for edge_index in iter_set_bits(boundary):
            coordinate = positive_coordinate_by_edge[edge_index]
            if coordinate >= 0:
                cycle_coordinates ^= 1 << coordinate
        pivot = boundary_basis.insert(cycle_coordinates)
        if pivot is None:
            omega2_zero_columns.add(omega2_index)
            continue
        if pivot in paired_positive_coordinates:
            raise RuntimeError("an H1 birth was paired twice")
        paired_positive_coordinates.add(pivot)
        birth_edge = positive_edges[pivot]
        h1_intervals.append(
            PersistenceInterval(
                dimension=1,
                birth=one_births[birth_edge],
                death=death,
                birth_index=birth_edge,
                death_index=omega2_index,
                kind="native_omega2_boundary",
            )
        )

    for coordinate, edge_index in enumerate(positive_edges):
        if coordinate not in paired_positive_coordinates:
            h1_intervals.append(
                PersistenceInterval(
                    dimension=1,
                    birth=one_births[edge_index],
                    death=inf,
                    birth_index=edge_index,
                    death_index=None,
                    kind="h1_fundamental_cycle",
                )
            )

    return _LowDimensionalReduction(
        intervals=tuple(h0_intervals + h1_intervals),
        reduction_column_count=len(events) + len(chain.omega_generators[2]),
        finite_pair_count=(
            len(h0_intervals)
            - len(disjoint_set.roots())
            + len(paired_positive_coordinates)
        ),
        omega2_zero_columns=frozenset(omega2_zero_columns),
    )


def compute_persistence(
    hyperdigraph: FilteredSequenceHyperdigraph,
    max_dimension: int = 1,
    *,
    include_diagonal: bool = False,
    omega_backend: OmegaBackend = "auto",
    reduction_backend: ReductionBackend = "auto",
) -> PersistenceResult:
    """Compute native persistent hyperdigraph homology over GF(2)."""

    if not isinstance(hyperdigraph, FilteredSequenceHyperdigraph):
        raise TypeError("hyperdigraph must be a FilteredSequenceHyperdigraph")
    _validate_options(max_dimension, include_diagonal, reduction_backend)
    chain = build_filtered_chain_complex(
        hyperdigraph,
        max_dimension,
        omega_backend=omega_backend,
    )

    use_native_h1 = (
        chain.face_compatible_one_skeleton
        and reduction_backend in {"auto", "native_h1"}
    )
    if reduction_backend == "native_h1" and not use_native_h1:
        raise ValueError(
            "native_h1 requires every edge to be born after its singleton faces"
        )

    if use_native_h1:
        low_dimensional_result = _native_h0_h1_intervals(
            hyperdigraph, chain, min(max_dimension, 1)
        )
        raw_intervals = low_dimensional_result.intervals
        reduction_columns = low_dimensional_result.reduction_column_count
        finite_pairs = low_dimensional_result.finite_pair_count
        low_backend = "modified_dlw_native_omega2"
        higher_backend = "not_requested"
        notes: tuple[str, ...] = (
            "Union--find and fundamental-cycle coordinates are retained from "
            "the Dey--Li--Wang strategy.",
            "Native Omega_2 generators replace graph-induced bigon/triangle/"
            "quadrangle enumeration.",
        )
        if max_dimension >= 2:
            high_intervals, high_columns, high_pairs = (
                _ordinary_intervals_for_dimensions(
                    chain,
                    tuple(range(2, max_dimension + 1)),
                    known_zero_columns={2: low_dimensional_result.omega2_zero_columns},
                )
            )
            raw_intervals += high_intervals
            reduction_columns += high_columns
            finite_pairs += high_pairs
            higher_backend = "ordinary_filtered_boundary_reduction"
            notes += (
                "The native d_2 reduction was reused to identify H_2 births; "
                "d_3 and higher boundary matrices used ordinary persistence "
                "reduction.",
            )
    else:
        raw_intervals, reduction_columns, finite_pairs = _standard_intervals(
            chain, max_dimension
        )
        low_backend = "standard_filtered_chain_reduction"
        higher_backend = (
            "ordinary_filtered_boundary_reduction"
            if max_dimension >= 2
            else "not_requested"
        )
        notes = (
            "A filtration-compatible basis was constructed for every Omega space.",
            "The resulting free filtered chain complex used ordinary column reduction.",
        )

    omega2_generator_type_counts = (
        tuple(sorted(Counter(chain.omega_generator_kinds[2]).items()))
        if len(chain.omega_generator_kinds) > 2
        else ()
    )
    if len(chain.omega_backends) > 2:
        omega2_backend = chain.omega_backends[2]
        if omega2_backend == "face_closed_units":
            notes += (
                "Omega_2 used unit explicit 2-hyperedges (the native triangle family).",
            )
        elif omega2_backend == "single_missing_face_grouping":
            notes += (
                "Omega_2 used the exact one-missing-face unit/pair grouping.",
            )
        else:
            notes += (
                "Omega_2 used one-pass filtration-aware general circuit reduction.",
            )

    intervals = tuple(
        sorted(
            (
                interval
                for interval in raw_intervals
                if include_diagonal or interval.birth < interval.death
            ),
            key=lambda item: (
                item.dimension,
                item.birth,
                item.death,
                -1 if item.birth_index is None else item.birth_index,
            ),
        )
    )
    return PersistenceResult(
        intervals=intervals,
        max_dimension=max_dimension,
        diagnostics=PersistenceDiagnostics(
            definition_id=DEFINITION_ID,
            coefficient_field="GF(2)",
            critical_value_count=len(
                hyperdigraph.thresholds(through_dimension=max_dimension + 1)
            ),
            chain_generator_counts=tuple(
                len(generators) for generators in chain.omega_generators
            ),
            omega_backends=chain.omega_backends,
            reduction_column_count=reduction_columns,
            finite_pair_count=finite_pairs,
            low_dimensional_backend=low_backend,
            omega2_generator_type_counts=omega2_generator_type_counts,
            omega2_max_late_faces_at_birth=(
                chain.max_late_faces_at_birth[2]
                if len(chain.max_late_faces_at_birth) > 2
                else 0
            ),
            higher_dimensional_backend=higher_backend,
            notes=notes
            + (
                "Diagonal intervals from tied-event refinements were retained."
                if include_diagonal
                else "Diagonal intervals from tied events were omitted.",
            ),
        ),
    )


__all__ = ["ReductionBackend", "compute_persistence"]
