"""Validated data models for sequence-based hyperdigraphs.

The mathematical object is the one used by Chen, Liu, Wu, and Wei: a directed
``p``-hyperedge is an ordered sequence of ``p + 1`` *distinct* vertices.  It is
not a tail/head directed hypergraph.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Set
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import cast


DEFINITION_ID = "chen-liu-wu-wei-2023-sequence-embedded"
Hyperedge = tuple[Hashable, ...]
_RealValue = Real | float


def _reject_ambiguous_iterable(value: object, name: str) -> None:
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must not be a string or bytes object")
    if isinstance(value, (Set, Mapping)):
        raise TypeError(f"{name} must be ordered; sets and mappings are not accepted")


def _validate_vertices(vertices: Iterable[Hashable]) -> tuple[tuple[Hashable, ...], dict[Hashable, int]]:
    _reject_ambiguous_iterable(vertices, "vertices")
    result = tuple(vertices)
    if not result:
        raise ValueError("a hyperdigraph must have at least one ambient vertex")
    index: dict[Hashable, int] = {}
    for position, vertex in enumerate(result):
        try:
            duplicate = vertex in index
        except TypeError as exc:
            raise TypeError(f"vertex {vertex!r} is not hashable") from exc
        if duplicate:
            raise ValueError(f"duplicate ambient vertex: {vertex!r}")
        index[vertex] = position
    return result, index


def _validate_hyperedge(
    raw_hyperedge: Iterable[Hashable],
    vertex_index: Mapping[Hashable, int],
) -> Hyperedge:
    _reject_ambiguous_iterable(raw_hyperedge, "each directed hyperedge")
    hyperedge = tuple(raw_hyperedge)
    if not hyperedge:
        raise ValueError("directed hyperedges must be nonempty sequences")
    try:
        distinct = len(set(hyperedge)) == len(hyperedge)
    except TypeError as exc:
        raise TypeError(f"vertices in {hyperedge!r} must be hashable") from exc
    if not distinct:
        raise ValueError(
            "sequence-based directed hyperedges cannot repeat a vertex: "
            f"{hyperedge!r}"
        )
    missing = [vertex for vertex in hyperedge if vertex not in vertex_index]
    if missing:
        raise ValueError(
            f"directed hyperedge {hyperedge!r} uses undeclared vertices {missing!r}"
        )
    return hyperedge


def _validate_birth(value: _RealValue) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("filtration births must be finite real numbers")
    birth = float(value)
    if not isfinite(birth):
        raise ValueError("filtration births must be finite real numbers")
    return birth


@dataclass(frozen=True, slots=True)
class WeightedHyperedge:
    """One directed hyperedge and the filtration value at which it appears."""

    vertices: Hyperedge
    birth: float

    @property
    def dimension(self) -> int:
        return len(self.vertices) - 1


class SequenceHyperdigraph:
    """A finite native sequence hyperdigraph.

    By default, ambient vertices are not silently inserted into ``F_0``.  Pass
    ``include_all_vertices=True`` for the common graph-like convention.
    """

    __slots__ = (
        "_vertices",
        "_vertex_index",
        "_by_dimension",
        "_singleton_policy",
        "_synthesized_zero_hyperedges",
    )

    def __init__(
        self,
        vertices: Iterable[Hashable],
        directed_hyperedges: Iterable[Iterable[Hashable]] = (),
        *,
        include_all_vertices: bool = False,
    ) -> None:
        if not isinstance(include_all_vertices, bool):
            raise TypeError("include_all_vertices must be a boolean")
        _reject_ambiguous_iterable(directed_hyperedges, "directed_hyperedges")
        vertex_tuple, vertex_index = _validate_vertices(vertices)

        seen: set[Hyperedge] = set()
        by_dimension: dict[int, list[Hyperedge]] = {}
        for raw_hyperedge in directed_hyperedges:
            hyperedge = _validate_hyperedge(raw_hyperedge, vertex_index)
            if hyperedge in seen:
                raise ValueError(f"duplicate directed hyperedge: {hyperedge!r}")
            seen.add(hyperedge)
            by_dimension.setdefault(len(hyperedge) - 1, []).append(hyperedge)

        explicit_zero = {edge[0] for edge in by_dimension.get(0, ())}
        if include_all_vertices:
            zero_vertices = vertex_tuple
            synthesized = tuple(v for v in vertex_tuple if v not in explicit_zero)
            singleton_policy = "all_ambient_vertices"
        else:
            zero_vertices = tuple(v for v in vertex_tuple if v in explicit_zero)
            synthesized = ()
            singleton_policy = "explicit_only"
        if zero_vertices:
            by_dimension[0] = [(vertex,) for vertex in zero_vertices]
        else:
            by_dimension.pop(0, None)

        self._vertices = vertex_tuple
        self._vertex_index = vertex_index
        self._by_dimension = {
            dimension: tuple(edges)
            for dimension, edges in by_dimension.items()
            if edges
        }
        self._singleton_policy = singleton_policy
        self._synthesized_zero_hyperedges = synthesized

    @property
    def vertices(self) -> tuple[Hashable, ...]:
        return self._vertices

    @property
    def vertex_index(self) -> dict[Hashable, int]:
        return dict(self._vertex_index)

    @property
    def definition_id(self) -> str:
        return DEFINITION_ID

    @property
    def singleton_policy(self) -> str:
        return self._singleton_policy

    @property
    def synthesized_zero_hyperedges(self) -> tuple[Hashable, ...]:
        return self._synthesized_zero_hyperedges

    @property
    def dimensions(self) -> tuple[int, ...]:
        return tuple(sorted(self._by_dimension))

    @property
    def max_dimension(self) -> int:
        return max(self._by_dimension, default=-1)

    def directed_hyperedges(self, dimension: int | None = None) -> tuple[Hyperedge, ...]:
        if dimension is None:
            return tuple(
                edge
                for current_dimension in sorted(self._by_dimension)
                for edge in self._by_dimension[current_dimension]
            )
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise TypeError("dimension must be an integer")
        return self._by_dimension.get(dimension, ())

    def number_of_hyperedges(self, dimension: int | None = None) -> int:
        return len(self.directed_hyperedges(dimension))

    def as_filtered(
        self, birth: _RealValue = 0.0
    ) -> FilteredSequenceHyperdigraph:
        filtration_birth = _validate_birth(birth)
        return FilteredSequenceHyperdigraph(
            self._vertices,
            (
                WeightedHyperedge(edge, filtration_birth)
                for edge in self.directed_hyperedges()
            ),
        )


class FilteredSequenceHyperdigraph:
    """A fixed-vertex filtration specified by directed-hyperedge birth values."""

    __slots__ = (
        "_vertices",
        "_vertex_index",
        "_by_dimension",
        "_birth_by_hyperedge",
        "_singleton_policy",
        "_synthesized_zero_hyperedges",
    )

    def __init__(
        self,
        vertices: Iterable[Hashable],
        weighted_hyperedges: Iterable[
            WeightedHyperedge | tuple[Iterable[Hashable], _RealValue]
        ] = (),
        *,
        include_all_vertices: bool = False,
        vertex_birth: _RealValue = 0.0,
    ) -> None:
        if not isinstance(include_all_vertices, bool):
            raise TypeError("include_all_vertices must be a boolean")
        _reject_ambiguous_iterable(weighted_hyperedges, "weighted_hyperedges")
        vertex_tuple, vertex_index = _validate_vertices(vertices)
        default_vertex_birth = _validate_birth(vertex_birth)

        records: list[WeightedHyperedge] = []
        seen: set[Hyperedge] = set()
        for raw_record in weighted_hyperedges:
            raw_edge: Iterable[Hashable]
            raw_birth: _RealValue
            if isinstance(raw_record, WeightedHyperedge):
                raw_edge, raw_birth = raw_record.vertices, raw_record.birth
            else:
                _reject_ambiguous_iterable(raw_record, "each weighted hyperedge")
                raw_pair = tuple(raw_record)
                if len(raw_pair) != 2:
                    raise ValueError(
                        "each weighted hyperedge must be WeightedHyperedge or "
                        "(hyperedge, birth)"
                    )
                raw_edge = cast(Iterable[Hashable], raw_pair[0])
                raw_birth = cast(_RealValue, raw_pair[1])
            hyperedge = _validate_hyperedge(raw_edge, vertex_index)
            if hyperedge in seen:
                raise ValueError(f"duplicate directed hyperedge: {hyperedge!r}")
            seen.add(hyperedge)
            records.append(WeightedHyperedge(hyperedge, _validate_birth(raw_birth)))

        explicit_singletons = {
            record.vertices[0]: record.birth
            for record in records
            if record.dimension == 0
        }
        synthesized: list[Hashable] = []
        if include_all_vertices:
            rewritten: list[WeightedHyperedge] = [
                record for record in records if record.dimension != 0
            ]
            for vertex in vertex_tuple:
                if vertex not in explicit_singletons:
                    synthesized.append(vertex)
                birth = min(
                    explicit_singletons.get(vertex, default_vertex_birth),
                    default_vertex_birth,
                )
                rewritten.append(WeightedHyperedge((vertex,), birth))
            records = rewritten
            singleton_policy = "all_ambient_vertices"
        else:
            singleton_policy = "explicit_only"

        order = {edge.vertices: position for position, edge in enumerate(records)}
        records.sort(
            key=lambda item: (
                item.dimension,
                item.birth,
                order[item.vertices],
            )
        )
        by_dimension: dict[int, list[WeightedHyperedge]] = {}
        for record in records:
            by_dimension.setdefault(record.dimension, []).append(record)

        self._vertices = vertex_tuple
        self._vertex_index = vertex_index
        self._by_dimension = {
            dimension: tuple(items) for dimension, items in by_dimension.items()
        }
        self._birth_by_hyperedge = {
            record.vertices: record.birth for record in records
        }
        self._singleton_policy = singleton_policy
        self._synthesized_zero_hyperedges = tuple(synthesized)

    @classmethod
    def from_hyperdigraph(
        cls,
        hyperdigraph: SequenceHyperdigraph,
        birth: _RealValue = 0.0,
    ) -> FilteredSequenceHyperdigraph:
        if not isinstance(hyperdigraph, SequenceHyperdigraph):
            raise TypeError("hyperdigraph must be a SequenceHyperdigraph")
        return hyperdigraph.as_filtered(birth)

    @property
    def vertices(self) -> tuple[Hashable, ...]:
        return self._vertices

    @property
    def vertex_index(self) -> dict[Hashable, int]:
        return dict(self._vertex_index)

    @property
    def definition_id(self) -> str:
        return DEFINITION_ID

    @property
    def singleton_policy(self) -> str:
        return self._singleton_policy

    @property
    def synthesized_zero_hyperedges(self) -> tuple[Hashable, ...]:
        return self._synthesized_zero_hyperedges

    @property
    def dimensions(self) -> tuple[int, ...]:
        return tuple(sorted(self._by_dimension))

    @property
    def max_dimension(self) -> int:
        return max(self._by_dimension, default=-1)

    def weighted_hyperedges(
        self, dimension: int | None = None
    ) -> tuple[WeightedHyperedge, ...]:
        if dimension is None:
            return tuple(
                record
                for current_dimension in sorted(self._by_dimension)
                for record in self._by_dimension[current_dimension]
            )
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise TypeError("dimension must be an integer")
        return self._by_dimension.get(dimension, ())

    def directed_hyperedges(self, dimension: int | None = None) -> tuple[Hyperedge, ...]:
        return tuple(record.vertices for record in self.weighted_hyperedges(dimension))

    def birth_of(self, hyperedge: Iterable[Hashable]) -> float | None:
        edge = tuple(hyperedge)
        return self._birth_by_hyperedge.get(edge)

    def thresholds(self, through_dimension: int | None = None) -> tuple[float, ...]:
        return tuple(
            sorted(
                {
                    record.birth
                    for record in self.weighted_hyperedges()
                    if through_dimension is None or record.dimension <= through_dimension
                }
            )
        )

    def snapshot(self, threshold: Real) -> SequenceHyperdigraph:
        if isinstance(threshold, bool) or not isinstance(threshold, Real):
            raise TypeError("threshold must be a real number")
        value = float(threshold)
        return SequenceHyperdigraph(
            self._vertices,
            (
                record.vertices
                for record in self.weighted_hyperedges()
                if record.birth <= value
            ),
        )


Hyperdigraph = SequenceHyperdigraph
FilteredHyperdigraph = FilteredSequenceHyperdigraph


__all__ = [
    "DEFINITION_ID",
    "FilteredHyperdigraph",
    "FilteredSequenceHyperdigraph",
    "Hyperdigraph",
    "Hyperedge",
    "SequenceHyperdigraph",
    "WeightedHyperedge",
]
