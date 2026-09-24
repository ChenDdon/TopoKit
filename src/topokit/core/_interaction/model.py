"""Validated filtered simplicial-complex inputs.

The interaction construction needs more than a collection of simplex tuples:
all factors must use the same global vertex labels, every factor must be
face-closed, and every filtration must be non-decreasing along inclusions.
This module validates those invariants once and builds the secondary indexes
used by the interaction join.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from math import isfinite
from numbers import Real
from types import MappingProxyType


Simplex = tuple[int, ...]


def _canonical_simplex(vertices: Iterable[int]) -> Simplex:
    try:
        simplex = tuple(vertices)
    except TypeError as error:
        raise TypeError("a simplex must be an iterable of integer vertex labels") from error
    if not simplex:
        raise ValueError("the empty simplex is not part of the chain complex")
    if any(isinstance(vertex, bool) or not isinstance(vertex, int) for vertex in simplex):
        raise TypeError("vertex labels must be integers shared by every factor")
    if len(set(simplex)) != len(simplex):
        raise ValueError("a simplex cannot contain a vertex more than once")
    return tuple(sorted(simplex))


def _birth_value(value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("a filtration value must be a finite real number")
    birth = float(value)
    if not isfinite(birth):
        raise ValueError("a filtration value must be finite")
    return birth


@dataclass(frozen=True, slots=True, init=False)
class FilteredSimplicialComplex:
    """Frozen, indexed representation of one filtered simplicial complex.

    Simplex identifiers are dense and stable for a fixed underlying complex.
    ``postings_by_dimension[p][v]`` is the central interaction-search index: it
    lists all ``p``-simplices containing the global vertex ``v``.
    """

    simplices: tuple[Simplex, ...]
    births: tuple[float, ...]
    simplex_dimensions: tuple[int, ...]
    ids_by_dimension: tuple[tuple[int, ...], ...]
    face_ids: tuple[tuple[int, ...], ...]
    postings_by_dimension: tuple[Mapping[int, tuple[int, ...]], ...]
    _id_of: Mapping[Simplex, int]

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError(
            "use SimplicialComplexBuilder or "
            "FilteredSimplicialComplex.from_dimension_dict()"
        )

    @classmethod
    def _from_simplex_births(
        cls,
        ordered: tuple[Simplex, ...],
        births: tuple[float, ...],
    ) -> "FilteredSimplicialComplex":
        """Build all mutually consistent indexes from validated minimal data."""

        if len(ordered) != len(births):
            raise ValueError("indexed construction requires one birth per simplex")
        id_of = {simplex: index for index, simplex in enumerate(ordered)}
        if len(id_of) != len(ordered):
            raise ValueError("a simplicial complex cannot contain duplicate simplices")
        dimensions = tuple(len(simplex) - 1 for simplex in ordered)
        # A legitimately empty sublevel complex has no simplex dimensions.
        # It is distinct from the forbidden empty simplex and tensors to zero.
        max_dimension = max(dimensions, default=-1)

        ids_by_dimension_lists: list[list[int]] = [
            [] for _ in range(max_dimension + 1)
        ]
        face_ids: list[tuple[int, ...]] = []
        posting_lists: list[dict[int, list[int]]] = [
            defaultdict(list) for _ in range(max_dimension + 1)
        ]

        for simplex_id, (simplex, dimension, birth) in enumerate(
            zip(ordered, dimensions, births)
        ):
            ids_by_dimension_lists[dimension].append(simplex_id)
            if dimension == 0:
                face_ids.append(())
            else:
                faces = tuple(
                    simplex[:omitted] + simplex[omitted + 1 :]
                    for omitted in range(len(simplex))
                )
                missing = tuple(face for face in faces if face not in id_of)
                if missing:
                    raise ValueError(
                        "simplicial input is not face-closed; "
                        f"missing faces include {missing[0]}"
                    )
                oriented_face_ids = tuple(id_of[face] for face in faces)
                for face_id in oriented_face_ids:
                    if births[face_id] > birth:
                        raise ValueError(
                            "filtration is not non-decreasing: "
                            "a face is born after its coface"
                        )
                face_ids.append(oriented_face_ids)
            for vertex in simplex:
                posting_lists[dimension][vertex].append(simplex_id)

        ids_by_dimension = tuple(
            tuple(sorted(ids, key=lambda sid: (births[sid], ordered[sid])))
            for ids in ids_by_dimension_lists
        )
        postings = tuple(
            MappingProxyType(
                {
                    vertex: tuple(ids)
                    for vertex, ids in sorted(layer.items())
                }
            )
            for layer in posting_lists
        )

        instance = object.__new__(cls)
        object.__setattr__(instance, "simplices", ordered)
        object.__setattr__(instance, "births", births)
        object.__setattr__(instance, "simplex_dimensions", dimensions)
        object.__setattr__(instance, "ids_by_dimension", ids_by_dimension)
        object.__setattr__(instance, "face_ids", tuple(face_ids))
        object.__setattr__(instance, "postings_by_dimension", postings)
        object.__setattr__(instance, "_id_of", MappingProxyType(id_of))
        return instance

    @property
    def max_dimension(self) -> int:
        return len(self.ids_by_dimension) - 1

    @property
    def number_of_simplices(self) -> int:
        return len(self.simplices)

    def simplex_id(self, simplex: Iterable[int]) -> int | None:
        return self._id_of.get(_canonical_simplex(simplex))

    def simplex(self, simplex_id: int) -> Simplex:
        return self.simplices[simplex_id]

    def dimension(self, simplex_id: int) -> int:
        return self.simplex_dimensions[simplex_id]

    def filtration(self, simplex_id: int) -> float:
        return self.births[simplex_id]

    def simplices_of_dimension(self, dimension: int) -> tuple[int, ...]:
        if dimension < 0 or dimension >= len(self.ids_by_dimension):
            return ()
        return self.ids_by_dimension[dimension]

    def postings(self, dimension: int, vertex: int) -> tuple[int, ...]:
        if dimension < 0 or dimension >= len(self.postings_by_dimension):
            return ()
        return self.postings_by_dimension[dimension].get(vertex, ())

    @classmethod
    def from_dimension_dict(
        cls,
        simplices: Mapping[int, Sequence[Iterable[int]]],
        filtrations: Mapping[int, Sequence[Real]],
        *,
        complete_faces: bool = False,
    ) -> "FilteredSimplicialComplex":
        """Create a factor from the dictionary format used by the legacy code.

        By default the input is required to contain its complete face closure.
        Set ``complete_faces=True`` when the supplied entries are maximal cells;
        missing faces are then inserted with the same filtration value and an
        existing face filtration is lowered when necessary.
        """

        builder = SimplicialComplexBuilder()
        simplex_dimensions = set(simplices)
        filtration_dimensions = set(filtrations)
        if simplex_dimensions != filtration_dimensions:
            raise ValueError("simplices and filtrations must have identical dimension keys")
        for dimension in sorted(simplex_dimensions):
            if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0:
                raise ValueError("dimension keys must be non-negative integers")
            layer = simplices[dimension]
            values = filtrations[dimension]
            if len(layer) != len(values):
                raise ValueError("each simplex must have exactly one filtration value")
            for simplex, birth in zip(layer, values):
                canonical = _canonical_simplex(simplex)
                if len(canonical) - 1 != dimension:
                    raise ValueError("a simplex does not match its supplied dimension key")
                builder.insert(canonical, birth, with_faces=complete_faces)
        return builder.freeze()


class SimplicialComplexBuilder:
    """Mutable builder with GUDHI-like insertion and face completion."""

    __slots__ = ("_birth_by_simplex",)

    def __init__(self) -> None:
        self._birth_by_simplex: dict[Simplex, float] = {}

    def insert(
        self,
        simplex: Iterable[int],
        birth: Real,
        *,
        with_faces: bool = True,
    ) -> None:
        canonical = _canonical_simplex(simplex)
        value = _birth_value(birth)
        cells: Iterable[Simplex]
        if with_faces:
            cells = (
                face
                for size in range(1, len(canonical) + 1)
                for face in combinations(canonical, size)
            )
        else:
            cells = (canonical,)
        for cell in cells:
            existing = self._birth_by_simplex.get(cell)
            if existing is None or value < existing:
                self._birth_by_simplex[cell] = value

    def freeze(self) -> FilteredSimplicialComplex:
        ordered = tuple(sorted(self._birth_by_simplex, key=lambda item: (len(item), item)))
        births = tuple(self._birth_by_simplex[simplex] for simplex in ordered)
        return FilteredSimplicialComplex._from_simplex_births(ordered, births)


__all__ = [
    "FilteredSimplicialComplex",
    "Simplex",
    "SimplicialComplexBuilder",
]
