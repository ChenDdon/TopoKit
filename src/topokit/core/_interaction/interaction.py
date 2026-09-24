"""Construction of filtered interaction chains with GF(2) and signed boundaries.

The surviving interaction cells are product keys, not ordinary simplices.  A
dimension-aware inverted index is therefore used instead of forcing the cells
into a simplex tree.  The recursive join below is a *virtual product trie*: it
caches the common vertex set along one DFS branch but does not materialize
pointer-heavy prefix nodes.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .model import FilteredSimplicialComplex, Simplex


InteractionKey = tuple[int, ...]


def _intersect_sorted(first: Simplex, second: Simplex) -> Simplex:
    left_index = 0
    right_index = 0
    common: list[int] = []
    while left_index < len(first) and right_index < len(second):
        first_vertex = first[left_index]
        second_vertex = second[right_index]
        if first_vertex == second_vertex:
            common.append(first_vertex)
            left_index += 1
            right_index += 1
        elif first_vertex < second_vertex:
            left_index += 1
        else:
            right_index += 1
    return tuple(common)


def _bounded_compositions(total: int, maxima: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
    composition = [0] * len(maxima)

    def visit(position: int, remaining: int) -> Iterator[tuple[int, ...]]:
        if position == len(maxima) - 1:
            if remaining <= maxima[position]:
                composition[position] = remaining
                yield tuple(composition)
            return
        upper = min(maxima[position], remaining)
        for value in range(upper + 1):
            composition[position] = value
            yield from visit(position + 1, remaining - value)

    if maxima:
        yield from visit(0, total)


@dataclass(frozen=True, slots=True)
class InteractionDegree:
    """Cells and filtration values for one total interaction degree."""

    keys: tuple[InteractionKey, ...]
    births: tuple[float, ...]
    _index: Mapping[InteractionKey, int]

    def index_of(self, key: InteractionKey) -> int | None:
        return self._index.get(key)


@dataclass(frozen=True, slots=True)
class InteractionConstructionDiagnostics:
    factor_count: int
    factor_simplex_counts: tuple[int, ...]
    interaction_cell_counts: tuple[int, ...]
    posting_entry_visits: int
    unique_join_candidates: int
    dimension_compositions: int
    join_backend: str = "adaptive_postings"


@dataclass(frozen=True, slots=True)
class FilteredInteractionChainComplex:
    """Filtered quotient chain complex, built through homology degree ``H+1``."""

    factors: tuple[FilteredSimplicialComplex, ...]
    degrees: tuple[InteractionDegree, ...]
    max_homology_dimension: int
    diagnostics: InteractionConstructionDiagnostics

    @property
    def max_cell_degree(self) -> int:
        return len(self.degrees) - 1

    def number_of_cells(self, degree: int | None = None) -> int:
        if degree is None:
            return sum(len(layer.keys) for layer in self.degrees)
        if degree < 0 or degree >= len(self.degrees):
            return 0
        return len(self.degrees[degree].keys)

    def interaction_vertices(self, key: InteractionKey) -> Simplex:
        if len(key) != len(self.factors):
            raise ValueError("an interaction key must contain one simplex per factor")
        common = self.factors[0].simplices[key[0]]
        for factor, simplex_id in zip(self.factors[1:], key[1:]):
            common = _intersect_sorted(common, factor.simplices[simplex_id])
            if not common:
                break
        return common

    def boundary_rows(
        self,
        degree: int,
        cell_index: int,
        *,
        validate_missing: bool = False,
    ) -> tuple[int, ...]:
        """Return the quotient boundary as sorted row indexes over ``GF(2)``.

        A missing face key is zero in the quotient.  With
        ``validate_missing=True``, its common intersection is recomputed so a
        missing *admissible* face is reported as a construction error.
        """

        if degree <= 0:
            return ()
        if degree >= len(self.degrees):
            raise IndexError("interaction degree is outside the constructed chain")
        layer = self.degrees[degree]
        if cell_index < 0 or cell_index >= len(layer.keys):
            raise IndexError("interaction cell index is outside its degree")

        mutable_key = list(layer.keys[cell_index])
        target_layer = self.degrees[degree - 1]
        rows: set[int] = set()
        for factor_index, (factor, simplex_id) in enumerate(zip(self.factors, mutable_key)):
            for face_id in factor.face_ids[simplex_id]:
                mutable_key[factor_index] = face_id
                face_key = tuple(mutable_key)
                row = target_layer._index.get(face_key)
                if row is None:
                    if validate_missing and self.interaction_vertices(face_key):
                        raise RuntimeError(
                            "an admissible interaction face is missing from the index"
                        )
                elif row in rows:
                    rows.remove(row)
                else:
                    rows.add(row)
                mutable_key[factor_index] = simplex_id
        return tuple(sorted(rows))

    def signed_boundary_entries(
        self,
        degree: int,
        cell_index: int,
        *,
        validate_missing: bool = False,
    ) -> tuple[tuple[int, int], ...]:
        """Return sorted ``(row, coefficient)`` entries of the integer boundary.

        Factor simplices use increasing global vertex order. Deleting position
        ``r`` in factor ``i`` has sign ``(-1)**(sum(dim(sigma_j), j<i) + r)``.
        Empty-intersection faces are zero in the quotient. These coefficients
        define the real boundary for Laplacians; ``boundary_rows`` remains the
        separate, sign-free GF(2) path used by persistence.
        """

        if isinstance(degree, bool) or not isinstance(degree, int):
            raise TypeError("interaction degree must be an integer")
        if degree < 0 or degree >= len(self.degrees):
            raise IndexError("interaction degree is outside the constructed chain")
        layer = self.degrees[degree]
        if isinstance(cell_index, bool) or not isinstance(cell_index, int):
            raise TypeError("interaction cell index must be an integer")
        if cell_index < 0 or cell_index >= len(layer.keys):
            raise IndexError("interaction cell index is outside its degree")
        if degree == 0:
            return ()

        mutable_key = list(layer.keys[cell_index])
        target_layer = self.degrees[degree - 1]
        entries: dict[int, int] = {}
        preceding_degree = 0
        for factor_index, (factor, simplex_id) in enumerate(zip(self.factors, mutable_key)):
            for omitted, face_id in enumerate(factor.face_ids[simplex_id]):
                mutable_key[factor_index] = face_id
                face_key = tuple(mutable_key)
                row = target_layer._index.get(face_key)
                if row is None:
                    if validate_missing and self.interaction_vertices(face_key):
                        raise RuntimeError(
                            "an admissible interaction face is missing from the index"
                        )
                else:
                    sign = -1 if (preceding_degree + omitted) % 2 else 1
                    entries[row] = entries.get(row, 0) + sign
                mutable_key[factor_index] = simplex_id
            preceding_degree += factor.simplex_dimensions[simplex_id]
        return tuple(sorted((row, value) for row, value in entries.items() if value))

    def validate_signed_boundary(self) -> None:
        """Audit closure, filtration, and ``boundary**2 == 0`` over integers.

        This deliberately expensive validation is opt-in and requires no
        numerical dependencies. Unlike a GF(2) audit it detects sign errors.
        """

        for degree in range(1, len(self.degrees)):
            source = self.degrees[degree]
            target = self.degrees[degree - 1]
            for cell_index, birth in enumerate(source.births):
                boundary_squared: dict[int, int] = {}
                for row, coefficient in self.signed_boundary_entries(
                    degree, cell_index, validate_missing=True
                ):
                    if target.births[row] > birth:
                        raise RuntimeError(
                            "an interaction boundary face is born after its source"
                        )
                    if degree > 1:
                        for lower_row, lower_coefficient in self.signed_boundary_entries(
                            degree - 1, row
                        ):
                            boundary_squared[lower_row] = (
                                boundary_squared.get(lower_row, 0) + coefficient * lower_coefficient
                            )
                if any(boundary_squared.values()):
                    raise RuntimeError(
                        "constructed signed interaction boundary does not square to zero"
                    )

    def validate_boundary(self) -> None:
        """Audit closure, filtration compatibility, and ``boundary^2 = 0``."""

        for degree in range(1, len(self.degrees)):
            source = self.degrees[degree]
            target = self.degrees[degree - 1]
            for cell_index, birth in enumerate(source.births):
                for row in self.boundary_rows(
                    degree, cell_index, validate_missing=True
                ):
                    if target.births[row] > birth:
                        raise RuntimeError(
                            "an interaction boundary face is born after its source"
                        )

        for degree in range(2, len(self.degrees)):
            for cell_index in range(len(self.degrees[degree].keys)):
                boundary_squared: set[int] = set()
                for row in self.boundary_rows(degree, cell_index):
                    for lower_row in self.boundary_rows(degree - 1, row):
                        if lower_row in boundary_squared:
                            boundary_squared.remove(lower_row)
                        else:
                            boundary_squared.add(lower_row)
                if boundary_squared:
                    raise RuntimeError("constructed interaction boundary does not square to zero")


class _AdaptivePostingJoin:
    __slots__ = (
        "factors",
        "marks",
        "epochs",
        "posting_entry_visits",
        "unique_join_candidates",
    )

    def __init__(self, factors: tuple[FilteredSimplicialComplex, ...]) -> None:
        self.factors = factors
        self.marks = [[0] * factor.number_of_simplices for factor in factors]
        self.epochs = [0] * len(factors)
        self.posting_entry_visits = 0
        self.unique_join_candidates = 0

    def _estimated_posting_work(
        self, factor_index: int, dimension: int, common: Simplex
    ) -> int:
        factor = self.factors[factor_index]
        return sum(len(factor.postings(dimension, vertex)) for vertex in common)

    def _candidates(
        self, factor_index: int, dimension: int, common: Simplex
    ) -> list[int]:
        factor = self.factors[factor_index]
        marks = self.marks[factor_index]
        self.epochs[factor_index] += 1
        epoch = self.epochs[factor_index]
        candidates: list[int] = []
        for vertex in common:
            posting = factor.postings(dimension, vertex)
            self.posting_entry_visits += len(posting)
            for simplex_id in posting:
                if marks[simplex_id] != epoch:
                    marks[simplex_id] = epoch
                    candidates.append(simplex_id)
        self.unique_join_candidates += len(candidates)
        return candidates

    def enumerate(self, dimensions: tuple[int, ...]) -> list[InteractionKey]:
        if len(dimensions) != len(self.factors):
            raise ValueError("a dimension composition must match the factor count")
        factor_layers = tuple(
            factor.simplices_of_dimension(dimension)
            for factor, dimension in zip(self.factors, dimensions)
        )
        if any(not layer for layer in factor_layers):
            return []

        if len(self.factors) == 1:
            return [(simplex_id,) for simplex_id in factor_layers[0]]

        if len(self.factors) == 2:
            # Hot path: avoid recursive calls and candidate-list allocation.
            root_factor_index = 0 if len(factor_layers[0]) <= len(factor_layers[1]) else 1
            target_factor_index = 1 - root_factor_index
            target_factor = self.factors[target_factor_index]
            marks = self.marks[target_factor_index]
            joined_keys: list[InteractionKey] = []
            for root_id in factor_layers[root_factor_index]:
                self.epochs[target_factor_index] += 1
                epoch = self.epochs[target_factor_index]
                for vertex in self.factors[root_factor_index].simplices[root_id]:
                    posting = target_factor.postings(dimensions[target_factor_index], vertex)
                    self.posting_entry_visits += len(posting)
                    for target_id in posting:
                        if marks[target_id] == epoch:
                            continue
                        marks[target_id] = epoch
                        self.unique_join_candidates += 1
                        if root_factor_index == 0:
                            joined_keys.append((root_id, target_id))
                        else:
                            joined_keys.append((target_id, root_id))
            return joined_keys

        root_factor_index = min(range(len(self.factors)), key=lambda index: (len(factor_layers[index]), index))
        key = [-1] * len(self.factors)
        other_factors = tuple(index for index in range(len(self.factors)) if index != root_factor_index)
        joined_keys: list[InteractionKey] = []

        def visit(common: Simplex, remaining: tuple[int, ...]) -> None:
            if not remaining:
                joined_keys.append(tuple(key))
                return
            selected = min(
                remaining,
                key=lambda index: (
                    self._estimated_posting_work(index, dimensions[index], common),
                    index,
                ),
            )
            later = tuple(index for index in remaining if index != selected)
            for simplex_id in self._candidates(selected, dimensions[selected], common):
                key[selected] = simplex_id
                if later:
                    next_common = _intersect_sorted(
                        common, self.factors[selected].simplices[simplex_id]
                    )
                    if next_common:
                        visit(next_common, later)
                else:
                    # Candidate construction already proves nonempty intersection.
                    joined_keys.append(tuple(key))
            key[selected] = -1

        for simplex_id in factor_layers[root_factor_index]:
            key[root_factor_index] = simplex_id
            visit(self.factors[root_factor_index].simplices[simplex_id], other_factors)
        key[root_factor_index] = -1
        return joined_keys


def build_interaction_chain_complex(
    factors: Iterable[FilteredSimplicialComplex],
    *,
    max_homology_dimension: int,
    validate: bool = False,
) -> FilteredInteractionChainComplex:
    """Build the interaction quotient chain through degree ``H+1``.

    The maximum interaction complex is constructed once.  A cell's filtration
    is the maximum of its factor-simplex filtrations, so rebuilding the complex
    independently at every threshold is unnecessary.
    """

    factor_tuple = tuple(factors)
    if not factor_tuple:
        raise ValueError("at least one factor simplicial complex is required")
    if isinstance(max_homology_dimension, bool) or not isinstance(
        max_homology_dimension, int
    ):
        raise TypeError("max_homology_dimension must be a non-negative integer")
    if max_homology_dimension < 0:
        raise ValueError("max_homology_dimension must be non-negative")

    join = _AdaptivePostingJoin(factor_tuple)
    factor_maxima = tuple(factor.max_dimension for factor in factor_tuple)
    degree_layers: list[InteractionDegree] = []
    composition_count = 0

    for degree in range(max_homology_dimension + 2):
        keys: list[InteractionKey] = []
        for dimensions in _bounded_compositions(degree, factor_maxima):
            composition_count += 1
            keys.extend(join.enumerate(dimensions))

        if validate and len(keys) != len(set(keys)):
            raise RuntimeError("the posting join emitted a duplicate interaction cell")
        # Birth bucketing avoids an O(M log M) comparison sort of product keys.
        # Enumeration is deterministic, so the order inside a tied grade is
        # deterministic as well; persistence only requires nondecreasing grade.
        birth_buckets: dict[float, list[InteractionKey]] = {}
        if len(factor_tuple) == 2:
            # This is the dominant PIH use case.  Avoid a generator, zip, max,
            # and an eagerly allocated setdefault list for every product cell.
            first_births = factor_tuple[0].births
            second_births = factor_tuple[1].births
            for key in keys:
                first_birth = first_births[key[0]]
                second_birth = second_births[key[1]]
                birth = first_birth if first_birth >= second_birth else second_birth
                bucket = birth_buckets.get(birth)
                if bucket is None:
                    birth_buckets[birth] = [key]
                else:
                    bucket.append(key)
        else:
            for key in keys:
                birth = factor_tuple[0].births[key[0]]
                for factor_index in range(1, len(factor_tuple)):
                    candidate = factor_tuple[factor_index].births[key[factor_index]]
                    if candidate > birth:
                        birth = candidate
                bucket = birth_buckets.get(birth)
                if bucket is None:
                    birth_buckets[birth] = [key]
                else:
                    bucket.append(key)
        ordered_keys_list: list[InteractionKey] = []
        ordered_births: list[float] = []
        for birth in sorted(birth_buckets):
            bucket = birth_buckets[birth]
            ordered_keys_list.extend(bucket)
            ordered_births.extend([birth] * len(bucket))
        ordered_keys = tuple(ordered_keys_list)
        births = tuple(ordered_births)
        index = {key: cell_index for cell_index, key in enumerate(ordered_keys)}
        if len(index) != len(ordered_keys):
            raise RuntimeError("interaction cells must have unique ordered factor keys")
        degree_layers.append(
            InteractionDegree(
                keys=ordered_keys,
                births=births,
                _index=MappingProxyType(index),
            )
        )

    diagnostics = InteractionConstructionDiagnostics(
        factor_count=len(factor_tuple),
        factor_simplex_counts=tuple(
            factor.number_of_simplices for factor in factor_tuple
        ),
        interaction_cell_counts=tuple(len(layer.keys) for layer in degree_layers),
        posting_entry_visits=join.posting_entry_visits,
        unique_join_candidates=join.unique_join_candidates,
        dimension_compositions=composition_count,
    )
    chain = FilteredInteractionChainComplex(
        factors=factor_tuple,
        degrees=tuple(degree_layers),
        max_homology_dimension=max_homology_dimension,
        diagnostics=diagnostics,
    )
    if validate:
        chain.validate_boundary()
    return chain


__all__ = [
    "FilteredInteractionChainComplex",
    "InteractionConstructionDiagnostics",
    "InteractionDegree",
    "InteractionKey",
    "build_interaction_chain_complex",
]
