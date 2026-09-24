"""Implicit persistent H0/H1 for score-directed path-compatible hyperdigraphs.

The input is a weighted point cloud with distinct scalar scores. Every retained
pair is directed from the lower score to the higher score and enters at its
Euclidean distance. Directed 2-hyperedges are represented implicitly: every
distinct allowed two-step path ``(a,b,c)`` is present when ``(a,b)`` and
``(b,c)`` are present. Consequently only triangle and quadrangle boundary
families are needed in dimension one; repeated-vertex path-homology bigons are
excluded.

The default point-cloud support is the Delaunay 1-skeleton.  Complete support
remains available explicitly.  In either case, the implementation stores
edges and reduction state in packed numeric arrays and never materializes the
cubic collection of directed 2-hyperedges.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass, replace
import math
from typing import Any, Iterable, Iterator, Sequence

from .model import DEFINITION_ID
from .results import (
    PersistenceDiagnostics,
    PersistenceInterval,
    PersistenceResult,
)


_EMPTY = -1
_UNIT = -2
_DENSE = -3
_MAX_PACKED_VERTEX_COUNT = 65_535
_MAX_SIGNED_INDEX = 2_147_483_647


def _set_bits(bits: int) -> Iterator[int]:
    while bits:
        low = bits & -bits
        yield low.bit_length() - 1
        bits ^= low


def _bits(values: Iterable[int]) -> int:
    result = 0
    for value in values:
        result |= 1 << value
    return result


@dataclass(frozen=True, slots=True)
class ScoreDirectedEdgeFiltration:
    """Compact edge filtration underlying a full distinct-path hyperdigraph."""

    vertex_count: int
    packed_pairs: Sequence[int]
    edge_weights: Sequence[float]
    score_order: tuple[int, ...]
    storage_backend: str
    critical_value_count: int
    connection_support: str = "explicit"
    ambient_dimension: int | None = None
    intrinsic_dimension: int | None = None
    support_edge_count: int | None = None
    support_maximal_simplex_count: int | None = None
    projected_to_affine_hull: bool = False

    def __post_init__(self) -> None:
        if self.vertex_count < 2:
            raise ValueError("vertex_count must be at least two")
        if self.vertex_count > _MAX_PACKED_VERTEX_COUNT:
            raise ValueError(
                "packed point-cloud filtrations support at most 65,535 points"
            )
        if len(self.packed_pairs) != len(self.edge_weights):
            raise ValueError("packed_pairs and edge_weights must have equal length")
        if len(self.score_order) != self.vertex_count:
            raise ValueError("score_order must contain every vertex")
        if self.connection_support not in {"explicit", "delaunay", "complete"}:
            raise ValueError("connection_support metadata is invalid")
        if self.support_edge_count is not None and self.support_edge_count < len(
            self.packed_pairs
        ):
            raise ValueError("support_edge_count cannot be smaller than edge_count")

    @property
    def edge_count(self) -> int:
        return len(self.packed_pairs)









def _support_notes(filtration: ScoreDirectedEdgeFiltration) -> tuple[str, ...]:
    if filtration.connection_support == "explicit":
        return ()
    retained = filtration.edge_count
    available = filtration.support_edge_count
    cutoff_note = (
        f"{retained} directed edges were retained from {available} support edges."
        if available is not None
        else f"{retained} directed edges were retained."
    )
    projection_note = (
        "Delaunay support was computed after projection to the intrinsic affine hull."
        if filtration.projected_to_affine_hull
        else "Delaunay support used the ambient-coordinate affine dimension."
    )
    notes = (
        f"Point-cloud connection support: {filtration.connection_support}.",
        cutoff_note,
    )
    if filtration.connection_support == "delaunay":
        notes += (projection_note,)
    return notes




class _DisjointSet:
    __slots__ = ("parent", "rank", "representative", "components")

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size
        self.representative = list(range(size))
        self.components = size

    def find(self, item: int) -> int:
        parent = self.parent
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(self, first: int, second: int) -> int | None:
        left = self.find(first)
        right = self.find(second)
        if left == right:
            return None
        left_rep = self.representative[left]
        right_rep = self.representative[right]
        killed = max(left_rep, right_rep)
        if self.rank[left] < self.rank[right] or (
            self.rank[left] == self.rank[right] and left_rep > right_rep
        ):
            left, right = right, left
        self.parent[right] = left
        self.representative[left] = min(left_rep, right_rep)
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1
        self.components -= 1
        return killed

    def surviving_representatives(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                self.representative[index]
                for index, parent in enumerate(self.parent)
                if index == parent
            )
        )


class _CompactGF2Basis:
    """Highest-pivot basis with array-indexed compressed sparse columns."""

    __slots__ = (
        "descriptors",
        "lengths",
        "sparse_data",
        "dense_columns",
        "rank",
        "unit_count",
        "sparse_count",
        "dense_count",
        "xor_count",
        "dense_promotions",
    )

    _MIN_DENSE_SUPPORT = 64

    def __init__(self) -> None:
        self.descriptors = array("q")
        self.lengths = array("I")
        self.sparse_data = array("I")
        self.dense_columns: dict[int, int] = {}
        self.rank = 0
        self.unit_count = 0
        self.sparse_count = 0
        self.dense_count = 0
        self.xor_count = 0
        self.dense_promotions = 0

    def add_coordinate(self) -> None:
        self.descriptors.append(_EMPTY)
        self.lengths.append(0)

    def _maybe_dense(self, work: set[int]) -> set[int] | int:
        if len(work) < self._MIN_DENSE_SUPPORT:
            return work
        highest = max(work)
        dense_bytes = 32 + (highest + 8) // 8
        sparse_bytes = 64 + 12 * len(work)
        if dense_bytes > sparse_bytes:
            return work
        self.dense_promotions += 1
        return _bits(work)

    def _store(self, pivot: int, work: set[int] | int) -> None:
        if isinstance(work, int):
            support_size = work.bit_count()
            dense_bytes = 32 + (work.bit_length() + 7) // 8
            sparse_bytes = 8 * support_size
            if dense_bytes <= sparse_bytes:
                self.descriptors[pivot] = _DENSE
                self.dense_columns[pivot] = work
                self.dense_count += 1
                return
            values = tuple(_set_bits(work))
        else:
            values = tuple(sorted(work))
        if len(values) == 1:
            self.descriptors[pivot] = _UNIT
            self.unit_count += 1
            return
        offset = len(self.sparse_data)
        self.sparse_data.extend(values)
        self.descriptors[pivot] = offset
        self.lengths[pivot] = len(values)
        self.sparse_count += 1

    def reduce_and_insert(self, coordinates: Iterable[int]) -> int | None:
        work: set[int] | int = set(coordinates)
        while work:
            pivot = work.bit_length() - 1 if isinstance(work, int) else max(work)
            descriptor = self.descriptors[pivot]
            if descriptor == _EMPTY:
                self._store(pivot, work)
                self.rank += 1
                return pivot
            self.xor_count += 1
            if descriptor == _UNIT:
                if isinstance(work, int):
                    work ^= 1 << pivot
                else:
                    work.remove(pivot)
                continue
            if descriptor == _DENSE:
                previous = self.dense_columns[pivot]
                if isinstance(work, int):
                    work ^= previous
                else:
                    work = _bits(work) ^ previous
                    self.dense_promotions += 1
                continue
            offset = descriptor
            length = self.lengths[pivot]
            if isinstance(work, int):
                for position in range(offset, offset + length):
                    work ^= 1 << self.sparse_data[position]
            else:
                for position in range(offset, offset + length):
                    coordinate = self.sparse_data[position]
                    if coordinate in work:
                        work.remove(coordinate)
                    else:
                        work.add(coordinate)
                work = self._maybe_dense(work)
        return None


def _row_offsets(vertex_count: int) -> tuple[int, ...]:
    return tuple(
        source * (2 * vertex_count - source - 1) // 2 for source in range(vertex_count)
    )


def _pair_slot(offsets: Sequence[int], source: int, target: int) -> int:
    return offsets[source] + target - source - 1


class _DenseNeighborIndex:
    __slots__ = ("outgoing_bits", "incoming_bits")

    def __init__(self, vertex_count: int) -> None:
        self.outgoing_bits = [0] * vertex_count
        self.incoming_bits = [0] * vertex_count

    def first_out_in(self, source: int, target: int) -> int | None:
        values = self.outgoing_bits[source] & self.incoming_bits[target]
        return (values & -values).bit_length() - 1 if values else None

    def common_out(self, first: int, second: int) -> Iterator[int]:
        return _set_bits(self.outgoing_bits[first] & self.outgoing_bits[second])

    def common_in(self, first: int, second: int) -> Iterator[int]:
        return _set_bits(self.incoming_bits[first] & self.incoming_bits[second])

    def outgoing(self, source: int) -> Iterator[int]:
        return _set_bits(self.outgoing_bits[source])

    def incoming(self, target: int) -> Iterator[int]:
        return _set_bits(self.incoming_bits[target])

    def has(self, source: int, target: int) -> bool:
        return bool((self.outgoing_bits[source] >> target) & 1)

    def any_common_in(self, first: int, second: int) -> bool:
        return bool(self.incoming_bits[first] & self.incoming_bits[second])

    def any_common_out(self, first: int, second: int) -> bool:
        return bool(self.outgoing_bits[first] & self.outgoing_bits[second])

    def add(self, source: int, target: int) -> None:
        self.outgoing_bits[source] |= 1 << target
        self.incoming_bits[target] |= 1 << source


class _SparseNeighborIndex:
    __slots__ = ("outgoing_sets", "incoming_sets")

    def __init__(self, vertex_count: int) -> None:
        self.outgoing_sets = [set() for _ in range(vertex_count)]
        self.incoming_sets = [set() for _ in range(vertex_count)]

    def first_out_in(self, source: int, target: int) -> int | None:
        return min(
            self.outgoing_sets[source] & self.incoming_sets[target],
            default=None,
        )

    def common_out(self, first: int, second: int) -> Iterator[int]:
        return iter(sorted(self.outgoing_sets[first] & self.outgoing_sets[second]))

    def common_in(self, first: int, second: int) -> Iterator[int]:
        return iter(sorted(self.incoming_sets[first] & self.incoming_sets[second]))

    def outgoing(self, source: int) -> Iterator[int]:
        return iter(sorted(self.outgoing_sets[source]))

    def incoming(self, target: int) -> Iterator[int]:
        return iter(sorted(self.incoming_sets[target]))

    def has(self, source: int, target: int) -> bool:
        return target in self.outgoing_sets[source]

    def any_common_in(self, first: int, second: int) -> bool:
        return not self.incoming_sets[first].isdisjoint(self.incoming_sets[second])

    def any_common_out(self, first: int, second: int) -> bool:
        return not self.outgoing_sets[first].isdisjoint(self.outgoing_sets[second])

    def add(self, source: int, target: int) -> None:
        self.outgoing_sets[source].add(target)
        self.incoming_sets[target].add(source)


def _interval(
    dimension: int,
    birth: float,
    death: float,
    birth_index: int | None,
    death_index: int | None,
    kind: str,
) -> PersistenceInterval:
    return PersistenceInterval(
        dimension=dimension,
        birth=birth,
        death=death,
        birth_index=birth_index,
        death_index=death_index,
        kind=kind,
    )


def compute_path_compatible_persistence(
    filtration: ScoreDirectedEdgeFiltration,
    max_dimension: int = 1,
    *,
    include_diagonal: bool = False,
) -> PersistenceResult:
    """Compute exact persistent H0/H1 without explicit 2-hyperedges."""

    if not isinstance(filtration, ScoreDirectedEdgeFiltration):
        raise TypeError("filtration must be a ScoreDirectedEdgeFiltration")
    if isinstance(max_dimension, bool) or max_dimension not in {0, 1}:
        raise ValueError(
            "the implicit path-compatible engine supports dimensions 0 and 1"
        )
    if not isinstance(include_diagonal, bool):
        raise TypeError("include_diagonal must be a boolean")
    count = filtration.vertex_count
    edge_count = filtration.edge_count
    if edge_count > _MAX_SIGNED_INDEX:
        raise ValueError("the compact reducer supports at most 2,147,483,647 edges")

    disjoint = _DisjointSet(count)
    h0: list[PersistenceInterval] = []
    negative_count = 0
    scanned = 0

    if max_dimension == 0:
        for step in range(edge_count):
            packed = int(filtration.packed_pairs[step])
            source, target = packed >> 16, packed & 0xFFFF
            killed = disjoint.union(source, target)
            scanned += 1
            if killed is None:
                continue
            negative_count += 1
            h0.append(
                _interval(
                    0,
                    0.0,
                    float(filtration.edge_weights[step]),
                    killed,
                    step,
                    "h0_union_find",
                )
            )
            if disjoint.components == 1:
                break
        for representative in disjoint.surviving_representatives():
            h0.append(
                _interval(0, 0.0, math.inf, representative, None, "h0_union_find")
            )
        return PersistenceResult(
            intervals=tuple(sorted(h0, key=lambda item: (item.birth, item.death))),
            max_dimension=0,
            diagnostics=PersistenceDiagnostics(
                definition_id=DEFINITION_ID,
                coefficient_field="GF(2)",
                critical_value_count=filtration.critical_value_count,
                chain_generator_counts=(count, edge_count),
                omega_backends=("implicit_vertices", "packed_score_directed_edges"),
                reduction_column_count=scanned,
                finite_pair_count=negative_count,
                low_dimensional_backend="score_dag_union_find_early_stop",
                notes=(
                    "H0 stopped once the spanning forest was fully determined.",
                    f"Edge storage backend: {filtration.storage_backend}.",
                )
                + _support_notes(filtration),
                vertex_count=count,
                edge_count=edge_count,
                negative_edge_count=negative_count,
                positive_edge_count=scanned - negative_count,
                connection_support=filtration.connection_support,
                ambient_dimension=filtration.ambient_dimension,
                intrinsic_dimension=filtration.intrinsic_dimension,
                support_edge_count=filtration.support_edge_count,
                support_maximal_simplex_count=(
                    filtration.support_maximal_simplex_count
                ),
                projected_to_affine_hull=filtration.projected_to_affine_hull,
            ),
        )

    pair_capacity = count * (count - 1) // 2
    use_sparse_graph_index = (
        filtration.connection_support == "delaunay"
        or pair_capacity > max(1, edge_count) * 16
    )
    offsets = () if use_sparse_graph_index else _row_offsets(count)
    edge_step_by_pair: dict[int, int] | array
    if use_sparse_graph_index:
        edge_step_by_pair = {}
        neighbors: _DenseNeighborIndex | _SparseNeighborIndex = _SparseNeighborIndex(
            count
        )
    else:
        edge_step_by_pair = array("i", [-1]) * pair_capacity
        neighbors = _DenseNeighborIndex(count)
    positive_coordinate_by_edge = array("i", [-1]) * edge_count
    positive_edges = array("i")
    paired_positive_edges = bytearray()
    basis = _CompactGF2Basis()
    h1: list[PersistenceInterval] = []
    generator_counts = {"triangle": 0, "quadrangle": 0}
    generator_count = 0
    paired_count = 0
    unpaired_count = 0

    def edge_index(source: int, target: int) -> int:
        if use_sparse_graph_index:
            index = edge_step_by_pair.get((source << 16) | target, -1)  # type: ignore[union-attr]
        else:
            index = edge_step_by_pair[_pair_slot(offsets, source, target)]
        if index < 0:
            raise RuntimeError(
                f"required old edge ({source}, {target}) was not indexed"
            )
        return index

    for step in range(edge_count):
        packed = int(filtration.packed_pairs[step])
        source, target = packed >> 16, packed & 0xFFFF
        death = float(filtration.edge_weights[step])
        killed = disjoint.union(source, target)
        if killed is not None:
            negative_count += 1
            h0.append(_interval(0, 0.0, death, killed, step, "h0_union_find"))
        else:
            coordinate = len(positive_edges)
            positive_edges.append(step)
            positive_coordinate_by_edge[step] = coordinate
            paired_positive_edges.append(0)
            basis.add_coordinate()
            unpaired_count += 1

            def submit(kind: str, edge_indices: tuple[int, ...]) -> bool:
                nonlocal generator_count, paired_count, unpaired_count
                generator_count += 1
                generator_counts[kind] += 1
                coordinates = (
                    positive_coordinate_by_edge[index]
                    for index in edge_indices
                    if positive_coordinate_by_edge[index] >= 0
                )
                pivot = basis.reduce_and_insert(coordinates)
                if pivot is not None:
                    paired_positive_edges[pivot] = 1
                    paired_count += 1
                    unpaired_count -= 1
                    birth_edge = positive_edges[pivot]
                    birth = float(filtration.edge_weights[birth_edge])
                    if include_diagonal or birth < death:
                        h1.append(
                            _interval(
                                1,
                                birth,
                                death,
                                birth_edge,
                                step,
                                f"h1_{kind}_boundary",
                            )
                        )
                return basis.rank == len(positive_edges)

            # Stop enumerating boundaries once every current positive edge is paired.
            all_cycles_paired = False
            middle = neighbors.first_out_in(source, target)
            if middle is not None:
                all_cycles_paired = submit(
                    "triangle",
                    (edge_index(source, middle), edge_index(middle, target), step),
                )

            if not all_cycles_paired:
                for sink in neighbors.common_out(source, target):
                    if submit(
                        "triangle",
                        (step, edge_index(target, sink), edge_index(source, sink)),
                    ):
                        all_cycles_paired = True
                        break

            if not all_cycles_paired:
                for old_source in neighbors.common_in(source, target):
                    if submit(
                        "triangle",
                        (
                            edge_index(old_source, source),
                            step,
                            edge_index(old_source, target),
                        ),
                    ):
                        all_cycles_paired = True
                        break

            if not all_cycles_paired:
                source_groups: dict[int, list[int]] = {}
                for sink in neighbors.outgoing(target):
                    if neighbors.has(source, sink):
                        continue
                    old_middle = neighbors.first_out_in(source, sink)
                    if old_middle is not None:
                        source_groups.setdefault(old_middle, []).append(sink)
                for old_middle in sorted(source_groups):
                    sinks = source_groups[old_middle]
                    retained = (
                        sinks[:1]
                        if neighbors.any_common_in(old_middle, target)
                        else sinks
                    )
                    for sink in retained:
                        if submit(
                            "quadrangle",
                            (
                                step,
                                edge_index(target, sink),
                                edge_index(source, old_middle),
                                edge_index(old_middle, sink),
                            ),
                        ):
                            all_cycles_paired = True
                            break
                    if all_cycles_paired:
                        break

            if not all_cycles_paired:
                sink_groups: dict[int, list[int]] = {}
                for old_source in neighbors.incoming(source):
                    if neighbors.has(old_source, target):
                        continue
                    old_middle = neighbors.first_out_in(old_source, target)
                    if old_middle is not None:
                        sink_groups.setdefault(old_middle, []).append(old_source)
                for old_middle in sorted(sink_groups):
                    sources = sink_groups[old_middle]
                    retained = (
                        sources[:1]
                        if neighbors.any_common_out(old_middle, source)
                        else sources
                    )
                    for old_source in retained:
                        if submit(
                            "quadrangle",
                            (
                                edge_index(old_source, source),
                                step,
                                edge_index(old_source, old_middle),
                                edge_index(old_middle, target),
                            ),
                        ):
                            all_cycles_paired = True
                            break
                    if all_cycles_paired:
                        break

        neighbors.add(source, target)
        if use_sparse_graph_index:
            edge_step_by_pair[(source << 16) | target] = step
        else:
            edge_step_by_pair[_pair_slot(offsets, source, target)] = step

    for representative in disjoint.surviving_representatives():
        h0.append(_interval(0, 0.0, math.inf, representative, None, "h0_union_find"))
    if unpaired_count:
        for coordinate, birth_edge in enumerate(positive_edges):
            if not paired_positive_edges[coordinate]:
                h1.append(
                    _interval(
                        1,
                        float(filtration.edge_weights[birth_edge]),
                        math.inf,
                        birth_edge,
                        None,
                        "h1_cycle",
                    )
                )

    intervals = tuple(
        sorted(
            (*h0, *h1),
            key=lambda item: (
                item.dimension,
                item.birth,
                item.death,
                -1 if item.birth_index is None else item.birth_index,
            ),
        )
    )
    positive_count = len(positive_edges)
    return PersistenceResult(
        intervals=intervals,
        max_dimension=1,
        diagnostics=PersistenceDiagnostics(
            definition_id=DEFINITION_ID,
            coefficient_field="GF(2)",
            critical_value_count=filtration.critical_value_count,
            chain_generator_counts=(count, edge_count),
            omega_backends=(
                "implicit_vertices",
                "packed_score_directed_edges",
                "implicit_distinct_two_paths",
            ),
            reduction_column_count=edge_count + generator_count,
            finite_pair_count=negative_count + paired_count,
            low_dimensional_backend="modified_dlw_implicit_two_family",
            omega2_generator_type_counts=tuple(generator_counts.items()),
            omega2_max_late_faces_at_birth=1,
            notes=(
                "Every implicit 2-hyperedge has both consecutive edge faces.",
                "Only triangle and quadrangle boundaries were generated; bigons were excluded.",
                (
                    "Diagonal intervals were retained."
                    if include_diagonal
                    else "Diagonal intervals were discarded at creation."
                ),
                f"Edge storage backend: {filtration.storage_backend}.",
                f"Boundary basis stored {basis.unit_count} unit, {basis.sparse_count} sparse, and {basis.dense_count} dense pivots.",
                f"Boundary reduction performed {basis.xor_count} pivot XORs and {basis.dense_promotions} dense promotions.",
                (
                    "Sparse edge and neighbor indexes were used."
                    if use_sparse_graph_index
                    else "Dense triangular edge and bitset-neighbor indexes were used."
                ),
            )
            + _support_notes(filtration),
            vertex_count=count,
            edge_count=edge_count,
            negative_edge_count=negative_count,
            positive_edge_count=positive_count,
            boundary_generator_count=generator_count,
            boundary_rank=basis.rank,
            boundary_generator_type_counts=tuple(generator_counts.items()),
            connection_support=filtration.connection_support,
            ambient_dimension=filtration.ambient_dimension,
            intrinsic_dimension=filtration.intrinsic_dimension,
            support_edge_count=filtration.support_edge_count,
            support_maximal_simplex_count=(filtration.support_maximal_simplex_count),
            projected_to_affine_hull=filtration.projected_to_affine_hull,
        ),
    )


__all__ = ['ScoreDirectedEdgeFiltration', 'compute_path_compatible_persistence']
