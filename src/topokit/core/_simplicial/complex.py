"""A native sorted-vertex trie with a face-closed, monotone filtration."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import combinations
from numbers import Integral

from ._validation import ComplexityLimitError, nonnegative_int, real

Simplex = tuple[int, ...]


def normalize_simplex(simplex) -> Simplex:
    vertices = tuple(simplex)
    if not vertices:
        raise ValueError("The empty simplex is not stored (ordinary, unreduced homology)")
    if any(isinstance(v, bool) or not isinstance(v, Integral) for v in vertices):
        raise TypeError("Vertex labels must be integers")
    canonical_simplex = tuple(sorted(int(v) for v in vertices))
    if len(set(canonical_simplex)) != len(canonical_simplex):
        raise ValueError("A simplex cannot contain repeated vertices")
    return canonical_simplex


def facets(simplex):
    """Codimension-one faces in signed-boundary order; no empty face."""
    if len(simplex) > 1:
        for i in range(len(simplex)):
            yield simplex[:i] + simplex[i + 1:]


@dataclass(slots=True)
class _Node:
    value: float = 0.0
    children: dict = field(default_factory=dict)


class SimplexTree:
    """Filtered simplicial complex; insertion supplies all nonempty faces.

    Pass maximal simplices for a fixed complex, or a mapping simplex -> birth.
    Inserting a simplex lowers existing face births if necessary, never raises
    them. Labels are arbitrary integers, sorted to define orientations.
    """

    def __init__(self, simplices=(), *, max_simplices=1_000_000):
        self.max_simplices = nonnegative_int(max_simplices, "max_simplices")
        self._root = _Node()
        self._nodes = {}
        self._ordered = None
        self.metadata = {}
        items = simplices.items() if isinstance(simplices, Mapping) else ((s, 0.0) for s in simplices)
        for simplex, value in items:
            self.insert(simplex, value)

    def __len__(self):
        return len(self._nodes)

    def __iter__(self):
        return (simplex for simplex, _ in self.get_filtration())

    def __contains__(self, simplex):
        return self.find(simplex)

    @property
    def dimension(self):
        return max((len(s) - 1 for s in self._nodes), default=-1)

    def _store(self, simplex, value):
        """Internal fast path: prefix must exist, caller guarantees closure."""
        stored_node = self._nodes.get(simplex)
        if stored_node is None:
            if len(self) >= self.max_simplices:
                raise ComplexityLimitError(f"Simplex limit {self.max_simplices:,} exceeded")
            parent_node = self._root if len(simplex) == 1 else self._nodes[simplex[:-1]]
            stored_node = _Node(float(value))
            parent_node.children[simplex[-1]] = stored_node
            self._nodes[simplex] = stored_node
        else:
            stored_node.value = min(stored_node.value, float(value))
        self._ordered = None

    def insert(self, simplex, filtration=0.0):
        simplex = normalize_simplex(simplex)
        birth_value = real(filtration, "filtration")
        # Even a single simplex entails 2**|simplex|-1 nonempty faces.
        if (1 << len(simplex)) - 1 > self.max_simplices:
            raise ComplexityLimitError("The simplex's face closure exceeds max_simplices")
        face_closure = [s for k in range(1, len(simplex) + 1) for s in combinations(simplex, k)]
        if len(self) + sum(s not in self._nodes for s in face_closure) > self.max_simplices:
            raise ComplexityLimitError("Insertion would exceed max_simplices; tree unchanged")
        is_new_simplex = simplex not in self._nodes
        for face in face_closure:
            self._store(face, birth_value)
        return is_new_simplex

    def find(self, simplex):
        current_node = self._root
        for vertex in normalize_simplex(simplex):
            current_node = current_node.children.get(vertex)
            if current_node is None:
                return False
        return True

    def filtration(self, simplex):
        return self._nodes[normalize_simplex(simplex)].value

    def get_filtration(self):
        """Immutable sequence sorted by birth, dimension, then vertex tuple."""
        if self._ordered is None:
            self._ordered = tuple(sorted(
                ((s, node.value) for s, node in self._nodes.items()),
                key=lambda item: (item[1], len(item[0]), item[0]),
            ))
        return self._ordered

    def simplices(self, dimension=None):
        if dimension is not None:
            dimension = nonnegative_int(dimension, "dimension")
        return tuple(sorted(s for s in self._nodes if dimension is None or len(s) == dimension + 1))

    def cofaces(self, simplex, codimension=None):
        simplex = normalize_simplex(simplex)
        if codimension is not None:
            codimension = nonnegative_int(codimension, "codimension")
        required_vertices = set(simplex)
        return tuple(s for s in self.simplices() if required_vertices.issubset(s)
                     and (codimension is None or len(s) - len(simplex) == codimension))

    def at(self, scale):
        scale = real(scale, "scale", finite=False)
        snapshot = SimplexTree(max_simplices=self.max_simplices)
        snapshot.metadata = self.metadata.copy()
        for simplex, birth_value in self.get_filtration():
            if birth_value > scale:
                break
            snapshot._store(simplex, birth_value)
        return snapshot

    def skeleton(self, max_dimension):
        max_dimension = nonnegative_int(max_dimension, "max_dimension")
        skeleton = SimplexTree(max_simplices=self.max_simplices)
        skeleton.metadata = self.metadata.copy()
        for simplex, birth_value in self.get_filtration():
            if len(simplex) <= max_dimension + 1:
                skeleton._store(simplex, birth_value)
        return skeleton

    def validate(self):
        for simplex, value in self.get_filtration():
            for face in facets(simplex):
                if face not in self._nodes or self.filtration(face) > value:
                    raise ValueError(f"Invalid face closure or filtration at {simplex}")
        return True


SimplicialComplex = SimplexTree


def as_complex(value):
    return value if isinstance(value, SimplexTree) else SimplexTree(value)
