"""Incremental ordinary L0 assembly on explicitly supplied edge filtrations."""
import numpy as np

from ...exceptions import ResourceLimitError


def delete_vertex_l0(matrix, index):
    """Induced L0 after deleting a singleton and all its incident edge columns.

    A principal submatrix alone retains the old degrees. Its diagonal must
    additionally lose each edge to the removed vertex. Negative off-diagonal
    entries already contain those multiplicities, including reciprocal edges.
    The caller supplies a face-closed ordinary L0; the input is never mutated.
    """
    keep = np.arange(len(matrix)) != index
    reduced = matrix[np.ix_(keep, keep)]
    reduced[np.diag_indices_from(reduced)] += matrix[keep, index]
    return reduced


def supports_incremental_l0(native):
    """Only edge boundaries with both singleton faces already present qualify."""
    births = {r.vertices[0]: r.birth for r in native.weighted_hyperedges(0)}
    return all(all(v in births and births[v] <= r.birth for v in r.vertices)
               for r in native.weighted_hyperedges(1))


class L0Accumulator:
    """One fixed singleton basis, one dense buffer, one insertion per edge.

    Ordered reciprocal edges are distinct columns of B1 and both contribute.
    Higher hyperedges are irrelevant to ordinary L0; no higher-degree or
    two-scale persistent operator is computed by this accumulator.
    """

    def __init__(self, native, max_dense_entries, max_sparse_entries):
        if not supports_incremental_l0(native):
            raise ValueError("incremental L0 requires singleton faces born no later than their edges")
        births = {r.vertices[0]: r.birth for r in native.weighted_hyperedges(0)}
        self.vertices = tuple(v for v in native.vertices if v in births)
        n = len(self.vertices)
        records = native.weighted_hyperedges(1)  # Native order is birth-sorted.
        if n * n > max_dense_entries:
            raise ResourceLimitError("Incremental L0 full singleton buffer exceeds max_dense_entries")
        if max(n, 2 * len(records)) > max_sparse_entries:
            raise ResourceLimitError("Incremental L0 singleton/edge schedule exceeds max_sparse_entries")
        index = {v: i for i, v in enumerate(self.vertices)}
        self.vertex_births = np.asarray([births[v] for v in self.vertices], dtype=float)
        self.edge_births = np.asarray([r.birth for r in records], dtype=float)
        self.rows = np.asarray([index[r.vertices[0]] for r in records], dtype=np.intp)
        self.columns = np.asarray([index[r.vertices[1]] for r in records], dtype=np.intp)
        self.buffer = np.zeros((n, n), dtype=float)
        self.edge_count = 0
        self.last_scale = -np.inf
        self.last_added = 0

    def advance(self, scale):
        if scale < self.last_scale:
            raise ValueError("incremental L0 scales must be nondecreasing")
        stop = int(np.searchsorted(self.edge_births, scale, side="right"))
        i = self.rows[self.edge_count:stop]
        j = self.columns[self.edge_count:stop]
        # add.at accumulates repeated endpoints and reciprocal pairs correctly.
        np.add.at(self.buffer, (i, i), 1.)
        np.add.at(self.buffer, (j, j), 1.)
        np.add.at(self.buffer, (i, j), -1.)
        np.add.at(self.buffer, (j, i), -1.)
        self.last_added = stop - self.edge_count
        self.edge_count = stop
        self.last_scale = scale
        active = np.flatnonzero(self.vertex_births <= scale)
        matrix = (self.buffer if len(active) == len(self.vertices)
                  else self.buffer[np.ix_(active, active)])
        labels = tuple((self.vertices[i],) for i in active)
        return matrix, labels

    @property
    def diagnostics(self):
        return {"assembly_backend": "incremental_l0", "edges_inserted": self.edge_count,
                "edges_added_at_scale": self.last_added, "allocated_vertex_count": len(self.vertices),
                "edge_update_entries": 4 * self.edge_count}
