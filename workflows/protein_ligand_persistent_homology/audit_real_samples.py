"""Independent graph-component oracle on full alpha constructions and real inputs."""
from pathlib import Path
import math
import time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from topokit import PointCloud
from topokit.builders import simplicial
import ph_features as f
import ph_common as c


def direct_counts(points, protein_count, cross_only):
    # Deliberately build through tetrahedra, not the production alpha 1-skeleton.
    n = len(points)
    if n == 0:
        return np.zeros(100, dtype=int)
    alpha = simplicial.from_points(PointCloud(points), complex_type='alpha',
        max_dimension=2, backend='native', filtration_range=(0., 25.),
        duplicates='error', geometry_tolerance=1e-12, max_simplices=1_000_000)
    edges = [(tuple(edge), math.sqrt(birth)) for edge, birth in alpha.metadata['raw_filtration']
             if len(edge) == 2 and (not cross_only or min(edge) < protein_count <= max(edge))]
    expected = []
    for right in f.BIN_EDGES[1:]:
        # A death exactly at right still covers [left,right), so omit that edge here.
        retained = [edge for edge, birth in edges if birth < right]
        rows = [i for i, j in retained] + [j for i, j in retained]
        cols = [j for i, j in retained] + [i for i, j in retained]
        adjacency = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
        expected.append(connected_components(adjacency, directed=False, return_labels=False))
    return np.asarray(expected)


def audit_real(output, receipt, union):
    started, checked = time.monotonic(), []
    source = Path(receipt['source_dataset'])
    for sample in ('1a30', '1w8l', '10gs'):
        row = union[sample]
        tensor, record = c.load_feature(sample, output, receipt, c.input_hashes(row, source))
        selected = f.select_atoms(source / row['protein_file'], source / row['ligand_mol2_file'])
        for pair in (('C', 'C'), ('C', 'H'), ('C', 'Null'), ('N', 'O'),
                     ('Null', 'C'), ('Null', 'H'), ('Null', 'Null')):
            channel = f.CHANNELS.index(pair)
            points, np_, cross, _ = f.channel_points(selected, channel)
            assert record['channel_atom_counts'][channel] == [np_, len(points)-np_]
            np.testing.assert_array_equal(tensor[channel], direct_counts(points, np_, cross))
            checked.append({'sample': sample, 'channel': channel, 'pair': pair,
                            'atoms': len(points), 'bins_checked': 100})
    return {'passed': True, 'method': 'full alpha simplices then independent scipy graph components before each right boundary',
            'checked_channels': checked, 'slice_count': len(checked)*100,
            'seconds': time.monotonic()-started}
