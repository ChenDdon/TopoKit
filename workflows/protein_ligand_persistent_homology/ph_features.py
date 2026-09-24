"""Molecular alpha H0 application composed from public TopoKit APIs."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from scipy.spatial import cKDTree

from topokit import PointCloud, Topology, PersistenceInterval, PersistenceResult, readers
from topokit.builders import simplicial
from topokit.core import hyperdigraph
from topokit.postprocessing import barcode_bin_counts

PROTEIN = ('C', 'N', 'O', 'S', 'Null')
LIGAND = ('C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'H', 'Null')
CHANNELS = tuple((p, l) for p in PROTEIN for l in LIGAND)
BIN_EDGES = np.arange(101, dtype=float) / 20
SHAPE = (55, 100)
CROP = 15.
MAX_RADIUS = 5.


def schema():
    value = {'schema': 'topokit.protein_ligand.alpha_h0_barcode.v1', 'version': '1.0.0',
        'protein_elements': PROTEIN, 'ligand_elements': LIGAND, 'channels': CHANNELS,
        'protein_records': 'ATOM', 'crop_angstrom': CROP, 'crop_inclusive': True,
        'crop_anchors': 'original supported ligand atoms, including explicit H',
        'deduplication': 'exact keep first across cropped protein then ligand before channels; preserve retained attributes',
        'alpha_backend': 'native', 'alpha_backend_version': '1.1.1',
        'alpha_support': 'computed separately on each selected channel; full-coface edge births',
        'geometry_tolerance': 1e-12, 'max_simplices': 1_000_000,
        'mixed_channels': 'protein-ligand edges only; all vertices retained',
        'explicit_null_channels': 'within-component alpha edges on the non-Null side',
        'missing_component': 'non-Null missing element leaves other-side vertices isolated',
        'edge_weights': 1, 'orientation': 'one increasing-row-index orientation per edge; H0 only',
        'homology': 'TopoKit sequence-hyperdigraph H0', 'field': 'GF(2)', 'max_dimension': 0,
        'vertex_birth': 0., 'filtration_coordinate': 'alpha_radius', 'scale_units': 'angstrom',
        'filtration_start': 0., 'filtration_end': MAX_RADIUS,
        'builder_filtration_end_squared_radius': MAX_RADIUS**2,
        'edge_birth_conversion': 'sqrt(native squared-alpha edge birth)',
        'hard_pair_distance_cutoff': None, 'bin_edges': BIN_EDGES.tolist(),
        'bin_counting': 'cover: birth <= left and death >= right; positive-length bars only',
        'interval_convention': '[birth,death)', 'bin_convention': '[left,right)',
        'infinite_death': 'survives through observed window; count coverage without assigning death at cutoff',
        'empty_channel': 'zero bars, zero bin counts', 'tensor_shape': SHAPE,
        'axes': ('channel', 'bin'), 'flatten_order': 'C', 'feature_count': 5500,
        'calculation_dtype': 'float64', 'storage_dtype': 'float32',
        'barcode_storage': 'NPZ offsets[56], births and deaths in alpha-radius angstrom; positive infinity retained',
        'exact_reuse': 'identical selected atom rows and identical edge admission rule',
        'engine_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    value = json.loads(json.dumps(value))
    value['recipe_id'] = 'alpha-h0-cover-' + hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]
    return value


def crop_and_deduplicate(pp, pe, lp, le):
    pp, lp = np.asarray(pp, dtype=float), np.asarray(lp, dtype=float)
    pe, le = np.asarray(pe, dtype=object), np.asarray(le, dtype=object)
    for points, labels, allowed in ((pp, pe, PROTEIN[:-1]), (lp, le, LIGAND[:-1])):
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError('Finite Nx3 atom coordinates required')
        if labels.shape != (len(points),) or not np.isin(labels, allowed).all():
            raise ValueError('Invalid supported element labels')
    if not len(lp):
        raise ValueError('No supported ligand crop anchors')
    original_protein = len(pp)
    keep = cKDTree(lp).query(pp, k=1)[0] <= CROP
    pp, pe = pp[keep], pe[keep]
    unique = PointCloud(np.vstack((pp, lp))).unique_coordinates()
    ids = np.asarray(unique.metadata['coordinate_deduplication']['retained_indices'], dtype=int)
    pi, li = ids[ids < len(pp)], ids[ids >= len(pp)] - len(pp)
    return {'protein_points': pp[pi], 'protein_elements': pe[pi],
        'ligand_points': lp[li], 'ligand_elements': le[li],
        'deduplication': {'policy': 'exact_keep_first', 'order': 'cropped_protein_then_ligand',
            'protein_atoms_before_crop': original_protein, 'cropped_protein_atoms': len(pp),
            'supported_ligand_anchors': len(lp), 'input_atoms': len(pp)+len(lp),
            'retained_atoms': len(ids), 'removed_atoms': len(pp)+len(lp)-len(ids),
            'retained_indices': ids.tolist()}}


def select_atoms(protein_file, ligand_file):
    protein = readers.read_pdb(protein_file, coordinate_units='angstrom')
    ligand = readers.read(ligand_file, coordinate_units='angstrom')
    pe = np.asarray(protein.metadata['labels'], dtype=object)
    le = np.asarray(ligand.metadata['labels'], dtype=object)
    pm = (np.asarray(protein.metadata['columns']['record_name']) == 'ATOM') & np.isin(pe, PROTEIN[:-1])
    lm = np.isin(le, LIGAND[:-1])
    return crop_and_deduplicate(protein.points[pm], pe[pm], ligand.points[lm], le[lm])


def channel_points(selected, channel):
    p, l = CHANNELS[channel]
    pi = np.flatnonzero(selected['protein_elements'] == p)
    li = np.flatnonzero(selected['ligand_elements'] == l)
    points = np.vstack((selected['protein_points'][pi], selected['ligand_points'][li]))
    return points, len(pi), p != 'Null' and l != 'Null', (tuple(pi), tuple(li))


def channel_topology(points, protein_count, cross_only):
    count = len(points)
    if not 0 <= protein_count <= count:
        raise ValueError('Invalid channel protein count')
    edges = []
    # With no possible retained edge, alpha geometry cannot affect H0.
    if count >= 2 and (not cross_only or 0 < protein_count < count):
        alpha = simplicial.from_points(PointCloud(points, metadata={'coordinate_units': 'angstrom'}),
            complex_type='alpha', max_dimension=0, backend='native',
            filtration_range=(0., MAX_RADIUS**2), duplicates='error',
            geometry_tolerance=1e-12, max_simplices=1_000_000)
        if alpha.metadata.get('geometry_backend_version') != '1.1.1':
            raise RuntimeError('Native alpha version differs from the frozen recipe')
        for simplex, birth in alpha.metadata['raw_filtration']:
            if len(simplex) != 2:
                continue
            i, j = sorted(simplex)
            if not cross_only or i < protein_count <= j:
                if birth < 0 or not math.isfinite(birth):
                    raise ValueError('Invalid squared-alpha birth')
                edges.append(((int(i), int(j)), math.sqrt(birth)))
    native = hyperdigraph.FilteredHyperdigraph(tuple(range(count)), edges,
        include_all_vertices=True, vertex_birth=0.)
    return Topology('hyperdigraph', native, metadata={
        'filtration_start': 0., 'filtration_end': MAX_RADIUS,
        'scale_units': 'angstrom', 'coordinate_units': 'angstrom',
        'filtration_coordinate': 'alpha_radius', 'complex_type': 'alpha_1_skeleton',
        'max_analysis_dimension': 0, 'connection_support': 'native_alpha',
        'edge_admission': 'cross_only' if cross_only else 'within_component',
        'vertex_count': count, 'edge_count': len(edges), 'vertex_birth': 0.,
        'edge_weight': 1., 'orientation': 'increasing row index',
        'cutoff_policy': 'alpha radius <= 5; no separate pair-distance cap'})


def compute(selected):
    tensor = np.zeros(SHAPE, dtype='<f4')
    counts = np.zeros((55, 2), dtype=int)
    cache, barcodes, diagnostics = {}, [], []
    for channel in range(len(CHANNELS)):
        points, np_, cross, atom_key = channel_points(selected, channel)
        counts[channel] = (np_, len(points)-np_)
        key = (atom_key, cross)
        if key not in cache:
            if len(points):
                topology = channel_topology(points, np_, cross)
                bars = hyperdigraph.persistence(topology, max_dimension=0, field=2)
                edge_count = topology.metadata['edge_count']
            else:
                # Native hyperdigraphs require an ambient vertex; do not invent one.
                bars = PersistenceResult((), field='GF(2)', max_dimension=0,
                    metadata={'filtration_start': 0., 'filtration_end': MAX_RADIUS,
                              'scale_units': 'angstrom', 'filtration_coordinate': 'alpha_radius',
                              'empty_channel_policy': 'zero bars; no artificial vertex'})
                edge_count = 0
            vector = barcode_bin_counts(bars, bin_edges=BIN_EDGES, dimensions=(0,), mode='cover')
            cache[key] = (bars, vector.values, {'vertices': len(points),
                'edges': edge_count, 'barcodes': len(bars.intervals)})
        bars, values, info = cache[key]
        tensor[channel] = values
        barcodes.append(bars)
        diagnostics.append(info)
    if not np.isfinite(tensor).all() or np.any(tensor < 0):
        raise ValueError('Invalid barcode feature values')
    return tensor, counts, selected['deduplication'], tuple(barcodes), diagnostics


def pack_barcodes(barcodes):
    if len(barcodes) != 55:
        raise ValueError('55 barcode channels required')
    offsets = np.zeros(56, dtype=np.int64)
    births, deaths = [], []
    for channel, result in enumerate(barcodes):
        if result.max_dimension != 0 or result.field != 'GF(2)':
            raise ValueError('Only GF(2) H0 is part of this recipe')
        births.extend(bar.birth for bar in result.intervals)
        deaths.extend(bar.death for bar in result.intervals)
        offsets[channel+1] = len(births)
    return {'offsets': offsets, 'births': np.asarray(births, dtype='<f8'),
            'deaths': np.asarray(deaths, dtype='<f8')}


def unpack_barcodes(arrays):
    offsets, births, deaths = (arrays[key] for key in ('offsets', 'births', 'deaths'))
    if (offsets.shape != (56,) or offsets.dtype.kind not in 'iu' or offsets[0] != 0
            or np.any(np.diff(offsets) < 0) or births.ndim != 1 or births.shape != deaths.shape
            or offsets[-1] != len(births) or not np.isfinite(births).all()
            or np.any(births != 0) or np.any(np.isnan(deaths)) or np.any(deaths <= births)
            or np.any(deaths[np.isfinite(deaths)] > MAX_RADIUS)):
        raise ValueError('Invalid stored H0 barcodes')
    return tuple(PersistenceResult(tuple(PersistenceInterval(0, float(b), float(d), True)
        for b, d in zip(births[offsets[i]:offsets[i+1]], deaths[offsets[i]:offsets[i+1]], strict=True)),
        field='GF(2)', max_dimension=0, metadata={'filtration_start': 0., 'filtration_end': MAX_RADIUS,
            'scale_units': 'angstrom', 'filtration_coordinate': 'alpha_radius'}) for i in range(55))
