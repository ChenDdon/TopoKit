"""Fixed structure-based protein–protein Topo features.

The public recipe retains every fifth radius of the verified dense PPI recipe,
with no partner-swap augmentation. Geometry and spectral algorithms remain in
TopoKit's builders, core, and shared spectral postprocessing. Coordinate files
must use angstroms; the reader does not infer or convert their units.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

from topokit import PointCloud, ResourceLimitError, readers
from topokit.builders import simplicial
from topokit.core import hyperdigraph
from topokit.workflows.protein_ligand_prediction.features import spectral_statistics, STATISTICS
from topokit.workflows.protein_ligand_prediction import features as pl_features
import topokit

RECIPE_ID = 'ppi-topo-cnos-ca-disjoint-crop20-alpha-r1p0-r7p5-step0p5-v1'
PARENT_RECIPE_ID = 'ppi-topo-cnos-ca-disjoint-crop20-alpha-r1p0-r7p9-v2'
DENSE_RADIUS_INDICES = tuple(range(0, 70, 5))
GROUPS = ('null', 'C', 'N', 'O', 'S', 'CA')
CHANNELS = tuple((a, b) for a in GROUPS for b in GROUPS)
RADII = tuple((10 + i) / 10 for i in DENSE_RADIUS_INDICES)
THRESHOLDS = np.square(RADII)
CUTOFF = 20.0
TENSOR_SHAPE = (len(STATISTICS), len(RADII), len(CHANNELS))


def schema():
    """Return a fresh JSON-ready contract for the 5,040-feature recipe."""
    return {
        'recipe_id': RECIPE_ID, 'version': '1.0.0', 'coordinate_units': 'angstrom',
        'parent_recipe': {'recipe_id': PARENT_RECIPE_ID,
            'tensor_shape': [len(STATISTICS), 70, len(CHANNELS)],
            'retained_radius_indices': list(DENSE_RADIUS_INDICES),
            'projection': 'dense_tensor[:, retained_radius_indices, :]; flatten in C order'},
        'augmentation': 'none', 'radius_evaluation': 'direct at retained radii',
        'categories': list(GROUPS), 'category_policy': 'CA is carbon named CA; C excludes CA; no calcium category',
        'channels': [list(c) for c in CHANNELS], 'alpha_radii_angstrom': list(RADII),
        'alpha_squared_thresholds': THRESHOLDS.tolist(), 'statistics': list(STATISTICS),
        'interface': {'cutoff_angstrom': CUTOFF, 'metric': 'cross-partner CA-CA distance',
                      'predicate': 'strictly_less_than', 'selection': 'all CNOS ATOM rows in qualifying residues',
                      'missing_CA': 'exclude unanchored residues and report their identities',
                      'missing_chain_policy': 'only explicitly allowed missing chains contribute empty sets; present chains remain',
                      'empty_partner_policy': 'if one partner is explicitly empty, retain all CA-anchored CNOS residues of the available partner for null-side channels'},
        'coordinate_selection': {'model': 'first encountered', 'chain_namespace': 'author',
            'records': 'ATOM only; include noncanonical ATOM residues',
            'alternate_locations': 'highest occupancy per residue/atom name; ties first input row; record removed rows'},
        'partner_order': 'descending total observed ATOM residue count before cropping; ties first chain appearance',
        'duplicate_coordinates': 'global exact keep-first after ordering/cropping; retain atom category and component',
        'geometry': 'native alpha 1.1.1 on each selected channel union; full coface births before edge filtering',
        'mixed_edges': 'cross-partner only', 'null_side_edges': 'within selected partner',
        'operator': 'complete ordinary unweighted Hyperdigraph L0 at each radius; not two-scale persistent L',
        'spectral_zero_tolerance': 1e-10, 'missing_channel': 'zero only when required category absent',
        'tensor': {'shape': list(TENSOR_SHAPE), 'dtype': 'float32', 'order': 'C', 'features': int(np.prod(TENSOR_SHAPE))},
        'null_null': 'intentional all-zero channel', 'alpha_engine': '1.1.1',
    }


def implementation_receipt():
    """Identify the implementation and installed numerical source files."""
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    root = Path(topokit.__file__).parent
    tree = hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        tree.update(str(path.relative_to(root)).encode() + b'\0' + path.read_bytes() + b'\0')
    return {'implementation_version': '1.0.0', 'recipe_id': RECIPE_ID,
            'implementation_sha256': digest(__file__),
            'spectral_statistics_source_sha256': digest(pl_features.__file__),
            'topokit_python_source_tree_sha256': tree.hexdigest()}


def _missing(value):
    return value is None or str(value) in ('', '.', '?')


def _read_atoms(path):
    """Normalize reader metadata; select first model, never infer chain mappings."""
    path = Path(path)
    if path.suffix.lower() == '.pdb':
        models = [line[10:14].strip() for line in path.read_text().splitlines() if line.startswith('MODEL ')]
        model = int(models[0]) if models else None
        cloud = readers.read_pdb(path, model=model, coordinate_units='angstrom')
        c = cloud.metadata['columns']
        atoms = [dict(index=i, point=np.asarray(p), element=str(cloud.metadata['labels'][i]),
            record=c['record_name'][i], chain=str(c['chain_id'][i] or ''),
            residue=(str(c['chain_id'][i] or ''), str(c['residue_sequence'][i]), str(c['insertion_code'][i] or '')),
            residue_name=c['residue_name'][i], name=c['atom_name'][i],
            alt=c['alternate_location'][i], occupancy=c['occupancy'][i]) for i, p in enumerate(cloud.points)]
        return atoms, {'format': 'PDB', 'model': model, 'available_models': models or [None]}
    if path.suffix.lower() not in ('.cif', '.mmcif'):
        raise ValueError('structure_file must be PDB or atom-site mmCIF')
    cloud = readers.read_cif(path, coordinate_units='angstrom')
    c = cloud.metadata['columns']
    def col(tag, default=None, fallback=None):
        value = c.get('_atom_site.' + tag)
        if value is None and fallback: value = c.get('_atom_site.' + fallback)
        return value if value is not None else [default]*len(cloud)
    records, chains = col('group_PDB'), col('auth_asym_id')
    numbers, insertions = col('auth_seq_id'), col('pdbx_PDB_ins_code')
    names, residues = col('auth_atom_id', fallback='label_atom_id'), col('auth_comp_id', fallback='label_comp_id')
    alts, occupancy = col('label_alt_id'), col('occupancy')
    models = col('pdbx_PDB_model_num', '1')
    first = models[0]
    atoms = []
    for i, p in enumerate(cloud.points):
        if models[i] != first: continue
        if _missing(chains[i]) or _missing(numbers[i]):
            raise ValueError('mmCIF needs explicit author chain/residue identifiers')
        atoms.append(dict(index=i, point=np.asarray(p), element=str(cloud.metadata['labels'][i]),
            record=records[i], chain=str(chains[i]),
            residue=(str(chains[i]), str(numbers[i]), '' if _missing(insertions[i]) else str(insertions[i])),
            residue_name=residues[i], name=str(names[i]), alt=None if _missing(alts[i]) else str(alts[i]),
            occupancy=None if _missing(occupancy[i]) else float(occupancy[i])))
    return atoms, {'format': 'mmCIF', 'model': str(first), 'available_models': list(dict.fromkeys(map(str, models)))}


def _category(atom):
    return 'CA' if atom['element'] == 'C' and atom['name'] == 'CA' else atom['element']


def select_atoms(atoms, partner_a_chains, partner_b_chains, *, input_metadata=None,
                 allowed_missing_chains=(), empty_partner=None):
    """Select explicit partners while preserving row and residue identities.

    Missing chains raise unless listed in ``allowed_missing_chains``. Setting
    ``empty_partner="A"`` or ``"B"`` deliberately discards that entire partner;
    this is never inferred from a dataset or a chain-name convention.
    """
    groups = [list(partner_a_chains), list(partner_b_chains)]
    if any(not g or len(g) != len(set(g)) for g in groups) or set(groups[0]) & set(groups[1]):
        raise ValueError('two nonempty disjoint partner chain lists are required')
    allowed_missing_chains = list(allowed_missing_chains)
    if len(allowed_missing_chains) != len(set(allowed_missing_chains)) or not set(allowed_missing_chains) <= set(groups[0]+groups[1]):
        raise ValueError('allowed missing chains must be distinct members of the declared partner chains')
    if empty_partner not in (None, 'A', 'B'):
        raise ValueError('empty_partner must be A, B, or None')
    empty_index = None if empty_partner is None else ('A', 'B').index(empty_partner)
    atoms = [a for a in atoms if a['record'] == 'ATOM']
    chain_order = list(dict.fromkeys(a['chain'] for a in atoms))
    missing_requested_chains = [[ch for ch in g if ch not in chain_order] for g in groups]
    unauthorized_missing = {ch for i, missing in enumerate(missing_requested_chains)
                            if i != empty_index for ch in missing} - set(allowed_missing_chains)
    if unauthorized_missing:
        raise ValueError('requested protein ATOM chains absent: ' + ','.join(sorted(unauthorized_missing)))
    atoms = [a for a in atoms if a['chain'] in set(groups[0]+groups[1])]
    # Bio.PDB-like selected atom convention; retain original row order after selection.
    chosen, alt_removed = {}, []
    for a in atoms:
        key = (a['residue'], a['name'])
        if key not in chosen:
            chosen[key] = a
            continue
        old = chosen[key]
        if old['residue_name'] != a['residue_name']:
            raise ValueError(f'conflicting residue identities at {a["residue"]}')
        score = lambda x: float(x['occupancy']) if x['occupancy'] is not None else 0.0
        if score(a) > score(old):
            alt_removed.append(old['index']); chosen[key] = a
        else: alt_removed.append(a['index'])
    atoms = sorted(chosen.values(), key=lambda a: a['index'])
    partners = [[a for a in atoms if a['chain'] in g] for g in groups]
    observed_lengths = [len({a['residue'] for a in p}) for p in partners]
    if empty_index is not None:
        partners[empty_index] = []
    if not any(partners):
        raise ValueError('at least one partner must contain observed protein ATOM records')
    lengths = [len({a['residue'] for a in p}) for p in partners]
    ranks = [min((chain_order.index(ch) for ch in g if ch in chain_order), default=len(chain_order)+i)
             for i,g in enumerate(groups)]
    order = sorted(range(2), key=lambda i: (-lengths[i], ranks[i]))
    partners = [partners[i] for i in order]
    ca_lists = [[a for a in p if a['name'] == 'CA' and a['element'] == 'C'] for p in partners]
    if any(partners[i] and not ca_lists[i] for i in range(2)):
        raise ValueError('each nonempty protein partner needs at least one C-alpha anchor')
    anchor_points = [np.asarray([a['point'] for a in p], dtype=float).reshape(-1,3) for p in ca_lists]
    single_partner = any(not p for p in partners)
    crop_mode = 'available_partner_CA_residues_no_opponent' if single_partner else 'cross_partner_CA_distance'
    keep_residues = []
    missing_ca = []
    for i in range(2):
        if single_partner:
            keep_residues.append({a['residue'] for a in ca_lists[i]})
        else:
            distances = cKDTree(anchor_points[1-i]).query(anchor_points[i], k=1)[0]
            keep_residues.append({a['residue'] for a, d in zip(ca_lists[i], distances) if d < CUTOFF})
        missing_ca.append(sorted({a['residue'] for a in partners[i]} - {a['residue'] for a in ca_lists[i]}))
    cropped = [[a for a in partners[i] if a['residue'] in keep_residues[i] and a['element'] in ('C','N','O','S')] for i in range(2)]
    if not any(cropped) or (not single_partner and any(not p for p in cropped)):
        raise ValueError('no interacting residues within the declared C-alpha cutoff')
    counts_before = [len(p) for p in cropped]
    all_atoms = cropped[0] + cropped[1]
    cloud = PointCloud(np.asarray([a['point'] for a in all_atoms], dtype=float))
    unique = cloud.unique_coordinates()
    info = unique.metadata['coordinate_deduplication']
    retained = info['retained_indices']
    retained_set = set(retained)
    removed = [{'input_index': all_atoms[i]['index'], 'kept_input_index': all_atoms[j]['index'],
                'removed_partner': 1 if i < counts_before[0] else 2,
                'category': _category(all_atoms[i]), 'point': all_atoms[i]['point'].tolist()}
               for i,j in info['original_to_retained_id'].items() if i != j]
    cropped = [[a for i,a in enumerate(p, start=0 if k==0 else counts_before[0]) if i in retained_set] for k,p in enumerate(cropped)]
    result = {}
    for i, name in enumerate(('partner1','partner2')):
        result[name+'_points'] = np.asarray([a['point'] for a in cropped[i]], dtype=float).reshape(-1,3)
        result[name+'_categories'] = np.asarray([_category(a) for a in cropped[i]], dtype=object)
    result['diagnostics'] = {
        'input': input_metadata or {}, 'source_partner_chains': groups,
        'ordered_source_partners': ['A' if i==0 else 'B' for i in order],
        'ordered_partner_chains': [[ch for ch in chain_order if ch in groups[i]] if i != empty_index else [] for i in order],
        'observed_requested_partner_residue_counts': observed_lengths,
        'source_partner_residue_counts': lengths, 'ordered_partner_residue_counts': [lengths[i] for i in order],
        'allowed_missing_chains': allowed_missing_chains, 'missing_requested_chains': missing_requested_chains,
        'explicit_empty_partner': empty_partner, 'interface_crop_mode': crop_mode,
        'length_basis': 'observed ATOM residues before crop; not SEQRES length',
        'chain_order': chain_order, 'cropped_residues': [len(k) for k in keep_residues],
        'cropped_atoms_before_deduplication': counts_before, 'selected_atoms': [len(p) for p in cropped],
        'missing_CA_residues_excluded': missing_ca,
        'alternate_atom_rows_removed': sorted(alt_removed), 'duplicate_coordinate_rows_removed': removed,
        'nonpositive_occupancy_selected': sum(a['occupancy'] is not None and a['occupancy']<=0 for p in cropped for a in p),
    }
    return result


def read_selected_atoms(structure_file, partner_a_chains, partner_b_chains, *, allowed_missing_chains=(), empty_partner=None):
    """Read angstrom PDB/mmCIF coordinates and apply the fixed selection rules."""
    atoms, metadata = _read_atoms(structure_file)
    return select_atoms(atoms, partner_a_chains, partner_b_chains, input_metadata=metadata,
                        allowed_missing_chains=allowed_missing_chains, empty_partner=empty_partner)


def channel_summaries(partner1, partner2, *, cross_only,
                      max_dense_entries=25_000_000, max_simplices=1_000_000):
    """Evaluate ten complete L0 statistics at the fourteen retained radii."""
    points = np.vstack((partner1, partner2))
    count, first_count = len(points), len(partner1)
    result = np.zeros(TENSOR_SHAPE[:2], dtype=float)
    if not count or (cross_only and (not first_count or first_count==count)): return result
    if count*count > max_dense_entries:
        raise ResourceLimitError(f'Full spectrum needs {count} x {count} dense entries')
    cloud = PointCloud(points, weights=np.arange(count, dtype=float), metadata={
        'coordinate_units': 'angstrom', 'weight_semantics': 'row-order direction tags only'})
    alpha = simplicial.from_points(cloud, complex_type='alpha', max_dimension=0,
        backend='native', filtration_range=(0., float(THRESHOLDS[-1])), duplicates='merge',
        geometry_tolerance=1e-12, max_simplices=max_simplices)
    if alpha.metadata.get('geometry_backend_version') != '1.1.1':
        raise ValueError('This recipe requires native alpha engine 1.1.1')
    if set(alpha.native.simplices(0)) != {(i,) for i in range(count)}:
        raise ValueError('alpha construction lost selected vertices')
    edges = [(tuple(e),float(b)) for e,b in alpha.metadata['raw_filtration'] if len(e)==2]
    if cross_only: edges = [(e,b) for e,b in edges if e[0] < first_count <= e[1]]
    edges.sort(key=lambda x:x[1])
    births = np.asarray([b for _,b in edges], dtype=float)
    filtered = hyperdigraph.FilteredHyperdigraph(tuple(range(count)),edges,include_all_vertices=True,vertex_birth=0.)
    sweep = hyperdigraph.L0Sweep(filtered,max_dense_entries=max_dense_entries)
    previous = -1
    for i,t in enumerate(THRESHOLDS):
        active = int(np.searchsorted(births,t,side='right'))
        if active != previous:
            values = np.zeros(count) if not active else sweep.laplacian(float(t),tol=1e-10).eigenvalues
            summary = spectral_statistics(values)
            previous = active
        result[:,i] = summary
    if not np.isfinite(result).all(): raise ValueError('nonfinite L0 summary')
    return result


def compute(selected, *, max_dense_entries=25_000_000, max_simplices=1_000_000):
    """Return the float64 tensor, channel-presence flags and partner atom counts.

    Use :func:`featurize` for the canonical float32 C-order stored tensor.
    """
    tensor = np.zeros(TENSOR_SHAPE,dtype=float)
    present = np.zeros(36,dtype=bool)
    counts = np.zeros((36,2),dtype=np.int32)
    for i,(a,b) in enumerate(CHANNELS):
        first = selected['partner1_points'][selected['partner1_categories']==a]
        second = selected['partner2_points'][selected['partner2_categories']==b]
        counts[i] = len(first),len(second)
        if a==b=='null' or (a!='null' and not len(first)) or (b!='null' and not len(second)): continue
        present[i] = True
        tensor[:,:,i] = channel_summaries(first,second,cross_only=a!='null' and b!='null',
            max_dense_entries=max_dense_entries,max_simplices=max_simplices)
    return tensor,present,counts


def featurize(structure_file, partner_a_chains, partner_b_chains, *, allowed_missing_chains=(), empty_partner=None, **limits):
    """Return a canonical float32 C-order tensor of shape ``(10, 14, 36)``."""
    selected = read_selected_atoms(structure_file,partner_a_chains,partner_b_chains,
                                  allowed_missing_chains=allowed_missing_chains, empty_partner=empty_partner)
    return np.ascontiguousarray(compute(selected,**limits)[0],dtype='<f4')
