"""Explicit manifests, checksummed artifacts and shared experiment contracts."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile
from datetime import datetime, timezone

import numpy as np
import loo_features as features

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
SOURCE = PROJECT / 'datasets/protein_ligand_prediction'
OUTPUT = PROJECT / 'datasets/protein_ligand_prediction_version2'
EXPECTED = {'train_refined_2007': 1105, 'test_casf_2007': 195,
            'train_refined_2016': 3772, 'test_casf_2016': 285}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(tmp, path)


def atomic_npy(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.', suffix='.npy', delete=False) as f:
        temporary = Path(f.name)
        np.save(f, value, allow_pickle=False)
    os.replace(temporary, path)


def rows(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def manifests(directory):
    result = {name: rows(Path(directory) / (name + '.csv')) for name in EXPECTED}
    union = {}
    for name, values in result.items():
        if len(values) != EXPECTED[name] or len({r['pdb_id'] for r in values}) != len(values):
            raise ValueError(f'Incorrect or duplicate manifest membership: {name}')
        for row in values:
            sample = row['pdb_id']
            if len(sample) != 4 or not sample.isalnum():
                raise ValueError(f'Invalid PDB ID: {sample}')
            if row['label_relation'] != '=' or not np.isfinite(float(row['label_logka'])):
                raise ValueError(f'Invalid exact pK label: {sample}')
            if sample in union:
                old = union[sample]
                for key in ('protein_file', 'ligand_mol2_file'):
                    if old[key] != row[key]:
                        raise ValueError(f'Conflicting structure: {sample}')
            union[sample] = row
    for year in (2007, 2016):
        train = {r['pdb_id'] for r in result[f'train_refined_{year}']}
        test = {r['pdb_id'] for r in result[f'test_casf_{year}']}
        if train & test:
            raise ValueError(f'Matched train/test overlap: {year}')
    if len(union) != 4324:
        raise ValueError(f'Expected 4324 unique complexes, got {len(union)}')
    return result, union


def source_inventory():
    package = HERE.parents[1]
    paths = sorted((package / 'src/topokit').rglob('*.py')) + sorted(HERE.rglob('*.py'))
    return {str(p.relative_to(package)): sha(p) for p in paths}


def environment():
    import scipy
    import sklearn
    import joblib
    import sys
    return {'python': platform.python_version(), 'executable': sys.executable,
            'numpy': np.__version__, 'scipy': scipy.__version__,
            'sklearn': sklearn.__version__, 'joblib': joblib.__version__,
            'conda_prefix': os.environ.get('CONDA_PREFIX'),
            'threads': {key: os.environ.get(key) for key in
                       ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}}


def assert_environment():
    import sys
    prefix = os.environ.get('CONDA_PREFIX', '')
    if Path(prefix).name != 'topokit' or Path(sys.prefix).resolve() != Path(prefix).resolve():
        raise RuntimeError('Activate the topokit conda environment before running this study')


def prepare(source=SOURCE, output=OUTPUT):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output:
        raise ValueError('New experiment needs its own output directory')
    loaded, union = manifests(source / 'labels/final')
    receipt = {'schema': features.schema(), 'source_dataset': str(source),
               'output_dataset': str(output), 'software': environment(),
               'source_sha256': source_inventory(), 'expected_counts': EXPECTED,
               'unique_complexes': len(union),
               'manifest_sha256': {name: sha(source / 'labels/final' / (name + '.csv')) for name in EXPECTED}}
    receipt['identity'] = digest(receipt)
    receipt_path = output / 'experiments/study/provenance.json'
    if receipt_path.exists():
        if read_json(receipt_path) != receipt:
            raise ValueError('Study identity changed; preserve old run and use a new output directory')
        check_contract(output, receipt)
        return receipt
    # Atomic features/models must never be reused without their original receipt.
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Nonempty output without provenance: {output}')
    for name in ('labels/final', 'features/samples', 'features/records', 'features/failures',
                 'models', 'results', 'experiments/study', 'experiments/study/logs'):
        (output / name).mkdir(parents=True, exist_ok=True)
    for name in EXPECTED:
        shutil.copyfile(source / 'labels/final' / (name + '.csv'), output / 'labels/final' / (name + '.csv'))
    # Preserve structure paths in byte-identical manifests without duplicating structures.
    (output / 'structures').symlink_to(source / 'structures', target_is_directory=True)
    for relative in receipt['source_sha256']:
        target = output / 'experiments/study/source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(HERE.parents[1] / relative, target)
    atomic_json(output / 'features/feature_schema.json', receipt['schema'])
    atomic_json(receipt_path, receipt)
    (output / 'README.md').write_text(
        '# Protein–ligand atom-deletion experiment\n\n'
        'Separate 12,960-feature VR atom-deletion experiment. See the workflow README at\n'
        '../../topokit/workflows/protein_ligand_prediction_version2/README.md.\n\n'
        'labels/final contains byte-identical existing 2007/2016 manifests; structures is a\n'
        f'read-only-by-contract symlink to {source}/structures.\n'
        'features contains the new tensors and their records; models contains the four fits;\n'
        'results contains predictions and separate RMSE/PCC values. experiments/study holds\n'
        'frozen source, provenance, logs, progress and final audit. Remote results remain here.\n')
    return receipt


def check_contract(output, receipt=None):
    output = Path(output)
    receipt = receipt or read_json(output / 'experiments/study/provenance.json')
    if source_inventory() != receipt['source_sha256'] or features.schema() != receipt['schema']:
        raise ValueError('Scientific source changed during study')
    if environment() != receipt['software']:
        raise ValueError('Runtime environment differs from frozen study')
    for name, expected in receipt['manifest_sha256'].items():
        for root in (Path(receipt['source_dataset']), output):
            if sha(root / 'labels/final' / (name + '.csv')) != expected:
                raise ValueError(f'Manifest changed: {name}')
    if read_json(output / 'features/feature_schema.json') != receipt['schema']:
        raise ValueError('Feature schema changed')
    return receipt


def input_hashes(row, source):
    source = Path(source).resolve()
    result = {}
    for key in ('protein_file', 'ligand_mol2_file'):
        path = (source / row[key]).resolve()
        if not path.is_relative_to(source) or not path.is_file():
            raise ValueError(f'Unavailable/invalid input: {path}')
        result[key] = {'path': row[key], 'sha256': sha(path)}
    return result


def load_feature(sample, output, receipt, expected_inputs=None):
    output = Path(output)
    path = output / 'features/samples' / (sample + '.npy')
    record = read_json(output / 'features/records' / (sample + '.json'))
    if (record['sample'] != sample or record['identity'] != receipt['identity']
            or record['recipe_id'] != receipt['schema']['recipe_id']
            or record['feature_sha256'] != sha(path)
            or (expected_inputs is not None and record['inputs'] != expected_inputs)):
        raise ValueError(f'Feature provenance mismatch: {sample}')
    value = np.load(path, allow_pickle=False)
    if (value.shape != features.SHAPE or value.dtype != np.dtype('<f4')
            or not value.flags.c_contiguous or not np.isfinite(value).all()):
        raise ValueError(f'Feature tensor invalid: {sample}')
    return value, record


def load_matrix(values, output, receipt):
    matrix = np.empty((len(values), 12960), dtype=np.float32)
    hashes = {}
    for i, row in enumerate(values):
        value, record = load_feature(row['pdb_id'], output, receipt)
        matrix[i] = value.ravel(order='C')
        hashes[row['pdb_id']] = record['feature_sha256']
    return matrix, hashes


def scores(observed, predicted):
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError('Nonfinite observations or predictions')
    error = predicted - observed
    pcc = (float(np.corrcoef(observed, predicted)[0, 1])
           if len(observed) > 1 and np.ptp(observed) > 0 and np.ptp(predicted) > 0 else None)
    return {'rmse': float(np.sqrt(np.mean(error ** 2))),
            'pcc': pcc,
            'mae': float(np.mean(np.abs(error))), 'count': len(observed)}
