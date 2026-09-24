"""Artifact identity and train/test isolation checks with tiny synthetic data."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import ph_features as f
import ph_common as c
import ph_train as ml


def fixture_features(tmp_path):
    receipt = {'identity': 'fixture', 'schema': f.schema()}
    rows = []
    rng = np.random.default_rng(1200)
    for i in range(22):
        sample = f'{i:04d}'
        row = {'pdb_id': sample, 'label_logka': str(2 + i*.31)}
        rows.append(row)
        # Deliberately shifted test distribution reveals train/test scaler leakage.
        tensor = rng.normal(0 if i < 14 else 50, 1, size=f.SHAPE).astype('<f4')
        path = tmp_path / 'features/samples' / (sample + '.npy')
        c.atomic_npy(path, tensor)
        barcode_path = tmp_path / 'barcodes/samples' / (sample + '.npz')
        c.atomic_npz(barcode_path, {'offsets': np.zeros(56, dtype=np.int64), 'births': np.empty(0), 'deaths': np.empty(0)})
        c.atomic_json(tmp_path / 'features/records' / (sample + '.json'),
                      {'sample': sample, 'identity': receipt['identity'],
                       'recipe_id': receipt['schema']['recipe_id'], 'inputs': {},
                       'feature_sha256': c.sha(path), 'barcode_sha256': c.sha(barcode_path)})
    return receipt, rows[:14], rows[14:]


def test_identity_shape_and_matrix_order(tmp_path):
    receipt, train, test = fixture_features(tmp_path)
    X, hashes = c.load_matrix(train, tmp_path, receipt)
    assert X.shape == (14, 5500) and len(hashes) == 14
    tensor, _ = c.load_feature(train[0]['pdb_id'], tmp_path, receipt)
    np.testing.assert_array_equal(X[0], tensor.ravel(order='C'))
    record_path = tmp_path / 'features/records/0000.json'
    record = c.read_json(record_path); record['identity'] = 'other-experiment'
    c.atomic_json(record_path, record)
    with pytest.raises(ValueError, match='provenance'):
        c.load_feature('0000', tmp_path, receipt)


def test_training_and_independent_audit(tmp_path, monkeypatch):
    pytest.importorskip('sklearn')
    receipt, train, test = fixture_features(tmp_path)
    monkeypatch.setattr(c, 'assert_environment', lambda: None)
    monkeypatch.setattr(c, 'check_contract', lambda *args: receipt)
    monkeypatch.setattr(c, 'manifests', lambda *args: ({'train_refined_2007': train, 'test_casf_2007': test}, {}))
    meta = ml.fit_job(2007, 30000, tmp_path, smoke=True)
    assert meta['training_count'] == 12 and meta['test_count'] == 8 and meta['trees'] == 5
    target = tmp_path / 'experiments/smoke/models/refined_2007_30000_trees'
    assert ml.audit_model(target, tmp_path, receipt, smoke=True)['passed']
    # Resume must validate the exact saved model, not refit or trust only a marker.
    assert ml.fit_job(2007, 30000, tmp_path, smoke=True) == meta
    predictions = target / 'predictions.csv'
    predictions.write_text(predictions.read_text() + '\n')
    with pytest.raises(ValueError, match='artifact changed'):
        ml.audit_model(target, tmp_path, receipt, smoke=True)


def test_full_protocol_settings_are_explicit():
    assert ml.JOBS == [(2007, 10000), (2007, 30000), (2013, 10000), (2013, 30000), (2016, 10000), (2016, 30000)]
    assert ml.SETTINGS == dict(learning_rate=.002, max_depth=7, min_samples_split=5,
        min_samples_leaf=1, subsample=.8, max_features='sqrt', loss='squared_error',
        criterion='friedman_mse', n_iter_no_change=None, random_state=0, verbose=0)


def test_constant_predictions_have_explicit_undefined_pcc():
    assert c.scores(np.array([1., 2.]), np.array([3., 3.]))['pcc'] is None
