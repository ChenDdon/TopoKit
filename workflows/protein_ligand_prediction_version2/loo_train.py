"""Four explicit complete-manifest GBDT fits; no test-based model selection."""
from __future__ import annotations
import csv
import os
from pathlib import Path
import time
import uuid

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
import study_common as common

SETTINGS = dict(learning_rate=.002, max_depth=7, min_samples_split=5,
                min_samples_leaf=1, subsample=.8, max_features='sqrt',
                loss='squared_error', criterion='friedman_mse',
                n_iter_no_change=None, random_state=0, verbose=0)
JOBS = [(year, trees) for year in (2007, 2016) for trees in (10000, 30000)]


def fit_job(year, trees, output, smoke=False):
    import joblib
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    common.assert_environment()
    output = Path(output)
    receipt = common.check_contract(output)
    loaded, _ = common.manifests(output / 'labels/final')
    train, test = loaded[f'train_refined_{year}'], loaded[f'test_casf_{year}']
    if smoke:
        train, test = train[:12], test[:8]
    name = f'refined_{year}_{trees}_trees'
    base = output / ('experiments/smoke/models' if smoke else 'models')
    final = base / name
    if (final / 'metadata.json').exists():
        audit_model(final, output, receipt, smoke=smoke)
        return common.read_json(final / 'metadata.json')
    if final.exists():
        raise FileExistsError(f'Incomplete model directory: {final}')
    stage = base / ('.' + name + '.staging-' + uuid.uuid4().hex)
    stage.mkdir(parents=True)
    started = time.monotonic()
    settings = {**SETTINGS, 'n_estimators': 5 if smoke else trees}
    progress = output / ('experiments/smoke' if smoke else 'experiments/study') / (name + '.progress.json')
    common.atomic_json(progress, {'state': 'loading', 'utc': common.utc(), 'job': name, 'pid': os.getpid()})
    X, train_hashes = common.load_matrix(train, output, receipt)
    T, test_hashes = common.load_matrix(test, output, receipt)
    y = np.array([float(r['label_logka']) for r in train])
    yt = np.array([float(r['label_logka']) for r in test])
    plan = {'identity': receipt['identity'], 'job': name, 'smoke': smoke,
            'year': year, 'settings': settings, 'training_count': len(train), 'test_count': len(test),
            'train_ids': [r['pdb_id'] for r in train], 'test_ids': [r['pdb_id'] for r in test],
            'train_feature_sha256': train_hashes, 'test_feature_sha256': test_hashes,
            'X_sha256': __import__('hashlib').sha256(X.tobytes(order='C')).hexdigest(),
            'T_sha256': __import__('hashlib').sha256(T.tobytes(order='C')).hexdigest(),
            'feature_count': 12960, 'normalization': 'StandardScaler fit on all training rows only',
            'target': 'manifest label_logka, unscaled pK', 'early_stopping': False,
            'validation_split': None, 'test_used_during_training': False,
            'comparison': 'user-prespecified 10000 versus 30000 trees; all scores reported'}
    common.atomic_json(stage / 'plan.json', plan)
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('gbdt', GradientBoostingRegressor(**settings))])

    def monitor(i, estimator, _locals):
        if i == 0 or (i + 1) % 500 == 0 or i + 1 == settings['n_estimators']:
            common.atomic_json(progress, {'state': 'fitting', 'utc': common.utc(), 'job': name,
                'pid': os.getpid(), 'trees_finished': i + 1, 'trees_total': settings['n_estimators'],
                'elapsed_seconds': time.monotonic() - started})
        return False  # Progress only; never implements stopping.

    pipe.fit(X, y, gbdt__monitor=monitor)
    scaler, model = pipe.named_steps['scaler'], pipe.named_steps['gbdt']
    if (int(scaler.n_samples_seen_) != len(train) or model.n_estimators_ != settings['n_estimators']
            or model.n_features_in_ != 12960):
        raise ValueError('Training contract violation')
    np.testing.assert_allclose(scaler.mean_, X.mean(axis=0, dtype=np.float64), rtol=1e-12, atol=1e-12)
    predicted = pipe.predict(T)
    metrics = common.scores(yt, predicted)
    joblib.dump(pipe, stage / 'pipeline.joblib', compress=3)
    np.testing.assert_array_equal(joblib.load(stage / 'pipeline.joblib').predict(T), predicted)
    with (stage / 'predictions.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('pdb_id', 'observed_pK', 'predicted_pK', 'residual_predicted_minus_observed'))
        writer.writerows((r['pdb_id'], a, b, b-a) for r, a, b in zip(test, yt, predicted, strict=True))
    common.atomic_json(stage / 'metrics.json', metrics)
    common.check_contract(output, receipt)
    for sample, expected in {**train_hashes, **test_hashes}.items():
        if common.sha(output / 'features/samples' / (sample + '.npy')) != expected:
            raise ValueError('Feature changed during fit')
    meta = {'state': 'complete', 'job': name, 'identity': receipt['identity'], 'smoke': smoke,
            'training_count': len(train), 'test_count': len(test), 'metrics': metrics,
            'trees': settings['n_estimators'], 'feature_count': 12960,
            'completed_utc': common.utc(), 'seconds': time.monotonic() - started,
            'roundtrip_predictions_identical': True,
            'artifact_sha256': {p.name: common.sha(p) for p in stage.iterdir() if p.is_file()}}
    common.atomic_json(stage / 'metadata.json', meta)
    os.rename(stage, final)
    common.atomic_json(progress, meta)
    return meta


def audit_model(target, output, receipt, smoke=False):
    import hashlib
    import joblib
    from sklearn.preprocessing import StandardScaler

    target, output = Path(target), Path(output)
    meta = common.read_json(target / 'metadata.json')
    plan = common.read_json(target / 'plan.json')
    if meta['identity'] != receipt['identity'] or plan['identity'] != receipt['identity'] or plan['smoke'] != smoke:
        raise ValueError('Model identity mismatch')
    for name, expected in meta['artifact_sha256'].items():
        if common.sha(target / name) != expected:
            raise ValueError(f'Model artifact changed: {target / name}')
    loaded, _ = common.manifests(output / 'labels/final')
    year = plan['year']
    train, test = loaded[f'train_refined_{year}'], loaded[f'test_casf_{year}']
    if smoke:
        train, test = train[:12], test[:8]
    if plan['train_ids'] != [r['pdb_id'] for r in train] or plan['test_ids'] != [r['pdb_id'] for r in test]:
        raise ValueError('Model sample membership/order changed')
    X, train_hashes = common.load_matrix(train, output, receipt)
    T, test_hashes = common.load_matrix(test, output, receipt)
    assert train_hashes == plan['train_feature_sha256'] and test_hashes == plan['test_feature_sha256']
    assert hashlib.sha256(X.tobytes(order='C')).hexdigest() == plan['X_sha256']
    assert hashlib.sha256(T.tobytes(order='C')).hexdigest() == plan['T_sha256']
    pipe = joblib.load(target / 'pipeline.joblib')
    scaler, model = pipe.named_steps['scaler'], pipe.named_steps['gbdt']
    independent = StandardScaler().fit(X)
    for attr in ('mean_', 'var_', 'scale_'):
        np.testing.assert_allclose(getattr(scaler, attr), getattr(independent, attr), rtol=1e-12, atol=1e-12)
    assert int(scaler.n_samples_seen_) == len(train)
    expected_settings = {**SETTINGS, 'n_estimators': 5 if smoke else int(meta['job'].split('_')[2])}
    assert plan['settings'] == expected_settings
    assert all(model.get_params()[key] == value for key, value in expected_settings.items())
    assert model.n_estimators_ == expected_settings['n_estimators'] and model.n_features_in_ == 12960
    assert model.max_features_ == 113 and model.n_iter_no_change is None
    assert plan['validation_split'] is None and plan['test_used_during_training'] is False
    predicted = model.predict(independent.transform(T))
    saved = common.rows(target / 'predictions.csv')
    assert [r['pdb_id'] for r in saved] == [r['pdb_id'] for r in test]
    observed = np.array([float(r['label_logka']) for r in test])
    np.testing.assert_array_equal(observed, [float(r['observed_pK']) for r in saved])
    np.testing.assert_array_equal(predicted, [float(r['predicted_pK']) for r in saved])
    np.testing.assert_allclose(predicted - observed, [float(r['residual_predicted_minus_observed']) for r in saved], rtol=0, atol=1e-14)
    recomputed = common.scores(observed, predicted)
    for key, value in recomputed.items():
        if value is None:
            assert meta['metrics'][key] is None
        else:
            np.testing.assert_allclose(value, meta['metrics'][key], rtol=0, atol=1e-14)
    return {'job': meta['job'], 'passed': True, 'training_count': len(train), 'test_count': len(test),
            'independent_scaler_and_predictions': True, 'metrics': recomputed}
