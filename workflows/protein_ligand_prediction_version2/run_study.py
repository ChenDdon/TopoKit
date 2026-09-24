#!/usr/bin/env python3
"""Resumable Cornell experiment: exact features, four fits, independent audit."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import fcntl
import multiprocessing as mp
import os
from pathlib import Path
import resource
import time
import traceback

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
import loo_features as features
import study_common as common
from loo_train import JOBS, fit_job, audit_model


def generate_one(row, output, receipt):
    started = time.monotonic()
    output = Path(output)
    sample = row['pdb_id']
    failure = output / 'features/failures' / (sample + '.json')
    try:
        source = Path(receipt['source_dataset'])
        inputs = common.input_hashes(row, source)
        record_path = output / 'features/records' / (sample + '.json')
        if record_path.exists():
            _, record = common.load_feature(sample, output, receipt, inputs)
            failure.unlink(missing_ok=True)
            return {'sample': sample, 'state': 'verified_existing', 'seconds': record['seconds']}
        selected = features.select_atoms(source / row['protein_file'], source / row['ligand_mol2_file'])
        value, counts, dedup = features.compute(selected)
        if common.input_hashes(row, source) != inputs:
            raise ValueError('Input changed during feature generation')
        path = output / 'features/samples' / (sample + '.npy')
        common.atomic_npy(path, value)
        record = {'sample': sample, 'identity': receipt['identity'],
                  'recipe_id': receipt['schema']['recipe_id'], 'inputs': inputs,
                  'feature_sha256': common.sha(path), 'shape': list(value.shape),
                  'channel_atom_counts': counts.tolist(), 'deduplication': dedup,
                  'seconds': time.monotonic() - started, 'completed_utc': common.utc(),
                  'max_rss_MB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                  'pid': os.getpid()}
        common.atomic_json(record_path, record)
        common.load_feature(sample, output, receipt, inputs)
        failure.unlink(missing_ok=True)
        return {'sample': sample, 'state': 'complete', 'seconds': record['seconds'],
                'max_rss_MB': record['max_rss_MB']}
    except Exception:
        error = {'sample': sample, 'state': 'failed', 'utc': common.utc(), 'traceback': traceback.format_exc()}
        common.atomic_json(failure, error)
        return error


def generate(values, output, receipt, workers, progress_path):
    started = time.monotonic()
    completed = []
    failed = []
    log = Path(output) / 'experiments/study/feature_events.jsonl'
    common.atomic_json(progress_path, {'stage': 'features', 'utc': common.utc(), 'total': len(values),
        'completed': 0, 'remaining': len(values), 'failed': 0, 'workers': workers})
    context = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = {pool.submit(generate_one, row, str(output), receipt): row['pdb_id'] for row in values}
        for future in as_completed(futures):
            result = future.result()
            with log.open('a') as f:
                f.write(__import__('json').dumps(result, allow_nan=False) + '\n')
            (failed if result['state'] == 'failed' else completed).append(result)
            status = {'stage': 'features', 'utc': common.utc(), 'total': len(values),
                      'completed': len(completed), 'failed': len(failed),
                      'remaining': len(values)-len(completed), 'workers': workers,
                      'elapsed_seconds': time.monotonic()-started, 'latest': result}
            common.atomic_json(progress_path, status)
            if len(completed) % 25 == 0 or result['state'] == 'failed':
                print(__import__('json').dumps(status), flush=True)
    common.check_contract(output, receipt)
    if failed:
        raise RuntimeError(f'{len(failed)} feature failures; no samples dropped; inspect features/failures')
    return completed


def audit_features(output, receipt, values):
    results = []
    for row in values:
        sample = row['pdb_id']
        inputs = common.input_hashes(row, receipt['source_dataset'])
        value, record = common.load_feature(sample, output, receipt, inputs)
        counts = np.asarray(record['channel_atom_counts'])
        if counts.shape != (40, 2) or np.any(counts < 0):
            raise ValueError(f'Invalid channel counts: {sample}')
        # Check the stored sum/mean relationship independently of spectral evaluation.
        for side in range(2):
            n = counts[:, side]
            sums, means = value[..., 2*side], value[..., 2*side+1]
            np.testing.assert_allclose(means*n[:, None, None], sums, rtol=3e-7, atol=1e-4)
            if np.any(value[n == 0, :, :, 2*side:2*side+2] != 0):
                raise ValueError(f'Absent-side aggregation is nonzero: {sample}')
        if (output / 'features/failures' / (sample + '.json')).exists():
            raise ValueError(f'Unresolved failure record: {sample}')
        results.append({'sample': sample, 'feature_sha256': record['feature_sha256'], 'inputs': inputs})
    return results


def publish(output, receipt, audits):
    result_dir = output / 'results'
    for audit in audits:
        name = audit['job']
        for filename in ('predictions.csv', 'metrics.json'):
            import shutil
            target = result_dir / name / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(output / 'models' / name / filename, target)
    rows = [{'job': a['job'], 'training_count': a['training_count'], 'test_count': a['test_count'],
             **{k: a['metrics'][k] for k in ('rmse', 'pcc', 'mae')}} for a in audits]
    with (result_dir / 'performance.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    common.atomic_json(result_dir / 'performance.json', {'identity': receipt['identity'], 'rows': rows})
    lines = ['# VR atom-deletion results', '', 'All models use 12,960 features and train-only standardization.',
             'Seed 0; no early stopping; full matched manifests; target is unscaled pK.', '',
             '| Model | Train | Test | RMSE | PCC | MAE |', '| --- | ---: | ---: | ---: | ---: | ---: |']
    def number(value):
        return 'undefined' if value is None else f'{value:.6f}'
    lines.extend(f"| {r['job']} | {r['training_count']} | {r['test_count']} | {number(r['rmse'])} | {number(r['pcc'])} | {number(r['mae'])} |" for r in rows)
    lines += ['', 'All four saved models passed independent scaler, reload, membership, prediction and metric audits.',
              'CASF is a reused comparison benchmark; these scores are not a new untouched external validation set.',
              'See experiments/study/provenance.json and final_audit.json for frozen source and input verification.']
    (result_dir / 'RESULTS.md').write_text('\n'.join(lines)+'\n')


def run(args):
    common.assert_environment()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Adjacent lock avoids making an otherwise empty new output look initialized.
    with (output.parent / ('.' + output.name + '.lock')).open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another controller holds this study lock') from error
        receipt = common.prepare(args.source, output)
        loaded, union = common.manifests(output / 'labels/final')
        progress_path = output / 'experiments/study/progress.json'
        if args.action == 'prepare':
            print(receipt['identity'], flush=True); return
        smoke = args.action == 'smoke'
        values = list(union.values())
        if smoke:
            ids = {r['pdb_id'] for name, rows in loaded.items() for r in rows[:12 if name.startswith('train') else 8]}
            values = [row for row in values if row['pdb_id'] in ids]
        values.sort(key=lambda row: row['pdb_id'])
        started = time.monotonic()
        try:
            if args.action != 'audit':
                generate(values, output, receipt, args.workers, progress_path)
            common.atomic_json(progress_path, {'stage': 'feature_audit', 'utc': common.utc(), 'completed': len(values)})
            inventory = audit_features(output, receipt, values)
            common.atomic_json(output / 'experiments/study/feature_inventory.json', inventory)
            from audit_real_samples import audit_real
            common.atomic_json(progress_path, {'stage': 'independent_real_feature_audit', 'utc': common.utc()})
            real_audit = audit_real(output, receipt, union)
            common.atomic_json(output / 'experiments/study/real_feature_audit.json', real_audit)
            if args.action != 'audit':
                common.atomic_json(progress_path, {'stage': 'training', 'utc': common.utc(), 'features_complete': len(values),
                    'models_total': 4, 'models_complete': 0, 'workers': min(4, args.ml_workers), 'smoke': smoke})
                completed = []
                with ProcessPoolExecutor(max_workers=min(4, args.ml_workers), mp_context=mp.get_context('spawn')) as pool:
                    futures = [pool.submit(fit_job, year, trees, str(output), smoke) for year, trees in JOBS]
                    for future in as_completed(futures):
                        completed.append(future.result())
                        common.atomic_json(progress_path, {'stage': 'training', 'utc': common.utc(),
                            'features_complete': len(values), 'models_total': 4, 'models_complete': len(completed),
                            'latest': completed[-1], 'smoke': smoke})
                        print(__import__('json').dumps(completed[-1]), flush=True)
            common.atomic_json(progress_path, {'stage': 'model_audit', 'utc': common.utc(), 'smoke': smoke})
            base = output / ('experiments/smoke/models' if smoke else 'models')
            audits = [audit_model(base / f'refined_{year}_{trees}_trees', output, receipt, smoke)
                      for year, trees in JOBS]
            common.check_contract(output, receipt)
            final = {'state': 'complete', 'utc': common.utc(), 'identity': receipt['identity'], 'smoke': smoke,
                     'feature_count': len(values), 'model_count': 4, 'failures': 0,
                     'independent_real_feature_slices': real_audit['slice_count'],
                     'elapsed_seconds_this_invocation': time.monotonic()-started, 'models': audits}
            common.atomic_json(output / 'experiments/study/final_audit.json', final)
            if not smoke:
                publish(output, receipt, audits)
            common.atomic_json(progress_path, final)
            print(__import__('json').dumps(final), flush=True)
        except BaseException:
            common.atomic_json(progress_path, {'stage': 'failed', 'utc': common.utc(), 'traceback': traceback.format_exc()})
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('prepare', 'smoke', 'run', 'audit'), default='run')
    parser.add_argument('--source', type=Path, default=common.SOURCE)
    parser.add_argument('--output', type=Path, default=common.OUTPUT)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--ml-workers', type=int, default=4)
    args = parser.parse_args()
    if min(args.workers, args.ml_workers) < 1:
        parser.error('Worker counts must be positive')
    run(args)
