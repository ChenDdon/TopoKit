"""Compare revised molecular tensors with a supplied frozen pre-update engine.

The frozen engine uses the retained general L0 API. No datasets or old study
artifacts are written. Input/source identities and comparison results are saved.
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import time
import types

import numpy as np

WORKFLOW=Path(__file__).resolve().parents[1]/'workflows/protein_ligand_prediction_v2_alpha_ablation'
sys.path.insert(0,str(WORKFLOW))
import alpha_strategies as current


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--baseline-engine',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--samples',nargs='+',default=['1a30','4ej8','1hps','4fys'])
    args=parser.parse_args()
    assert digest(args.baseline_engine)=='60dca28af2a6b1863ffa854751091ef2e0953b7b5706e44ba8ac8eed82238c13'
    old=types.ModuleType('frozen_alpha_before_incremental')
    # Preserve the original relative imports, but execute frozen source bytes.
    old.__file__=str(WORKFLOW/'alpha_strategies.py')
    exec(compile(args.baseline_engine.read_text(),str(args.baseline_engine),'exec'),old.__dict__)
    manifests=[args.dataset/'labels/refined_nmi_complete_toptransformer_20260913/train_pdbbind_v2016_refined_minus_casf2016_current_features.csv',
               args.dataset/'labels/test_casf_2016_nmi.csv']
    rows={}
    for manifest in manifests:
        with manifest.open(newline='') as stream:
            rows.update({r['pdb_id']:r for r in csv.DictReader(stream)})
    checks=[]
    for sample in args.samples:
        row=rows[sample]
        protein=args.dataset/row['protein_file'];ligand=args.dataset/row['ligand_mol2_file']
        selected=current.v2._read_selected_atoms(protein,ligand,20.)
        start=time.perf_counter()
        expected=old.compute_feature_tensors(selected,old.enumerate_strategies(),max_dense_entries=25_000_000,max_hyperedges=1_000_000)
        old_seconds=time.perf_counter()-start
        start=time.perf_counter()
        actual=current.compute_feature_tensors(selected,current.enumerate_strategies(),max_dense_entries=25_000_000,max_hyperedges=1_000_000)
        new_seconds=time.perf_counter()-start
        max_difference=0.
        for mask in current.enumerate_strategies():
            for a,b in zip(expected[mask][1:],actual[mask][1:]):
                np.testing.assert_array_equal(a,b)
            np.testing.assert_allclose(expected[mask][0],actual[mask][0],rtol=1e-12,atol=1e-12)
            np.testing.assert_array_equal(expected[mask][0].astype('<f4'),actual[mask][0].astype('<f4'))
            max_difference=max(max_difference,float(np.max(np.abs(expected[mask][0]-actual[mask][0]))))
        record={'sample_id':sample,'protein_sha256':digest(protein),'ligand_sha256':digest(ligand),
                'strategies':7,'float32_tensors_bitwise_equal':True,'float64_max_absolute_difference':max_difference,
                'baseline_seconds':old_seconds,'incremental_seconds':new_seconds,'selection_counts':selected['counts']}
        checks.append(record);print(json.dumps(record),flush=True)
    result={'status':'passed','recorded_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'baseline_engine_sha256':digest(args.baseline_engine),'current_engine_sha256':digest(Path(current.__file__)),
            'checks':checks,'verified_tensors':7*len(checks),
            'timing_scope':'Single-run end-to-end feature checks, not a controlled speed benchmark; construction-only timings are in benchmark.json.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
