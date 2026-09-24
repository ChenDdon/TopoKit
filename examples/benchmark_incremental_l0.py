"""Reproducible L0 matrix-assembly benchmark, excluding geometry/eigensolvers.

Both paths return independent dense matrices at every requested scale. The
baseline is the existing general embedded-chain operator, not a dense graph
implementation invented for this benchmark. No scientific outputs are changed.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time

import numpy as np

from topokit import PointCloud
from topokit.builders import simplicial
from topokit.core import hyperdigraph as hd
from topokit.core._hyperdigraph.spectral import ordinary_operator


def run_case(points, mode, repeats):
    n = len(points)
    scales = np.linspace(0., 1.5, 50)
    if mode == "alpha":
        topology = simplicial.from_points(PointCloud(points), complex_type="alpha",
            max_dimension=0, backend="gudhi_exact", max_scale=(scales[-1]/2)**2)
        edges = [(edge, birth) for edge, birth in topology.metadata["raw_filtration"] if len(edge)==2]
        thresholds = (scales/2)**2
    else:
        rows, cols = np.triu_indices(n, 1)
        distances = np.linalg.norm(points[rows]-points[cols], axis=1)
        edges = [((int(i),int(j)),float(d)) for i,j,d in zip(rows,cols,distances) if d<=scales[-1]]
        thresholds = scales
    native = hd.FilteredHyperdigraph(tuple(range(n)), edges, include_all_vertices=True)

    def general():
        matrices = []
        for scale in thresholds:
            operator = ordinary_operator(native.snapshot(float(scale)), 0)
            matrices.append(operator.matrix.matmat(np.eye(n)))
        return matrices

    def incremental():
        sweep = hd.L0Sweep(native)
        matrices = [sweep.matrix_at(float(scale)) for scale in thresholds]
        assert sweep.diagnostics["edges_inserted"] == len(edges)
        return matrices

    expected = general()
    actual = incremental()
    assert all(np.array_equal(a,b) for a,b in zip(expected,actual))
    digest = hashlib.sha256(b"".join(a.astype('<f8').tobytes() for a in expected)).hexdigest()
    timings = {"general": [], "incremental": []}
    # Alternate order after the equality/warm-up run.
    for repeat in range(repeats):
        for name in (("general","incremental") if repeat%2==0 else ("incremental","general")):
            start = time.perf_counter()
            values = general() if name=="general" else incremental()
            timings[name].append(time.perf_counter()-start)
            assert all(np.array_equal(a,b) for a,b in zip(expected,values))
    medians = {key:statistics.median(value) for key,value in timings.items()}
    return {"mode":mode,"vertices":n,"edges":len(edges),"scales":len(scales),
            "all_matrices_bitwise_equal":True,"matrices_sha256":digest,
            "seconds":timings,"median_seconds":medians,
            "construction_speedup":medians['general']/medians['incremental'],
            "accumulated_edge_visits_general":int(sum(sum(b<=s for _,b in edges) for s in thresholds)),
            "edge_insertions_incremental":len(edges)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=5)
    args = parser.parse_args()
    if args.repeats<1:
        parser.error('--repeats must be positive')
    cases=[]
    for n in (64,256):
        points=np.random.default_rng(20260915+n).uniform(0.,3.,size=(n,3))
        for mode in ('alpha','distance'):
            case=run_case(points,mode,args.repeats);cases.append(case)
            print(json.dumps(case),flush=True)
    result={"status":"passed","recorded_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
            "platform":platform.platform(),"numpy":np.__version__,"cases":cases,
            "scope":"Construction only; geometry and eigenvalue calculation excluded; includes dense snapshot copies. Single numerical thread requested externally. Synthetic fixed-seed point clouds; not a full molecular throughput benchmark."}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
