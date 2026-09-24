import os
from pathlib import Path
import subprocess
import sys


def test_all_workflows_run_with_topology_packages_blocked():
    source = Path(__file__).resolve().parents[2] / "src"
    code = '''
import sys
class BlockTopologyImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'gudhi', 'ripser', 'dionysus', 'gtda', 'simplicial_topo'}:
            raise AssertionError('Forbidden topology dependency: ' + fullname)
sys.meta_path.insert(0, BlockTopologyImports())
from topokit.core._simplicial import persistent_laplacian_at
from topokit.workflows.simplicial import analyze_points
points = [[0.,0.], [1.,0.], [1.,1.], [0.,1.]]
for kind, a, b in [('alpha', .3, .6), ('rips', 1., 1.5)]:
    result = analyze_points(points, complex_type=kind, scales=[a,b])
    assert result.snapshots[a].homology.betti_numbers == (1,1)
    assert persistent_laplacian_at(result.complex,a,b,1).betti_number == 0
'''
    env = dict(os.environ, PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE="1")
    subprocess.run([sys.executable, "-c", code], env=env, check=True, capture_output=True, text=True)
