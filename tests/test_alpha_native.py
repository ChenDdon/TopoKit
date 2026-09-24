"""Independent contracts for the GUDHI-free alpha filtration implementation."""
import importlib.util
import subprocess
import sys

import numpy as np
import pytest
from scipy.spatial import QhullError

from topokit.builders import _alpha_native as implementation
from topokit.builders._simplicial import alpha_complex, GeometryError
from topokit.core._simplicial import homology, persistent_homology


def test_native_thin_tetrahedron_preserves_true_affine_dimension():
    e = 2.**-48
    points = [[0,0,0],[1,0,0],[0,1,0],[.25,.25,e]]
    tree = alpha_complex(points)
    z = (e*e-.375)/(2*e)
    assert tree.metadata["affine_dimension"] == 3
    assert tree.filtration((0,1,2,3)) == pytest.approx(.5+z*z,rel=1e-14)
    # The long edge has a non-Gabriel birth; a dimension cutoff must not
    # replace it by distance squared / 4.
    edges = alpha_complex(points,max_dimension=1)
    for edge in edges.simplices(1):
        assert edges.filtration(edge) == tree.filtration(edge)


def test_closed_form_obtuse_triangle_boundaries_and_metadata():
    points = [[-1,0,0],[1,0,0],[0,.5,0]]
    full = alpha_complex(points)
    assert full.filtration((0,1)) == full.filtration((0,1,2)) == 1.5625
    assert full.filtration((0,2)) == .3125
    assert (0,1) not in alpha_complex(points,max_scale=np.nextafter(1.5625,0.))
    assert (0,1) in alpha_complex(points,max_scale=1.5625)
    assert full.metadata["full_simplex_count"] == 7
    assert not full.metadata["coordinate_perturbation"]


@pytest.mark.parametrize("dimension",[2,3,4])
def test_small_cloud_repair_matches_regular_delaunay(monkeypatch,dimension):
    points = np.random.default_rng(882).normal(size=(9,dimension))
    expected = alpha_complex(points)
    def unavailable(*args,**kwargs):
        raise QhullError("deliberate candidate triangulation failure")
    monkeypatch.setattr(implementation,"Delaunay",unavailable)
    actual = alpha_complex(points)
    assert actual.metadata["triangulation_repaired"]
    assert set(actual) == set(expected)
    for simplex in actual:
        assert actual.filtration(simplex) == pytest.approx(expected.filtration(simplex),rel=1e-12,abs=1e-12)
    assert homology(actual).betti_numbers == (1,)+(0,)*dimension


@pytest.mark.parametrize("embedded",[False,True])
def test_cospherical_symbolic_repair_is_a_triangulation(monkeypatch,embedded):
    square = np.array([[0.,0],[1,0],[1,1],[0,1]])
    if embedded:
        square = np.column_stack((square,square[:,0]+square[:,1]))
    def unavailable(*args,**kwargs):
        raise QhullError("force rational tie handling")
    monkeypatch.setattr(implementation,"Delaunay",unavailable)
    tree = alpha_complex(square)
    assert len(tree.simplices(2)) == 2
    assert len(tree.simplices(1)) == 5
    assert homology(tree).betti_numbers == (1,0,0)
    if not embedded:
        np.testing.assert_array_equal(persistent_homology(tree).diagram(1),[[.25,.5]])


def test_bounded_repair_fails_explicitly_instead_of_jitter(monkeypatch):
    def unavailable(*args,**kwargs):
        raise QhullError("force bounded repair")
    monkeypatch.setattr(implementation,"Delaunay",unavailable)
    with pytest.raises(GeometryError,match="300000"):
        alpha_complex(np.random.default_rng(3).normal(size=(100,3)))


def test_large_cloud_local_cospherical_repair_keeps_expected_edge():
    # A six-site near-cospherical molecular geometry that Qhull can merge
    # numerically. Far sites make whole-cloud enumeration exceed the bound.
    sites = np.array([[-58.767,97.989,81.123],[-58.481,99.655,79.264],
                      [-54.357,103.927,82.651],[-56.216,103.641,80.985],
                      [-52.829,99.517,76.713],[-54.495,101.376,76.999]])
    points = np.vstack((sites,np.random.default_rng(21).uniform(200,250,(100,3))))
    tree = alpha_complex(points,max_dimension=1,max_scale=24.01)
    assert (0,5) in tree
    assert tree.filtration((0,5)) == pytest.approx(23.061042615475984,abs=1e-10)
    assert len(tree.simplices(0)) == len(points)


@pytest.mark.parametrize("dimension",[2,3,4])
@pytest.mark.parametrize("seed",range(3))
@pytest.mark.skipif(importlib.util.find_spec("gudhi") is None,reason="optional independent oracle")
def test_all_dimensions_against_exact_oracle(dimension,seed):
    import gudhi
    points = np.random.default_rng(seed).normal(size=(12,dimension))
    native = alpha_complex(points)
    oracle = gudhi.AlphaComplex(points=points,precision="exact").create_simplex_tree()
    expected = {tuple(s):float(b) for s,b in oracle.get_filtration()}
    assert set(native) == set(expected)
    for simplex in native:
        assert native.filtration(simplex) == pytest.approx(expected[simplex],rel=1e-9,abs=1e-10)


@pytest.mark.skipif(importlib.util.find_spec("gudhi") is None,reason="optional independent oracle")
def test_l1_l2_operators_keep_full_coface_contributions():
    from topokit.builders.simplicial import from_points
    from topokit.core.simplicial import laplacian
    points = np.random.default_rng(872).normal(size=(10,3))
    native = from_points(points,max_dimension=2,backend="native")
    oracle = from_points(points,max_dimension=2,backend="gudhi_exact")
    for scale in (.7,1.8,3.2):
        for degree in (0,1,2):
            ours = laplacian(native,degree,scale=scale,return_matrix=True)
            theirs = laplacian(oracle,degree,scale=scale,return_matrix=True)
            assert ours.basis == theirs.basis
            assert (ours.matrix-theirs.matrix).nnz == 0
            np.testing.assert_allclose(ours.eigenvalues,theirs.eigenvalues,atol=1e-12)


def test_workflow_runs_when_gudhi_import_is_forbidden():
    code = '''
import importlib.abc,sys
from pathlib import Path
class RejectGUDHI(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='gudhi' or fullname.startswith('gudhi.'):
            raise AssertionError('production workflow tried to import GUDHI')
sys.meta_path.insert(0,RejectGUDHI())
sys.path.insert(0,str(Path('src').resolve()))
from topokit.workflows.protein_ligand_prediction import features as fs
import numpy as np
result=fs.channel_summaries(np.array([[0.,0,0]]),np.array([[2.,0,0]]),cross_only=True)
assert result.shape==(10,50)
assert result[1,8]==2 and result[1,9]==1
assert fs.schema()['construction']['backend']=='native'
'''
    from pathlib import Path
    process = subprocess.run([sys.executable,"-c",code],cwd=Path(__file__).resolve().parents[1],
                             capture_output=True,text=True)
    assert process.returncode == 0,process.stdout+process.stderr


def test_near_planar_hull_sliver_uses_local_repair_only_after_old_failure(monkeypatch):
    from pathlib import Path
    points = np.loadtxt(Path(__file__).resolve().parents[1]/
                        "examples/data/alpha_near_planar_hull_73.csv", delimiter=",", skiprows=1)
    repair = implementation._repair_cells
    def without_new_recovery(*args,**kwargs):
        if kwargs.get("ill_conditioned_stars"):
            raise GeometryError("previous native repair exhausted")
        return repair(*args,**kwargs)
    with monkeypatch.context() as m:
        m.setattr(implementation,"_repair_cells",without_new_recovery)
        with pytest.raises(GeometryError,match="previous native repair exhausted"):
            alpha_complex(points,max_scale=24.01)
    tree = alpha_complex(points,max_scale=24.01)
    assert tree.metadata["triangulation_repaired"]
    assert len(tree.simplices(0)) == 73
    assert len(tree) == 1239  # independent exact-alpha oracle count
    # This hull sliver's exact sphere contains every other point, so it is
    # not a Delaunay tetrahedron. Its near-zero volume confused Qhull.
    assert (66,67,69,70) not in tree
    assert not tree.metadata["coordinate_perturbation"]


@pytest.mark.skipif(importlib.util.find_spec("gudhi") is None,reason="optional independent oracle")
def test_near_planar_hull_full_complex_matches_exact_alpha():
    from pathlib import Path
    import gudhi
    points = np.loadtxt(Path(__file__).resolve().parents[1]/
                        "examples/data/alpha_near_planar_hull_73.csv", delimiter=",", skiprows=1)
    native = dict(alpha_complex(points).get_filtration())
    external = gudhi.AlphaComplex(points=points,precision="exact").create_simplex_tree()
    expected = {tuple(s):float(b) for s,b in external.get_filtration()}
    assert native.keys() == expected.keys()
    for simplex in native:
        assert native[simplex] == pytest.approx(expected[simplex],rel=1e-9,abs=1e-10)
