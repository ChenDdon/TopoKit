"""Exact-alpha adapter contracts, independent closed forms and degeneracy."""
import importlib.util
import subprocess
import sys

import numpy as np
import pytest

from topokit import PointCloud, ResourceLimitError
from topokit.builders import simplicial as builder


requires_gudhi = pytest.mark.skipif(importlib.util.find_spec("gudhi") is None,
                                   reason="optional exact alpha dependency unavailable")


def test_native_geometry_never_imports_gudhi_or_falls_back():
    code = """
import importlib.abc, sys
class RejectGUDHI(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'gudhi' or fullname.startswith('gudhi.'):
            raise ImportError('GUDHI deliberately unavailable')
sys.meta_path.insert(0, RejectGUDHI())
from topokit.builders.simplicial import from_points
assert from_points([[0,0],[2,0]],max_dimension=0).native.simplices(1)==((0,1),)
try:
    from_points([[0,0],[2,0]],max_dimension=0,backend='gudhi_exact')
except ImportError as error:
    assert 'alpha_exact' in str(error)
else:
    raise AssertionError('exact backend must not silently use native geometry')
"""
    result = subprocess.run([sys.executable,"-c",code],text=True,capture_output=True)
    assert result.returncode == 0, result.stdout+result.stderr
    with pytest.raises(ValueError,match="backend"):
        builder.from_points([[0,0],[2,0]],backend="automatic")


@requires_gudhi
def test_exact_obtuse_births_and_cofaces_survive_requested_skeleton():
    points=PointCloud([[-1,0,0],[1,0,0],[0,.5,0]],ids=["a","b","c"],weights=[5,8,13])
    topology=builder.from_points(points,max_dimension=0,backend="gudhi_exact")
    raw=dict(topology.metadata["raw_filtration"])
    assert raw[(0,1)] == 1.5625
    assert raw[(0,2)] == raw[(1,2)] == .3125
    assert topology.native.simplices(2)==()
    assert topology.metadata["full_simplex_count"]==7
    assert topology.metadata["vertex_id_map"] == {0:"a",1:"b",2:"c"}
    assert not topology.metadata["geometry_weighted"]
    assert not topology.metadata["geometry_tolerance_applied"]
    assert topology.metadata["geometry_backend"] == "gudhi_exact"
    assert topology.metadata["geometry_precision"] == "exact"
    assert not topology.metadata["coordinate_perturbation"]
    # Check the full coface budget even when only vertices or a tiny radius is requested.
    with pytest.raises(ResourceLimitError,match="max_simplices"):
        builder.from_points(points,max_dimension=0,filtration_range=(0,0),
                            backend="gudhi_exact",max_simplices=6)


@requires_gudhi
def test_exact_backend_keeps_thin_full_dimension_without_projection():
    epsilon=2.**-48
    points=np.array([[0,0,0],[1,0,0],[0,1,0],[.25,.25,epsilon]])
    topology=builder.from_points(points,max_dimension=2,backend="gudhi_exact")
    raw=dict(topology.metadata["raw_filtration"])
    # Independent circumsphere equations give center (.5,.5,z).
    z=(epsilon**2-.375)/(2*epsilon)
    assert raw[(0,1,2,3)] == pytest.approx(.5+z*z,rel=1e-14)
    assert topology.metadata["affine_dimension"] == 3
    assert len(topology.native.simplices(0))==4
    np.testing.assert_array_equal(topology.cloud.points,points)


@requires_gudhi
def test_exact_bounds_duplicate_policy_empty_and_collinear_clouds():
    obj=builder.from_points([[0,0],[2,0]],max_dimension=0,backend="gudhi_exact",
                            filtration_range=(0,1))
    assert dict(obj.metadata["raw_filtration"])[(0,1)]==1
    assert not builder.from_points([[0,0],[2,0]],max_dimension=0,backend="gudhi_exact",
                                   filtration_range=(0,np.nextafter(1.,0.))).native.simplices(1)
    points=[[0,0],[0,0],[2,0]]
    with pytest.raises(ValueError,match="Duplicate"):
        builder.from_points(points,backend="gudhi_exact",duplicates="error")
    merged=builder.from_points(points,max_dimension=0,backend="gudhi_exact")
    assert merged.metadata["original_to_vertex"]==(0,0,2)
    assert merged.native.simplices(1)==((0,2),)
    empty=builder.from_points(np.empty((0,3)),max_dimension=0,backend="gudhi_exact")
    assert empty.native.simplices(0)==()
    single=builder.from_points([[2,3,4]],max_dimension=0,backend="gudhi_exact")
    assert single.native.simplices(0)==((0,),)
    line=builder.from_points([[0,0],[2,0],[5,0]],max_dimension=0,backend="gudhi_exact")
    assert dict(line.metadata["raw_filtration"])[(1,2)]==2.25
    assert line.native.simplices(1)==((0,1),(1,2))
