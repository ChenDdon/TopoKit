"""Exact-coordinate input handling and unchanged higher-dimensional operators."""
import numpy as np
import pytest

from topokit import PointCloud
from topokit.builders import simplicial as builder
from topokit.core import simplicial as core


def test_first_point_preserves_attributes_order_and_reader_metadata():
    cloud = PointCloud([[2,0],[0,0],[2,0],[0,1e-12]],
        ids=["first","second","duplicate","near"], weights=[2,4,99,8],
        metadata={"labels":("C","O","N","S"), "columns":{"serial":(10,20,30,40)},
                  "source_atom_count":4, "source_bonds":({"atoms":("first","second")},
                                                           {"atoms":("duplicate","near")})})
    unique = cloud.unique_coordinates()
    assert unique.ids == ("first","second","near")
    np.testing.assert_array_equal(unique.weights,[2,4,8])
    assert unique.metadata["labels"] == ("C","O","S")
    assert unique.metadata["columns"]["serial"] == (10,20,40)
    assert unique.metadata["atom_count"] == 3
    assert unique.metadata["source_atom_count"] == 4
    assert len(unique.metadata["bonds"]) == 1
    assert unique.metadata["coordinate_deduplication"]["original_to_retained_id"]["duplicate"] == "first"
    assert len(cloud) == 4 and cloud.metadata["labels"][2] == "N"


@pytest.mark.parametrize("points,expected", [([],0), ([[1,2,3]],1), ([[1,2,3]]*5,1),
                                             ([[0,0,0],[-0.,0,0]],1)])
def test_empty_single_repeated_and_signed_zero(points,expected):
    cloud = PointCloud(np.asarray(points,dtype=float).reshape(-1,3))
    unique = cloud.unique_coordinates()
    assert len(unique) == expected
    assert len(builder.from_points(cloud).native.simplices(0)) == expected


@pytest.mark.parametrize("bad", [np.nan,np.inf,-np.inf])
def test_nonfinite_coordinates_rejected_before_unique(bad):
    with pytest.raises(ValueError,match="finite"):
        PointCloud([[0,0],[bad,0]]).unique_coordinates()


@pytest.mark.parametrize("degree", [0,1,2])
def test_duplicate_alpha_matches_manual_unique_homology_and_laplacian(degree):
    points = [[0,0,0],[2,0,0],[0,0,0],[0,2,0],[0,0,2],[2,0,0]]
    cloud = PointCloud(points,ids=["a","b","a2","c","d","b2"])
    actual = builder.from_points(cloud,max_dimension=2)
    expected = builder.from_points(cloud.subset(["a","b","c","d"]),max_dimension=2,
                                   duplicates="error")
    assert actual.metadata["duplicate_point_count"] == 2
    assert actual.metadata["original_to_vertex"] == (0,1,0,3,4,1)
    for scale in (0.,1.,2.,3.):
        a = core.laplacian(actual,degree,scale=scale,return_matrix=True)
        b = core.laplacian(expected,degree,scale=scale,return_matrix=True)
        assert a.basis == b.basis
        np.testing.assert_array_equal(a.matrix.toarray(),b.matrix.toarray())
        assert core.homology(actual,scale=scale).betti_numbers == core.homology(expected,scale=scale).betti_numbers
    a = core.persistent_laplacian(actual,degree,start=1.,end=3.,return_matrix=True)
    b = core.persistent_laplacian(expected,degree,start=1.,end=3.,return_matrix=True)
    assert a.basis == b.basis
    np.testing.assert_array_equal(a.matrix.toarray(),b.matrix.toarray())


def test_unique_vertex_budget_is_checked_after_deduplication():
    obj = builder.from_points([[1,2,3]]*100,max_dimension=0,max_simplices=1)
    assert obj.native.simplices(0) == ((0,),)
    assert obj.metadata["input_point_count"] == 100
