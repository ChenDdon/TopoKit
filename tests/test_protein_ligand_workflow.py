"""Selected workflow contracts: radius units, alpha births, crop and site identity."""
import json
from importlib.resources import files
import numpy as np
import pytest
from topokit import PointCloud
from topokit.workflows.protein_ligand_prediction import features as fs


def test_selected_schema_and_model_profile_are_available_without_dataset():
    schema=fs.schema()
    assert fs.STRATEGY_ID == "FS-AN"
    assert schema['recipe_id']=='hpcc-alpha15-cb4760366e92bd2e'
    assert schema['selection']['protein_crop_angstrom']==15
    assert schema['scales_angstrom']==[i/10 for i in range(1,51)]
    assert schema['tensor']['features']==27500
    assert schema['statistics']==list(fs.STATISTICS)
    assert schema['channels']==[list(x) for x in fs.CHANNELS]
    schema['selection']['protein_crop_angstrom']=999
    assert fs.schema()['selection']['protein_crop_angstrom']==15
    assert fs.implementation_receipt()['implementation_sha256']!=schema['engine_sha256']
    model=json.loads(files('topokit.workflows.protein_ligand_prediction').joinpath('model_profile.json').read_text())
    assert model['default_training_set']=='general_v2020R1'
    assert model['parameters']['n_estimators']==10000
    assert model['parameters']['learning_rate']==0.005
    assert model['parameters']['min_samples_split']==2
    assert model['parameters']['subsample']==0.4
    assert model['seeds']==[0,1,2] and model['seed_parameter']=='random_state'
    assert model['aggregation']=='arithmetic_mean' and model['weights']==[1/3]*3
    # Historical feature selection keeps its original modeling protocol.
    assert model['feature_selection_parameters']['learning_rate']==0.002
    assert model['feature_selection_parameters']['min_samples_split']==5


def test_deduplication_precedes_channels_and_keeps_protein_first():
    raw={'protein_points':np.array([[0.,0,0],[2.,0,0],[2.,0,0]]),
         'protein_elements':np.array(['C','N','O']),
         'ligand_points':np.array([[2.,0,0],[.3,.5,2.1],[2.2,1.4,.7]]),
         'ligand_elements':np.array(['O','H','C'])}
    dedup=fs.deduplicate_selected_atoms(raw)
    assert dedup['coordinate_deduplication']['removed_count']==2
    assert dedup['protein_elements'].tolist()==['C','N']
    assert dedup['ligand_elements'].tolist()==['H','C']
    new,present,counts=fs.compute(raw)
    assert not new[:,:,0].any()
    assert not present[fs.CHANNELS.index(('O','C'))]
    assert not present[fs.CHANNELS.index(('C','O'))]
    assert present[fs.CHANNELS.index(('C','H'))]
    assert counts[fs.CHANNELS.index(('N','null'))].tolist()==[1,0]

def test_scale_endpoints_use_inclusive_squared_radius():
    for length,index in [(.2,0),(2.,9),(10.,49)]:
        out=fs.channel_summaries(np.array([[0.,0,0]]),np.array([[length,0,0]]),cross_only=True)
        np.testing.assert_array_equal(out[1,:index],np.full(index,2.))
        np.testing.assert_array_equal(out[1,index:],np.ones(50-index))
        np.testing.assert_array_equal(out[2,index:],np.full(50-index,2.))
    outside=fs.channel_summaries(np.array([[0.,0,0]]),np.array([[10.001,0,0]]),cross_only=True)
    np.testing.assert_array_equal(outside[1],np.full(50,2.))


@pytest.mark.parametrize('cross_only',[False,True])
def test_obtuse_alpha_triangle_against_closed_form_incidence(cross_only):
    # Long edge has alpha birth 5, not its half-distance squared (=4).
    points=np.array([[0.,0,0],[4,0,0],[1,1,0]])
    edges=[((0,2),.5),((1,2),2.5)]+([] if cross_only else [((0,1),5.)])
    actual=fs.channel_summaries(points[:2],points[2:],cross_only=cross_only)
    for i,t in enumerate(fs.THRESHOLDS):
        active=[e for e,b in edges if b<=t];inc=np.zeros((3,len(active)))
        for j,(u,v) in enumerate(active):inc[u,j],inc[v,j]=-1,1
        eig=np.linalg.eigvalsh(inc@inc.T)
        np.testing.assert_allclose(actual[:,i],fs.spectral_statistics(eig),atol=1e-12,rtol=1e-12)


def test_crop_includes_15_and_supported_h_anchor_but_excludes_larger(monkeypatch):
    points=np.array([[15.,0,0],[-15.,0,0],[-15.001,0,0],[25.,0,0],[25.001,0,0],[0,1,0]])
    protein=PointCloud(points,metadata={'labels':['C']*6,'columns':{'record_name':['ATOM']*5+['HETATM']}})
    ligand=PointCloud([[0.,0,0],[10.,0,0]],metadata={'labels':['C','H']})
    monkeypatch.setattr(fs.readers,'read_pdb',lambda *a,**k:protein)
    monkeypatch.setattr(fs.readers,'read',lambda *a,**k:ligand)
    selected=fs.read_selected_atoms('protein.pdb','ligand.mol2')
    np.testing.assert_array_equal(selected['protein_points'],points[[0,1,3]])
    assert selected['counts']['protein_CNOS_ATOM_before_crop']==5
    assert selected['counts']['protein_CNOS_ATOM_after_crop']==3


def test_dense_budget_and_exact_duplicate_policy():
    p=np.array([[0.,0,0],[0,0,0]]);l=np.array([[0.,0,0],[2,0,0]])
    a=fs.channel_summaries(p,l,cross_only=True,max_dense_entries=4)
    b=fs.channel_summaries(p[:1],l[1:],cross_only=True)
    np.testing.assert_array_equal(a,b)
    with pytest.raises(fs.ResourceLimitError):
        fs.channel_summaries(p[:1],l[1:],cross_only=True,max_dense_entries=3)
