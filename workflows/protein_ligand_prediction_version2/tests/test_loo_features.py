from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from topokit import PointCloud
import loo_features as f


def direct_stats(adj,protein_count):
    n=len(adj);values=[]
    for v in range(n):
        keep=np.arange(n)!=v;a=adj[np.ix_(keep,keep)].astype(float)
        # Independent full induced-graph D-A oracle, including all isolates.
        eig=np.linalg.eigvalsh(np.diag(a.sum(axis=1))-a)
        p=eig[eig>1e-10];s=np.zeros(9);s[8]=np.count_nonzero(abs(eig)<=1e-10)
        if len(p):s[:8]=(sum(p),np.mean(p),np.median(p),np.std(p),np.var(p),max(p),min(p),sum(p*p))
        values.append(s)
    out=np.zeros((9,4))
    for indices,j in ((range(protein_count),0),(range(protein_count,n),2)):
        chosen=[values[i] for i in indices]
        if chosen:out[:,j]=np.sum(chosen,axis=0);out[:,j+1]=np.mean(chosen,axis=0)
    return out


@pytest.mark.parametrize('n',range(0,13))
@pytest.mark.parametrize('density',[0.,.15,.45,1.])
def test_all_deletion_statistics_against_independent_full_matrix(n,density):
    rng=np.random.default_rng(310+n)
    a=np.triu(rng.random((n,n))<density,1);a=a|a.T
    cache=f.ComponentCache()
    for count in sorted(set((0,n//2,n))):
        np.testing.assert_allclose(f.summarize_graph(a,count,cache),direct_stats(a,count),rtol=1e-10,atol=1e-9)


def test_statistics_and_empty_spectra():
    np.testing.assert_array_equal(f.statistics([0,0,1,3]),[4,2,2,1,1,3,1,10,2])
    np.testing.assert_array_equal(f.statistics([]),np.zeros(9))
    np.testing.assert_array_equal(f.statistics([0,-1e-12,1e-10]),[0,0,0,0,0,0,0,0,3])
    with pytest.raises(ValueError):f.statistics([-1e-4])


def test_vertex_removal_recalculates_degree_and_keeps_isolates():
    a=np.array([[0,1,0],[1,0,1],[0,1,0]],dtype=bool)
    out=f.summarize_graph(a,1)
    # Deleting middle vertex leaves two isolates, not the nonzero diagonal minor.
    np.testing.assert_allclose(out,direct_stats(a,1),atol=1e-12)
    assert out[8,2]==3 and out[8,3]==1.5


def test_component_reuse_twins_and_cache_eviction():
    a=np.ones((5,5),dtype=bool);np.fill_diagonal(a,False)
    assert len(f.twin_groups(a))==1
    cache=f.ComponentCache(max_bytes=2000)
    np.testing.assert_allclose(f.summarize_graph(a,2,cache),direct_stats(a,2),atol=1e-10)
    before=cache.hits;f.summarize_graph(a,3,cache);assert cache.hits>before
    star=np.zeros((5,5),dtype=bool);star[0,1:]=True;star[1:,0]=True
    assert sorted(map(len,f.twin_groups(star)))==[1,4]
    np.testing.assert_allclose(f.summarize_graph(star,1,cache),direct_stats(star,1),atol=1e-10)
    assert cache.bytes<=cache.max_bytes


def selected():
    return {'protein_points':np.array([[0,0,0],[2,0,0],[12,0,0]],float),
            'protein_elements':np.array(['C','C','N']),
            'ligand_points':np.array([[0,2,0],[0,4.5,0],[9,0,0]],float),
            'ligand_elements':np.array(['C','H','N'])}


def test_complete_tensor_order_missing_components_and_inclusive_thresholds():
    s=selected();out,counts,_=f.compute(s,cache=f.ComponentCache())
    assert out.shape==(40,9,9,4) and out.dtype==np.float32 and out.flags.c_contiguous
    assert out.size==12960
    for channel,(p,l) in enumerate(f.CHANNELS):
        pp=s['protein_points'][s['protein_elements']==p];lp=s['ligand_points'][s['ligand_elements']==l]
        x=np.vstack((pp,lp));assert counts[channel].tolist()==[len(pp),len(lp)]
        for k,r in enumerate(f.SCALES):
            a=np.linalg.norm(x[:,None,:]-x[None,:,:],axis=2)<=r;np.fill_diagonal(a,False)
            np.testing.assert_allclose(out[channel,k],direct_stats(a,len(pp)),rtol=1e-6,atol=1e-6)
    assert not np.any(out[f.CHANNELS.index(('S','Br'))])
    assert np.any(out[f.CHANNELS.index(('C','Br')),:,:,0])
    assert not np.any(out[f.CHANNELS.index(('C','Br')),:,:,2:])
    # Only a single ligand H remains in S/H, hence all deletion summaries zero.
    assert not np.any(out[f.CHANNELS.index(('S','H'))])


def test_exact_dedup_before_channels_first_labels_win():
    s=selected();s['ligand_points'][0]=s['protein_points'][0]
    s['ligand_elements'][0]='F'
    r=f.deduplicate(s)
    assert len(r['ligand_points'])==2 and r['deduplication']['removed_atoms']==1
    assert 'F' not in r['ligand_elements']
    assert len(s['ligand_points'])==3
    s['ligand_points'][0]=[1e-8,0,0]
    assert f.deduplicate(s)['deduplication']['removed_atoms']==0


def test_crop_uses_original_h_anchors_atom_records_and_inclusive_radius(monkeypatch):
    protein=PointCloud([[12,0,0],[12.00001,0,0],[0,1,0],[100,1,0]],metadata={
        'labels':['C','C','N','O'],'columns':{'record_name':['ATOM','ATOM','HETATM','ATOM']}})
    ligand=PointCloud([[0,0,0],[100,0,0]],metadata={'labels':['H','C']})
    monkeypatch.setattr(f.readers,'read_pdb',lambda *a,**kw:protein)
    monkeypatch.setattr(f.readers,'read',lambda *a,**kw:ligand)
    r=f.select_atoms('x.pdb','x.mol2')
    np.testing.assert_array_equal(r['protein_points'],[[12,0,0],[100,1,0]])


def test_schema_and_no_mutation():
    s=selected();before={k:v.copy() for k,v in s.items()};f.compute(s)
    for k in s:np.testing.assert_array_equal(s[k],before[k])
    schema=f.schema();assert schema['feature_count']==12960 and schema['tensor_shape']==[40,9,9,4]
    assert schema['scales_angstrom']==[2.,2.5,3.,3.5,4.,4.5,5.,5.5,6.]
