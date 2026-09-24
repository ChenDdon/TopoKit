"""VR atom-deletion application; every spectrum uses the public TopoKit L0 core."""
from collections import OrderedDict
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.spatial import cKDTree, distance
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from topokit import PointCloud, readers, ResourceLimitError
from topokit.core import hyperdigraph as core

PROTEIN = ('C','N','O','S')
LIGAND = ('C','N','O','S','P','F','Cl','Br','I','H')
CHANNELS = tuple((p,l) for p in PROTEIN for l in LIGAND)
SCALES = np.arange(2.,6.01,.5)
STATISTICS = ('positive_sum','positive_mean','positive_median','positive_std',
              'positive_variance','positive_max','positive_min','positive_sum_squares','zero_count')
AGGREGATIONS = ('sum_protein','mean_protein','sum_ligand','mean_ligand')
TOL = 1e-10
SHAPE = (40,9,9,4)


def schema():
    value={'schema':'topokit.protein_ligand.vr_atom_deletion.v1','version':'1.0.0',
      'protein_elements':PROTEIN,'ligand_elements':LIGAND,'channels':CHANNELS,
      'protein_records':'ATOM','crop_angstrom':12.,'crop_inclusive':True,
      'crop_anchors':'original supported ligand atoms, including explicit H',
      'deduplication':'exact keep first across cropped protein then ligand, before channels; preserve retained attributes',
      'graph':'one unit edge per distinct unordered pair at distance <= cutoff; all three edge types',
      'scales_angstrom':SCALES.tolist(),'deletion':'remove vertex and incident edges; independent deletions; original crop fixed',
      'operator':'ordinary TopoKit Hyperdigraph L0; one orientation per undirected edge',
      'missing_component':'available side contributes; absent-side aggregation zero',
      'statistics':STATISTICS,'std_ddof':0,'zero_tolerance':TOL,'aggregations':AGGREGATIONS,
      'empty_positive':'first eight statistics zero','empty_graph':'all nine zero',
      'tensor_shape':SHAPE,'axes':('channel','scale','statistic','aggregation'),
      'flatten_order':'C','feature_count':12960,'calculation_dtype':'float64','storage_dtype':'float32',
      'exact_reuse':'connected-component spectra; graph twins; repeated adjacency and absent-element channels',
      'engine_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    value=json.loads(json.dumps(value))
    value['recipe_id']='vr-atom-loo-'+hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()[:16]
    return value


def select_atoms(protein_file,ligand_file):
    protein=readers.read_pdb(protein_file,coordinate_units='angstrom')
    ligand=readers.read(ligand_file,coordinate_units='angstrom')
    pe=np.asarray(protein.metadata['labels'],dtype=object)
    le=np.asarray(ligand.metadata['labels'],dtype=object)
    mask=(np.asarray(protein.metadata['columns']['record_name'])=='ATOM') & np.isin(pe,PROTEIN)
    pp=np.asarray(protein.points[mask],dtype=np.float64);pe=pe[mask]
    mask=np.isin(le,LIGAND);lp=np.asarray(ligand.points[mask],dtype=np.float64);le=le[mask]
    if not len(lp):raise ValueError('No supported ligand crop anchors')
    if not len(pp):raise ValueError('No protein CNOS ATOM records')
    mask=cKDTree(lp).query(pp,k=1)[0]<=12.
    pp,pe=pp[mask],pe[mask]
    return deduplicate({'protein_points':pp,'protein_elements':pe,'ligand_points':lp,'ligand_elements':le})


def deduplicate(selected):
    points=[];elements=[];n=[]
    for side,allowed in (('protein',PROTEIN),('ligand',LIGAND)):
        x=np.asarray(selected[side+'_points'],dtype=np.float64)
        labels=np.asarray(selected[side+'_elements'],dtype=object)
        if x.ndim!=2 or x.shape[1]!=3 or not np.isfinite(x).all():raise ValueError('finite Nx3 coordinates required')
        if labels.shape!=(len(x),) or not np.isin(labels,allowed).all():raise ValueError('invalid element labels')
        points.append(x);elements.append(labels);n.append(len(x))
    cloud=PointCloud(np.vstack(points));unique=cloud.unique_coordinates()
    kept=np.asarray(unique.metadata['coordinate_deduplication']['retained_indices'],dtype=int)
    out={}
    for side,x,labels,ids in (('protein',points[0],elements[0],kept[kept<n[0]]),
                             ('ligand',points[1],elements[1],kept[kept>=n[0]]-n[0])):
        out[side+'_points']=x[ids];out[side+'_elements']=labels[ids]
    out['deduplication']={'policy':'exact_keep_first','order':'cropped_protein_then_ligand',
        'input_atoms':sum(n),'retained_atoms':len(kept),'removed_atoms':sum(n)-len(kept),
        'retained_indices':kept.tolist()}
    if out['deduplication']['removed_atoms']==0 and 'deduplication' in selected:
        out['deduplication']=selected['deduplication']
    return out


def statistics(eigenvalues):
    x=np.asarray(eigenvalues,dtype=np.float64)
    if not np.isfinite(x).all() or np.any(x < -TOL):raise ValueError('invalid non-PSD spectrum')
    p=x[x>TOL];out=np.zeros(9)
    out[8]=np.count_nonzero(np.abs(x)<=TOL)
    if len(p):
        out[:8]=(p.sum(),p.mean(),np.median(p),p.std(ddof=0),p.var(ddof=0),p.max(),p.min(),np.dot(p,p))
    return out


def native_sweep(adj):
    n=len(adj)
    edges=[((int(i),int(j)),0.) for i,j in zip(*np.where(np.triu(adj,1)))]
    native=core.FilteredHyperdigraph(tuple(range(n)),edges,include_all_vertices=True,vertex_birth=0.)
    return core.L0Sweep(native,max_dense_entries=25_000_000)


def native_spectrum(adj):
    if not len(adj):return np.empty(0,dtype=np.float64)
    return np.asarray(native_sweep(adj).laplacian(0.,tol=TOL).eigenvalues)


def twin_groups(adj):
    # Equal open or closed neighborhoods give a vertex-swap graph automorphism.
    # Consequently their vertex-deletion spectra are identical.
    n=len(adj);parent=list(range(n))
    def find(v):
        while parent[v]!=v:parent[v]=parent[parent[v]];v=parent[v]
        return v
    for closed in (False,True):
        rows=adj.copy();np.fill_diagonal(rows,closed);seen={}
        for v,row in enumerate(np.packbits(rows,axis=1)):
            key=row.tobytes()
            if key in seen:parent[find(v)]=find(seen[key])
            else:seen[key]=v
    groups={}
    for v in range(n):groups.setdefault(find(v),[]).append(v)
    return tuple(tuple(v) for v in groups.values())


class ComponentCache:
    """Bounded exact adjacency-keyed spectra; no truncation or spectral approximation."""
    def __init__(self,max_bytes=64*1024**2):
        self.max_bytes=max_bytes;self.bytes=0;self.cache=OrderedDict();self.hits=0;self.misses=0
    def get(self,adj):
        n=len(adj);key=(n,np.packbits(adj,axis=None).tobytes())
        if key in self.cache:
            self.hits+=1;result,size=self.cache.pop(key);self.cache[key]=(result,size);return result
        self.misses+=1
        sweep=native_sweep(adj);base=np.asarray(sweep.laplacian(0.,tol=TOL).eigenvalues);groups=[]
        for members in twin_groups(adj):
            deleted=np.asarray(sweep.vertex_deleted_laplacian(0.,members[0],tol=TOL).eigenvalues)
            groups.append((members,deleted))
        result=(base,groups);size=base.nbytes+sum(e.nbytes+32*len(g) for g,e in groups)+len(key[1])+512
        if size<=self.max_bytes:
            while self.cache and self.bytes+size>self.max_bytes:
                _,(_,old)=self.cache.popitem(last=False);self.bytes-=old
            self.cache[key]=(result,size);self.bytes+=size
        return result


CACHE=ComponentCache()


def summarize_graph(adj,protein_count,cache=None):
    """Compose independent component spectra, retaining all isolated zero modes."""
    cache=CACHE if cache is None else cache
    n=len(adj);out=np.zeros((9,4),dtype=np.float64)
    if not 0<=protein_count<=n:raise ValueError('invalid component count')
    if not n:return out
    count,labels=connected_components(csr_matrix(adj),directed=False)
    components=[np.flatnonzero(labels==k) for k in range(count)]
    spectra=[cache.get(adj[np.ix_(ids,ids)]) for ids in components]
    for k,ids in enumerate(components):
        other=np.concatenate([s[0] for j,s in enumerate(spectra) if j!=k]) if count>1 else np.empty(0)
        for members,deleted in spectra[k][1]:
            values=statistics(np.concatenate((other,deleted)))
            nprotein=sum(ids[v]<protein_count for v in members)
            nligand=len(members)-nprotein
            out[:,0]+=nprotein*values;out[:,2]+=nligand*values
    if protein_count:out[:,1]=out[:,0]/protein_count
    if n-protein_count:out[:,3]=out[:,2]/(n-protein_count)
    return out


def compute(selected,cache=None):
    selected=deduplicate(selected);tensor=np.zeros(SHAPE,dtype=np.float64);channel_cache={}
    pp,lp=selected['protein_points'],selected['ligand_points']
    pe,le=selected['protein_elements'],selected['ligand_elements']
    counts=np.zeros((40,2),dtype=int)
    for channel,(p,l) in enumerate(CHANNELS):
        pi=np.flatnonzero(pe==p);li=np.flatnonzero(le==l);counts[channel]=(len(pi),len(li))
        key=(tuple(pi),tuple(li))
        if key in channel_cache:tensor[channel]=channel_cache[key];continue
        points=np.vstack((pp[pi],lp[li]));n=len(points)
        if n*n>25_000_000:raise ResourceLimitError('channel exceeds explicit dense-entry budget')
        if n:
            distances=distance.squareform(distance.pdist(points));previous=None;summary=None
            for k,scale in enumerate(SCALES):
                adj=distances<=scale;np.fill_diagonal(adj,False)
                signature=np.packbits(adj,axis=None).tobytes()
                if signature!=previous:summary=summarize_graph(adj,len(pi),cache);previous=signature
                tensor[channel,k]=summary
        channel_cache[key]=tensor[channel].copy()
    if not np.isfinite(tensor).all():raise ValueError('nonfinite features')
    return np.ascontiguousarray(tensor,dtype='<f4'),counts,selected['deduplication']
