"""Batched unweighted alpha geometry, using NumPy/SciPy and Python rationals.

Qhull supplies candidate Delaunay cells; this module computes the alpha
filtration itself. Circumspheres are evaluated on the original coordinates,
not on their numerically projected affine hull. Small, ill-conditioned clouds
can be retriangulated by enumerating empty circumspheres. No jitter is used.
"""
from fractions import Fraction
from itertools import combinations, islice
import math

import numpy as np
from scipy.spatial import Delaunay, QhullError, cKDTree

from ..core._simplicial._validation import ComplexityLimitError, nonnegative_int, tolerance
from ..core._simplicial.complex import SimplexTree
from ..data import _coordinate_representatives

VERSION = "1.1.1"
_ENUMERATION_LIMIT = 300_000


def _solve(matrix, rhs):
    """Small exact linear system; None means genuinely rank deficient."""
    a = [list(row) + [value] for row, value in zip(matrix, rhs)]
    n = len(a)
    for col in range(n):
        pivot = next((i for i in range(col, n) if a[i][col]), None)
        if pivot is None:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        divisor = a[col][col]
        a[col] = [v / divisor for v in a[col]]
        for i in range(n):
            if i != col and a[i][col]:
                factor = a[i][col]
                a[i] = [v - factor*w for v, w in zip(a[i], a[col])]
    return [row[-1] for row in a]


class _Exact:
    def __init__(self, points):
        self.points = points
        self.rows = {}
        self.spheres = {}

    def row(self, i):
        i = int(i)
        if i not in self.rows:
            self.rows[i] = tuple(Fraction(float(x)) for x in self.points[i])
        return self.rows[i]

    def sphere(self, simplex):
        key = tuple(int(i) for i in simplex)
        if key not in self.spheres:
            base = self.row(key[0])
            edges = [tuple(a-b for a, b in zip(self.row(i), base)) for i in key[1:]]
            gram = [[sum(a*b for a, b in zip(u, v)) for v in edges] for u in edges]
            coef = _solve(gram, [gram[i][i]/2 for i in range(len(edges))])
            if coef is None:
                self.spheres[key] = None
            else:
                offset = [sum(c*e[j] for c, e in zip(coef, edges)) for j in range(len(base))]
                self.spheres[key] = (tuple(a+b for a,b in zip(base,offset)),
                                     sum(v*v for v in offset))
        return self.spheres[key]

    def power(self, sphere, vertex):
        center, radius = sphere
        return sum((a-b)**2 for a,b in zip(self.row(vertex), center)) - radius

    def rank(self):
        basis = []
        pivots = []
        origin = self.row(0)
        for i in range(1, len(self.points)):
            row = [a-b for a,b in zip(self.row(i),origin)]
            for pivot, vector in zip(pivots, basis):
                factor = row[pivot]
                row = [a-factor*b for a,b in zip(row,vector)]
            pivot = next((j for j,x in enumerate(row) if x), None)
            if pivot is not None:
                divisor = row[pivot]
                basis.append([x/divisor for x in row])
                pivots.append(pivot)
                if len(basis) == self.points.shape[1]:
                    break
        return len(basis)

    def tie_outside(self, simplex, vertex):
        """Lexicographic infinitesimal lifting chooses one cospherical triangulation.

        Largest lexicographic coordinate has the dominant positive lift;
        original coordinates/births do not
        change. Positive power in this symbolic lift means outside.
        """
        base = self.row(simplex[0])
        edges = [tuple(a-b for a,b in zip(self.row(i),base)) for i in simplex[1:]]
        delta = tuple(a-b for a,b in zip(self.row(vertex),base))
        gram = [[sum(a*b for a,b in zip(u,v)) for v in edges] for u in edges]
        weights = _solve(gram, [sum(a*b for a,b in zip(e,delta)) for e in edges])
        coefficients = {int(vertex): Fraction(1), int(simplex[0]): sum(weights)-1}
        coefficients.update({int(i): -w for i,w in zip(simplex[1:],weights)})
        order = sorted(coefficients, key=self.row, reverse=True)
        return next(coefficients[i] for i in order if coefficients[i]) > 0


def _spheres(points, cells, exact):
    """Vectorized minimum circumspheres, with rational repair of ill-conditioning."""
    p = points[cells].astype(np.longdouble)
    edges = p[:,1:] - p[:,:1]
    norms = np.sum(edges*edges, axis=2)
    degree = cells.shape[1]-1
    bad = np.zeros(len(cells), dtype=bool)
    if degree == 1:
        offset = edges[:,0]/2
    elif degree == 2 and points.shape[1] == 3:
        a,b = edges[:,0],edges[:,1]
        cross = np.cross(a,b)
        denominator = 2*np.sum(cross*cross,axis=1)
        bad = denominator <= 1e-20*np.maximum(norms[:,0]*norms[:,1], np.finfo(float).tiny)
        with np.errstate(divide="ignore",invalid="ignore"):
            offset = (norms[:,0,None]*np.cross(b,cross) + norms[:,1,None]*np.cross(cross,a))/denominator[:,None]
    elif degree == 3 and points.shape[1] == 3:
        a,b,c = edges[:,0],edges[:,1],edges[:,2]
        cross = np.cross(b,c)
        determinant = np.sum(a*cross,axis=1)
        bad = determinant**2 <= 1e-20*np.maximum(np.prod(norms,axis=1),np.finfo(float).tiny)
        with np.errstate(divide="ignore",invalid="ignore"):
            offset = (norms[:,0,None]*cross + norms[:,1,None]*np.cross(c,a) + norms[:,2,None]*np.cross(a,b))/(2*determinant[:,None])
    else:
        gram = np.asarray(edges @ np.swapaxes(edges,1,2),dtype=float)
        bad = np.linalg.cond(gram) > 1e10
        offset = np.zeros((len(cells),points.shape[1]),dtype=np.longdouble)
        good = ~bad
        if good.any():
            coefficients = np.linalg.solve(gram[good], np.asarray(norms[good]/2,dtype=float)[...,None])[...,0]
            offset[good] = np.sum(coefficients[:,:,None]*edges[good],axis=1)
    centers = p[:,0] + offset
    radii = np.sum(offset*offset,axis=1)
    valid = np.ones(len(cells),dtype=bool)
    for i in np.flatnonzero(bad | ~np.isfinite(radii)):
        sphere = exact.sphere(cells[i])
        if sphere is None:
            valid[i] = False
            centers[i], radii[i] = 0, np.inf
        else:
            center,radius = sphere
            centers[i] = [float(x) for x in center]
            radii[i] = float(radius)
    return centers,radii,valid,bad


def _empty(points, cells, centers, radii, valid, bad, index, exact, *, ties=False):
    """Empty-sphere test excluding incident vertices; exact near-boundary signs."""
    # At most |simplex| incident vertices can precede a nonincident neighbor.
    k = min(len(points), cells.shape[1]+1)
    _, nearest = index.query(np.asarray(centers,dtype=float),k=k)
    nearest = np.asarray(nearest).reshape(len(cells),k)
    incident = np.any(nearest[:,:,None] == cells[:,None,:],axis=2)
    delta = points[nearest].astype(np.longdouble) - centers[:,None,:]
    powers = np.sum(delta*delta,axis=2)-radii[:,None]
    powers[incident] = np.inf
    minimum = np.min(powers,axis=1)
    error = 1e-11*np.maximum(1,radii)
    empty = valid & (minimum >= 0)
    uncertain = valid & ((np.abs(minimum) <= error) | bad)
    for i in np.flatnonzero(uncertain):
        sphere = exact.sphere(cells[i])
        if sphere is None:
            empty[i] = False
            continue
        # Rare ill-conditioned spheres use all points, avoiding a rounded
        # center's nearest-neighbor ordering as a geometric decision.
        candidates = range(len(points)) if bad[i] else index.query_ball_point(
            np.asarray(centers[i],dtype=float),float(np.sqrt(max(0,radii[i])+error[i])*1.00000000001))
        members = set(map(int,cells[i]))
        empty[i] = True
        for j in candidates:
            if j in members:
                continue
            power = exact.power(sphere,j)
            if power < 0 or (ties and power == 0 and not exact.tie_outside(cells[i],j)):
                empty[i] = False
                break
    return empty


def _enumerate(points, rank, index, exact, vertices=None):
    from ._simplicial import GeometryError
    vertices = range(len(points)) if vertices is None else vertices
    count = math.comb(len(vertices),rank+1)
    if count > _ENUMERATION_LIMIT:
        raise GeometryError("Unreliable Qhull triangulation; exact small-cloud repair exceeds 300000 candidate cells (no jitter or GUDHI fallback)")
    iterator = combinations(vertices,rank+1)
    accepted = []
    while True:
        batch = list(islice(iterator,2048))
        if not batch:
            break
        cells = np.asarray(batch,dtype=int)
        centers,radii,valid,bad = _spheres(points,cells,exact)
        empty = _empty(points,cells,centers,radii,valid,bad,index,exact,ties=True)
        accepted.extend(cells[empty].tolist())
        # Do not retain rational data for every rejected candidate.
        exact.spheres.clear()
    if not accepted:
        raise GeometryError("No reliable Delaunay cells in exact small-cloud repair")
    return np.asarray(accepted,dtype=int)


def _repair_cells(points, maximal, rank, index, exact, centers, radii, valid, bad,
                  *, ill_conditioned_stars=False):
    """Repair local near-cospherical cavities without enumerating the whole cloud.

    New cells must have empty spheres against ALL original points. Symbolic
    tie decisions are also applied to retained cells, so the repaired region
    cannot overlap a different cospherical triangulation outside the cavity.
    """
    from ._simplicial import GeometryError
    keep = _empty(points,maximal,centers,radii,valid,bad,index,exact,ties=True)
    groups = set()
    budget = 0
    for i in np.flatnonzero(~keep):
        if valid[i] and np.isfinite(radii[i]) and not (ill_conditioned_stars and bad[i]):
            radius = float(np.sqrt(radii[i]+1e-10*max(1,radii[i])))
            vertices = set(index.query_ball_point(np.asarray(centers[i],dtype=float),radius))
        else:
            # A flat cell has no sphere. On the explicit last-resort repair,
            # an ill-conditioned hull sliver can have an enormous erroneous
            # nonempty circumsphere; its facet star is a bounded local cavity.
            # Candidate cells still pass empty-ball tests against ALL points.
            neighbors = np.sum(np.isin(maximal,maximal[i]),axis=1) >= rank
            vertices = set(map(int,maximal[neighbors].ravel()))
        vertices.update(map(int,maximal[i]))
        group = tuple(sorted(vertices))
        if group not in groups:
            budget += math.comb(len(group),rank+1)
            if budget > _ENUMERATION_LIMIT:
                raise GeometryError("Local alpha repair exceeds 300000 candidate cells (no jitter or GUDHI fallback)")
            groups.add(group)
    repaired = [maximal[keep]]
    for vertices in sorted(groups):
        repaired.append(_enumerate(points,rank,index,exact,vertices))
    result = np.unique(np.concatenate(repaired),axis=0)
    # A hole would expose a non-convex boundary face. Check manifold incidence
    # and supporting boundary planes after local replacement.
    faces = np.concatenate([np.delete(result,j,axis=1) for j in range(rank+1)])
    faces,counts = np.unique(faces,axis=0,return_counts=True)
    if np.any(counts>2):
        raise GeometryError("Nonmanifold boundary after native alpha repair")
    for face in faces[counts==1]:
        vertices = points[face]
        # Test whether this facet supports the full point set within its affine
        # hull. The inward vector comes from its incident maximal simplex.
        adjacent = result[np.sum(np.isin(result,face),axis=1)==rank][0]
        other = next(int(v) for v in adjacent if v not in face)
        edges = vertices[1:]-vertices[0]
        vector = points[other]-vertices[0]
        if len(edges):
            vector = vector - np.linalg.lstsq(edges.T,vector,rcond=None)[0] @ edges
        powers = (points-vertices[0]) @ vector
        error = 1e-10*max(1,float(np.max(np.abs(powers))))
        if np.min(powers) < -error:
            raise GeometryError("Incomplete boundary after native alpha repair")
    return result


def alpha_complex(points, *, max_dimension=None, max_scale=math.inf,
                  duplicates="merge", geometry_tolerance=1e-12, max_simplices=1_000_000):
    from ._simplicial import GeometryError, _cutoff, _points
    points = _points(points)
    maximum = _cutoff(max_scale)
    tolerance(geometry_tolerance)  # compatibility; does not enlarge empty balls
    if max_dimension is not None:
        max_dimension = nonnegative_int(max_dimension,"max_dimension")
    if duplicates not in {"error","merge"}:
        raise ValueError("duplicates must be 'error' or 'merge'")
    tree = SimplexTree(max_simplices=max_simplices)
    labels, representatives = _coordinate_representatives(points)
    original_count = len(points)
    if len(labels) != original_count and duplicates == "error":
        raise ValueError("Duplicate coordinates; pass duplicates='merge' to merge explicitly")
    points = points[labels]
    n = len(points)
    if n > tree.max_simplices:
        raise ComplexityLimitError("Unique vertex count exceeds max_simplices before Delaunay construction")
    tree.metadata = {"complex_type":"alpha","scale_units":"squared_radius","max_scale":maximum,
        "original_to_vertex":tuple(int(i) for i in representatives),
        "duplicate_policy":duplicates, "duplicate_representative":"first_input_row",
        "input_point_count":original_count, "unique_point_count":n,
        "duplicate_point_count":original_count-n,
        "geometry_backend":"native","geometry_backend_version":VERSION,
        "geometry_precision":"floating point with rational degeneracy repair",
        "geometry_tolerance_applied":False,"coordinate_perturbation":False,
        "cofaces_processed_before_truncation":True,"full_simplex_count":n,
        "simplex_budget_scope":"full Delaunay closure before filtration/skeleton truncation",
        "triangulation_repaired":False,"affine_dimension":0}
    if n == 0:
        return tree
    exact = _Exact(points)
    centered = points.astype(np.longdouble)-points[0].astype(np.longdouble)
    scale = np.max(np.abs(centered))
    normalized = np.asarray(centered/scale if scale else centered,dtype=float)
    _,singular,vt = np.linalg.svd(normalized,full_matrices=False)
    rank = int(np.count_nonzero(singular > np.finfo(float).eps*max(normalized.shape)*singular[0])) if singular[0] else 0
    if rank < min(n-1,points.shape[1]):
        rank = exact.rank()
    tree.metadata["affine_dimension"] = rank
    if rank == 0:
        tree._store((int(labels[0]),),0.)
        return tree
    if (1 << (rank+1))-1 > max_simplices:
        raise ComplexityLimitError("A maximal alpha simplex alone exceeds max_simplices")
    index = cKDTree(points)
    if rank == 1:
        # Any varying coordinate orders an exactly collinear cloud.
        axis = int(np.argmax(np.ptp(normalized,axis=0)))
        order = np.argsort(points[:,axis])
        maximal = np.sort(np.column_stack((order[:-1],order[1:])),axis=1)
    elif n == rank+1:
        maximal = np.arange(n,dtype=int)[None,:]
    else:
        coordinates = normalized if rank == points.shape[1] else normalized @ vt[:rank].T
        maximal = None
        try:
            maximal = np.sort(Delaunay(coordinates).simplices,axis=1)
            centers,radii,valid,bad = _spheres(points,maximal,exact)
            empty = _empty(points,maximal,centers,radii,valid,bad,index,exact,ties=True)
            if len(np.unique(maximal)) != n:
                raise GeometryError("Qhull cells require repair")
            if not empty.all():
                maximal = _repair_cells(points,maximal,rank,index,exact,centers,radii,valid,bad)
                tree.metadata["triangulation_repaired"] = True
        except (QhullError, GeometryError):
            try:
                maximal = _enumerate(points,rank,index,exact)
            except GeometryError:
                # Only a path that previously RAISED can reach this repair.
                # Every previously successful construction is unchanged.
                if maximal is None or not bad.any():
                    raise
                maximal = _repair_cells(points,maximal,rank,index,exact,
                    centers,radii,valid,bad,ill_conditioned_stars=True)
            tree.metadata["triangulation_repaired"] = True
    # Compact dimension arrays replace a temporary full SimplexTree. Cofaces
    # are still processed before returning only vertices/edges for L0.
    layers = {rank:np.unique(maximal,axis=0)}
    maps = {}
    total = len(layers[rank])
    for degree in range(rank,0,-1):
        if total > max_simplices:
            raise ComplexityLimitError("Full alpha closure exceeds max_simplices")
        cells = layers[degree]
        faces = np.concatenate([np.delete(cells,j,axis=1) for j in range(degree+1)])
        layers[degree-1],inverse = np.unique(faces,axis=0,return_inverse=True)
        maps[degree] = inverse
        total += len(layers[degree-1])
    if total > max_simplices:
        raise ComplexityLimitError("Full alpha closure exceeds max_simplices")
    if len(layers[0]) != n:
        raise GeometryError("Alpha construction omitted input vertices")
    tree.metadata["full_simplex_count"] = total
    births = {degree:np.full(len(cells),np.inf) for degree,cells in layers.items()}
    for degree in range(rank,0,-1):
        cells = layers[degree]
        centers,radii,valid,bad = _spheres(points,cells,exact)
        empty = _empty(points,cells,centers,radii,valid,bad,index,exact)
        if not valid.all():
            raise GeometryError("Affinely dependent simplex in Delaunay closure")
        values = np.asarray(radii,dtype=float)
        births[degree] = np.minimum(births[degree],np.where(empty,values,np.inf))
        if not np.isfinite(births[degree]).all():
            raise GeometryError("Nonempty maximal circumsphere or overflowing squared alpha radius")
        np.minimum.at(births[degree-1],maps[degree],np.tile(births[degree],degree+1))
    births[0][:] = 0
    for degree in range(rank+1):
        if max_dimension is not None and degree > max_dimension:
            break
        for cell,value in zip(layers[degree],births[degree]):
            if value <= maximum:
                tree._store(tuple(int(labels[i]) for i in cell),float(value))
    tree.validate()
    return tree
