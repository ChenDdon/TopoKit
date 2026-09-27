# Native alpha hull-sliver regression fixture

`alpha_near_planar_hull_73.csv` contains 73 explicit ligand hydrogen coordinates
from the null/H channel of PDBbind v2020R1 complex 1tps. The source ligand file
is `1tps_ligand.mol2`; extraction uses the fixed molecular reader without
coordinate perturbation. These coordinates are a geometry regression fixture,
not affinity labels or a benchmark dataset. Source-data terms are distinct
from the toolkit code license.

Rows 66, 67, 69 and 70 (zero-based) form a nearly planar hull quadrilateral.
Qhull proposes a sliver tetrahedron whose exact circumsphere contains all
69 other points. Its squared radius is approximately 5.002269e22. Radius-ball
repair of this candidate requires more than 300000 candidates, exceeding the
bounded repair budget. The local facet-star fallback handles this case only
after the other repair paths are exhausted. All candidates must pass empty-ball
checks against the complete input; no coordinate perturbation is permitted.

Independent GUDHI exact geometry has 1239 simplices at squared cutoff 24.01.
The optional oracle test also compares the full, untruncated native complex
against GUDHI. Higher cofaces remain present before any skeleton extraction.

The CSV SHA-256 digest is
`0648db773f30190eaeac370fd4427649c79956dfcd0e0d3ff036fd223f281092`.
