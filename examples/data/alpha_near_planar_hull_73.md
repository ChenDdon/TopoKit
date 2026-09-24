# Native alpha hull-sliver regression fixture

The 73 explicit ligand H coordinates in this CSV are the selected null/H
channel of the user's local PDBbind v2020R1 complex 1tps. They are an input
geometry fixture, not affinity labels or a new benchmark dataset. Source:
`datasets/protein_ligand_prediction/structures/1tps/1tps_ligand.mol2`.
Extraction uses the fixed molecular reader without coordinate perturbation.

Rows 66, 67, 69 and 70 (zero-based) form a nearly planar hull quadrilateral.
Qhull proposes a sliver tetrahedron whose exact circumsphere contains all
69 other points. Its radius squared is about 5.002269e22. The old radius-ball
repair consequently attempted more than 300000 candidates. The added local
facet-star repair is attempted only after all pre-existing paths have failed;
all candidates still face empty-ball checks against the complete input.

Independent GUDHI exact geometry has 1239 simplices at squared cutoff 24.01;
the full, untruncated native complex is also compared in the optional oracle
test. Higher cofaces remain present before any skeleton extraction.
