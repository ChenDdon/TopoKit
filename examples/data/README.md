# Example data inventory and provenance

These files are compact inputs for the reader and workflow examples.  They are
kept outside the installed `topokit` package and are not reference-quality
chemical databases.  The toolkit's MIT license covers the code, not the source
structures.  Consult the cited source or original provider before
redistributing a structure.

## Inventory

| File | Reader role | Provenance recorded in this workspace |
| --- | --- | --- |
| `data_6PT3_ligand.pdb` | PDB ligand fixture | The filename identifies PDB entry 6PT3.  It is coordinate-identical to the ligand in `data_6PT3_ligand.mol2`; the extraction/conversion history and redistribution terms were not retained. |
| `data_6PT3_ligand.mol2` | MOL2 ligand fixture | The filename identifies PDB entry 6PT3.  The file contains Tripos atom types and Gasteiger charges; its extraction/conversion history and redistribution terms were not retained. |
| `data_6PT3_receptor.pdb` | Additional PDB receptor fixture | The filename identifies PDB entry 6PT3.  The file contains only coordinate records, so the original download/conversion history and redistribution terms cannot be reconstructed from it. |
| `data_protein.pdbqt` | PDBQT receptor fixture | The embedded `REMARK` names a prepared 6PT3 receptor.  This is a docking-preparation derivative; the preparation software, settings, and redistribution terms were not retained. |
| `data_molecule.sdf` | SDF/V3000 fixture | The embedded record name is `train__NCGC00013037-01`, with an RDKit 3D header.  The original collection and redistribution terms were not retained. |
| `data_molecule.mol` | MOL/V2000 fixture | Exact renamed copy of `D-Glucose_Conformer3D_CID_5793.mol` in the integration source workspace; source: [PubChem CID 5793](https://pubchem.ncbi.nlm.nih.gov/compound/5793). |
| `data_material.cif` | CIF fractional-coordinate fixture | Exact renamed copy of `C8H27B10BrNiP2.cif` in the integration source workspace.  The file identifies COD entry 4066832 and DOI [10.1021/om100669x](https://doi.org/10.1021/om100669x), and carries the COD public-domain notice. |
| `AQUCOG_clean.cif` | CIF/MOF fixture | Byte-identical to the AQUCOG member of the public CoRE MOF 2019 v1.1.4 all-solvent-removed archive: 162 P1 atom sites (Ni18 C72 H18 O54), distributed under CC BY 4.0. See the attribution and derivative note below. |
| `data_material_from_cif.json` | Native JSON PointCloud fixture | Derived from `data_material.cif`: 29 Cartesian coordinate rows with the original CIF site IDs, canonical element labels, and explicit Pauling-electronegativity weights. |
| `data_cluster.xyz` | XYZ comparison fixture | Exact renamed copy of `B12N12_cage.xyz` in the integration source workspace; source: [JACS article](https://doi.org/10.1021/ja410088y). |
| `point_cloud_24.csv` | Small numerical CSV fixture | First 12 source point IDs from the B12N12 coordinates, with assigned uniform weights; see `point_cloud_24.json`. |
| `point_cloud_24.json` | Sidecar provenance | Records the source reference, selection rule, units, and weight semantics for `point_cloud_24.csv`. It intentionally has no top-level `coordinates` (or `points` alias), so the native JSON point-cloud reader rejects it; read the paired CSV for coordinates. |

PDB entry 6PT3 can be consulted at the
[RCSB Protein Data Bank](https://www.rcsb.org/structure/6PT3).  The identifier
alone does not establish how the four local 6PT3-derived files were prepared.

### AQUCOG provenance

`AQUCOG_clean.cif` is byte-identical to
`structure_10143/AQUCOG_clean.cif` in the public CoRE MOF 2019 v1.1.4
all-solvent-removed archive on
[Zenodo (DOI 10.5281/zenodo.7691378)](https://doi.org/10.5281/zenodo.7691378).
That dataset is licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); cite Chung et al.,
*J. Chem. Eng. Data* **64**, 5985–5998 (2019), DOI
[10.1021/acs.jced.9b00835](https://doi.org/10.1021/acs.jced.9b00835).
The originating experiment is CSD refcode AQUCOG from Wood et al.,
*Chem. Commun.* **52**, 10048 (2016), DOI
[10.1039/C6CC04808K](https://doi.org/10.1039/C6CC04808K).

This fixture is the computation-ready CoRE derivative, not a raw CCDC/WebCSD
export: it is symmetry-expanded to P1 and omits the neon guest and solvent from
the deposited structure. Its SHA-256 digest is
`8afe10ec5002a0154fb96e5f067f5d96c7789ad42676bda8a94cf7712e2b725b`.

## Reproducibility checksums

The SHA-256 digests below describe the fixtures as used by the 2026-09-11
reader validation.

```text
8afe10ec5002a0154fb96e5f067f5d96c7789ad42676bda8a94cf7712e2b725b  AQUCOG_clean.cif
c101bf3471a1fe49108b8c1303c9074f0cc98b2d1fea0099e0804a4ad9800fa6  data_6PT3_ligand.mol2
e68b2c4504c9d3a523c8436552b15d1a4bc0c619dd74e6ec4c281062f2ffe949  data_6PT3_ligand.pdb
e8c802c9d5b11f664a59e55781b18af74cd784feb9df07149d0502a0673c3c5c  data_6PT3_receptor.pdb
20854100fe7561e0bdb09626a15ac17fb949b8801305b9db12ad0a10ce21ebb4  data_cluster.xyz
f2a0fa280826322ce6ada7de55502176a9a2a62d69b63655017f0c1f8469e5cb  data_material.cif
c6c7cf43c2c1e84964e29b988b2b7f99f13b9b0f3c4e01c78a569501afcac682  data_material_from_cif.json
c51cc9dc5d21a0b3b7b3723b066d860c270036269ad7bc936b041bd06d60d570  data_molecule.mol
2d08c4177dae6ff6a48b13a6534ff18eb8d8ae78a28f12bd44ce22d7b6c531cb  data_molecule.sdf
fbe4c6dbab7a71d2bc548df6a8958ba37ca4b8b96c362b53294152adbaa016f0  data_protein.pdbqt
292005cb2dbd4096bf2b3b26e0068d80e7166e66650581651bb27c9967b64a6b  point_cloud_24.csv
986147cad3309d1dc3c045181fa56ccbdbb2539dd08405c942b25ff2fa4b94df  point_cloud_24.json
```

## Scope

The reader tests use these files to verify parsing, row alignment, coordinate
conversion, and metadata retention.  They do not validate chemical identity,
bond perception, docking quality, crystal symmetry expansion, or predictive
performance.  Files with incomplete provenance should remain local test/demo
fixtures unless their upstream source and redistribution rights are confirmed.
