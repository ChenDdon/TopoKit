# Using the supplied scientific workflows

For a user's protein–ligand feature request, first distinguish the input and goal:

- **Bound/docked 3D complex → topology features:** follow
  [topokit-protein-ligand](skills/topokit-protein-ligand/SKILL.md). Use the
  installed FS-AN recipe and the manifest exporter. No pretrained model is needed.
- **Protein sequences + ligand SMILES → pretrained embeddings:** follow
  [topokit-sequence](skills/topokit-sequence/SKILL.md). Explicitly select the
  ESM-2 + CPZ profile and verify the supplied local checkpoint identities.
- **Affinity prediction:** first require completed compatible features and a
  trusted trained bundle. Follow the relevant prediction guide; extraction does
  not implicitly authorize fitting or selecting a model.

Run examples from the directory containing `pyproject.toml`, with that copy of
TopoKit installed in the active environment. `topokit` and `topokit_dev` share
the same Python import name: use separate environments, or deliberately install
the intended copy before running. Check `topokit.__file__` when diagnosing a
version mismatch. Repository scripts and examples are not wheel contents.

Preserve the user's inputs and scientific question. Do not silently substitute
the topology and sequence modalities, alter crop/radii/channels, change the
pretrained encoder, impute failed features, or choose protein chains or ligand
conformers on the user's behalf. Explain an ambiguity that affects the recipe
and obtain the missing input; ordinary valid manifest runs need no extra approval.

Use a new output directory and report feature dimensions, row IDs, recipe,
warnings and failures. Inspect the completion receipt, not just whether an NPY
exists. User-supplied datasets and checkpoints retain their own terms; copying
examples locally does not establish permission to publish them.

The atom-deletion and H0 study folders reproduce fixed historical cohorts.
They are not the default controllers for arbitrary new user datasets. Maintain
scientific algorithms in `src/topokit`, keeping example assets and execution
receipts outside the installed library, as required by the root maintenance rules.
