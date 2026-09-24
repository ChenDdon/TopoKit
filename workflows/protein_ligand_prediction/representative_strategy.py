"""Repository compatibility entry point for the installed FS-AN workflow."""
import json
from topokit.workflows.protein_ligand_prediction import (
    compute, featurize, implementation_receipt, read_selected_atoms, schema,
)

if __name__ == "__main__":
    print(json.dumps(schema(), indent=2))
