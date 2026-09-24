"""Layer 1: extensible file readers preserving data and scientific metadata."""
from .formats import read_csv, read_json, read_xyz
from .molecular import read_cif, read_mol, read_mol2, read_pdb, read_pdbqt, read_sdf
from .registry import Reader, ReaderRegistry
from .atomic_properties import PAULING_ELECTRONEGATIVITY, PAULING_SOURCE, assign_element_weights

registry = ReaderRegistry()
registry.register("csv", read_csv, extensions=("csv",))
registry.register("xyz", read_xyz, extensions=("xyz",))
registry.register("json", read_json, extensions=("json",))
registry.register("pdb", read_pdb, extensions=("pdb",))
registry.register("mol2", read_mol2, extensions=("mol2",))
registry.register("sdf", read_sdf, extensions=("sdf",))
registry.register("mol", read_mol, extensions=("mol",))
registry.register("pdbqt", read_pdbqt, extensions=("pdbqt",))
registry.register("cif", read_cif, extensions=("cif", "mmcif"))
read = registry.read
register_reader = registry.register
available_readers = registry.available

__all__ = ["Reader", "ReaderRegistry", "read", "read_csv", "read_json", "read_xyz",
           "read_pdb", "read_mol2", "read_sdf", "read_mol", "read_pdbqt", "read_cif",
           "register_reader", "available_readers", "assign_element_weights",
           "PAULING_ELECTRONEGATIVITY", "PAULING_SOURCE"]
