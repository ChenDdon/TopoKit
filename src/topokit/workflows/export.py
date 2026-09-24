"""Result-bundle orchestration; preserves semantic metadata on every result."""
from dataclasses import replace


def compact_analysis(result):
    """Store shared construction provenance once instead of once per spectrum."""
    common_keys = set(result.topology.metadata)
    semantic_keys = {"filtration_start", "filtration_end", "scale_units", "route",
                     "definition_id", "coefficient_field", "max_dimension"}
    def slim(item):
        return replace(item, metadata={key: value for key, value in item.metadata.items()
                                       if key not in common_keys or key in semantic_keys})
    snapshots = {scale: {"homology": slim(snapshot["homology"]),
                         "laplacians": {q: slim(spectrum) for q, spectrum in snapshot["laplacians"].items()}}
                 for scale, snapshot in result.snapshots.items()}
    return {"topology": result.topology, "persistence": slim(result.persistence),
            "snapshots": snapshots, "environment": result.metadata}
