"""Independent provenance snapshots for derived numerical representations."""
from copy import deepcopy


def source_semantics(result, *, field):
    source = deepcopy(result.metadata)
    return {
        "source_metadata": source,
        **{key: deepcopy(source.get(key)) for key in
           ("route", "definition_id", "scale_units", "coordinate_units",
            "filtration_start", "filtration_end", "complex_type", "object_type",
            "construction", "connection_support", "cutoff_distance", "cutoff_policy",
            "graph_expansion", "filtration_parameters", "coupling", "progression",
            "filtration_a", "filtration_b", "factor_filtration_ranges", "factor_scale_units")},
        "coefficient_field": field,
        "field": field,
    }
