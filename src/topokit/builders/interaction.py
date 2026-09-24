"""Construct two-factor interaction objects with explicit shared identity.

Point-cloud factors are built independently. Their scalar filtrations are
coupled by maximum birth, never by copying grades between matching vertices.
The private quotient-tensor implementation remains distinct from simplicial
topology; an interaction cell is an ordered pair of simplices.
"""
from collections.abc import Mapping, Set
from bisect import bisect_left
from copy import deepcopy
from dataclasses import asdict
import math
import numbers

from ..data import as_point_cloud
from ..results import Topology
from ..exceptions import ResourceLimitError
from .._validation import dimension as _dimension, start_scale
from ..filtrations import resolve_range
from ..core import _interaction as _native


DEFINITION_ID = _native.DEFINITION_ID


def _end_scale(value, start):
    if isinstance(value, bool):
        raise ValueError("max_scale must be a number no smaller than filtration_start")
    end = float(value)
    if math.isnan(end) or end < start:
        raise ValueError("max_scale must be no smaller than filtration_start")
    return end


def _limit(value, name):
    value = _dimension(value, name)
    if value == 0:
        raise ValueError(f"{name} must be positive")
    return value


def _point_id(value):
    if isinstance(value, bool) or not isinstance(value, (str, numbers.Integral)):
        raise ValueError("overlap IDs must be actual string or integer point IDs")
    return int(value) if isinstance(value, numbers.Integral) else value


def _overlap(a, b, pairs, self_mode):
    if self_mode:
        if pairs is not None:
            raise ValueError("one-cloud self-interaction uses all IDs; do not supply overlap_pairs")
        pairs = tuple((label, label) for label in a.ids)
    else:
        if pairs is None:
            raise ValueError("two clouds require explicit overlap_pairs; overlap is never inferred")
        if isinstance(pairs, (str, bytes, Mapping, Set)):
            raise ValueError("overlap_pairs must be an ordered iterable of paired point IDs")
        checked_pairs = []
        left_ids, right_ids = set(), set()
        for pair in pairs:
            if isinstance(pair, (str, bytes, Mapping, Set)):
                raise ValueError("each overlap pair must contain two actual point IDs")
            try:
                left_id, right_id = pair
            except (TypeError, ValueError) as error:
                raise ValueError("each overlap pair must contain exactly two point IDs") from error
            left_id, right_id = _point_id(left_id), _point_id(right_id)
            if left_id not in a.index or right_id not in b.index:
                raise ValueError(f"unknown overlap point ID in {(left_id, right_id)!r}")
            if left_id in left_ids or right_id in right_ids:
                raise ValueError("overlap_pairs must be one-to-one with no repeated IDs")
            left_ids.add(left_id)
            right_ids.add(right_id)
            checked_pairs.append((left_id, right_id))
        pairs = tuple(checked_pairs)
    first_id_map = {label: index for index, label in enumerate(a.ids)}
    second_id_map = {right_id: first_id_map[left_id] for left_id, right_id in pairs}
    next_id = len(first_id_map)
    for label in b.ids:
        if label not in second_id_map:
            second_id_map[label] = next_id
            next_id += 1
    return pairs, (first_id_map, second_id_map)


def _factor_settings(options, dimensions, homology_dimension):
    if options is None:
        settings = ({}, {})
    elif isinstance(options, Mapping):
        settings = (dict(options), dict(options))
    else:
        settings = tuple(options)
        if len(settings) != 2 or any(not isinstance(item, Mapping) for item in settings):
            raise ValueError("factor_options must be one mapping or exactly two mappings")
        settings = tuple(dict(item) for item in settings)
    if dimensions is None:
        caps = (None, None)
    elif isinstance(dimensions, numbers.Integral) and not isinstance(dimensions, bool):
        caps = (dimensions, dimensions)
    else:
        try:
            caps = tuple(dimensions)
        except TypeError as error:
            raise ValueError("factor_max_dimensions must contain two nonnegative integers") from error
        if len(caps) != 2:
            raise ValueError("factor_max_dimensions must contain exactly two entries")
    for index, options in enumerate(settings):
        specified = options.pop("max_dimension", None)
        explicit_cap = options.pop("max_simplex_dimension", None)
        if specified is not None and explicit_cap is not None:
            raise ValueError("set factor maximum dimensions in only one place")
        specified = specified if explicit_cap is None else explicit_cap
        if caps[index] is not None and specified is not None:
            raise ValueError("set factor maximum dimensions in only one place")
        cap = caps[index] if caps[index] is not None else specified
        options["max_dimension"] = (homology_dimension + 1 if cap is None
                                    else _dimension(cap, "factor_max_dimension"))
    return settings


def _higher_overlap(records, factors, codecs):
    """Validate annotations; never select or insert quotient-tensor cells."""
    if records is None:
        return ()
    if isinstance(records, (str, bytes, Mapping, Set)):
        raise ValueError("overlap_simplices must be an ordered iterable of paired simplices")
    annotations, seen = [], set()
    for record in records:
        if isinstance(record, (str, bytes, Mapping, Set)):
            raise ValueError("each overlap simplex record must contain two simplex tuples")
        try:
            left, right = record
            if isinstance(left, (str, bytes, Mapping)) or isinstance(right, (str, bytes, Mapping)):
                raise ValueError
            left, right = tuple(map(_point_id, left)), tuple(map(_point_id, right))
        except (TypeError, ValueError) as error:
            raise ValueError("each overlap simplex record must contain two simplex tuples of actual IDs") from error
        if len(left) < 2 or len(left) != len(right):
            raise ValueError("higher overlap simplices must have equal size and at least two vertices")
        if len(set(left)) != len(left) or len(set(right)) != len(right):
            raise ValueError("an overlap simplex cannot repeat a vertex")
        try:
            global_left = tuple(sorted(codecs[0][v] for v in left))
            global_right = tuple(sorted(codecs[1][v] for v in right))
        except KeyError as error:
            raise ValueError("an overlap simplex contains an unknown point ID") from error
        if global_left != global_right:
            raise ValueError("overlap simplices must agree under the supplied vertex correspondence")
        if any(factor.simplex_id(simplex) is None
               for factor, simplex in zip(factors, (global_left, global_right))):
            raise ValueError("an overlap simplex is absent from an independently constructed factor; annotations never insert cells")
        if global_left in seen:
            raise ValueError("overlap_simplices contains a duplicate unordered simplex pair")
        seen.add(global_left)
        # Retain supplied stable IDs and tuple orders as annotations; order has
        # no mathematical orientation meaning and never reorders input arrays.
        annotations.append((left, right))
    return tuple(annotations)


def _numeric_sequence(values, name, *, strictly_increasing=False):
    if isinstance(values, (str, bytes, Mapping, Set)):
        raise ValueError(f"{name} must be a nonempty sequence of finite real numbers")
    try:
        values = tuple(values)
    except TypeError as error:
        raise ValueError(f"{name} must be a nonempty sequence of finite real numbers") from error
    if not values or any(isinstance(v, bool) or not isinstance(v, numbers.Real)
                         or not math.isfinite(float(v)) for v in values):
        raise ValueError(f"{name} must be a nonempty sequence of finite real numbers")
    values = tuple(map(float, values))
    if any(b <= a if strictly_increasing else b < a for a, b in zip(values, values[1:])):
        adjective = "strictly increasing" if strictly_increasing else "nondecreasing"
        raise ValueError(f"{name} must be {adjective}")
    return values


def _paired_schedules(filtration_a, filtration_b, progression):
    first = _numeric_sequence(filtration_a, "filtration_a")
    second = first if filtration_b is None else _numeric_sequence(filtration_b, "filtration_b")
    if len(first) != len(second):
        raise ValueError("filtration_a and filtration_b must have equal lengths")
    stages = (tuple(map(float, range(len(first)))) if progression is None else
              _numeric_sequence(progression, "progression", strictly_increasing=True))
    if len(stages) != len(first):
        raise ValueError("progression must have the same length as the paired schedules")
    return first, second, stages


def _stored_factors(factors):
    return tuple(tuple(zip(factor.simplices, factor.births)) for factor in factors)


def _factor_from_items(items):
    builder = _native.SimplicialComplexBuilder()
    for simplex, birth in items:
        builder.insert(simplex, birth, with_faces=False)
    return builder.freeze()


def _regrade_factors(factors, metadata, first, second, stages, *, explicit_progression,
                    reused_schedule):
    """Pull two independent filtrations back along one monotone sampled path."""
    ranges = metadata["factor_filtration_ranges"]
    for name, values, (low, high) in zip(("filtration_a", "filtration_b"), (first, second), ranges):
        if values[0] < low or values[-1] > high:
            raise ValueError(f"{name} lies outside its constructed factor range [{low}, {high}]; extrapolation is not allowed")
    regraded, retained_counts = [], []
    for factor, values in zip(factors, (first, second)):
        items = []
        for simplex, birth in zip(factor.simplices, factor.births):
            index = bisect_left(values, birth)
            if index < len(values):
                items.append((simplex, stages[index]))
        regraded.append(_factor_from_items(items))
        retained_counts.append(len(items))
    metadata = deepcopy(metadata)
    metadata.update({
        "source_factor_filtrations": _stored_factors(factors),
        "factor_scale_units": metadata.get("factor_scale_units", (metadata.get("scale_units", "user_scalar"),) * 2),
        "factor_thresholds": tuple(zip(first, second)),
        "filtration_a": first, "filtration_b": second,
        "filtration_b_reused": reused_schedule, "progression": stages,
        "scale_units": "user_progression" if explicit_progression else "progression_stage",
        "filtration_start": stages[0], "filtration_end": stages[-1],
        "filtration_range": (stages[0], stages[-1]),
        "filtration_exactness": "sampled_one_parameter_path",
        "filtration_parameters": 1, "path_monotonicity": "componentwise_nondecreasing",
        "start_policy": "first_observed_stage_not_intrinsic_birth",
        "initial_stage_semantics": "classes present initially may predate the first threshold pair",
        "right_censored": True,
        "terminal_interval_semantics": "infinite deaths mean survival beyond the last sampled pair, not proven essentiality",
        "retained_factor_simplex_counts": tuple(retained_counts),
        "coupling": "maximum_first_observed_stage",
    })
    return tuple(regraded), metadata


def _cell_estimate(factors, highest):
    """Witness-count upper bound, before allocating any interaction cells.

    Each surviving simplex pair has a shared vertex. Counting its vertex
    witnesses can overcount pairs, but cannot undercount the required cells.
    """
    first_factor, second_factor = factors
    shared_vertices = {v for s in first_factor.simplices if len(s) == 1 for v in s}
    shared_vertices.intersection_update(v for s in second_factor.simplices if len(s) == 1 for v in s)
    counts = []
    for degree in range(highest + 2):
        total = 0
        for left_degree in range(degree + 1):
            right_degree = degree - left_degree
            total += sum(len(first_factor.postings(left_degree, v)) * len(second_factor.postings(right_degree, v))
                         for v in shared_vertices)
        counts.append(total)
    return tuple(counts)


def _assemble(factors, metadata, cloud, highest, max_cells, validate):
    max_cells = _limit(max_cells, "max_cells")
    estimate = _cell_estimate(factors, highest)
    if sum(estimate) > max_cells:
        raise ResourceLimitError(
            f"Interaction construction upper bound is {sum(estimate):,} cells through "
            f"degree {highest + 1}, above max_cells={max_cells:,}; reduce the factors "
            "or explicitly raise the budget. No interaction cells were allocated."
        )
    chain = _native.build_interaction_chain_complex(
        factors, max_homology_dimension=highest, validate=validate)
    metadata = {**metadata, "definition_id": DEFINITION_ID, "factor_count": 2,
                "route": "interaction",
                "coupling": metadata.get("coupling", "maximum_factor_birth"), "filtration_parameters": 1,
                "max_dimension": highest, "max_cells": max_cells,
                "estimated_cells_by_degree_upper_bound": estimate,
                "construction_diagnostics": asdict(chain.diagnostics),
                "factor_max_dimensions": tuple(f.max_dimension for f in factors)}
    return Topology("interaction", chain, cloud=cloud, metadata=metadata)


def from_points(cloud_a, cloud_b=None, *, overlap_pairs=None, overlap_vertices=None,
                overlap_simplices=None, max_dimension=2,
                filtration_start=0.0, max_scale=math.inf, filtration_range=None,
                factor_max_dimensions=None, factor_options=None,
                filtration_a=None, filtration_b=None, progression=None,
                max_cells=1_000_000, max_simplices=1_000_000, validate=True):
    """Build two independently filtered point-cloud factors.

    One cloud creates two self-interaction factors sharing every ID. Two
    clouds require explicit ``(ID_in_a, ID_in_b)`` pairs; an explicitly empty
    mapping is valid and produces a zero interaction chain. Matching IDs do
    not assert coordinate equality, and unmatched IDs never collide.

    ``overlap_vertices`` is an alias for ``overlap_pairs``; provide only one.
    ``overlap_simplices`` contains pairs of unordered stable-ID tuples. It only
    validates/annotates simplices already present in both independent factors;
    it never inserts simplices or restricts interaction cells.

    ``factor_options`` is a shared mapping or two independent mappings for
    the simplicial builder (alpha by default, also rips/graph/flag). Construction
    settings, including bonds, cutoffs, and ranges, are applied independently.
    Alpha is unweighted. Point weights remain attributes. The legacy factor
    ``max_dimension`` option means maximum simplex dimension, not analysis degree.

    Without schedules, exact scalar maximum-birth coupling is preserved on
    the common factor domain; scalar and coordinate units must agree. Optional
    numeric ``filtration_a``/``filtration_b`` arrays specify a componentwise
    monotone sampled one-parameter path, NOT a two-parameter invariant. If b is
    None, its thresholds reuse a, not a's simplex births. Stage labels default
    to integers or use a strictly increasing ``progression`` array. Intervals
    are sampled/window-censored. ``max_cells`` bounds construction in advance.
    """
    from . import simplicial

    highest = _dimension(max_dimension)
    start, end = resolve_range(filtration_range, filtration_start=filtration_start, max_scale=max_scale)
    max_simplices = _limit(max_simplices, "max_simplices")
    if overlap_pairs is not None and overlap_vertices is not None:
        raise ValueError("provide only one of overlap_pairs and overlap_vertices")
    overlap_pairs = overlap_pairs if overlap_vertices is None else overlap_vertices
    schedules = None
    if filtration_a is not None:
        schedules = _paired_schedules(filtration_a, filtration_b, progression)
    elif filtration_b is not None or progression is not None:
        raise ValueError("filtration_b/progression require filtration_a")
    first_cloud = as_point_cloud(cloud_a)
    self_mode = cloud_b is None
    second_cloud = first_cloud if self_mode else as_point_cloud(cloud_b)
    if not len(first_cloud) or not len(second_cloud):
        raise ValueError("each interaction factor needs at least one point")
    pairs, id_maps = _overlap(first_cloud, second_cloud, overlap_pairs, self_mode)
    settings = _factor_settings(factor_options, factor_max_dimensions, highest)
    factors, raw_filtrations, applied_options, factor_topologies = [], [], [], []
    units = tuple(cloud.metadata.get("coordinate_units", cloud.metadata.get("coordinate_unit", "unspecified"))
                  for cloud in (first_cloud, second_cloud))
    if schedules is None and units[0] != units[1]:
        raise ValueError("interaction factors must have matching coordinate_units; convert and annotate inputs explicitly")
    for cloud, id_map, options in zip((first_cloud, second_cloud), id_maps, settings):
        options = dict(options)
        complex_type = options.pop("complex_type", "alpha")
        simplex_cap = options.pop("max_dimension")
        options.setdefault("max_simplices", max_simplices)
        if "filtration_range" not in options:
            options.setdefault("filtration_start", start)
            options.setdefault("max_scale", end)
        if options.get("duplicates", "error") != "error":
            raise ValueError("interaction factors preserve point identities; coordinate merging is not supported")
        factor_topology = simplicial.from_points(
            cloud, max_dimension=highest, max_simplex_dimension=simplex_cap,
            complex_type=complex_type, **options)
        factor_tree = factor_topology.native
        factor_builder = _native.SimplicialComplexBuilder()
        for simplex, birth in factor_tree.get_filtration():
            global_simplex = tuple(id_map[cloud.ids[row]] for row in simplex)
            factor_builder.insert(global_simplex, float(birth), with_faces=False)
        raw_factor = tuple((tuple(cloud.ids[row] for row in simplex), float(birth))
                           for simplex, birth in factor_topology.metadata["raw_filtration"])
        factors.append(factor_builder.freeze())
        raw_filtrations.append(raw_factor)
        factor_topologies.append(factor_topology)
        applied_options.append({"complex_type": complex_type, "max_dimension": simplex_cap, **options})
    factors = tuple(factors)
    ranges = tuple((factor.metadata["filtration_start"], factor.metadata["filtration_end"])
                   for factor in factor_topologies)
    scale_units = tuple(factor.metadata.get("scale_units", "user_scalar") for factor in factor_topologies)
    if schedules is None and scale_units[0] != scale_units[1]:
        raise ValueError("factor scalar units must agree; mixed scales require explicit paired filtration_a/filtration_b schedules")
    annotations = _higher_overlap(overlap_simplices, factors, id_maps)
    source_id_maps = tuple({global_id: point_id for point_id, global_id in id_map.items()} for id_map in id_maps)
    first_cloud, second_cloud = (factor.cloud for factor in factor_topologies)
    metadata = {
        "input_mode": "self" if self_mode else "paired", "overlap_pairs": pairs,
        "overlap_vertices": pairs, "overlap_simplices": annotations,
        "overlap_simplices_role": "validation_annotation_only",
        "interaction_admission_rule": "nonempty_shared_vertex_intersection",
        "overlap_count": len(pairs), "factor_point_ids": (first_cloud.ids, second_cloud.ids),
        "factor_vertex_id_maps": source_id_maps,
        "factor_id_to_global_vertex": tuple(dict(id_map) for id_map in id_maps),
        "factor_to_global": tuple(dict(id_map) for id_map in id_maps),
        "global_to_source": {v: tuple((i, mapping[v]) for i, mapping in enumerate(source_id_maps) if v in mapping)
                             for v in set(source_id_maps[0]).union(source_id_maps[1])},
        "factor_options": tuple(applied_options), "requested_factor_scale_ends": tuple(r[1] for r in ranges),
        "factor_construction_metadata": tuple(deepcopy(factor.metadata) for factor in factor_topologies),
        "factor_filtration_ranges": ranges, "factor_scale_units": scale_units,
        "source_factor_filtrations": _stored_factors(factors),
        "raw_factor_filtrations": tuple(raw_filtrations), "start_policy": "clamp_all_simplex_births",
        "scale_units": scale_units[0],
        "point_weights_used": False, "point_weights_role": "retained_attributes_only",
        "weights": (tuple(first_cloud.weights), tuple(second_cloud.weights)),
        "point_metadata": (dict(first_cloud.metadata), dict(second_cloud.metadata)),
        "coordinate_units": units[0] if units[0] == units[1] else "factor_specific",
        "factor_coordinate_units": units,
        "empty_overlap": not bool(pairs),
    }
    if schedules is not None:
        factors, metadata = _regrade_factors(
            factors, metadata, *schedules, explicit_progression=progression is not None,
            reused_schedule=filtration_b is None)
    else:
        common_start = max(low for low, _ in ranges)
        common_end = min(high for _, high in ranges)
        if common_end < common_start:
            raise ValueError("factor ranges have no common scalar domain; supply a paired schedule")
        factors = tuple(_factor_from_items((simplex, max(common_start, birth))
                         for simplex, birth in zip(factor.simplices, factor.births)
                         if birth <= common_end) for factor in factors)
        metadata.update({"filtration_start": common_start, "filtration_end": common_end,
                         "filtration_range": (common_start, common_end),
                         "filtration_exactness": "exact_common_scalar_domain"})
    return _assemble(factors, metadata, (first_cloud, second_cloud), highest, max_cells, validate)


def _explicit_items(value):
    metadata = {}
    if isinstance(value, Topology):
        if value.kind != "simplicial":
            raise ValueError("each factor must be an explicit simplicial complex")
        metadata, value = value.metadata, value.native
    if isinstance(value, _native.FilteredSimplicialComplex):
        items = zip(value.simplices, value.births)
    elif hasattr(value, "get_filtration"):
        items = value.get_filtration()
    elif isinstance(value, Mapping):
        items = value.items()
    else:
        raise TypeError("factor must be a simplicial Topology, filtered complex, or simplex-to-birth mapping")
    labels = metadata.get("vertex_id_map", {})
    normalized, local_ids = [], {}
    validator = _native.SimplicialComplexBuilder()
    for simplex, birth in items:
        if isinstance(birth, bool) or not isinstance(birth, numbers.Real):
            raise ValueError("factor births must be finite real numbers")
        simplex = tuple(_point_id(labels.get(v, v)) for v in simplex)
        for label in simplex:
            if label not in local_ids:
                local_ids[label] = len(local_ids)
        validator.insert(tuple(local_ids[label] for label in simplex), birth, with_faces=False)
        normalized.append((simplex, float(birth)))
    # Validate raw input before start-stage clamping: clamping must not hide
    # an originally missing face or an invalid face/coface birth ordering.
    validator.freeze()
    return tuple(normalized), metadata


def from_complexes(a, b, *, max_dimension=2, filtration_start=0.0,
                   max_scale=math.inf, max_cells=1_000_000, validate=True):
    """Use exactly two explicit filtered complexes with shared vertex labels.

    The two positional factors are never merged or required to be nested.
    A mapping input is ``{simplex_tuple: finite_birth}`` and must be face closed.
    """
    highest = _dimension(max_dimension)
    start = start_scale(filtration_start)
    end = _end_scale(max_scale, start)
    entries = tuple(_explicit_items(value) for value in (a, b))
    units = tuple(meta.get("scale_units") for _, meta in entries)
    if all(unit is not None for unit in units) and units[0] != units[1]:
        raise ValueError("factor scalar units must agree")
    ranges = tuple((max(start, meta.get("filtration_start", start)),
                    min(end, meta.get("filtration_end", end))) for _, meta in entries)
    if any(high < low for low, high in ranges):
        raise ValueError("factor filtration domains do not contain the requested range")
    common_start, common_end = max(low for low, _ in ranges), min(high for _, high in ranges)
    if common_end < common_start:
        raise ValueError("factor ranges have no common scalar domain")
    id_map = {}
    factors, raw_filtrations, source_id_maps = [], [], []
    for (items, _), (factor_start, factor_end) in zip(entries, ranges):
        builder = _native.SimplicialComplexBuilder()
        factor_map, retained_items = {}, []
        for simplex, birth in items:
            if not math.isfinite(birth):
                raise ValueError("factor simplex births must be finite")
            checked = tuple(_point_id(v) for v in simplex)
            if birth > factor_end:
                continue
            global_ids = []
            for label in checked:
                if label not in id_map:
                    id_map[label] = len(id_map)
                global_ids.append(id_map[label])
                factor_map[id_map[label]] = label
            builder.insert(global_ids, max(factor_start, birth), with_faces=False)
            retained_items.append((checked, birth))
        factors.append(builder.freeze())
        raw_filtrations.append(tuple(retained_items))
        source_id_maps.append(factor_map)
    shared_vertices = set(source_id_maps[0]).intersection(source_id_maps[1])
    metadata = {"input_mode": "explicit_complexes", "filtration_start": common_start,
                "filtration_end": common_end, "raw_factor_filtrations": tuple(raw_filtrations),
                "filtration_range": (common_start, common_end),
                "factor_filtration_ranges": ranges,
                "source_factor_filtrations": _stored_factors(factors),
                "factor_vertex_id_maps": tuple(source_id_maps),
                "factor_to_global": tuple({label: v for v, label in mapping.items()} for mapping in source_id_maps),
                "global_to_source": {v: tuple((i, mapping[v]) for i, mapping in enumerate(source_id_maps) if v in mapping)
                                     for v in set(source_id_maps[0]).union(source_id_maps[1])},
                "overlap_pairs": tuple((source_id_maps[0][v], source_id_maps[1][v]) for v in sorted(shared_vertices)),
                "overlap_count": len(shared_vertices), "empty_overlap": not bool(shared_vertices),
                "start_policy": "clamp_all_simplex_births",
                "scale_units": next((unit for unit in units if unit is not None), "user_scalar")}
    metadata["factor_scale_units"] = (metadata["scale_units"],) * 2
    metadata["filtration_exactness"] = "exact_common_scalar_domain"
    common_factors = tuple(_factor_from_items((simplex, max(common_start, birth))
                           for simplex, birth in zip(factor.simplices, factor.births)
                           if birth <= common_end) for factor in factors)
    return _assemble(common_factors, metadata, None, highest, max_cells, validate)


def regrade_pair_filtration(obj, filtration_a, filtration_b=None, *, progression=None,
                           max_dimension=None, max_cells=None, validate=True):
    """Return the pullback along an explicit monotone factor-threshold path.

    The equal-length numeric schedules are interpreted in each factor's own
    units. ``filtration_b=None`` reuses the first threshold schedule, never its
    simplex births. ``progression`` supplies strictly increasing scalar stage
    labels (otherwise 0, 1, ...). Each factor simplex first appears at the first
    sampled threshold at least its independently computed birth; interaction
    cells retain the unchanged quotient and maximum-stage coupling.

    This is sampled single-parameter persistence, not full multiparameter or
    original exact continuous persistence. Classes at the first pair may predate
    observation; infinite deaths are right-censored at the last pair. Requested
    thresholds must lie within each retained construction range. Regrading a
    regraded object uses the retained original factor filtrations, not the
    already quantized stages; no geometry is rerun or outside cells invented.
    """
    if not isinstance(obj, Topology) or obj.kind != "interaction":
        raise TypeError("expected an interaction Topology")
    chain = obj.native
    if not isinstance(chain, _native.FilteredInteractionChainComplex) or len(chain.factors) != 2:
        raise ValueError("paired filtration requires exactly two interaction factors")
    highest = chain.max_homology_dimension if max_dimension is None else _dimension(max_dimension)
    if highest > chain.max_homology_dimension:
        raise ValueError("requested dimension exceeds the constructed interaction chain")
    first_schedule, second_schedule, stages = _paired_schedules(filtration_a, filtration_b, progression)
    metadata = deepcopy(obj.metadata)
    metadata.setdefault("factor_filtration_ranges", (
        (metadata.get("filtration_start", 0.0), metadata.get("filtration_end", math.inf)),) * 2)
    stored_factors = metadata.get("source_factor_filtrations")
    factors = (tuple(_factor_from_items(items) for items in stored_factors)
               if stored_factors is not None else chain.factors)
    factors, metadata = _regrade_factors(
        factors, metadata, first_schedule, second_schedule, stages,
        explicit_progression=progression is not None, reused_schedule=filtration_b is None)
    cell_budget = obj.metadata.get("max_cells", 1_000_000) if max_cells is None else max_cells
    return _assemble(factors, metadata, obj.cloud, highest, cell_budget, validate)


def snapshot_at_pair(obj, scale_a, scale_b=None, *, max_dimension=None,
                     max_cells=None, validate=True):
    """Construct the exact static quotient at two explicit factor thresholds.

    Returned cells have scalar stage 0, not either factor's original units.
    Use core homology/Laplacian at stage 0. Metadata retains the threshold pair
    and construction ranges; this snapshot alone is not an intrinsic barcode.
    """
    result = regrade_pair_filtration(
        obj, (scale_a,), None if scale_b is None else (scale_b,),
        max_dimension=max_dimension, max_cells=max_cells, validate=validate)
    result.metadata.update({"observation_kind": "factor_pair_snapshot",
                            "factor_scale_pair": result.metadata["factor_thresholds"][0],
                            "static_snapshot_exact": True})
    return result


__all__ = ["from_points", "from_complexes", "regrade_pair_filtration", "snapshot_at_pair"]
