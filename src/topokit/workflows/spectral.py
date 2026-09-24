"""Compose spectral computations over defined filtrations, without rebuilding."""
import math
from numbers import Real
import numpy as np

from .. import core
from .._validation import dimension, validate_query
from ..exceptions import ResourceLimitError
from ..results import Topology


def _number(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError("observation scales must be finite real numbers")
    return float(value)


def _bounded(values, budget):
    for index, value in enumerate(values):
        if index >= budget:
            raise ResourceLimitError("too many spectral observations; select scales or explicitly raise max_snapshots")
        yield value


def critical_scales(topology):
    """Finite event grades of the already constructed object in its native units."""
    if not isinstance(topology, Topology):
        raise TypeError("critical_scales expects a built Topology")
    native = topology.native
    if topology.kind == "simplicial":
        values = [birth for _, birth in native.get_filtration()]
    elif topology.kind == "hyperdigraph":
        values = list(native.thresholds()) if hasattr(native, "thresholds") else []
    elif topology.kind == "interaction":
        values = [birth for layer in native.degrees for birth in layer.births]
    else:
        raise ValueError("unsupported topology kind")
    beginning = topology.metadata.get("filtration_start", 0.)
    ending = topology.metadata.get("filtration_end", math.inf)
    values.append(beginning)
    return tuple(sorted({float(value) for value in values
                         if math.isfinite(value) and beginning <= value <= ending}))


def laplacian_series(topology, *, max_dimension=2, scales=None,
                     mode="ordinary", scale_pairs=None, max_snapshots=256, **options):
    """Return scale -> degree -> SpectrumResult (ragged eigenvalue lengths).

    Default ordinary mode evaluates explicit increasing ``scales`` or all
    finite critical grades when omitted. Genuine persistent mode requires an
    explicit list of ``(start, end)`` pairs; it never creates a quadratic grid.
    ``return_eigenvectors=False`` and ``return_matrix=False`` remain the core
    defaults. This workflow never changes the object's barcode/filtration.
    Ordinary hyperdigraph L0 uses an incremental edge sweep when singleton
    faces precede their edges and the full sweep buffer fits the resource
    limits. Other cases, higher degrees, explicit reference backends and
    genuine persistent pairs retain the general core calculations.
    For independent interaction thresholds use the paired helpers in
    ``workflows.interaction`` or pass a builder-regraded one-parameter object.
    """
    if not isinstance(topology, Topology):
        raise TypeError("laplacian_series expects a built Topology")
    maximum_dimension = dimension(max_dimension)
    snapshot_budget = dimension(max_snapshots, "max_snapshots")
    if not snapshot_budget:
        raise ValueError("max_snapshots must be positive")
    # An observation is one ordinary scale or one persistent source/target pair.
    if mode == "ordinary":
        if scale_pairs is not None:
            raise ValueError("scale_pairs requires mode='persistent'")
        requested_observations = (critical_scales(topology) if scales is None
                     else tuple(_number(v) for v in _bounded(scales, snapshot_budget)))
        if any(target_scale <= source_scale for source_scale, target_scale in zip(requested_observations, requested_observations[1:])):
            raise ValueError("scales must be strictly increasing")
        for value in requested_observations:
            validate_query(topology.metadata, value)
    elif mode == "persistent":
        if scales is not None or scale_pairs is None:
            raise ValueError("persistent mode requires scale_pairs, not scales")
        validated_pairs = []
        for scale_pair in _bounded(scale_pairs, snapshot_budget):
            try:
                source_scale, target_scale = scale_pair
            except (TypeError, ValueError) as error:
                raise ValueError("each persistent pair must contain (start, end)") from error
            source_scale, target_scale = _number(source_scale), _number(target_scale)
            if source_scale > target_scale:
                raise ValueError("persistent pair start must not exceed end")
            validate_query(topology.metadata, source_scale)
            validate_query(topology.metadata, target_scale)
            validated_pairs.append((source_scale, target_scale))
        requested_observations = tuple(validated_pairs)
        if len(set(requested_observations)) != len(requested_observations):
            raise ValueError("persistent pairs must be distinct")
    else:
        raise ValueError("mode must be 'ordinary' or 'persistent'")
    if len(requested_observations) > snapshot_budget:
        raise ResourceLimitError("too many spectral observations; select scales or explicitly raise max_snapshots")
    if mode == "ordinary":
        sweep = None
        if topology.kind == "hyperdigraph" and options.get("backend", "auto") == "auto":
            from ..core.hyperdigraph import L0Sweep
            if L0Sweep.supports(topology) and requested_observations:
                try:
                    sweep = L0Sweep(topology,
                        max_dense_entries=options.get("max_dense_entries", 4_000_000),
                        max_sparse_entries=options.get("max_sparse_entries", 2_000_000))
                except ResourceLimitError:
                    # The full future-vertex buffer can exceed a budget even
                    # when individual requested snapshots still fit it.
                    pass
        if sweep is not None:
            l0_options = {key: value for key, value in options.items()
                          if key not in {"backend", "max_dense_entries", "max_sparse_entries"}}
            return {scale: {0: sweep.laplacian(scale, **l0_options), **{
                q: core.laplacian(topology, dimension=q, scale=scale, **options)
                for q in range(1, maximum_dimension + 1)}}
                for scale in requested_observations}
        return {scale: core.laplacians(topology, max_dimension=maximum_dimension, scale=scale, **options)
                for scale in requested_observations}
    return {scale_pair: core.persistent_laplacians(topology, max_dimension=maximum_dimension,
                                           start=scale_pair[0], end=scale_pair[1], **options)
            for scale_pair in requested_observations}
