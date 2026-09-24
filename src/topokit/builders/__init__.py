"""Layer 2: construct defined objects/filtrations, never compute ML features."""
from importlib import import_module
from ..results import Topology
from .connections import bonds_from_adjacency

_CUSTOM_BUILDERS = {}
_BUILTINS = ("simplicial", "hyperdigraph", "interaction")


def __getattr__(name):
    """Resolve public route modules without eagerly importing every builder."""
    if name in _BUILTINS:
        module = import_module(f"topokit.builders.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_BUILTINS))


def register_builder(name, builder, *, replace=False):
    """Register a recipe callable returning a Topology; no core registration implied."""
    if not isinstance(name, str) or not name.strip() or not callable(builder):
        raise ValueError("builder needs a nonempty name and callable")
    if not replace and (name in _BUILTINS or name in _CUSTOM_BUILDERS):
        raise ValueError("builder name exists; use replace=True explicitly")
    _CUSTOM_BUILDERS[name] = builder


def build(data, *, kind="simplicial", **options):
    """Apply the chosen construction to data; custom recipes may reuse any builder."""
    if kind in _CUSTOM_BUILDERS:
        builder = _CUSTOM_BUILDERS[kind]
    elif kind in _BUILTINS:
        builder = import_module(f"topokit.builders.{kind}").from_points
    else:
        raise ValueError(f"Unknown builder {kind!r}; register a builder first")
    result = builder(data, **options)
    if not isinstance(result, Topology):
        raise TypeError("builders must return a defined Topology object")
    return result


def from_points(points, *, kind="simplicial", **options):
    """Construct one of the point-cloud routes (an alias of explicit build)."""
    return build(points, kind=kind, **options)


# Route modules stay lazy attributes; wildcard imports retain the function-only API.
__all__ = ["build", "from_points", "register_builder", "bonds_from_adjacency"]
