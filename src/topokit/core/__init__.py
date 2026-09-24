"""Layer 3: analyze well-defined objects. No reading or geometric construction."""
from importlib import import_module
from ..results import Topology
from .._validation import dimension as _dimension

_ROUTES = ("simplicial", "hyperdigraph", "interaction")


def __getattr__(name):
    """Resolve public route modules without eagerly importing every core."""
    if name in _ROUTES:
        module = import_module(f"topokit.core.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_ROUTES))


def _module(obj):
    if isinstance(obj, Topology):
        kind = obj.kind
    else:
        namespace = type(obj).__module__
        kind = next((name for name in _ROUTES
                     if namespace.startswith(f"topokit.core._{name}.")), None)
    if kind not in _ROUTES:
        raise TypeError("core accepts a defined topology object; construct it in topokit.builders first")
    return import_module(f"topokit.core.{kind}")


def homology(obj, **options):
    return _module(obj).homology(obj, **options)


def persistence(obj, **options):
    return _module(obj).persistence(obj, **options)


def laplacian(obj, dimension=0, **options):
    return _module(obj).laplacian(obj, dimension=dimension, **options)


def persistent_laplacian(obj, dimension=0, **options):
    return _module(obj).persistent_laplacian(obj, dimension=dimension, **options)


def laplacians(obj, max_dimension=2, **options):
    """Ordinary L0..Lq of one supplied object/snapshot, keyed by degree."""
    maximum = _dimension(max_dimension)
    return {q: laplacian(obj, dimension=q, **options) for q in range(maximum + 1)}


def persistent_laplacians(obj, max_dimension=2, *, start, end, **options):
    """Genuine two-stage persistent L0..Lq, keyed by degree."""
    maximum = _dimension(max_dimension)
    return {q: persistent_laplacian(obj, dimension=q, start=start, end=end, **options)
            for q in range(maximum + 1)}


# Route modules stay lazy attributes; wildcard imports retain the function-only API.
__all__ = ["homology", "persistence", "laplacian", "persistent_laplacian",
           "laplacians", "persistent_laplacians"]
