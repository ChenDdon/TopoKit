"""Executable dependency boundaries for the six-layer architecture.

These checks inspect nested and relative imports as well as cold-process
behavior. Defined-object constructors and boundary assembly belong to core;
geometric construction belongs to builders. Root convenience aliases are
deliberately permitted and are not implementation layers.
"""
import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "src" / "topokit"
LAYERS = ("readers", "builders", "core", "postprocessing", "visualization", "workflows")
UPPER = {"readers", "builders", "postprocessing", "visualization", "workflows"}
STALE = ("topokit._core", "topokit.simplicial", "topokit.hyperdigraph", "topokit.interaction")
ANALYSIS_NAMES = {
    "homology", "persistence", "persistent_homology", "laplacian", "persistent_laplacian",
    "compute_homology", "compute_persistence", "compute_laplacian", "compute_laplacian_filtration",
    "compute_persistent_laplacian", "compute_path_compatible_persistence",
    "compute_interaction_laplacian", "compute_persistent_interaction_laplacian",
    "InteractionLaplacianEngine", "solve_symmetric", "ordinary_operator", "pairwise_operator",
    "reference_operator", "histogram_features", "barcode_bin_counts", "PersistenceVectorizer", "summarize_spectrum",
    "summarize_spectra", "spectral_entropy", "spectral_variance", "spectral_moment",
    "spectral_moments", "spectral_energy", "laplacian_energy", "graph_entropy",
}


def _module(path):
    parts = path.relative_to(SOURCE).with_suffix("").parts
    return ".".join(("topokit", *(parts[:-1] if parts[-1] == "__init__" else parts)))


def _package(path):
    name = _module(path)
    return name if path.name == "__init__.py" else name.rpartition(".")[0]


def _matches(name, prefix):
    return name == prefix or name.startswith(prefix + ".")


def _imports(path, tree):
    """Yield every concrete dependency, including imports inside functions."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            target = "." * node.level + (node.module or "")
            target = importlib.util.resolve_name(target, _package(path)) if node.level else target
            yield node.lineno, target
            for alias in node.names:
                if alias.name != "*":
                    yield node.lineno, target + "." + alias.name


def _dynamic_imports(tree):
    """Recognize literal/prefix-qualified importlib calls, including aliases."""
    functions = {"__import__", "importlib.import_module"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "importlib":
            functions.update(alias.asname or alias.name for alias in node.names if alias.name == "import_module")
        elif isinstance(node, ast.Import):
            functions.update((alias.asname or "importlib") + ".import_module"
                             for alias in node.names if alias.name == "importlib")
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or ast.unparse(node.func) not in functions:
            continue
        arg = node.args[0] if node.args else next((kw.value for kw in node.keywords if kw.arg == "name"), None)
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            yield node.lineno, arg.value, False
        elif isinstance(arg, ast.JoinedStr) and arg.values and isinstance(arg.values[0], ast.Constant):
            yield node.lineno, arg.values[0].value, True
        else:
            yield node.lineno, None, True


def _sources(layer=None):
    return sorted((SOURCE / layer if layer else SOURCE).rglob("*.py"))


@pytest.mark.parametrize("layer", LAYERS)
def test_six_layers_are_real_source_packages(layer):
    directory = SOURCE / layer
    assert (directory / "__init__.py").is_file()
    assert any(path.name != "__init__.py" for path in directory.rglob("*.py"))


def test_package_contains_no_datasets_or_demonstration_implementations():
    # Ignore generated caches and empty historical directories during cleanup.
    forbidden = []
    for path in SOURCE.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(SOURCE)
        if (set(relative.parts) & {"datasets", "examples", "demos"}
                or path.name in {"demo.py", "dataset.py"}
                or path.suffix.lower() in {".xyz", ".mol2", ".pdb", ".csv", ".npy", ".npz"}):
            forbidden.append(str(relative))
    assert not forbidden, "Examples and scientific data must live outside src/: " + repr(forbidden)


def test_source_imports_use_canonical_paths_including_lazy_imports():
    violations = []
    for path in _sources():
        tree = ast.parse(path.read_text(), filename=str(path))
        dependencies = list(_imports(path, tree))
        dependencies.extend((line, name) for line, name, _ in _dynamic_imports(tree) if name)
        for line, name in dependencies:
            if any(_matches(name, old) for old in STALE):
                violations.append(f"{path.relative_to(SOURCE)}:{line}: {name}")
    assert not violations, "Stale source import paths:\n" + "\n".join(violations)


@pytest.mark.parametrize("layer", ("core", "builders", "readers", "postprocessing", "visualization"))
def test_layer_import_directions_include_nested_and_dynamic_imports(layer):
    forbidden = {
        "core": UPPER | {"api", "cli"},
        "builders": {"readers", "postprocessing", "visualization", "workflows", "api", "cli"},
        # Consumers use shared records, not the algorithm implementations.
        "readers": {"core", "builders", "postprocessing", "visualization", "workflows", "api", "cli"},
        "postprocessing": {"core", "builders", "readers", "visualization", "workflows", "api", "cli"},
        "visualization": {"core", "builders", "readers", "workflows", "api", "cli"},
    }[layer]
    violations = []
    for path in _sources(layer):
        tree = ast.parse(path.read_text(), filename=str(path))
        dependencies = list(_imports(path, tree))
        for line, prefix, interpolated in _dynamic_imports(tree):
            if prefix is None or (interpolated and not prefix.startswith(f"topokit.{layer}.")):
                violations.append(f"{path.relative_to(SOURCE)}:{line}: unbounded dynamic import {prefix!r}")
            elif prefix:
                dependencies.append((line, prefix))
        for line, name in dependencies:
            if any(_matches(name, f"topokit.{target}") for target in forbidden):
                violations.append(f"{path.relative_to(SOURCE)}:{line}: {name}")
            if layer == "core" and _matches(name, "scipy.spatial"):
                violations.append(f"{path.relative_to(SOURCE)}:{line}: geometric construction in core")
    assert not violations, "Layer dependencies violated:\n" + "\n".join(violations)


def test_builders_do_not_import_or_call_analysis_or_feature_entrypoints():
    violations = []
    for path in _sources("builders"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for line, name in _imports(path, tree):
            if name.rsplit(".", 1)[-1] in ANALYSIS_NAMES:
                violations.append(f"{path.name}:{line}: imports {name}")
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else (
                    node.func.attr if isinstance(node.func, ast.Attribute) else None)
                if name in ANALYSIS_NAMES:
                    violations.append(f"{path.name}:{node.lineno}: calls {name}")
    assert not violations, "Builders must return objects without computing results:\n" + "\n".join(violations)


def test_core_does_not_define_point_or_matrix_adapters():
    forbidden = {"from_points", "from_point_cloud", "from_adjacency_matrix", "from_distance_matrix",
                 "build_point_cloud_support", "build_complex", "alpha_complex", "rips_complex"}
    violations = []
    for path in _sources("core"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in forbidden:
                violations.append(f"{path.relative_to(SOURCE)}:{node.lineno}: {node.name}")
    assert not violations, "Construction entrypoints remain in math core:\n" + "\n".join(violations)


def _cold_process(body, blocked):
    """Import the real root package only after installing a rejecting finder."""
    program = """
import importlib.abc
import sys
class RejectLayers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == prefix or fullname.startswith(prefix + '.') for prefix in BLOCKED):
            raise AssertionError('Forbidden dependency loaded: ' + fullname)
sys.meta_path.insert(0, RejectLayers())
""".replace("BLOCKED", repr(tuple(blocked))) + "\n" + textwrap.dedent(body)
    environment = {**os.environ, "PYTHONPATH": str(SOURCE.parent), "PYTHONDONTWRITEBYTECODE": "1",
                   "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    result = subprocess.run([sys.executable, "-B", "-c", program], cwd=SOURCE.parent,
                            env=environment, text=True, capture_output=True, timeout=40)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("namespace_name", ("builders", "core"))
def test_public_route_namespaces_resolve_lazily_in_cold_process(namespace_name):
    body = """
        import importlib
        import sys

        namespace_name = NAMESPACE_NAME
        routes = ('simplicial', 'hyperdigraph', 'interaction')
        namespace = importlib.import_module('topokit.' + namespace_name)
        qualified = tuple(f'topokit.{namespace_name}.{route}' for route in routes)

        assert not any(name in sys.modules for name in qualified)
        assert all(route in dir(namespace) for route in routes)
        assert all(route not in namespace.__all__ for route in routes)
        for index, (route, module_name) in enumerate(zip(routes, qualified)):
            module = getattr(namespace, route)
            assert module.__name__ == module_name
            assert sys.modules[module_name] is module
            assert all(name not in sys.modules for name in qualified[index + 1:])

        try:
            namespace.not_a_topokit_route
        except AttributeError as error:
            assert 'not_a_topokit_route' in str(error)
        else:
            raise AssertionError('unknown public namespace attribute did not fail')
    """.replace("NAMESPACE_NAME", repr(namespace_name))
    _cold_process(body, [])


def test_core_analyzes_explicit_objects_with_all_upper_layers_unavailable():
    _cold_process("""
        from topokit import core
        from topokit.core import simplicial, hyperdigraph, interaction
        from topokit.core._interaction import build_interaction_chain_complex
        from topokit.results import Topology
        simplex = simplicial.SimplicialComplex([(0, 1)])
        hyper = hyperdigraph.Hyperdigraph((0, 1), ((0,), (1,), (0, 1)))
        factors = interaction.FactorComplexBuilder()
        factors.insert((0,), 0.0, with_faces=True)
        factor = factors.freeze()
        chain = build_interaction_chain_complex((factor, factor), max_homology_dimension=0)
        for kind, obj in [('simplicial', simplex), ('hyperdigraph', hyper), ('interaction', chain)]:
            assert core.homology(obj, max_dimension=0).betti_numbers == (1,)
            assert core.laplacian(obj, dimension=0).nullity == 1
            # Exercise both dynamic dispatch paths: native type and wrapper.
            wrapped = Topology(kind, obj)
            assert core.homology(wrapped, max_dimension=0).betti_numbers == (1,)
        filtered_hyper = hyperdigraph.FilteredHyperdigraph(
            (0, 1), (((0, 1), 1.0),), include_all_vertices=True)
        sweep = hyperdigraph.L0Sweep(filtered_hyper)
        assert sweep.matrix_at(0.).sum() == 0
        assert sweep.laplacian(1.).nullity == 1
        for obj in (simplex, filtered_hyper, chain):
            bars = core.persistence(obj, max_dimension=0)
            assert bars.betti_at(1.0) == (1,)
            spectrum = core.persistent_laplacian(obj, dimension=0, start=0.0, end=1.0)
            assert spectrum.kind == 'persistent' and spectrum.nullity == 1
        assert not any(name.startswith('topokit.builders') for name in sys.modules)
    """, [*(f"topokit.{layer}" for layer in UPPER), "topokit.api", "scipy.spatial", "matplotlib", "sklearn"])


def test_readers_features_and_views_operate_without_topology_algorithms():
    _cold_process("""
        import math
        from topokit import PointCloud
        from topokit.readers import ReaderRegistry
        from topokit.results import PersistenceInterval, PersistenceResult
        from topokit.postprocessing import histogram_features, spectral_entropy
        from topokit.visualization import plot_points
        cloud = PointCloud([[0, 0], [1, 0]], ids=['left', 'right'])
        registry = ReaderRegistry()
        registry.register('custom', lambda path: cloud, extensions=['example'])
        assert registry.read('opaque.example') is cloud
        bars = PersistenceResult((PersistenceInterval(0, 0, 1), PersistenceInterval(0, 0, math.inf)),
                                 max_dimension=0)
        features = histogram_features(bars, birth_edges=[0, 2], death_edges=[0, 2])
        assert features.finite_histograms.sum() == 1
        assert features.essential_histograms.sum() == 1
        assert abs(spectral_entropy([0, 1, 1]) - math.log(2)) < 1e-12
        # Supplying an axis also checks the plotting layer without a plot dependency.
        class Axis:
            def __getattr__(self, name):
                return lambda *args, **kwargs: None
        axis = Axis()
        assert plot_points(cloud, ax=axis, labels=True) is axis
        assert not any(name.startswith('topokit.core._') for name in sys.modules)
    """, ["topokit.builders", "topokit.workflows", "topokit.api", "matplotlib", "sklearn",
           "topokit.core._simplicial", "topokit.core._hyperdigraph", "topokit.core._interaction",
           "topokit.core.simplicial", "topokit.core.hyperdigraph", "topokit.core.interaction",
           "topokit.core._spectral"])


def test_all_builders_construct_with_analysis_entrypoints_disabled():
    _cold_process("""
        from topokit import PointCloud
        from topokit import core
        from topokit.core import simplicial, hyperdigraph, interaction
        from topokit.builders import from_points
        from topokit.builders.simplicial import from_graph
        from topokit.builders.hyperdigraph import from_digraph
        def forbidden(*args, **kwargs):
            raise AssertionError('Builder invoked an analysis/feature entrypoint')
        forbidden_names = FORBIDDEN_NAMES
        for name, module in list(sys.modules.items()):
            if name == 'topokit.core' or name.startswith('topokit.core.'):
                for symbol, value in list(vars(module).items()):
                    if callable(value) and (symbol in forbidden_names or
                                           getattr(value, '__name__', None) in forbidden_names):
                        setattr(module, symbol, forbidden)
        cloud = PointCloud([[0, 0], [1, 0], [0, 1]], ids=['a', 'b', 'c'])
        for kind in ('simplicial', 'hyperdigraph', 'interaction'):
            obj = from_points(cloud, kind=kind, max_dimension=0)
            assert obj.kind == kind and obj.native is not None
        assert from_graph([(0, 1)], max_dimension=0).native is not None
        assert from_digraph([(0, 1)]).number_of_hyperedges(1) == 1
    """.replace("FORBIDDEN_NAMES", repr(ANALYSIS_NAMES)),
        ["topokit.readers", "topokit.postprocessing", "topokit.visualization", "topokit.workflows", "sklearn", "matplotlib"])
