"""The core import and homology API remain usable without site-packages."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest


class OptionalDependencyTests(unittest.TestCase):
    def test_core_import_and_actionable_numerical_error_without_site_packages(self):
        source = Path(__file__).resolve().parents[2] / "src"
        program = """
import sys, os, types
# topokit itself intentionally requires NumPy. Isolate the dependency-free
# private reference core without changing its relative imports.
root = os.environ['PYTHONPATH']
for name, relative in [('topokit', 'topokit'), ('topokit.core', 'topokit/core')]:
    module = types.ModuleType(name)
    module.__path__ = [os.path.join(root, relative)]
    sys.modules[name] = module
import topokit.core._interaction as interaction
from topokit.core._interaction import *
assert 'numpy' not in sys.modules
assert 'scipy' not in sys.modules
assert interaction.__version__ == '0.2.0'
builder = SimplicialComplexBuilder()
builder.insert((0,), 0)
factor = builder.freeze()
chain = build_interaction_chain_complex((factor, factor), max_homology_dimension=0)
chain.validate_signed_boundary()
assert compute_persistence(chain).betti_at(0, 0) == 1
try:
    InteractionLaplacianEngine(chain)
except ImportError as error:
    assert 'interaction-topology[laplacian]' in str(error)
else:
    raise AssertionError('NumPy/SciPy unexpectedly available with site-packages disabled')
"""
        result = subprocess.run(
            [sys.executable, "-S", "-c", program],
            env={**os.environ, "PYTHONPATH": str(source), "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
