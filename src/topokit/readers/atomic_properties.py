"""Offline elemental attributes. Assignment is explicit, never a builder default.

See markdown/REFERENCE_DATA.md for the PubChem snapshot and missing-value policy.
These are scientific reference constants, not sample or demonstration datasets.
"""
from collections.abc import Mapping
from copy import deepcopy
from numbers import Real
from types import MappingProxyType
import math

from ..data import PointCloud


PAULING_SOURCE = MappingProxyType({
    "provider": "PubChem (NCBI/NLM/NIH)",
    "property": "electronegativity", "scale": "Pauling", "units": "dimensionless",
    "url": "https://pubchem.ncbi.nlm.nih.gov/rest/pug/periodictable/JSON",
    "definition_url": "https://pubchem.ncbi.nlm.nih.gov/periodic-table/electronegativity",
    "retrieved": "2026-09-03",
    "response_sha256": "ff0f75976583b8a6493b18d0b42e83e1b68bd6e112b45d33578d98494ed5d321",
})

# Blank PubChem entries remain None, even when another table has an estimate.
PAULING_ELECTRONEGATIVITY = MappingProxyType({
    "H": 2.2, "He": None, "Li": 0.98, "Be": 1.57, "B": 2.04, "C": 2.55,
    "N": 3.04, "O": 3.44, "F": 3.98, "Ne": None, "Na": 0.93, "Mg": 1.31,
    "Al": 1.61, "Si": 1.9, "P": 2.19, "S": 2.58, "Cl": 3.16, "Ar": None,
    "K": 0.82, "Ca": 1.0, "Sc": 1.36, "Ti": 1.54, "V": 1.63, "Cr": 1.66,
    "Mn": 1.55, "Fe": 1.83, "Co": 1.88, "Ni": 1.91, "Cu": 1.9, "Zn": 1.65,
    "Ga": 1.81, "Ge": 2.01, "As": 2.18, "Se": 2.55, "Br": 2.96, "Kr": 3.0,
    "Rb": 0.82, "Sr": 0.95, "Y": 1.22, "Zr": 1.33, "Nb": 1.6, "Mo": 2.16,
    "Tc": 1.9, "Ru": 2.2, "Rh": 2.28, "Pd": 2.2, "Ag": 1.93, "Cd": 1.69,
    "In": 1.78, "Sn": 1.96, "Sb": 2.05, "Te": 2.1, "I": 2.66, "Xe": 2.6,
    "Cs": 0.79, "Ba": 0.89, "La": 1.1, "Ce": 1.12, "Pr": 1.13, "Nd": 1.14,
    "Pm": None, "Sm": 1.17, "Eu": None, "Gd": 1.2, "Tb": None, "Dy": 1.22,
    "Ho": 1.23, "Er": 1.24, "Tm": 1.25, "Yb": None, "Lu": 1.27, "Hf": 1.3,
    "Ta": 1.5, "W": 2.36, "Re": 1.9, "Os": 2.2, "Ir": 2.2, "Pt": 2.28,
    "Au": 2.54, "Hg": 2.0, "Tl": 1.62, "Pb": 2.33, "Bi": 2.02, "Po": 2.0,
    "At": 2.2, "Rn": None, "Fr": 0.7, "Ra": 0.9, "Ac": 1.1, "Th": 1.3,
    "Pa": 1.5, "U": 1.38, "Np": 1.36, "Pu": 1.28, "Am": 1.3, "Cm": 1.3,
    "Bk": 1.3, "Cf": 1.3, "Es": 1.3, "Fm": 1.3, "Md": 1.3, "No": 1.3,
    "Lr": 1.3, "Rf": None, "Db": None, "Sg": None, "Bh": None, "Hs": None,
    "Mt": None, "Ds": None, "Rg": None, "Cn": None, "Nh": None, "Fl": None,
    "Mc": None, "Lv": None, "Ts": None, "Og": None,
})


def assign_element_weights(cloud, *, symbols=None, overrides=None):
    """Return a new point cloud with explicitly assigned Pauling weights.

    Symbols default to row-aligned canonical element ``metadata['labels']``
    supplied by XYZ or a molecular/crystallographic reader. No atom-name
    guessing or missing-value imputation occurs here. ``overrides`` supplies
    finite weights for chosen element symbols and is recorded in result
    metadata. This helper does not infer bonds, alter coordinates, or construct
    topology.
    """
    if not isinstance(cloud, PointCloud):
        raise TypeError("assign_element_weights expects a PointCloud")
    if symbols is None:
        symbols = cloud.metadata.get("labels")
    if symbols is None or isinstance(symbols, (str, bytes, set, frozenset, Mapping)):
        raise ValueError("provide one element symbol per point (or reader labels)")
    labels = tuple(symbols)
    if len(labels) != len(cloud) or any(not isinstance(x, str) for x in labels):
        raise ValueError("provide one canonical element symbol per point")
    if overrides is not None and not isinstance(overrides, Mapping):
        raise ValueError("overrides must map element symbols to finite real weights")
    weight_overrides = dict(overrides or {})
    for symbol, value in weight_overrides.items():
        if symbol not in PAULING_ELECTRONEGATIVITY:
            raise ValueError(f"unknown element symbol in overrides: {symbol!r}")
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError("override weights must be finite real numbers")
    unknown_symbols = tuple(dict.fromkeys(x for x in labels if x not in PAULING_ELECTRONEGATIVITY))
    if unknown_symbols:
        raise ValueError(f"unknown element symbols: {unknown_symbols}; supply canonical symbols explicitly")
    assigned_weights = [weight_overrides.get(x, PAULING_ELECTRONEGATIVITY[x]) for x in labels]
    missing_symbols = tuple(dict.fromkeys(x for x, value in zip(labels, assigned_weights) if value is None))
    if missing_symbols:
        raise ValueError(f"no PubChem Pauling value for {missing_symbols}; provide explicit overrides")
    metadata = deepcopy(cloud.metadata)
    metadata["weight_assignment"] = {
        **dict(PAULING_SOURCE), "overrides": {k: float(v) for k, v in weight_overrides.items()},
        "missing_value_policy": "error_unless_overridden",
        "contains_user_overrides": bool(weight_overrides),
        "symbols_by_point_id": dict(zip(cloud.ids, labels)),
    }
    return PointCloud(cloud.points, ids=cloud.ids, weights=assigned_weights, metadata=metadata)
