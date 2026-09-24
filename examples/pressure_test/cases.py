"""Declared, reproducible pressure-test matrix; no scientific computations."""
from copy import deepcopy


# These are explicit budgets, not silent topology/spectrum truncations.
LIMITS = {
    "max_cells": 2_000_000,
    "max_simplices": 1_000_000,
    "max_hyperedges": 1_000_000,
    "max_chain_bytes": 4 * 1024**3,
    "max_dense_entries": 16_000_000,
    "max_dense_bytes": 4 * 1024**3,
    "max_sparse_entries": 20_000_000,
}

PROFILES = {
    "simplicial-alpha": dict(family="simplicial", construction="alpha",
        sizes={0: (1000, 5000, 20000), 1: (128, 512, 2048), 2: (32, 128, 512)}),
    "simplicial-rips": dict(family="simplicial", construction="rips",
        sizes={0: (256, 1024, 4096), 1: (32, 64, 128), 2: (16, 32, 64)}),
    "hyperdigraph-delaunay-distinct": dict(family="hyperdigraph", construction="delaunay",
        weights="distinct", sizes={0: (512, 2048, 8192), 1: (32, 128, 512), 2: (16, 32, 64)}),
    "hyperdigraph-delaunay-equal": dict(family="hyperdigraph", construction="delaunay",
        weights="equal", sizes={0: (512, 2048, 8192), 1: (32, 128, 512), 2: (16, 32, 64)}),
    "hyperdigraph-complete-distinct": dict(family="hyperdigraph", construction="complete",
        weights="distinct", sizes={0: (128, 512, 2048), 1: (16, 32, 64), 2: (8, 16, 32)}),
    "hyperdigraph-complete-equal": dict(family="hyperdigraph", construction="complete",
        weights="equal", sizes={0: (128, 512, 2048), 1: (16, 32, 64), 2: (8, 16, 32)}),
    "interaction-alpha": dict(family="interaction", construction="alpha", overlap="full",
        sizes={0: (128, 512, 2048), 1: (32, 128, 512), 2: (16, 64, 256)}),
    "interaction-rips": dict(family="interaction", construction="rips", overlap="full",
        sizes={0: (32, 128, 512), 1: (12, 32, 64), 2: (8, 16, 32)}),
}
OPERATIONS = ("persistence", "laplacian", "persistent_laplacian")

# Homology and full spectra have different costs. Keep H0 and H1 scaling
# independent from L0/L1, and do not force the larger PH clouds into a solver.
PERSISTENCE_SIZES = {
    "simplicial-alpha": {0: (1000, 10000, 30000), 1: (512, 2048, 8192)},
    "simplicial-rips": {0: (256, 1024, 1536), 1: (32, 128, 192)},
    "hyperdigraph-delaunay-distinct": {0: (1024, 8192, 32768), 1: (128, 1024, 8192)},
    "hyperdigraph-delaunay-equal": {0: (512, 4096, 16384), 1: (64, 256, 1024)},
    "hyperdigraph-complete-distinct": {0: (128, 384, 768), 1: (24, 64, 128)},
    "hyperdigraph-complete-equal": {0: (96, 256, 512), 1: (16, 40, 64)},
    "interaction-alpha": {0: (512, 4096, 16384), 1: (64, 256, 1024)},
    "interaction-rips": {0: (128, 512, 1024), 1: (16, 64, 112)},
}
COMPACT_SIZES = {
    "delaunay": {0: (5000, 20000, 60000), 1: (1000, 5000, 20000)},
    "complete": {0: (1000, 3000, 10000), 1: (300, 1000, 5000)},
}


def build_cases(*, suite="pressure", profiles=None, repeats=1, seed=20260905):
    """Return a fully specified matrix, including the meaning of each setting.

    Smoke uses six points for every profile/operation/degree. Pressure has
    three increasing sizes per series. Partial spectra and half-overlap
    interaction are separate supplemental cases, never automatic fallbacks.
    """
    if suite not in {"smoke", "pressure", "supplemental"}:
        raise ValueError("suite must be smoke, pressure, or supplemental")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    selected = list(PROFILES) if profiles is None else list(profiles)
    if not selected or len(selected) != len(set(selected)) or set(selected) - set(PROFILES):
        raise ValueError("profiles must be unique declared profile names")
    cases = []
    for profile_name in selected:
        profile = PROFILES[profile_name]
        for degree in (0, 1, 2):
            for operation in OPERATIONS:
                sizes = profile["sizes"][degree]
                if suite == "pressure" and operation == "persistence":
                    sizes = PERSISTENCE_SIZES[profile_name].get(degree, sizes)
                if suite == "smoke":
                    sizes = (6,)
                elif suite == "supplemental":
                    sizes = sizes[1:2]
                if suite == "supplemental" and operation == "persistence" and profile["family"] != "interaction":
                    continue
                for point_count in sizes:
                    for repetition in range(repeats):
                        spectrum = ("partial" if suite == "supplemental" and operation != "persistence"
                                    else "full")
                        overlap = ("half" if suite == "supplemental" and profile["family"] == "interaction"
                                   else profile.get("overlap", "full"))
                        case_id = (f"{profile_name}-{operation}-q{degree}-n{point_count}"
                                   f"-{spectrum}-{overlap}-r{repetition}")
                        cases.append({
                            "case_id": case_id, "profile": profile_name, "suite": suite,
                            "execution_route": "canonical",
                            "family": profile["family"], "construction": profile["construction"],
                            "points": point_count, "ambient_dimension": 3, "seed": seed,
                            "repetition": repetition, "operation": operation, "degree": degree,
                            "weights": profile.get("weights", "equal"), "overlap": overlap,
                            "spectrum": spectrum, "k": 8, "limits": deepcopy(LIMITS),
                            "construction_rule": (
                                "unrestricted Rips through q+1; edge-distance births" if profile["construction"] == "rips"
                                else "unweighted alpha through q+1; squared-radius births" if profile["construction"] == "alpha"
                                else f"{profile['construction']} support; ordered distinct-vertex sequences through q+1; consecutive-distance births"),
                            "filtration_cutoff": None,
                        })
    return cases


def build_compact_cases(*, smoke=False, seed=20260905):
    """Separate public compact H0/H1 workflow, eligible distinct weights only.

    Its API exposes no canonical allocation guards. Only the parent process
    wall/RSS/address-space limits and documented packed-storage caps apply.
    """
    cases = []
    for support, degrees in COMPACT_SIZES.items():
        template = next(case for case in build_cases(suite="smoke")
            if case["profile"] == f"hyperdigraph-{support}-distinct"
            and case["operation"] == "persistence" and case["degree"] == 0)
        for degree, sizes in degrees.items():
            for points in ((8,) if smoke else sizes):
                cases.append({**template,
                    "case_id": f"compact-hyperdigraph-{support}-persistence-q{degree}-n{points}-r0",
                    "suite": "compact-smoke" if smoke else "compact-pressure",
                    "execution_route": "compact_hyperdigraph", "limits": {},
                    "points": points, "degree": degree, "seed": seed,
                    "construction_rule": f"{support} packed score-DAG edges; implicit directed two-path boundaries for H1; consecutive-distance births"})
    return cases
