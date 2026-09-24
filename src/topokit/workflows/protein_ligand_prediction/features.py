"""Fixed FS-AN protein–ligand application; compose public layers only.

The numerical functions preserve the audited 15 Å implementation. Schema
engine hashes identify its frozen reference source, not this reorganized file;
implementation_receipt() reports the current implementation separately.
"""
from __future__ import annotations
import hashlib
import json
import math
from importlib.resources import files
from pathlib import Path
from typing import Any, Sequence
import numpy as np
from scipy.spatial import cKDTree, distance
from topokit import PointCloud, ResourceLimitError, readers
from topokit.builders import simplicial
from topokit.core import hyperdigraph

STRATEGY_ID = "FS-AN"
CROP_ANGSTROM = 15.0
ALPHA_RADII = tuple(i / 10 for i in range(1, 51))
THRESHOLDS = np.square(ALPHA_RADII)
PROTEIN_BASE_ELEMENTS = ("C", "N", "O", "S")
LIGAND_FEATURE_ELEMENTS = ("C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "H")
LIGAND_CROP_ELEMENTS = ("H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I")
PROTEIN_GROUPS = ("null",) + PROTEIN_BASE_ELEMENTS
LIGAND_GROUPS = ("null",) + LIGAND_FEATURE_ELEMENTS
CHANNELS = tuple((p, l) for p in PROTEIN_GROUPS for l in LIGAND_GROUPS)
DEFAULT_SPECTRAL_ZERO_TOLERANCE = 1e-10
STATISTICS = ("matrix_size_n0", "zero_eigenvalue_count", "positive_mean",
              "positive_min_over_mean", "positive_MAD_over_mean", "positive_max_over_mean",
              "positive_q25", "positive_median", "positive_q75", "spectral_entropy_nats")

def spectral_statistics(eigenvalues):
    result = np.zeros(10, dtype=np.float64)
    result[:6] = compact_spectral_statistics(eigenvalues)
    values = np.asarray(eigenvalues, dtype=np.float64)
    positive = values[values > DEFAULT_SPECTRAL_ZERO_TOLERANCE]
    if len(positive):
        result[6:9] = np.quantile(positive, [0.25, 0.5, 0.75], method="linear")
        probabilities = positive / float(np.max(positive))
        probabilities /= float(np.sum(probabilities))
        nonzero = probabilities[probabilities > 0]
        result[9] = -float(np.sum(nonzero * np.log(nonzero)))
    return result


def deduplicate_selected_atoms(selected):
    """Validate then keep each exact site once across components/elements.

    Stable order is cropped protein rows, then ligand rows. The first atom's
    element/component wins even when a later duplicate has different labels.
    Cropping uses the original supported ligand anchors; their duplicate
    multiplicity never changes the distance predicate. Input files are intact.
    """
    result = dict(selected)
    for side, allowed in (("protein", PROTEIN_BASE_ELEMENTS), ("ligand", LIGAND_FEATURE_ELEMENTS)):
        points = np.asarray(selected[side + "_points"], dtype=np.float64)
        elements = np.asarray(selected[side + "_elements"])
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError(f"{side} coordinates must be finite N x 3")
        if elements.shape != (len(points),) or not np.isin(elements, allowed).all():
            raise ValueError(f"invalid {side} element labels")
        result[side + "_points"], result[side + "_elements"] = points, elements
    protein, ligand = result["protein_points"], result["ligand_points"]
    points = np.vstack((protein, ligand))
    unique = PointCloud(points).unique_coordinates()
    info = unique.metadata["coordinate_deduplication"]
    retained = np.asarray(info["retained_indices"], dtype=int)
    elements = np.concatenate((result["protein_elements"], result["ligand_elements"]))
    def atom(i):
        return {"component": "protein" if i < len(protein) else "ligand",
                "selected_row": int(i if i < len(protein) else i-len(protein)),
                "element": str(elements[i])}
    removed = [{"coordinates": points[i].tolist(), "removed": atom(i), "retained": atom(j)}
               for i, j in info["original_to_retained_id"].items() if i != j]
    for side, indices in (("protein", retained[retained < len(protein)]),
                          ("ligand", retained[retained >= len(protein)]-len(protein))):
        result[side + "_points"] = result[side + "_points"][indices]
        result[side + "_elements"] = result[side + "_elements"][indices]
    # Preserve the original read-time removal receipt when compute() rechecks
    # an already-unique selection. The caller's dictionary is never mutated.
    if removed or "coordinate_deduplication" not in result:
        result["coordinate_deduplication"] = {
            "policy": "exact_keep_first", "order": "cropped_protein_then_ligand",
            "input_count": len(points), "unique_count": len(unique),
            "removed_count": len(removed), "removed_atoms": removed}
    result["counts"] = {**result.get("counts", {}),
        "protein_unique_atoms": len(result["protein_points"]),
        "ligand_unique_atoms": len(result["ligand_points"]),
        "duplicate_atoms_removed": result["coordinate_deduplication"]["removed_count"]}
    if removed and "cross_distances" in result:
        result["cross_distances"] = distance.cdist(result["protein_points"], result["ligand_points"])
    return result


def compact_spectral_statistics(
    eigenvalues: Sequence[float],
    *,
    zero_tolerance: float = DEFAULT_SPECTRAL_ZERO_TOLERANCE,
) -> np.ndarray:
    """Summarize one complete spectrum using the six v2 descriptors.

    The kernel/positive split uses an absolute tolerance and no decimal
    pre-rounding.  This preserves the design note's topological meaning for
    beta_0 while still treating eigensolver-scale residuals as numerical zero.
    """
    values = np.asarray(eigenvalues, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("a finite one-dimensional spectrum is required")
    if not math.isfinite(zero_tolerance) or zero_tolerance <= 0.0:
        raise ValueError("zero_tolerance must be finite and positive")
    if np.any(values < -zero_tolerance):
        minimum = float(np.min(values))
        raise ValueError(
            f"Laplacian spectrum contains an eigenvalue below -zero_tolerance: {minimum}"
        )

    positive = values[values > zero_tolerance]
    result = np.zeros(6, dtype=np.float64)
    result[0] = len(values)
    result[1] = len(values) - len(positive)
    if len(positive):
        positive_mean = float(np.mean(positive))
        result[2] = positive_mean
        result[3] = float(np.min(positive)) / positive_mean
        result[4] = float(np.mean(np.abs(positive - positive_mean))) / positive_mean
        result[5] = float(np.max(positive)) / positive_mean
    return result


def _reader_elements(cloud: Any, context: str) -> np.ndarray:
    labels = cloud.metadata.get("labels")
    if labels is None or len(labels) != len(cloud):
        raise ValueError(f"{context} reader did not return aligned element labels")
    return np.asarray(labels, dtype=object)


def _read_selected_atoms(protein_file: Path, ligand_file: Path, crop_radius: float) -> dict[str, Any]:
    if protein_file.suffix.lower() != ".pdb":
        raise ValueError("the protein input must be a PDB file")
    if ligand_file.suffix.lower() not in {".mol2", ".sdf", ".mol"}:
        raise ValueError("the ligand input must be MOL2, SDF, or MOL")

    protein_cloud = readers.read_pdb(protein_file, coordinate_units="angstrom")
    protein_elements = _reader_elements(protein_cloud, "protein")
    record_names = np.asarray(protein_cloud.metadata["columns"]["record_name"], dtype=object)
    protein_mask = (record_names == "ATOM") & np.isin(protein_elements, PROTEIN_BASE_ELEMENTS)
    raw_protein_points = np.asarray(protein_cloud.points[protein_mask], dtype=np.float64)
    raw_protein_elements = protein_elements[protein_mask]
    if not len(raw_protein_points):
        raise ValueError("no protein ATOM records with elements C/N/O/S were found")

    ligand_cloud = readers.read(ligand_file, coordinate_units="angstrom")
    ligand_elements = _reader_elements(ligand_cloud, "ligand")
    ligand_crop_mask = np.isin(ligand_elements, LIGAND_CROP_ELEMENTS)
    ligand_crop_points = np.asarray(ligand_cloud.points[ligand_crop_mask], dtype=np.float64)
    if not len(ligand_crop_points):
        raise ValueError("no supported ligand atoms were found for the protein-field selection")

    nearest_ligand_distance = cKDTree(ligand_crop_points).query(raw_protein_points, k=1)[0]
    protein_field_mask = nearest_ligand_distance <= crop_radius
    protein_points = raw_protein_points[protein_field_mask]
    protein_elements = raw_protein_elements[protein_field_mask]
    if not len(protein_points):
        raise ValueError("the declared ligand-centered protein field contains no C/N/O/S atoms")

    ligand_feature_mask = np.isin(ligand_elements, LIGAND_FEATURE_ELEMENTS)
    ligand_points = np.asarray(ligand_cloud.points[ligand_feature_mask], dtype=np.float64)
    ligand_feature_elements = ligand_elements[ligand_feature_mask]
    if not len(ligand_points):
        raise ValueError("the ligand contains no configured feature elements")

    cross_distances = distance.cdist(protein_points, ligand_points, metric="euclidean")
    return deduplicate_selected_atoms({
        "protein_points": protein_points,
        "protein_elements": protein_elements,
        "ligand_points": ligand_points,
        "ligand_elements": ligand_feature_elements,
        "cross_distances": cross_distances,
        "counts": {
            "protein_reader_atoms": len(protein_cloud),
            "protein_CNOS_ATOM_before_crop": len(raw_protein_points),
            "protein_CNOS_ATOM_after_crop": len(protein_points),
            "ligand_reader_atoms": len(ligand_cloud),
            "ligand_crop_anchor_atoms_including_supported_H": len(ligand_crop_points),
            "ligand_feature_atoms_including_explicit_hydrogen": len(ligand_points),
        },
    })


def read_selected_atoms(protein_file, ligand_file):
    return _read_selected_atoms(Path(protein_file), Path(ligand_file), CROP_ANGSTROM)


def channel_summaries(protein, ligand, *, cross_only, max_dense_entries=25_000_000,
                      max_simplices=1_000_000):
    unique = PointCloud(np.vstack((protein, ligand))).unique_coordinates()
    protein_count = sum(i < len(protein) for i in unique.ids)
    count = len(unique)
    result = np.zeros((10, 50), dtype=np.float64)
    if not count or (cross_only and (not protein_count or protein_count == count)):
        return result
    if count * count > max_dense_entries:
        raise ResourceLimitError(f'Full spectrum needs {count} x {count} dense entries')
    cloud = PointCloud(unique.points, weights=np.arange(count, dtype=np.float64), metadata={
        'coordinate_units': 'angstrom', 'weight_semantics': 'row-order direction tags only'})
    alpha = simplicial.from_points(cloud, complex_type='alpha', max_dimension=0,
        backend='native', filtration_range=(0., 25.), duplicates='merge',
        geometry_tolerance=1e-12, max_simplices=max_simplices)
    if alpha.metadata.get('geometry_backend_version') != '1.1.1':
        raise ValueError('This recipe requires native alpha engine 1.1.1')
    if set(alpha.native.simplices(0)) != {(i,) for i in range(count)}:
        raise ValueError('alpha construction lost selected vertices')
    edges = [(tuple(e), float(b)) for e, b in alpha.metadata['raw_filtration'] if len(e) == 2]
    if cross_only:
        edges = [(e, b) for e, b in edges if e[0] < protein_count <= e[1]]
    edges.sort(key=lambda x: x[1])
    births = np.asarray([b for _, b in edges], dtype=np.float64)
    filtered = hyperdigraph.FilteredHyperdigraph(tuple(range(count)), edges,
        include_all_vertices=True, vertex_birth=0.)
    sweep = hyperdigraph.L0Sweep(filtered, max_dense_entries=max_dense_entries)
    previous = -1
    for i, threshold in enumerate(THRESHOLDS):
        active = int(np.searchsorted(births, threshold, side='right'))
        if active != previous:
            eigenvalues = (np.zeros(count) if active == 0 else
                           sweep.laplacian(float(threshold), tol=1e-10).eigenvalues)
            summary = spectral_statistics(eigenvalues)
            previous = active
        result[:, i] = summary
    if not np.isfinite(result).all():
        raise ValueError('nonfinite L0 summary')
    return result


def compute(selected, *, max_dense_entries=25_000_000, max_simplices=1_000_000):
    selected = deduplicate_selected_atoms(selected)
    tensor = np.zeros((10, 50, 55), dtype=np.float64)
    present = np.zeros(55, dtype=bool)
    counts = np.zeros((55, 2), dtype=np.int32)
    for i, (p, l) in enumerate(CHANNELS):
        protein = selected['protein_points'][np.asarray(selected['protein_elements']) == p]
        ligand = selected['ligand_points'][np.asarray(selected['ligand_elements']) == l]
        counts[i] = len(protein), len(ligand)
        if p == l == 'null' or (p != 'null' and not len(protein)) or (l != 'null' and not len(ligand)):
            continue
        present[i] = True
        tensor[:, :, i] = channel_summaries(protein, ligand, cross_only=p != 'null' and l != 'null',
            max_dense_entries=max_dense_entries, max_simplices=max_simplices)
    return tensor, present, counts



def schema():
    """Return a fresh copy of the frozen reference feature schema.

    Historical engine_sha256 fields preserve the original generation identity;
    see implementation_receipt() for this package's source identity.
    """
    return json.loads(files(__package__).joinpath("reference_schema.json").read_text())


def implementation_receipt():
    """Separate the installed implementation from the unchanged reference recipe."""
    return {"strategy_id": STRATEGY_ID, "reference_recipe_id": schema()["recipe_id"],
            "implementation_version": "1.0.0", "implementation_module": __name__,
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "reference_schema_sha256": hashlib.sha256(
                files(__package__).joinpath("reference_schema.json").read_bytes()).hexdigest()}


def featurize(protein_file, ligand_file, *, max_dense_entries=25_000_000,
              max_simplices=1_000_000):
    """Return the selected C-order float32 (10, 50, 55) tensor for one complex."""
    selected = read_selected_atoms(protein_file, ligand_file)
    tensor, _, _ = compute(selected, max_dense_entries=max_dense_entries,
                           max_simplices=max_simplices)
    return np.ascontiguousarray(tensor, dtype="<f4")
