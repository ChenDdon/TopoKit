"""Check distributable contents: python tests/check_distribution.py WHEEL [SDIST]."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import PurePosixPath
import tarfile
import zipfile


ALLOWED_WHEEL_JSON = {
    "topokit/workflows/protein_ligand_prediction/reference_schema.json",
    "topokit/workflows/protein_ligand_prediction/model_profile.json",
}
SAMPLE_DIRECTORY = "examples/protein_ligand"


def check_wheel(path):
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
    forbidden_prefixes = (
        "examples/", "tests/", "markdown/", "workflows/",
        "topokit/datasets/", "topokit/_core/",
    )
    forbidden_suffixes = (
        ".pyc", ".csv", ".ipynb", ".png", ".svg", ".pdf", ".joblib", ".pkl",
        ".npy", ".npz", ".pdb", ".pdbqt", ".mol2", ".sdf", ".mol", ".cif",
        ".xyz", ".pt", ".pth", ".safetensors",
    )
    unexpected = sorted(
        name for name in names
        if name.startswith(forbidden_prefixes) or "__pycache__" in name
        or name.endswith(forbidden_suffixes)
        or (name.endswith(".json") and name not in ALLOWED_WHEEL_JSON)
    )
    assert not unexpected, f"Non-library artifacts in wheel: {unexpected}"
    required = ALLOWED_WHEEL_JSON | {
        "topokit/workflows/protein_ligand_prediction/features.py",
        "topokit/workflows/protein_ligand_prediction/prediction.py",
        "topokit/workflows/protein_ligand_prediction/sequence.py",
        *(f"topokit/{layer}/__init__.py" for layer in
          ("readers", "builders", "core", "postprocessing", "visualization", "workflows")),
    }
    assert required <= names, f"Missing wheel files: {sorted(required - names)}"
    assert "topokit/demo.py" not in names and "topokit/io.py" not in names
    print("Wheel: six layers and workflow APIs; no datasets, notebooks, weights or caches.")


def check_sdist(path):
    with tarfile.open(path) as archive:
        members = {item.name.partition("/")[2]: item for item in archive.getmembers()}
        names = set(members)

        def read(name):
            handle = archive.extractfile(members[name])
            assert handle is not None, f"Expected a regular source file: {name}"
            return handle.read()

        required = {
            "README.md", "NOTICE.md", "LICENSE", "environment.yml",
            ".github/workflows/tests.yml", "markdown/CHANGELOG.md",
            "examples/point_cloud.py", "examples/data/README.md",
            "examples/data/point_cloud_24.csv", "examples/data/point_cloud_24.json",
            "examples/data/data_6PT3_receptor.pdb", "examples/data/data_6PT3_ligand.mol2",
            "examples/data/data_molecule.sdf", "examples/data/data_molecule.mol",
            "examples/data/data_protein.pdbqt", "examples/data/data_cluster.xyz",
            "examples/data/AQUCOG_clean.cif", "examples/data/data_material.cif",
            "examples/data/data_material_from_cif.json",
            "examples/point_cloud_topology_workflow.ipynb",
            "examples/fixed_topological_objects_analysis.ipynb",
            "examples/different_input_formats_workflow.ipynb",
            "tests/check_distribution.py", "tests/wheel_smoke.py",
            "tests/protein_ligand_wheel_smoke.py",
            "workflows/skills/topokit-protein-ligand/SKILL.md",
            "workflows/skills/topokit-sequence/SKILL.md",
            *(f"{SAMPLE_DIRECTORY}/{name}" for name in
              ("README.md", "manifest.csv", "SOURCE.json", "EXPECTED.json")),
            *("workflows/protein_ligand_prediction/" + name for name in
              ("representative_strategy.py", "README.md", "extract_features.py",
               "ml/predict.py", "ml/tests/test_predict.py", "MODEL_SELECTION.json",
               "sequence/README.md", "sequence/RECIPE.json", "sequence/MODEL_SELECTION.json",
               "sequence/extract_features.py")),
            *("examples/pressure_test/" + name for name in
              ("run_pressure.py", "worker.py", "fast_worker.py", "cases.py",
               "summarize.py", "aggregate_aws.py", "matrices/smoke.json")),
        }
        assert required <= names, f"Missing source files: {sorted(required - names)}"
        forbidden_prefixes = (
            "archive/", "examples/output/", "src/topokit/datasets/",
            "examples/pressure_test/results/", "examples/pressure_test/transfer/",
            "examples/pressure_test/readability/",
            "workflows/protein_ligand_prediction/hpcc_15a/",
            "workflows/protein_ligand_prediction/cornell_15a/",
        )
        unexpected = sorted(name for name in names if name.startswith(forbidden_prefixes)
                            or "__pycache__" in name or name.endswith(".pyc"))
        assert not unexpected, f"Generated or stale source artifacts: {unexpected}"
        assert "workflows/protein_ligand_prediction/feature_strategy.py" not in names
        assert "workflows/protein_ligand_prediction/ml_feature_comparison.py" not in names

        rows = list(csv.DictReader(io.StringIO(read(f"{SAMPLE_DIRECTORY}/manifest.csv").decode())))
        assert len(rows) == 10 and len({row["sample_id"] for row in rows}) == 10
        provenance = json.loads(read(f"{SAMPLE_DIRECTORY}/SOURCE.json"))
        expected_files = {
            entry["example_relative_path"]: entry["sha256"]
            for sample in provenance["samples"] for entry in sample["files"].values()
        }
        for row in rows:
            for key in ("protein_file", "ligand_file"):
                relative = PurePosixPath(row[key])
                assert not relative.is_absolute() and ".." not in relative.parts
                member = f"{SAMPLE_DIRECTORY}/{relative}"
                assert member in names, f"Manifest input absent from source archive: {member}"
                assert hashlib.sha256(read(member)).hexdigest() == expected_files[str(relative)]
    print("Source archive: tutorials, recipes, agent skills and ten verified pairs; generated pressure outputs excluded.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel")
    parser.add_argument("sdist", nargs="?")
    args = parser.parse_args()
    check_wheel(args.wheel)
    if args.sdist:
        check_sdist(args.sdist)


if __name__ == "__main__":
    main()
