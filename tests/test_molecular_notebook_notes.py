"""Documentation contract for the molecular-format notebook's spectral notes."""

import json
from pathlib import Path


NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "different_input_formats_workflow.ipynb"
)


def test_spectral_notes_separate_builtin_custom_and_curve_descriptors():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = [cell for cell in notebook["cells"] if cell.get("id") == "interpretation"]
    assert len(cells) == 1
    notes = "".join(cells[0]["source"])

    for heading in (
        "### Laplacian and spectrum primer",
        "### Four spectral curves shown",
        "### Descriptor reference and TopoKit status",
        "### Defining a summary that is not built in",
        "### Summaries of a curve across filtration scale",
    ):
        assert heading in notes

    for built_in in (
        "`zero_count`",
        "`positive_count`",
        "`spectral_entropy`",
        "`laplacian_energy`",
    ):
        assert built_in in notes
    for custom in (
        "Inverse positive-spectrum sum",
        "Log pseudo-determinant",
        "Heat trace",
        "Application-level custom calculation",
    ):
        assert custom in notes

    assert "statistics={" in notes
    assert "positive_only=True" in notes
    assert "positive_only=False" in notes
    assert "two-scale `persistent_laplacian" in notes
    assert "not molecular energies in kcal/mol" in notes
    assert "Built-in summary name: `energy`" in notes
    assert "public array-level helper: `spectral_energy`" in notes
    assert "a zero or empty selected spectrum returns 0" in notes
    assert "adjusted[np.abs(adjusted) <= zero_tolerance] = 0.0" in notes

    example = notes.split("```python\n", 1)[1].split("```", 1)[0]
    compile(example, str(NOTEBOOK), "exec")
