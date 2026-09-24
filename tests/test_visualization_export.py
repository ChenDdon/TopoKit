"""Actual file exports preserve editable labels and caller plotting state."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from topokit.visualization import save_figure


@pytest.fixture
def figure():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot([0, 1], [0, 1])
    ax.set_title("Editable cell")
    yield fig, ax, plt
    plt.close(fig)


@pytest.mark.parametrize("format_", ("png", "svg", "pdf"))
@pytest.mark.parametrize("source", ("figure", "axes"))
def test_actual_exports_keep_text_and_plotting_state(figure, tmp_path, format_, source):
    fig, ax, plt = figure
    target = tmp_path / "new" / f"cell.{format_}"
    # These intentional non-export defaults must be restored even though the
    # written files use editable SVG text and embedded TrueType PDF fonts.
    with plt.rc_context({"svg.fonttype": "path", "pdf.fonttype": 3}):
        before = dict(plt.rcParams)
        result = save_figure(fig if source == "figure" else ax, target, dpi=90)
        assert dict(plt.rcParams) == before
    assert isinstance(result, Path)
    assert result == target
    assert plt.fignum_exists(fig.number)
    assert ax.get_title() == "Editable cell"
    content = target.read_bytes()
    assert len(content) > 1000
    if format_ == "png":
        assert content.startswith(b"\x89PNG\r\n\x1a\n")
        pixels = plt.imread(target)
        assert pixels.shape[0] > 100 and pixels.shape[1] > 100
        assert float(np.std(pixels[..., :3])) > 0.01
    elif format_ == "svg":
        svg = ET.fromstring(content)
        texts = svg.findall(".//{http://www.w3.org/2000/svg}text")
        assert "Editable cell" in ["".join(text.itertext()) for text in texts]
    else:
        assert content.startswith(b"%PDF-")
        assert b"/FontFile2" in content


@pytest.mark.parametrize("filename", ("no_extension", "cell.unsupported"))
def test_invalid_format_fails_before_creating_directory(figure, tmp_path, filename):
    fig, _, _ = figure
    target = tmp_path / "not_created" / filename
    with pytest.raises(ValueError, match="extension|format"):
        save_figure(fig, target)
    assert not target.parent.exists()


@pytest.mark.parametrize("dpi", (0, -1, np.inf, np.nan, True, "300"))
def test_invalid_resolution_fails_before_creating_directory(figure, tmp_path, dpi):
    fig, _, _ = figure
    target = tmp_path / "not_created" / "cell.png"
    with pytest.raises(ValueError, match="dpi"):
        save_figure(fig, target, dpi=dpi)
    assert not target.parent.exists()


@pytest.mark.parametrize("invalid_figure", (None, object(), "not a figure"))
def test_invalid_figure_fails_before_creating_directory(tmp_path, invalid_figure):
    target = tmp_path / "not_created" / "cell.png"
    with pytest.raises(TypeError, match="figure or axes"):
        save_figure(invalid_figure, target)
    assert not target.parent.exists()


def test_invalid_transparency_fails_before_creating_directory(figure, tmp_path):
    fig, _, _ = figure
    target = tmp_path / "not_created" / "cell.png"
    with pytest.raises(ValueError, match="transparent"):
        save_figure(fig, target, transparent="yes")
    assert not target.parent.exists()


def test_transparent_png_and_case_insensitive_format(figure, tmp_path):
    fig, _, plt = figure
    target = tmp_path / "cell.PNG"
    save_figure(fig, str(target), dpi=90, transparent=True)
    pixels = plt.imread(target)
    assert pixels.shape[-1] == 4
    assert np.any(pixels[..., 3] == 0)
    assert np.any(pixels[..., 3] > 0)


def test_failed_write_restores_local_settings_and_keeps_figure(figure, tmp_path, monkeypatch):
    fig, _, plt = figure

    def failed_save(*args, **kwargs):
        raise OSError("test write failure")

    monkeypatch.setattr(fig, "savefig", failed_save)
    with plt.rc_context({"svg.fonttype": "path", "pdf.fonttype": 3}):
        before = dict(plt.rcParams)
        with pytest.raises(OSError, match="test write failure"):
            save_figure(fig, tmp_path / "cell.svg")
        assert dict(plt.rcParams) == before
    assert plt.fignum_exists(fig.number)
