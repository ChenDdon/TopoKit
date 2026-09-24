"""Scientific checks for molecular selection, alpha births and full-bin H0."""
from pathlib import Path
import sys
import math
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ph_features as f
from audit_real_samples import direct_counts


def sample(pp, lp, pe=None, le=None):
    pp, lp = np.asarray(pp, dtype=float).reshape(-1, 3), np.asarray(lp, dtype=float).reshape(-1, 3)
    return f.crop_and_deduplicate(pp, pe if pe is not None else ['C']*len(pp),
                                  lp, le if le is not None else ['C']*len(lp))


def test_crop_includes_hydrogen_anchor_and_boundary_before_keep_first_dedup():
    selected = sample([[0,0,0], [0,0,0], [15,0,0], [15.001,0,0], [1e-10,0,0]],
                      [[100,0,0], [0,0,0]], ['C','N','O','S','S'], ['C','H'])
    assert selected['protein_elements'].tolist() == ['C','O','S']
    assert selected['ligand_elements'].tolist() == ['C']
    assert selected['deduplication']['removed_atoms'] == 2
    assert selected['deduplication']['supported_ligand_anchors'] == 2
    np.testing.assert_array_equal(selected['protein_points'][:, 0], [0,15,1e-10])


def test_mixed_null_missing_isolates_and_no_accidental_cache_alias():
    selected = sample([[0,0,0], [2,0,0]], [[4,0,0]])
    tensor, counts, _, bars, _ = f.compute(selected)
    assert tensor.shape == (55,100) and tensor.dtype == np.dtype('<f4')
    mixed = f.CHANNELS.index(('C','C'))
    only_p = f.CHANNELS.index(('C','Null'))
    missing_l = f.CHANNELS.index(('C','I'))
    only_l = f.CHANNELS.index(('Null','C'))
    np.testing.assert_array_equal(tensor[mixed, :20], np.full(20, 3))
    np.testing.assert_array_equal(tensor[mixed, 20:], np.full(80, 2))
    np.testing.assert_array_equal(tensor[only_p, :20], np.full(20, 2))
    np.testing.assert_array_equal(tensor[only_p, 20:], np.ones(80))
    np.testing.assert_array_equal(tensor[missing_l], np.full(100, 2))
    np.testing.assert_array_equal(tensor[only_l], np.ones(100))
    assert not tensor[-1].any() and not bars[-1].intervals
    assert counts[mixed].tolist() == [2,1]
    assert len(bars[mixed].intervals) == 3


def test_radius_not_pair_distance_and_exact_final_boundary():
    for distance, expected_death in ((8.,4.), (10.,5.), (10.2, math.inf)):
        tensor, _, _, bars, _ = f.compute(sample([[0,0,0]], [[distance,0,0]]))
        finite = [bar.death for bar in bars[0].intervals if math.isfinite(bar.death)]
        if math.isfinite(expected_death):
            assert finite == [expected_death]
            end = int(expected_death*20)
            np.testing.assert_array_equal(tensor[0, :end], np.full(end, 2))
            np.testing.assert_array_equal(tensor[0, end:], np.ones(100-end))
        else:
            assert finite == []
            np.testing.assert_array_equal(tensor[0], np.full(100, 2))


def test_obtuse_alpha_edge_inherits_coface_birth_before_h0_truncation():
    selected = sample([[-1,0,0],[0,.2,0]], [[1,0,0]])
    tensor, _, _, bars, _ = f.compute(selected)
    finite = sorted(bar.death for bar in bars[0].intervals if math.isfinite(bar.death))
    # Circumradius (1 + .2^2)/(2*.2) = 2.6, not the long edge's half-length 1.
    np.testing.assert_allclose(finite, [math.sqrt(1.04)/2, 2.6], rtol=1e-12)
    points, np_, cross, _ = f.channel_points(selected, 0)
    np.testing.assert_array_equal(tensor[0], direct_counts(points, np_, cross))


@pytest.mark.parametrize('seed', [7,18,101])
def test_all_55_channels_against_full_alpha_graph_component_oracle(seed):
    rng = np.random.default_rng(seed)
    selected = sample(rng.normal(size=(8,3)), rng.normal(size=(9,3)),
                      ['C','N','O','S']*2, ['C','N','O','S','H','F','C','C','N'])
    tensor, _, _, bars, _ = f.compute(selected)
    for channel in range(55):
        points, np_, cross, _ = f.channel_points(selected, channel)
        np.testing.assert_array_equal(tensor[channel], direct_counts(points, np_, cross))
    restored = f.unpack_barcodes(f.pack_barcodes(bars))
    assert [b.intervals for b in restored] == [b.intervals for b in bars]


def test_empty_protein_keeps_ligand_and_explicit_null_allows_its_edges():
    tensor, _, _, _, _ = f.compute(sample([], [[0,0,0],[2,0,0]]))
    np.testing.assert_array_equal(tensor[0], np.full(100,2))
    only_l = f.CHANNELS.index(('Null','C'))
    np.testing.assert_array_equal(tensor[only_l,:20], np.full(20,2))
    np.testing.assert_array_equal(tensor[only_l,20:], np.ones(80))


def test_frozen_recipe_and_invalid_barcode_storage():
    schema = f.schema()
    assert schema['feature_count'] == 5500
    assert schema['field'] == 'GF(2)' and schema['max_dimension'] == 0
    assert schema['builder_filtration_end_squared_radius'] == 25
    assert schema['hard_pair_distance_cutoff'] is None
    assert len(schema['bin_edges']) == 101 and schema['bin_edges'][-1] == 5
    arrays = {'offsets': np.zeros(56, dtype=int), 'births': np.array([0.]), 'deaths': np.array([6.])}
    with pytest.raises(ValueError, match='Invalid'):
        f.unpack_barcodes(arrays)
