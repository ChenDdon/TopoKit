"""Independent D-A deletion oracle on real molecular feature slices."""
from pathlib import Path
import time
import numpy as np
import loo_features as f
import study_common as c


def direct_summary(points, protein_count, scale):
    # No TopoKit core, spectrum helper, component splitting, twin or cache reuse.
    delta = points[:, None, :] - points[None, :, :]
    adjacency = np.sqrt(np.sum(delta*delta, axis=2)) <= scale
    np.fill_diagonal(adjacency, False)
    values = []
    for removed in range(len(points)):
        keep = np.arange(len(points)) != removed
        induced = adjacency[np.ix_(keep, keep)].astype(float)
        laplacian = np.diag(induced.sum(axis=1)) - induced
        spectrum = np.linalg.eigvalsh(laplacian)
        if np.any(spectrum < -1e-10):
            raise ValueError('Oracle encountered non-PSD spectrum')
        positive = spectrum[spectrum > 1e-10]
        stats = np.zeros(9)
        stats[8] = np.count_nonzero(np.abs(spectrum) <= 1e-10)
        if positive.size:
            stats[:8] = (positive.sum(), positive.mean(), np.median(positive),
                          positive.std(ddof=0), positive.var(ddof=0), positive.max(),
                          positive.min(), np.sum(positive**2))
        values.append(stats)
    result = np.zeros((9, 4))
    values = np.asarray(values)
    for side, part in enumerate((values[:protein_count], values[protein_count:])):
        if len(part):
            result[:, side*2] = part.sum(axis=0)
            result[:, side*2+1] = part.mean(axis=0)
    return result


def audit_real(output, receipt, union):
    started = time.monotonic()
    source = Path(receipt['source_dataset'])
    checked = []
    for sample in ('1a30', '1w8l', '10gs'):
        row = union[sample]
        tensor, record = c.load_feature(sample, output, receipt, c.input_hashes(row, source))
        selected = f.select_atoms(source / row['protein_file'], source / row['ligand_mol2_file'])
        for channel in (0, 9, 16, 23, 39):
            p, l = f.CHANNELS[channel]
            pp = selected['protein_points'][selected['protein_elements'] == p]
            lp = selected['ligand_points'][selected['ligand_elements'] == l]
            points = np.vstack((pp, lp))
            assert record['channel_atom_counts'][channel] == [len(pp), len(lp)]
            for k in (0, 4, 8):
                reference = direct_summary(points, len(pp), f.SCALES[k])
                np.testing.assert_allclose(tensor[channel, k], reference, rtol=2e-6, atol=2e-4)
                np.testing.assert_array_equal(tensor[channel, k, 8, [0, 2]], reference[8, [0, 2]])
                checked.append({'sample': sample, 'channel': channel, 'scale': float(f.SCALES[k]),
                                'deletion_count': len(points)})
    return {'passed': True, 'method': 'independent full induced adjacency D-A and numpy eigvalsh',
            'checked_slices': checked, 'slice_count': len(checked), 'seconds': time.monotonic()-started}
