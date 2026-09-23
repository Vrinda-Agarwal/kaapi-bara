import numpy as np

from cafe_reviews.explain.ppi import mean_ci, ppi_mean


def test_ppi_equals_unlabeled_mean_when_predictions_are_perfect():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200).astype(float)
    f_unl = rng.integers(0, 2, 5000).astype(float)
    theta, lo, hi = ppi_mean(f_unl, y.copy(), y)
    assert theta == f_unl.mean()
    assert lo < theta < hi


def test_ppi_corrects_systematic_undercount():
    rng = np.random.default_rng(1)
    truth_unl = rng.random(20000) < 0.30
    truth_lab = rng.random(2000) < 0.30
    # Extractor misses half of the positives.
    f_unl = (truth_unl & (rng.random(20000) < 0.5)).astype(float)
    f_lab = (truth_lab & (rng.random(2000) < 0.5)).astype(float)
    naive, _, _ = mean_ci(f_unl)
    theta, lo, hi = ppi_mean(f_unl, f_lab, truth_lab.astype(float))
    assert naive < 0.2
    assert lo < 0.30 < hi
