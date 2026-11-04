import numpy as np
import pandas as pd
import pytest
from optbinning import OptimalBinning

from src.woe import BASE_ODDS, BASE_SCORE, FACTOR, PDO, FeatureBinning, Scorecard, fit_binning, iv_label


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(1)
    n = 20_000
    x1 = rng.gamma(2, 1, n)
    x2 = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(-3 + 0.9 * x1 - 0.5 * x2)))
    y = pd.Series((rng.random(n) < p).astype(int))
    X = pd.DataFrame({"x1": x1, "x2": x2})
    X.loc[rng.random(n) < 0.1, "x1"] = np.nan
    return X, y


def test_transform_matches_optbinning_including_cutpoints_and_nan(data):
    X, y = data
    fb = fit_binning(X["x1"], y, "x1")
    ob = OptimalBinning(name="x1", dtype="numerical", solver="cp", monotonic_trend="auto_asc_desc",
                        max_n_bins=8, min_bin_size=0.01, min_prebin_size=0.001).fit(X["x1"].to_numpy(), y.to_numpy())
    probe = np.r_[X["x1"].to_numpy()[:2000], fb.splits, np.nextafter(fb.splits, 1e9), np.nan]
    ref = ob.transform(probe, metric="woe", metric_missing="empirical")
    np.testing.assert_allclose(fb.transform(probe), ref, atol=1e-9)


def test_woe_sign_positive_means_safer(data):
    X, y = data
    fb = fit_binning(X["x2"], y, "x2")
    rates = [s["event_rate"] for s in fb.stats if s["bin"] != "Missing"]
    woes = [s["woe"] for s in fb.stats if s["bin"] != "Missing"]
    assert np.corrcoef(rates, woes)[0, 1] < -0.9          # higher default rate -> lower WoE


def test_iv_is_sum_of_bin_iv_and_nonnegative(data):
    X, y = data
    fb = fit_binning(X["x1"], y, "x1")
    assert fb.iv == pytest.approx(sum(s["iv"] for s in fb.stats)) and fb.iv > 0


def test_iv_label_thresholds():
    assert iv_label(0.01) == "useless" and iv_label(0.05) == "weak" and iv_label(0.2) == "medium" and iv_label(0.4) == "strong"


@pytest.fixture(scope="module")
def card(data):
    X, y = data
    B = {c: fit_binning(X[c], y, c) for c in X.columns}
    return Scorecard.fit(X, y, B, list(X.columns)), X, y


def test_coefficients_negative(card):
    sc, _, _ = card
    assert all(c < 0 for c in sc.coef.values())


def test_scaling_base_score_and_pdo():
    pd_at_base = Scorecard.pd_from_score(BASE_SCORE)
    assert pd_at_base == pytest.approx(1 / (1 + BASE_ODDS))              # 600 pts <-> 50:1 odds
    odds = lambda s: (1 - Scorecard.pd_from_score(s)) / Scorecard.pd_from_score(s)
    assert odds(BASE_SCORE + PDO) / odds(BASE_SCORE) == pytest.approx(2)  # +20 pts doubles good:bad odds


def test_points_table_matches_exact_score_within_rounding(card):
    sc, X, _ = card
    err = np.abs(sc.score(X) - sc.exact_score(X))
    assert err.max() <= 0.5 * len(sc.features) + 1e-6


def test_higher_score_means_lower_pd(card):
    sc, X, _ = card
    assert np.corrcoef(sc.score(X), sc.predict_pd(X))[0, 1] < -0.95


def test_json_roundtrip_identical_predictions(card, tmp_path):
    sc, X, _ = card
    sc.to_json(tmp_path / "s.json")
    sc2 = Scorecard.from_json(tmp_path / "s.json")
    np.testing.assert_allclose(sc.predict_pd(X), sc2.predict_pd(X))
    np.testing.assert_array_equal(sc.score(X), sc2.score(X))


def test_nan_row_scores_without_error(card):
    sc, _, _ = card
    r = pd.DataFrame({"x1": [np.nan], "x2": [np.nan]})
    assert np.isfinite(sc.score(r)).all() and np.isfinite(sc.predict_pd(r)).all()


def test_contributions_sum_to_score(card):
    sc, X, _ = card
    np.testing.assert_array_equal(sc.contributions(X).sum(axis=1).to_numpy(), sc.score(X))


def test_breakdown_reason_codes(card):
    sc, X, _ = card
    row = X.iloc[[0]]
    b = sc.breakdown(row)
    assert b.points.sum() == sc.score(row)[0]
    assert (b.points_lost >= 0).all() and b.points_lost.is_monotonic_decreasing
    assert set(b.feature) == set(sc.features)
    nan_row = pd.DataFrame({"x1": [np.nan], "x2": [0.0]})
    assert sc.breakdown(nan_row).set_index("feature").loc["x1", "bin"] == "Missing"
