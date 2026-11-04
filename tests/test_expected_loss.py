import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.evaluate import decide
from src.expected_loss import (BAND_DECISION, EAD_MAX_USD, EAD_MIN_USD, LGD, MEDIAN_INCOME_USD, USD_TO_BHD, band_table,
                               basel_retail_correlation, capital_rate, estimate_ead_bhd, expected_loss, portfolio_summary,
                               risk_band, tradeoff_table)


def test_el_formula_hand_example():
    assert expected_loss(0.05, 10_000) == pytest.approx(0.05 * 0.45 * 10_000) == pytest.approx(225.0)


def test_ead_floor_cap_missing_and_fx():
    e = estimate_ead_bhd([100, 5400, 1_000_000, np.nan])
    assert e[0] == pytest.approx(EAD_MIN_USD * USD_TO_BHD)                 # floored
    assert e[1] == pytest.approx(6 * 5400 * USD_TO_BHD)                    # 6x monthly income, in BHD
    assert e[2] == pytest.approx(EAD_MAX_USD * USD_TO_BHD)                 # capped
    assert e[3] == pytest.approx(6 * MEDIAN_INCOME_USD * USD_TO_BHD)       # NaN -> median


def test_band_edges():
    labels = [str(b)[0] for b in risk_band([0.0, 0.0199, 0.02, 0.0499, 0.05, 0.0999, 0.10, 0.1999, 0.20, 0.99, 1.0])]
    assert labels == list("AABBCCDDEEE")


def test_bands_agree_with_decision_policy():
    grid = np.linspace(0, 1, 2001)
    assert [BAND_DECISION[b] for b in risk_band(grid)] == list(decide(grid))


def test_band_table_totals_reconcile():
    rng = np.random.default_rng(0)
    p = rng.beta(1, 12, 5000); y = (rng.random(5000) < p).astype(int); ead = rng.uniform(1000, 15000, 5000)
    t = band_table(p, y, ead)
    assert t.borrowers.sum() == 5000 and t.expected_loss_bhd.sum() == pytest.approx(expected_loss(p, ead).sum())
    assert t.share_of_ead.sum() == pytest.approx(1) and t.share_of_el.sum() == pytest.approx(1)
    s = portfolio_summary(p, y, ead)
    assert s["expected_loss_bhd"] == pytest.approx(t.expected_loss_bhd.sum()) and s["unexpected_loss_999_bhd"] > s["expected_loss_bhd"] * 0.5


def test_backtest_expected_close_to_realised_when_pd_is_true():
    rng = np.random.default_rng(1)
    p = rng.beta(1, 12, 200_000); y = (rng.random(p.size) < p).astype(int); ead = np.full(p.size, 10_000.0)
    s = portfolio_summary(p, y, ead)
    assert s["expected_loss_bhd"] == pytest.approx(s["realised_loss_bhd"], rel=0.03)


def test_basel_correlation_endpoints_and_capital_hand_calc():
    assert basel_retail_correlation(0.0) == pytest.approx(0.16) and basel_retail_correlation(1.0) == pytest.approx(0.03)
    pdv, r = 0.02, float(basel_retail_correlation(0.02))
    expected = LGD * (norm.cdf((norm.ppf(pdv) + np.sqrt(r) * norm.ppf(0.999)) / np.sqrt(1 - r)) - pdv)
    assert float(capital_rate(pdv)) == pytest.approx(expected) and 0 < expected < LGD
    assert (capital_rate([0.001, 0.01, 0.05, 0.2]) > 0).all()


def test_tradeoff_monotone_and_endpoints():
    rng = np.random.default_rng(2)
    p = rng.beta(1, 12, 20_000); y = (rng.random(p.size) < p).astype(int); ead = np.full(p.size, 1.0)
    t = tradeoff_table(p, y, ead)
    assert t.approval_rate.is_monotonic_increasing
    assert t.default_rate_approved.dropna().iloc[0] < t.default_rate_approved.dropna().iloc[-1]    # stricter cutoff -> safer book
    assert tradeoff_table(p, y, ead, cutoffs=[1.0]).approval_rate.iloc[0] == 1.0
