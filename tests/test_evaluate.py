import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.evaluate import (CUTOFF_DECLINE, CUTOFF_REVIEW, LGD, MARGIN, auc, bootstrap_auc, calibration_table,
                          cutoff_metrics, decide, gini, ks)


def test_perfect_and_random_ranking():
    y = np.array([0, 0, 0, 1, 1]); p = np.array([.1, .2, .3, .8, .9])
    assert auc(y, p) == 1 and gini(y, p) == 1 and ks(y, p) == 1
    rng = np.random.default_rng(0); y = (rng.random(50_000) < .07).astype(int); p = rng.random(50_000)
    assert abs(auc(y, p) - .5) < .02 and abs(gini(y, p)) < .04 and ks(y, p) < .03


def test_gini_is_2auc_minus_1():
    rng = np.random.default_rng(1); y = (rng.random(5000) < .1).astype(int); p = rng.random(5000) + y * .3
    assert gini(y, p) == pytest.approx(2 * roc_auc_score(y, p) - 1)


def test_ks_known_value():
    # goods all score .1/.2, bads all score .8/.9 except one bad at .15 -> hand-computed KS
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1]); p = np.array([.1, .1, .2, .2, .15, .8, .8, .9])
    assert ks(y, p) == pytest.approx(0.75)


def test_cutoff_metrics_counts():
    y = np.array([0, 0, 0, 0, 1, 1]); p = np.array([.01, .02, .11, .2, .05, .5])
    m = cutoff_metrics(y, p, 0.10)
    assert (m["TN"], m["FP"], m["FN"], m["TP"]) == (2, 2, 1, 1)
    assert m["decline_rate"] == pytest.approx(3 / 6) and m["defaults_caught"] == 0.5 and m["precision"] == pytest.approx(1 / 3)
    assert m["bad_rate_approved"] == pytest.approx(1 / 3)
    assert m["cost_pct_exposure"] == pytest.approx((1 * LGD + 2 * MARGIN) / 6)
    assert m["cutoff"] == 0.10


def test_policy_constants_and_decide():
    assert CUTOFF_DECLINE == pytest.approx(0.10) and CUTOFF_REVIEW == 0.05
    # decline exactly where expected loss of approving equals expected margin lost by declining
    assert CUTOFF_DECLINE * LGD == pytest.approx((1 - CUTOFF_DECLINE) * MARGIN)
    assert list(decide([0.01, 0.0499, 0.05, 0.0999, 0.10, 0.9])) == ["Approve", "Approve", "Review", "Review", "Decline", "Decline"]


def test_calibration_table_perfectly_calibrated_data():
    rng = np.random.default_rng(2); p = rng.uniform(0.01, 0.3, 100_000); y = (rng.random(p.size) < p).astype(int)
    t = calibration_table(y, p)
    assert len(t) == 10 and (t.mean_pd - t.actual).abs().max() < 0.01 and t.n.sum() == p.size


def test_bootstrap_ci_contains_point_estimate_and_diff_sign():
    rng = np.random.default_rng(3); y = (rng.random(4000) < .1).astype(int)
    pa = rng.random(4000) + y * .5; pb = rng.random(4000) + y * 1.0
    ci = bootstrap_auc(y, pa, pb, n=200)
    assert ci["A"][0] < auc(y, pa) < ci["A"][1] and ci["B-A"][0] > 0
