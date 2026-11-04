import numpy as np
import pandas as pd
import pytest

from src.explain import explain, global_importance, sigmoid, top_drivers, waterfall_figure
from src.features import FEATURE_COLUMNS, TARGET
from src.train import fit_lgbm


@pytest.fixture(scope="module")
def setup():
    rng = np.random.default_rng(0)
    n = 5000
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    X.iloc[::9, 2] = np.nan
    p = 1 / (1 + np.exp(-(-2.5 + 1.5 * X.iloc[:, 0] - 0.8 * X.iloc[:, 1])))
    X[TARGET] = (rng.random(n) < p).astype(int)
    tr, va = X.iloc[:3500], X.iloc[3500:]
    booster, _ = fit_lgbm(tr, va, dict(num_leaves=8, min_data_in_leaf=50))
    return booster, va.iloc[:300]


def test_shap_is_exactly_additive_and_matches_pd(setup):
    b, X = setup
    e = explain(b, X)
    logodds = e.base_values + e.values.sum(axis=1)
    np.testing.assert_allclose(logodds, b.predict(X[FEATURE_COLUMNS], raw_score=True), atol=1e-8)
    np.testing.assert_allclose(sigmoid(logodds), b.predict(X[FEATURE_COLUMNS]), atol=1e-8)


def test_the_two_true_drivers_rank_top(setup):
    b, X = setup
    top2 = set(global_importance(explain(b, X)).index[:2])
    from src.explain import NICE
    assert top2 == {NICE[FEATURE_COLUMNS[0]], NICE[FEATURE_COLUMNS[1]]}


def test_top_drivers_sorted_and_directions(setup):
    b, X = setup
    d = top_drivers(explain(b, X), 0, k=5)
    assert len(d) == 5 and (d.shap_log_odds.abs().diff().dropna() <= 1e-12).all()
    assert ((d.shap_log_odds > 0) == (d.direction == "raises risk")).all()


def test_waterfall_returns_figure_for_single_row(setup):
    import matplotlib.figure
    b, X = setup
    fig = waterfall_figure(explain(b, X.iloc[[0]]), 0)
    assert isinstance(fig, matplotlib.figure.Figure)
