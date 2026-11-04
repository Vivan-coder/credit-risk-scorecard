import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_COLUMNS, TARGET
from src.train import MODELS, fit_lgbm, load_lgbm


@pytest.fixture(scope="module")
def frames():
    rng = np.random.default_rng(0)
    n = 6000
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    X.iloc[::7, 3] = np.nan                                            # NaNs must be accepted natively
    p = 1 / (1 + np.exp(-(-2.8 + 1.2 * X.iloc[:, 0].fillna(0))))
    X[TARGET] = (rng.random(n) < p).astype(int)
    return X.iloc[:4000], X.iloc[4000:]


def test_fit_lgbm_learns_and_outputs_probabilities(frames):
    tr, va = frames
    b, auc = fit_lgbm(tr, va, dict(num_leaves=8, min_data_in_leaf=50))
    p = b.predict(va[FEATURE_COLUMNS], num_iteration=b.best_iteration)
    assert auc > 0.7 and ((p > 0) & (p < 1)).all()
    assert abs(p.mean() - va[TARGET].mean()) < 0.02                    # unweighted => calibrated on average


def test_fit_is_deterministic(frames):
    tr, va = frames
    p1 = fit_lgbm(tr, va, dict(num_leaves=8))[0].predict(va[FEATURE_COLUMNS])
    p2 = fit_lgbm(tr, va, dict(num_leaves=8))[0].predict(va[FEATURE_COLUMNS])
    np.testing.assert_allclose(p1, p2)


def test_saved_model_roundtrip_and_calibration():
    if not (MODELS / "lgbm.txt").exists():
        pytest.skip("model not trained yet")
    d = pd.read_parquet(MODELS.parent / "data" / "processed.parquet")
    va = d[d.split == "validation"]
    b = load_lgbm()
    assert b.feature_name() == FEATURE_COLUMNS
    p = b.predict(va[FEATURE_COLUMNS])
    assert abs(p.mean() - va[TARGET].mean()) < 0.005
