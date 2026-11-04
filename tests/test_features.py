import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_COLUMNS, RAW_FEATURES, TARGET, build_features, split_data


def row(**kw):
    base = dict(utilization=0.3, age=40, late_30_59=0, debt_ratio=0.4, monthly_income=5000.0,
                open_credit_lines=5, late_90_plus=0, real_estate_loans=1, late_60_89=0, dependents=2.0)
    base.update(kw)
    return pd.DataFrame([base])


def test_placeholder_codes_become_nan_and_flagged():
    f = build_features(row(late_30_59=98, late_60_89=98, late_90_plus=98))
    assert f.loc[0, "late_code_flag"] == 1
    assert f[["late_30_59", "late_60_89", "late_90_plus", "total_late"]].isna().all(axis=None)


def test_normal_lates_untouched():
    f = build_features(row(late_30_59=1, late_60_89=2, late_90_plus=3))
    assert f.loc[0, "total_late"] == 6 and f.loc[0, "any_late"] == 1 and f.loc[0, "late_code_flag"] == 0


def test_missing_income_makes_debt_ratio_nan():
    f = build_features(row(monthly_income=np.nan, debt_ratio=1159.0))
    assert f.loc[0, "income_missing"] == 1 and np.isnan(f.loc[0, "debt_ratio"])


def test_caps_and_age():
    f = build_features(row(utilization=50708, monthly_income=3_000_000, age=0, debt_ratio=329664))
    assert f.loc[0, "utilization"] == 10 and f.loc[0, "monthly_income"] == 100_000
    assert np.isnan(f.loc[0, "age"]) and f.loc[0, "debt_ratio"] == 10


def test_over_limit_and_ratios():
    f = build_features(row(utilization=1.5, monthly_income=6000, dependents=2, real_estate_loans=2, open_credit_lines=4))
    assert f.loc[0, "over_limit"] == 1
    assert f.loc[0, "income_per_dependent"] == 2000
    assert f.loc[0, "real_estate_share"] == 0.5


def test_zero_open_lines_no_divide_by_zero():
    f = build_features(row(open_credit_lines=0, real_estate_loans=0))
    assert f.loc[0, "real_estate_share"] == 0


def test_missing_dependents_flagged():
    f = build_features(row(dependents=np.nan))
    assert f.loc[0, "dependents_missing"] == 1 and np.isnan(f.loc[0, "income_per_dependent"])


def test_output_columns_and_missing_input_error():
    f = build_features(row())
    assert list(f.columns) == FEATURE_COLUMNS
    with pytest.raises(KeyError):
        build_features(row().drop(columns=["age"]))


def test_row_wise_independence():
    """A row's features must not depend on the other rows (no leakage / no fitted stats)."""
    many = pd.concat([row(age=a, utilization=u) for a, u in [(25, .1), (60, 2.0), (33, 9.0)]], ignore_index=True)
    together = build_features(many)
    for i in range(len(many)):
        alone = build_features(many.iloc[[i]])
        pd.testing.assert_frame_equal(together.iloc[[i]].reset_index(drop=True), alone.reset_index(drop=True))


def test_split_is_stratified_disjoint_and_complete():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=10_000), TARGET: (rng.random(10_000) < 0.07).astype(int)})
    s = split_data(df)
    assert s["split"].value_counts(normalize=True).round(2).to_dict() == {"train": 0.6, "validation": 0.2, "test": 0.2}
    rates = s.groupby("split")[TARGET].mean()
    assert rates.max() - rates.min() < 0.003
    assert s.index.is_unique and len(s) == len(df)


def test_load_raw_encodes_target_as_int(tmp_path):
    from src.features import load_raw
    p = tmp_path / "r.csv"
    pd.DataFrame({"FinancialDistressNextTwoYears": ["No", "Yes", "No"],
                  "RevolvingUtilizationOfUnsecuredLines": [.1, .2, .3], "age": [30, 40, 50],
                  "NumberOfTime30-59DaysPastDueNotWorse": [0, 0, 0], "DebtRatio": [.1, .2, .3],
                  "MonthlyIncome": [1, 2, 3], "NumberOfOpenCreditLinesAndLoans": [1, 2, 3],
                  "NumberOfTimes90DaysLate": [0, 0, 0], "NumberRealEstateLoansOrLines": [0, 0, 0],
                  "NumberOfTime60-89DaysPastDueNotWorse": [0, 0, 0], "NumberOfDependents": [0, 1, 2]}).to_csv(p, index=False)
    df = load_raw(p)
    assert df[TARGET].tolist() == [0, 1, 0] and pd.api.types.is_integer_dtype(df[TARGET])
