"""Cleaning + feature engineering for the Give Me Some Credit dataset.

Design rules (all deliberate, see STUDY_GUIDE.md):
* Every transformation is ROW-WISE and uses FIXED, domain-based constants. Nothing is
  "fitted" on the data here (no medians, no quantiles), so there is no train/test leakage
  and a single applicant typed into the Streamlit form is processed exactly like the
  150k training rows.
* Missing values are kept as NaN (not imputed). optbinning gives NaN its own WoE bin and
  LightGBM handles NaN natively. Informative missingness also gets an explicit flag.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw_gmsc.csv"
PROCESSED_PATH = ROOT / "data" / "processed.parquet"
SAMPLE_PATH = ROOT / "data" / "sample.csv"

TARGET = "default"
SEED = 42

# Original column name -> readable snake_case name
RENAME = {
    "FinancialDistressNextTwoYears": TARGET,
    "RevolvingUtilizationOfUnsecuredLines": "utilization",
    "age": "age",
    "NumberOfTime30-59DaysPastDueNotWorse": "late_30_59",
    "DebtRatio": "debt_ratio",
    "MonthlyIncome": "monthly_income",
    "NumberOfOpenCreditLinesAndLoans": "open_credit_lines",
    "NumberOfTimes90DaysLate": "late_90_plus",
    "NumberRealEstateLoansOrLines": "real_estate_loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "late_60_89",
    "NumberOfDependents": "dependents",
}
RAW_FEATURES = [c for c in RENAME.values() if c != TARGET]
LATE_COLS = ["late_30_59", "late_60_89", "late_90_plus"]

# Fixed caps (domain judgement, not data-fitted quantiles)
PLACEHOLDER_MIN = 96        # 96 and 98 are placeholder codes in the late-payment columns
MIN_AGE = 18                # cannot lend to minors; age 0 is a data error
UTILIZATION_CAP = 10.0      # 1000% utilisation; beyond that is a data error (max raw = 50,708)
DEBT_RATIO_CAP = 10.0
INCOME_CAP = 100_000.0      # monthly; raw max is 3,008,750

FEATURE_COLUMNS = [
    "utilization", "over_limit", "age",
    "late_30_59", "late_60_89", "late_90_plus", "total_late", "any_late", "late_code_flag",
    "debt_ratio", "monthly_income", "income_missing", "income_per_dependent",
    "dependents", "dependents_missing",
    "open_credit_lines", "real_estate_loans", "real_estate_share",
]


def load_raw(path: Path | str = RAW_PATH) -> pd.DataFrame:
    """Read the raw CSV, rename columns, and encode the target as 0/1."""
    df = pd.read_csv(path).rename(columns=RENAME)
    if not pd.api.types.is_numeric_dtype(df[TARGET]):
        df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw (snake_case) applicant columns -> model features. Works for 1 row or 150k.

    Input must contain RAW_FEATURES. Any `default` column is carried through untouched.
    """
    missing = [c for c in RAW_FEATURES if c not in raw.columns]
    if missing:
        raise KeyError(f"Missing raw columns: {missing}")
    df = raw.copy()
    out = pd.DataFrame(index=df.index)

    # --- late-payment counts: 96/98 are placeholder codes, not real counts ---
    code_rows = (df[LATE_COLS] >= PLACEHOLDER_MIN).any(axis=1)
    out["late_code_flag"] = code_rows.astype(int)
    for c in LATE_COLS:
        out[c] = df[c].where(~code_rows)            # NaN for placeholder rows
    out["total_late"] = out[LATE_COLS].sum(axis=1, min_count=len(LATE_COLS))
    out["any_late"] = (out["total_late"] > 0).astype(float).where(out["total_late"].notna())

    # --- utilisation ---
    util = df["utilization"].clip(lower=0, upper=UTILIZATION_CAP)
    out["utilization"] = util
    out["over_limit"] = (util > 1).astype(int)

    # --- age (impossible ages -> NaN) ---
    out["age"] = df["age"].where(df["age"] >= MIN_AGE)

    # --- income and debt ratio ---
    income = df["monthly_income"]
    out["income_missing"] = income.isna().astype(int)
    out["monthly_income"] = income.clip(lower=0, upper=INCOME_CAP)
    # When income is missing, the dataset's DebtRatio is a raw dollar amount (median 1,159),
    # NOT a ratio. Treat it as unknown rather than feeding the model a fake ratio.
    out["debt_ratio"] = df["debt_ratio"].where(income.notna()).clip(lower=0, upper=DEBT_RATIO_CAP)

    # --- dependents ---
    out["dependents_missing"] = df["dependents"].isna().astype(int)
    out["dependents"] = df["dependents"].clip(upper=10)
    out["income_per_dependent"] = out["monthly_income"] / (out["dependents"] + 1)

    # --- credit lines ---
    out["open_credit_lines"] = df["open_credit_lines"]
    out["real_estate_loans"] = df["real_estate_loans"]
    lines = df["open_credit_lines"]
    out["real_estate_share"] = (df["real_estate_loans"] / lines.where(lines > 0)).fillna(0).clip(upper=1)

    out = out[FEATURE_COLUMNS]
    if TARGET in df.columns:
        out[TARGET] = df[TARGET].values
    return out


def split_data(feats: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Add a `split` column: 60% train / 20% validation / 20% test, stratified on default.

    Duplicates are NOT dropped: the 609 duplicate rows are mostly thin-file borrowers with
    identical (mostly zero) profiles, i.e. different people, not data-entry errors.
    """
    idx = feats.index
    train_idx, rest_idx = train_test_split(idx, test_size=0.4, stratify=feats[TARGET], random_state=seed)
    val_idx, test_idx = train_test_split(rest_idx, test_size=0.5, stratify=feats.loc[rest_idx, TARGET], random_state=seed)
    out = feats.copy()
    out["split"] = "train"
    out.loc[val_idx, "split"] = "validation"
    out.loc[test_idx, "split"] = "test"
    return out


def make_sample(raw: pd.DataFrame, n: int = 2000, seed: int = SEED) -> pd.DataFrame:
    """Small stratified sample in RAW schema, safe to commit and used by the Streamlit demo."""
    frac = n / len(raw)
    return raw.groupby(TARGET, group_keys=False).sample(frac=frac, random_state=seed).reset_index(drop=True)


def main() -> None:
    raw = load_raw()
    feats = split_data(build_features(raw))
    feats.to_parquet(PROCESSED_PATH)
    # demo sample is drawn from the TEST split only, so the app's portfolio numbers are out-of-sample
    make_sample(raw.loc[feats.index[feats["split"] == "test"]]).to_csv(SAMPLE_PATH, index=False)
    print(f"raw {raw.shape} -> processed {feats.shape} saved to {PROCESSED_PATH.name}")
    print(feats.groupby("split")[TARGET].agg(["size", "mean"]).round(4))
    print(f"sample.csv: {SAMPLE_PATH.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()

