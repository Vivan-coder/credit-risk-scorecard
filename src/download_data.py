"""Download the Give Me Some Credit dataset from OpenML (public, no login, no competition rules) to data/raw_gmsc.csv.

    python -m src.download_data

The raw file is git-ignored (never commit raw data); only the 2,000-row data/sample.csv is committed.
"""
from __future__ import annotations

from sklearn.datasets import fetch_openml

from src.features import RAW_PATH

OPENML_ID = 46929   # "GiveMeSomeCredit": 150,000 rows, 10 features + FinancialDistressNextTwoYears


def main() -> None:
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = fetch_openml(data_id=OPENML_ID, as_frame=True, parser="auto").frame
    assert df.shape == (150_000, 11), f"unexpected shape {df.shape}"
    df.to_csv(RAW_PATH, index=False)
    print(f"saved {df.shape[0]:,} rows x {df.shape[1]} columns to {RAW_PATH}")


if __name__ == "__main__":
    main()
