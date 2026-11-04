"""Weight-of-Evidence binning, Information Value, and the points-based scorecard (Model A).

Convention (matches optbinning): WoE = ln( %good / %bad ) per bin, so
    WoE > 0  -> bin is SAFER than average,   WoE < 0 -> bin is RISKIER than average.
A logistic regression of default on WoE features therefore has NEGATIVE coefficients.

optbinning is used only at FIT time to find the cut points. Scoring uses plain bin tables
(stored as JSON), so the Streamlit app needs no optbinning and the scorecard is fully auditable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
# NOTE: optbinning (and its heavy solver stack) is imported lazily inside fit_binning(), because it is
# only needed to FIT the bins. Scoring (and the deployed Streamlit app) needs only numpy/pandas.

# ---- scorecard scaling (industry convention) -------------------------------------------
BASE_SCORE = 600        # score at the base odds
BASE_ODDS = 50          # good:bad odds at the base score (50:1 -> PD ~ 1.96%)
PDO = 20                # points to double the good:bad odds
FACTOR = PDO / np.log(2)
OFFSET = BASE_SCORE - FACTOR * np.log(BASE_ODDS)

IV_MIN = 0.02           # keep features with IV > 0.02
CORR_MAX = 0.50         # drop a feature if |corr of WoE| with a higher-IV kept feature exceeds this (chosen on validation: see STUDY_GUIDE)


def iv_label(iv: float) -> str:
    """Standard rule-of-thumb for Information Value."""
    return ("useless" if iv < 0.02 else "weak" if iv < 0.1 else "medium" if iv < 0.3
            else "strong" if iv < 0.5 else "very strong (check for leakage)")


def _fmt(v: float) -> str:
    return f"{v:.4g}"


@dataclass
class FeatureBinning:
    name: str
    splits: list[float]          # ascending cut points; bin i = [splits[i-1], splits[i])
    woe: list[float]             # WoE per regular bin (len = len(splits) + 1)
    missing_woe: float           # WoE of the NaN bin (0 if no NaNs were seen in training)
    stats: list[dict] = field(default_factory=list)   # per-bin label/count/events/rate (incl. Missing)
    iv: float = 0.0

    def bin_index(self, x) -> np.ndarray:
        # optbinning bins are [a, b): a value equal to a cut point belongs to the UPPER bin
        return np.searchsorted(np.asarray(self.splits), np.asarray(x, dtype=float), side="right")

    def transform(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.asarray(self.woe)[self.bin_index(x)]
        return np.where(np.isnan(x), self.missing_woe, out)

    def labels(self) -> list[str]:
        s = self.splits
        if not s:
            return ["(-inf, inf)"]
        edges = ["-inf"] + [_fmt(v) for v in s] + ["inf"]
        return [("(" if i == 0 else "[") + f"{edges[i]}, {edges[i + 1]})" for i in range(len(edges) - 1)]


def fit_binning(x: pd.Series, y: pd.Series, name: str) -> FeatureBinning:
    """Optimal monotonic WoE binning for one numeric feature (fit on TRAIN only)."""
    from optbinning import OptimalBinning
    ob = OptimalBinning(
        name=name, dtype="numerical", solver="cp",
        monotonic_trend="auto_asc_desc",      # only ascending/descending: easy to explain to a regulator
        max_n_bins=8, min_bin_size=0.01,      # 1% so rare-but-powerful bins (e.g. 1+ times 90d late) survive
        min_prebin_size=0.001,
    )
    ob.fit(x.to_numpy(dtype=float), y.to_numpy())
    if ob.status not in ("OPTIMAL", "FEASIBLE"):
        raise RuntimeError(f"Binning of {name} failed with status {ob.status}")
    splits = [float(s) for s in ob.splits]
    n_bins = len(splits) + 1
    t = ob.binning_table.build()
    reg, miss = t.iloc[:n_bins], t[t["Bin"].astype(str) == "Missing"].iloc[0]
    fb = FeatureBinning(name=name, splits=splits, woe=[float(w) for w in reg["WoE"]],
                        missing_woe=float(miss["WoE"]))
    labels = fb.labels()
    for lab, (_, r) in zip(labels, reg.iterrows()):
        fb.stats.append(dict(bin=lab, count=int(r["Count"]), events=int(r["Event"]),
                             event_rate=float(r["Event rate"]), woe=float(r["WoE"]), iv=float(r["IV"])))
    fb.stats.append(dict(bin="Missing", count=int(miss["Count"]), events=int(miss["Event"]),
                         event_rate=float(miss["Event rate"]), woe=float(miss["WoE"]), iv=float(miss["IV"])))
    fb.iv = float(sum(s["iv"] for s in fb.stats))
    return fb


def fit_all_binnings(X: pd.DataFrame, y: pd.Series) -> dict[str, FeatureBinning]:
    return {c: fit_binning(X[c], y, c) for c in X.columns}


def woe_frame(X: pd.DataFrame, binnings: dict[str, FeatureBinning], cols: list[str] | None = None) -> pd.DataFrame:
    cols = cols or list(binnings)
    return pd.DataFrame({c: binnings[c].transform(X[c]) for c in cols}, index=X.index)


def select_features(binnings: dict[str, FeatureBinning], W: pd.DataFrame,
                    iv_min: float = IV_MIN, corr_max: float = CORR_MAX):
    """IV filter, then drop WoE-collinear features (keep the higher-IV one). Returns (kept, log)."""
    log, kept = [], []
    for name in sorted(binnings, key=lambda n: -binnings[n].iv):
        iv = binnings[name].iv
        if not iv > iv_min:
            log.append((name, iv, "dropped: IV <= %.2f" % iv_min))
            continue
        clash = [(k, abs(W[name].corr(W[k]))) for k in kept]
        clash = [(k, c) for k, c in clash if c > corr_max]
        if clash:
            k, c = max(clash, key=lambda t: t[1])
            log.append((name, iv, f"dropped: |corr|={c:.2f} with higher-IV '{k}'"))
            continue
        kept.append(name)
        log.append((name, iv, "kept"))
    return kept, log


@dataclass
class Scorecard:
    binnings: dict[str, FeatureBinning]
    features: list[str]
    coef: dict[str, float]
    intercept: float
    points: dict[str, list[int]]          # per feature, per regular bin
    missing_points: dict[str, int]
    meta: dict = field(default_factory=dict)

    # ---- fitting ---------------------------------------------------------------------
    @classmethod
    def fit(cls, X: pd.DataFrame, y: pd.Series, binnings: dict[str, FeatureBinning], features: list[str],
            C: float = 1.0) -> "Scorecard":
        W = woe_frame(X, binnings, features)
        feats = list(features)
        while True:   # logistic regression; WoE is "good-oriented" so every coefficient must be NEGATIVE
            lr = LogisticRegression(C=C, max_iter=1000).fit(W[feats], y)
            bad = [(f, c) for f, c in zip(feats, lr.coef_[0]) if c >= 0]
            if not bad:
                break
            worst = max(bad, key=lambda t: t[1])[0]
            feats.remove(worst)   # wrong-signed coefficient = multicollinearity artefact -> drop
        coef = dict(zip(feats, map(float, lr.coef_[0])))
        b0, n = float(lr.intercept_[0]), len(feats)
        points, missing_points = {}, {}
        for f in feats:
            b = binnings[f]
            # score = Offset + Factor*ln(odds_good), ln(odds_good) = -(b0 + sum b_i*WoE_i); spread intercept evenly
            pts = lambda woe: int(round(-(coef[f] * woe + b0 / n) * FACTOR + OFFSET / n))
            points[f] = [pts(w) for w in b.woe]
            missing_points[f] = pts(b.missing_woe)
        return cls(binnings, feats, coef, b0, points, missing_points,
                   meta=dict(base_score=BASE_SCORE, base_odds=BASE_ODDS, pdo=PDO, C=C,
                             factor=float(FACTOR), offset=float(OFFSET)))

    # ---- scoring ---------------------------------------------------------------------
    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        z = self.intercept + sum(self.coef[f] * self.binnings[f].transform(X[f]) for f in self.features)
        return 1.0 / (1.0 + np.exp(-z))

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Points earned per feature for each row (the 'reason codes' of the scorecard)."""
        cols = {}
        for f in self.features:
            idx = self.binnings[f].bin_index(X[f])
            pts = np.asarray(self.points[f])[idx]
            cols[f] = np.where(X[f].isna().to_numpy(), self.missing_points[f], pts)
        return pd.DataFrame(cols, index=X.index)

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return self.contributions(X).sum(axis=1).to_numpy()

    def breakdown(self, row: pd.DataFrame) -> pd.DataFrame:
        """Reason codes for ONE applicant: bin hit, points earned, best possible points, points lost (largest first)."""
        out = []
        for f in self.features:
            b, x = self.binnings[f], row[f].iloc[0]
            if pd.isna(x):
                label, pts = "Missing", self.missing_points[f]
                best = max(self.points[f] + [pts])
            else:
                i = int(b.bin_index([x])[0]); label, pts = b.labels()[i], self.points[f][i]
                best = max(self.points[f])
            out.append(dict(feature=f, value=x, bin=label, points=pts, best_points=best, points_lost=best - pts))
        return pd.DataFrame(out).sort_values("points_lost", ascending=False).reset_index(drop=True)

    def exact_score(self, X: pd.DataFrame) -> np.ndarray:
        """Un-rounded score straight from the model (used to measure rounding error)."""
        pd_ = self.predict_pd(X)
        return OFFSET + FACTOR * np.log((1 - pd_) / pd_)

    @staticmethod
    def pd_from_score(score) -> np.ndarray:
        """Invert the scaling: score -> good:bad odds -> PD."""
        odds_good = np.exp((np.asarray(score, dtype=float) - OFFSET) / FACTOR)
        return 1.0 / (1.0 + odds_good)

    # ---- reporting -------------------------------------------------------------------
    def table(self) -> pd.DataFrame:
        rows = []
        for f in self.features:
            b = self.binnings[f]
            pts = self.points[f] + [self.missing_points[f]]
            for s, p in zip(b.stats, pts):
                if s["bin"] == "Missing" and s["count"] == 0:
                    continue
                rows.append(dict(feature=f, bin=s["bin"], count=s["count"],
                                 events=s["events"], event_rate=s["event_rate"], woe=s["woe"],
                                 feature_iv=b.iv, coefficient=self.coef[f], points=p))
        df = pd.DataFrame(rows)
        df["pct"] = df["count"] / df.groupby("feature")["count"].transform("sum")
        return df

    # ---- persistence -----------------------------------------------------------------
    def to_json(self, path: Path | str) -> None:
        payload = dict(
            meta=self.meta, features=self.features, coef=self.coef, intercept=self.intercept,
            points=self.points, missing_points=self.missing_points,
            binnings={k: dict(name=v.name, splits=v.splits, woe=v.woe, missing_woe=v.missing_woe,
                              stats=v.stats, iv=v.iv) for k, v in self.binnings.items()})
        Path(path).write_text(json.dumps(payload, indent=1))

    @classmethod
    def from_json(cls, path: Path | str) -> "Scorecard":
        p = json.loads(Path(path).read_text())
        binnings = {k: FeatureBinning(**v) for k, v in p["binnings"].items()}
        return cls(binnings, p["features"], p["coef"], p["intercept"], p["points"], p["missing_points"], p["meta"])

