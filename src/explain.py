"""SHAP explainability for Model B (LightGBM). Reused by the Streamlit app.

SHAP values are in LOG-ODDS space and are exactly additive:
    log-odds(applicant) = base_value + sum_i SHAP_i        PD = sigmoid(log-odds)
base_value is the AVERAGE LOG-ODDS over the training data (not the average PD, because sigmoid is non-linear).
Positive SHAP = pushes this applicant towards DEFAULT (riskier); negative = pushes towards repayment.
"""
from __future__ import annotations

import threading

import matplotlib
import numpy as np
import pandas as pd
import shap

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.features import FEATURE_COLUMNS  # noqa: E402

_PLOT_LOCK = threading.RLock()   # matplotlib/shap plotting use global state; Streamlit serves sessions in threads

# Plain-English names used in plots and the app
NICE = {
    "utilization": "Credit-line utilisation", "over_limit": "Over credit limit", "age": "Age",
    "late_30_59": "# times 30-59d late", "late_60_89": "# times 60-89d late", "late_90_plus": "# times 90+d late",
    "total_late": "Total # late payments", "any_late": "Any late payment", "late_code_flag": "Late-count placeholder flag",
    "debt_ratio": "Debt ratio", "monthly_income": "Monthly income", "income_missing": "Income missing",
    "income_per_dependent": "Income per dependent", "dependents": "# dependents", "dependents_missing": "Dependents missing",
    "open_credit_lines": "# open credit lines", "real_estate_loans": "# real-estate loans", "real_estate_share": "Real-estate share of credit",
}


def get_explainer(booster) -> shap.TreeExplainer:
    return shap.TreeExplainer(booster)


def explain(booster, X: pd.DataFrame, explainer: shap.TreeExplainer | None = None) -> shap.Explanation:
    """SHAP Explanation (values in log-odds) for the rows of X."""
    explainer = explainer or get_explainer(booster)
    sv = explainer.shap_values(X[FEATURE_COLUMNS])
    sv = sv[1] if isinstance(sv, list) else sv          # older shap versions return one array per class
    base = float(np.ravel(explainer.expected_value)[-1])
    return shap.Explanation(values=sv, base_values=np.full(len(X), base), data=X[FEATURE_COLUMNS].to_numpy(dtype=float),
                            feature_names=[NICE.get(c, c) for c in FEATURE_COLUMNS])


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def waterfall_figure(expl: shap.Explanation, i: int = 0, max_display: int = 9, title: str | None = None):
    """Waterfall for one applicant: starts at the baseline log-odds and shows each feature's push to the final log-odds."""
    with _PLOT_LOCK:
        plt.close("all")
        shap.plots.waterfall(expl[i], max_display=max_display, show=False)
        fig = plt.gcf(); fig.set_size_inches(8.5, 4.8)
        pd_ = float(sigmoid(expl.base_values[i] + expl.values[i].sum()))
        fig.suptitle(title or f"Why this applicant has PD = {pd_:.1%}  (x-axis: log-odds of default)", y=1.02, fontsize=11)
        plt.close(fig)   # detach from pyplot's global registry; the caller keeps the Figure object
    return fig


def top_drivers(expl: shap.Explanation, i: int = 0, k: int = 5) -> pd.DataFrame:
    """The k features that moved this applicant's risk the most, as a tidy table."""
    d = pd.DataFrame({"feature": expl.feature_names, "value": expl.data[i], "shap_log_odds": expl.values[i]})
    d["direction"] = np.where(d.shap_log_odds > 0, "raises risk", "lowers risk")
    return d.reindex(d.shap_log_odds.abs().sort_values(ascending=False).index).head(k).reset_index(drop=True)


def global_importance(expl: shap.Explanation) -> pd.Series:
    """Mean |SHAP| per feature (average impact on log-odds), largest first."""
    return pd.Series(np.abs(expl.values).mean(axis=0), index=expl.feature_names).sort_values(ascending=False)


def summary_figures(expl: shap.Explanation, out_dir):
    plt.close("all")
    shap.plots.beeswarm(expl, max_display=12, show=False)
    fig = plt.gcf(); fig.set_size_inches(9, 5.5); fig.tight_layout(); fig.savefig(out_dir / "shap_summary.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    imp = global_importance(expl).head(12)[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.8)); ax.barh(imp.index, imp.values, color="#4C78A8")
    ax.set(xlabel="Mean |SHAP value| (average impact on log-odds of default)", title="Global feature importance (SHAP)")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", alpha=.3); fig.tight_layout(); fig.savefig(out_dir / "shap_importance.png", dpi=110); plt.close(fig)


def main(n_sample: int = 5000, seed: int = 42) -> None:
    from src.features import ROOT, TARGET
    from src.train import load_lgbm, load_splits

    figs = ROOT / "reports" / "figures"
    va = load_splits()["validation"]                                  # explainability does not need the test set
    s = va.sample(n_sample, random_state=seed); X, y = s[FEATURE_COLUMNS], s[TARGET]
    booster = load_lgbm(); expl = explain(booster, X)
    pdv = booster.predict(X)
    summary_figures(expl, figs)
    imp = global_importance(expl)
    print("Global importance (mean |SHAP|, log-odds):"); print(imp.round(3).head(10).to_string())

    # three illustrative applicants: one per policy outcome (closest to the middle of each band)
    from src.evaluate import CUTOFF_DECLINE, CUTOFF_REVIEW
    picks = {"low_risk_approve": 0.015, "borderline_review": (CUTOFF_REVIEW + CUTOFF_DECLINE) / 2, "high_risk_decline": 0.40}
    for name, target in picks.items():
        i = int(np.argmin(np.abs(pdv - target)))
        fig = waterfall_figure(expl, i); fig.savefig(figs / f"shap_waterfall_{name}.png", dpi=110, bbox_inches="tight"); plt.close(fig)
        raw = X.iloc[i]
        print(f"\n--- {name}: PD {pdv[i]:.1%} (actual outcome: {'DEFAULTED' if y.iloc[i] else 'repaid'}) | base PD {sigmoid(expl.base_values[i]):.1%}")
        print(top_drivers(expl, i).round(3).to_string(index=False))
    print("\nFigures written to", figs)


if __name__ == "__main__":
    main()
