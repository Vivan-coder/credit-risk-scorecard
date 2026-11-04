"""Expected loss:  EL = PD x LGD x EAD, in BHD.

ASSUMPTIONS (all stated in the README and the app's model card):
* LGD = 45%: the Basel foundation-IRB supervisory value for senior UNSECURED exposures.
* USD -> BHD at 0.376: the dinar's official peg (fixed, so no FX-risk modelling needed). NOTE: the dataset does not
  state its income currency; treating it as USD is an assumption (and a nominal conversion ignores purchasing power).
* EAD: the dataset has NO loan amount, so exposure is ASSUMED to be a multiple of monthly income
  (6x, floored at USD 1,000 and capped at USD 50,000), treated as a fully drawn term loan (CCF = 100%).
  In the app the user can overwrite it with the real requested amount.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

LGD = 0.45
USD_TO_BHD = 0.376
EXPOSURE_MULTIPLE = 6           # loan size as a multiple of monthly income (assumption)
EAD_MIN_USD, EAD_MAX_USD = 1_000.0, 50_000.0
MEDIAN_INCOME_USD = 5_400.0     # training-set median monthly income, used when income is missing

# Risk bands are aligned with the decision policy (src/evaluate.py): A,B approve (<5%), C review (5-10%), D,E decline (>=10%)
BAND_EDGES = [0.0, 0.02, 0.05, 0.10, 0.20, 1.0001]
BAND_LABELS = ["A  Very low (<2%)", "B  Low (2-5%)", "C  Medium (5-10%)", "D  High (10-20%)", "E  Very high (20%+)"]
BAND_DECISION = dict(zip(BAND_LABELS, ["Approve", "Approve", "Review", "Decline", "Decline"]))


def usd_to_bhd(usd):
    return np.asarray(usd, dtype=float) * USD_TO_BHD


def estimate_ead_bhd(monthly_income_usd) -> np.ndarray:
    """Assumed exposure in BHD from monthly income in USD (NaN income -> training median)."""
    inc = pd.Series(np.asarray(monthly_income_usd, dtype=float)).fillna(MEDIAN_INCOME_USD).to_numpy()
    return usd_to_bhd(np.clip(EXPOSURE_MULTIPLE * inc, EAD_MIN_USD, EAD_MAX_USD))


def expected_loss(pd_, ead, lgd: float = LGD) -> np.ndarray:
    """EL = PD x LGD x EAD (same currency as EAD)."""
    return np.asarray(pd_, dtype=float) * lgd * np.asarray(ead, dtype=float)


def risk_band(pd_) -> pd.Categorical:
    return pd.cut(np.asarray(pd_, dtype=float), BAND_EDGES, labels=BAND_LABELS, right=False)


# ---- unexpected loss (indicative Basel IRB "other retail" capital) -------------------------
def basel_retail_correlation(pd_) -> np.ndarray:
    """Basel II asset correlation for 'other retail' exposures: 3% at high PD up to 16% at low PD."""
    pd_ = np.asarray(pd_, dtype=float)
    w = (1 - np.exp(-35 * pd_)) / (1 - np.exp(-35))
    return 0.03 * w + 0.16 * (1 - w)


def capital_rate(pd_, lgd: float = LGD, confidence: float = 0.999) -> np.ndarray:
    """Unexpected loss at the 99.9% level as a fraction of EAD (Basel one-factor / Vasicek formula, retail: no maturity adjustment).

    K = LGD x [ Phi( (Phi^-1(PD) + sqrt(R) x Phi^-1(0.999)) / sqrt(1-R) ) - PD ]
    i.e. loss in a 1-in-1000 bad year MINUS the expected loss already provisioned. Indicative only: not a regulatory calculation.
    """
    pd_ = np.clip(np.asarray(pd_, dtype=float), 1e-6, 1 - 1e-6)
    r = basel_retail_correlation(pd_)
    cond_pd = norm.cdf((norm.ppf(pd_) + np.sqrt(r) * norm.ppf(confidence)) / np.sqrt(1 - r))
    return lgd * (cond_pd - pd_)


# ---- portfolio views -----------------------------------------------------------------------
def band_table(pd_, y, ead_bhd, lgd: float = LGD) -> pd.DataFrame:
    """EL by risk band. `y` = observed default flag (used only to back-test: realised loss = default x LGD x EAD)."""
    d = pd.DataFrame({"pd": np.asarray(pd_, dtype=float), "y": np.asarray(y), "ead": np.asarray(ead_bhd, dtype=float)})
    d["band"] = risk_band(d["pd"]); d["el"] = expected_loss(d.pd, d.ead, lgd); d["realised"] = d.y * lgd * d.ead
    g = d.groupby("band", observed=False).agg(borrowers=("pd", "size"), avg_pd=("pd", "mean"), actual_default_rate=("y", "mean"),
                                              ead_bhd=("ead", "sum"), expected_loss_bhd=("el", "sum"), realised_loss_bhd=("realised", "sum"))
    g["share_of_borrowers"] = g.borrowers / g.borrowers.sum()
    g["share_of_ead"] = g.ead_bhd / g.ead_bhd.sum()
    g["share_of_el"] = g.expected_loss_bhd / g.expected_loss_bhd.sum()
    g["el_rate"] = g.expected_loss_bhd / g.ead_bhd.where(g.ead_bhd > 0)
    g["decision"] = [BAND_DECISION[b] for b in g.index]
    return g


def portfolio_summary(pd_, y, ead_bhd, lgd: float = LGD) -> dict:
    pd_ = np.asarray(pd_, dtype=float); ead = np.asarray(ead_bhd, dtype=float)
    el = float(expected_loss(pd_, ead, lgd).sum()); ul = float((capital_rate(pd_, lgd) * ead).sum())
    return dict(borrowers=int(len(pd_)), total_ead_bhd=float(ead.sum()), expected_loss_bhd=el, el_rate=el / float(ead.sum()),
                realised_loss_bhd=float((np.asarray(y) * lgd * ead).sum()),
                unexpected_loss_999_bhd=ul, unexpected_loss_rate=ul / float(ead.sum()), avg_pd=float(pd_.mean()))


def tradeoff_table(pd_, y, ead_bhd, cutoffs=None, lgd: float = LGD) -> pd.DataFrame:
    """For each decline-cutoff: how many we approve, the default rate of the approved book, and its EL. Drives the app slider."""
    cutoffs = np.round(np.arange(0.02, 0.5001, 0.01), 2) if cutoffs is None else cutoffs
    pd_ = np.asarray(pd_, dtype=float); y = np.asarray(y); ead = np.asarray(ead_bhd, dtype=float); el = expected_loss(pd_, ead, lgd)
    rows = []
    for c in cutoffs:
        a = pd_ < c
        rows.append(dict(cutoff=float(c), approval_rate=float(a.mean()),
                         default_rate_approved=float(y[a].mean()) if a.any() else np.nan,
                         el_rate_approved=float(el[a].sum() / ead[a].sum()) if a.any() else np.nan,
                         el_approved_bhd=float(el[a].sum())))
    return pd.DataFrame(rows)


# ---- report generation (run: python -m src.expected_loss) ---------------------------------
def main() -> None:
    import json
    import warnings

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.evaluate import CUTOFF_DECLINE
    from src.features import FEATURE_COLUMNS, ROOT, TARGET
    from src.train import load_lgbm, load_splits
    from src.woe import Scorecard

    warnings.filterwarnings("ignore")
    reports, figs = ROOT / "reports", ROOT / "reports" / "figures"
    te = load_splits()["test"]; X, y = te[FEATURE_COLUMNS], te[TARGET].to_numpy()
    ead = estimate_ead_bhd(X["monthly_income"].to_numpy())
    models = {"A_scorecard": Scorecard.from_json(ROOT / "models" / "scorecard.json").predict_pd(X), "B_lightgbm": load_lgbm().predict(X)}

    summary, tables, trade = {}, [], {}
    for name, p in models.items():
        summary[name] = portfolio_summary(p, y, ead)
        t = band_table(p, y, ead); t.insert(0, "model", name); tables.append(t.reset_index())
        trade[name] = tradeoff_table(p, y, ead)
    pd.concat(tables).to_csv(reports / "expected_loss_by_band.csv", index=False)
    summary["assumptions"] = dict(lgd=LGD, usd_to_bhd=USD_TO_BHD, exposure_multiple_of_monthly_income=EXPOSURE_MULTIPLE,
                                  ead_min_usd=EAD_MIN_USD, ead_max_usd=EAD_MAX_USD, median_income_usd=MEDIAN_INCOME_USD, evaluated_on="test set (30,000 borrowers)")
    (reports / "expected_loss.json").write_text(json.dumps(summary, indent=1))

    # Figure 1: where the exposure sits vs where the loss comes from (Model A), plus back-test of EL vs realised loss
    t = band_table(models["A_scorecard"], y, ead); labels = [b.split("(")[0].strip() + "\n(" + b.split("(")[1] for b in t.index]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6)); x = np.arange(len(t)); w = 0.38
    axs[0].bar(x - w / 2, t.share_of_ead * 100, w, color="#4C78A8", label="Share of exposure (EAD)"); axs[0].bar(x + w / 2, t.share_of_el * 100, w, color="#E45756", label="Share of expected loss")
    axs[0].set(xticks=x, xticklabels=labels, ylabel="% of portfolio", title="Exposure vs expected loss by risk band (scorecard)"); axs[0].legend(fontsize=8)
    axs[1].bar(x - w / 2, t.expected_loss_bhd / 1e6, w, color="#E45756", label="Expected loss (model)"); axs[1].bar(x + w / 2, t.realised_loss_bhd / 1e6, w, color="#888888", label="Realised loss (actual defaults x LGD)")
    axs[1].set(xticks=x, xticklabels=labels, ylabel="BHD million", title="Back-test: expected vs realised loss (test set)"); axs[1].legend(fontsize=8)
    for a in axs: a.spines[["top", "right"]].set_visible(False); a.grid(axis="y", alpha=.3); a.tick_params(axis="x", labelsize=7)
    fig.tight_layout(); fig.savefig(figs / "expected_loss_by_band.png", dpi=110); plt.close(fig)

    # Figure 2: approval rate vs default rate trade-off (what the app slider shows)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for (name, tt), c in zip(trade.items(), ("#4C78A8", "#E45756")):
        ax.plot(tt.approval_rate * 100, tt.default_rate_approved * 100, color=c, lw=2, label=name.replace("_", " "))
        r = tt.iloc[(tt.cutoff - CUTOFF_DECLINE).abs().argmin()]; ax.plot(r.approval_rate * 100, r.default_rate_approved * 100, "o", color=c)
    ax.axhline(y.mean() * 100, color="#888888", ls="--", lw=1, label=f"Approve everyone ({y.mean():.1%})")
    ax.set(xlabel="Approval rate % (higher = more business)", ylabel="Default rate of approved loans %", title="Approval-rate vs default-rate trade-off (test set)\n(dots = 10% PD policy cutoff)")
    ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(figs / "tradeoff_curve.png", dpi=110); plt.close(fig)

    pd.set_option("display.width", 220); pd.set_option("display.float_format", "{:,.4f}".format)
    for name in models:
        s = summary[name]
        print(f"\n[{name}] borrowers {s['borrowers']:,} | EAD BHD {s['total_ead_bhd']/1e6:,.1f}M | EL BHD {s['expected_loss_bhd']/1e6:,.2f}M ({s['el_rate']:.2%} of EAD) | "
              f"realised BHD {s['realised_loss_bhd']/1e6:,.2f}M | 99.9% UL BHD {s['unexpected_loss_999_bhd']/1e6:,.2f}M ({s['unexpected_loss_rate']:.2%})")
    print("\nTrade-off @ 10% cutoff:"); print(pd.concat({k: v.iloc[(v.cutoff - CUTOFF_DECLINE).abs().argmin()] for k, v in trade.items()}, axis=1).to_string())


if __name__ == "__main__":
    main()
