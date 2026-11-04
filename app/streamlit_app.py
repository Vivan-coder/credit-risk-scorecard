"""Credit Risk Scorecard: live demo.   Run:  streamlit run app/streamlit_app.py

Tab 1  Score an applicant  -> score, PD, risk band, decision, expected loss (BHD), reason codes, SHAP waterfall
Tab 2  Portfolio view      -> expected loss by band, approval-rate vs default-rate trade-off slider
Tab 3  Model card          -> metrics, data, assumptions, limitations (numbers are read from reports/, never hard-coded)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))          # so `import src...` works locally and on Streamlit Cloud

import lightgbm as lgb          # noqa: E402
import matplotlib               # noqa: E402
import numpy as np              # noqa: E402
import pandas as pd             # noqa: E402
import streamlit as st          # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.evaluate import CUTOFF_DECLINE, CUTOFF_REVIEW, MARGIN, decide  # noqa: E402
from src.explain import NICE, explain, get_explainer, top_drivers, waterfall_figure  # noqa: E402
from src.expected_loss import (EAD_MAX_USD, EAD_MIN_USD, EXPOSURE_MULTIPLE, LGD, USD_TO_BHD, band_table,  # noqa: E402
                               estimate_ead_bhd, expected_loss, risk_band, tradeoff_table)
from src.features import FEATURE_COLUMNS, build_features  # noqa: E402
from src.woe import BASE_ODDS, BASE_SCORE, PDO, Scorecard  # noqa: E402

REPORTS, FIGS = ROOT / "reports", ROOT / "reports" / "figures"
BLUE, RED, GREY = "#4C78A8", "#E45756", "#888888"

st.set_page_config(page_title="Credit Risk Scorecard (BHD)", page_icon="🏦", layout="wide")


# ---------------------------------------------------------------------------------------------
# cached loaders
# ---------------------------------------------------------------------------------------------
@st.cache_resource
def load_models():
    sc = Scorecard.from_json(ROOT / "models" / "scorecard.json")
    booster = lgb.Booster(model_file=str(ROOT / "models" / "lgbm.txt"))
    return sc, booster, get_explainer(booster)


@st.cache_data
def load_portfolio():
    sc, booster, _ = load_models()
    s = pd.read_csv(ROOT / "data" / "sample.csv")
    f = build_features(s)
    return s, sc.predict_pd(f), booster.predict(f[FEATURE_COLUMNS]), estimate_ead_bhd(s["monthly_income"].to_numpy())


@st.cache_data
def load_reports():
    out = {"metrics": json.loads((REPORTS / "metrics.json").read_text()),
           "comparison": pd.read_csv(REPORTS / "model_comparison.csv"),
           "el": json.loads((REPORTS / "expected_loss.json").read_text())}
    return out


sc, booster, explainer = load_models()

# ---------------------------------------------------------------------------------------------
# example applicants (income / debt in BHD per month)
# ---------------------------------------------------------------------------------------------
EXAMPLES = {
    "Borderline (review)": dict(age=38, income=1200, income_unknown=False, util=85, l30=0, l60=0, l90=0, lines=6, re=0, dep=2, debt=500),
    "Low risk (approve)": dict(age=52, income=1800, income_unknown=False, util=10, l30=0, l60=0, l90=0, lines=8, re=1, dep=1, debt=400),
    "High risk (decline)": dict(age=33, income=900, income_unknown=False, util=95, l30=2, l60=1, l90=0, lines=4, re=0, dep=3, debt=500),
}


def load_example():
    for k, v in EXAMPLES[st.session_state["example"]].items():
        st.session_state[f"in_{k}"] = v


if "in_age" not in st.session_state:
    for k, v in EXAMPLES["Borderline (review)"].items():
        st.session_state[f"in_{k}"] = v
    st.session_state["use_est_ead"] = True


def fmt_bhd(x: float) -> str:
    return f"BHD {x:,.0f}"


def pct(v) -> str:
    return "n/a" if pd.isna(v) else f"{v:.2%}"


st.title("🏦 Credit Risk Scorecard")
st.caption("From a loan application to an approve / review / decline decision and an expected loss in BHD. "
           "**Demo on public non-Bahraini data (see the Model card tab); not for real lending decisions.**")
tab1, tab2, tab3 = st.tabs(["🧾 Score an applicant", "📊 Portfolio view", "📘 Model card"])

# =============================================================================================
# TAB 1: score an applicant
# =============================================================================================
with tab1:
    left, right = st.columns([1, 2], gap="large")
    with left:
        st.subheader("Applicant")
        st.selectbox("Load an example", list(EXAMPLES), key="example", on_change=load_example)
        st.number_input("Age", 18, 100, key="in_age", help="Must be 18 or over.")
        c1, c2 = st.columns(2)
        c1.number_input("Monthly income (BHD)", 0, 50_000, step=50, key="in_income", disabled=st.session_state["in_income_unknown"])
        c2.number_input("Monthly debt payments (BHD)", 0, 50_000, step=50, key="in_debt", disabled=st.session_state["in_income_unknown"])
        st.checkbox("Income not provided", key="in_income_unknown")
        st.slider("Credit-line utilisation (% of limits in use)", 0, 200, key="in_util",
                  help="Total balances on cards and credit lines divided by total limits. Over 100% = over the limit.")
        c1, c2, c3 = st.columns(3)
        c1.number_input("Times 30-59d late", 0, 20, key="in_l30", help="Past 2 years")
        c2.number_input("Times 60-89d late", 0, 20, key="in_l60")
        c3.number_input("Times 90+d late", 0, 20, key="in_l90")
        c1, c2, c3 = st.columns(3)
        c1.number_input("Open credit lines & loans", 0, 40, key="in_lines")
        c2.number_input("Real-estate loans", 0, 10, key="in_re")
        c3.number_input("Dependents", 0, 10, key="in_dep")
        st.checkbox(f"Use assumed loan size ({EXPOSURE_MULTIPLE}x monthly income)", key="use_est_ead",
                    help="The dataset has no loan amount, so exposure is an assumption. Untick to enter the requested amount.")
        if not st.session_state["use_est_ead"]:
            st.session_state.setdefault("in_loan", 8000.0)   # widget state is dropped while the widget is hidden
            st.number_input("Requested loan amount (BHD)", 100.0, 200_000.0, step=500.0, key="in_loan")

    # ---- build the raw applicant row (model expects the dataset's units: income in USD) ----
    s = st.session_state
    known = not s["in_income_unknown"]
    income_usd = s["in_income"] / USD_TO_BHD if known else np.nan
    debt_ratio = (s["in_debt"] / s["in_income"]) if known and s["in_income"] > 0 else np.nan
    row = pd.DataFrame([dict(utilization=s["in_util"] / 100, age=s["in_age"], late_30_59=s["in_l30"], debt_ratio=debt_ratio,
                             monthly_income=income_usd, open_credit_lines=s["in_lines"], late_90_plus=s["in_l90"],
                             real_estate_loans=s["in_re"], late_60_89=s["in_l60"], dependents=s["in_dep"])])
    feats = build_features(row)
    score = float(sc.score(feats)[0]); pd_a = float(sc.predict_pd(feats)[0]); pd_b = float(booster.predict(feats[FEATURE_COLUMNS])[0])
    decision, decision_b = str(decide(pd_a)), str(decide(pd_b))
    band = str(risk_band([pd_a])[0])
    ead = float(estimate_ead_bhd([income_usd])[0]) if s["use_est_ead"] else float(s["in_loan"])
    el = float(expected_loss(pd_a, ead))

    with right:
        msg = {"Approve": st.success, "Review": st.warning, "Decline": st.error}[decision]
        reason = {"Approve": f"PD is below the {CUTOFF_REVIEW:.0%} approve threshold.",
                  "Review": f"PD is between {CUTOFF_REVIEW:.0%} and {CUTOFF_DECLINE:.0%}: send to a human underwriter.",
                  "Decline": f"PD is at or above the {CUTOFF_DECLINE:.0%} decline cutoff (declining beats approving once PD x LGD exceeds the lender's margin)."}[decision]
        msg(f"### {decision}\n{reason}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Credit score", f"{score:.0f}", f"{score - BASE_SCORE:+.0f} vs base {BASE_SCORE}", help=f"{BASE_SCORE} pts = {BASE_ODDS}:1 good:bad odds; +{PDO} pts doubles the odds.")
        m2.metric("Probability of default", f"{pd_a:.2%}", help="From the WoE scorecard (primary decision model).")
        m3.metric("Risk band", band.split("(")[0].strip())
        m4.metric("Expected loss", fmt_bhd(el), help=f"PD x LGD ({LGD:.0%}) x EAD ({fmt_bhd(ead)})")
        agree = "agrees" if decision_b == decision else "**disagrees**"
        st.caption(f"Challenger model (LightGBM): PD {pd_b:.2%} -> {decision_b}; it {agree} with the scorecard. "
                   f"Exposure used: {fmt_bhd(ead)} ({'assumed from income' if s['use_est_ead'] else 'entered'}).")
        if decision_b != decision:
            st.info("Model disagreement is itself a signal: a lender would typically route these to manual review.")

        st.markdown("#### Why this score? Scorecard reason codes")
        bd = sc.breakdown(feats)
        bd["Feature"] = bd["feature"].map(lambda f: NICE.get(f, f))
        fig, ax = plt.subplots(figsize=(7, 2.4))
        ax.barh(bd["Feature"][::-1], bd["points_lost"][::-1], color=RED)
        ax.set(xlabel="Points lost versus the best possible bin for that feature"); ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", alpha=.3)
        fig.tight_layout(); st.pyplot(fig, width="stretch"); plt.close(fig)
        show = bd[["Feature", "bin", "points", "best_points", "points_lost"]].rename(columns={"bin": "Applicant's bin", "points": "Points", "best_points": "Best possible", "points_lost": "Points lost"})
        st.dataframe(show, hide_index=True, width="stretch")
        st.caption(f"Score = sum of points = {int(bd.points.sum())}. Maximum possible = {int(bd.best_points.sum())}.")

        st.markdown("#### Why this risk? SHAP explanation (LightGBM challenger)")
        expl = explain(booster, feats, explainer)
        st.pyplot(waterfall_figure(expl, 0), width="stretch")
        td = top_drivers(expl, 0, k=3)
        bullets = [f"**{r.feature}** ({'raises' if r.shap_log_odds > 0 else 'lowers'} risk, {r.shap_log_odds:+.2f} log-odds)" for r in td.itertuples()]
        st.caption("Biggest drivers: " + "; ".join(bullets) + ". The bar starts at the average applicant's log-odds and each feature pushes it up (red) or down (blue). "
                   f"Income values in this chart are in the model's units (USD-equivalent = BHD / {USD_TO_BHD}).")

# =============================================================================================
# TAB 2: portfolio view
# =============================================================================================
with tab2:
    sample, pa, pb, ead_s = load_portfolio()
    y = sample["default"].to_numpy()
    st.subheader("Portfolio view")
    st.caption(f"{len(sample):,} borrowers sampled from the **held-out test set** (out-of-sample), stratified on default. "
               f"Exposure per borrower is an assumption ({EXPOSURE_MULTIPLE}x monthly income, capped). Small sample = noisy; official metrics are on the Model card.")
    model_pick = st.radio("PD source", ["WoE scorecard (primary)", "LightGBM (challenger)"], horizontal=True)
    p = pa if model_pick.startswith("WoE") else pb

    st.markdown("##### Expected loss by risk band")
    bt = band_table(p, y, ead_s)
    c1, c2, c3 = st.columns(3)
    tot_el, tot_ead = bt.expected_loss_bhd.sum(), bt.ead_bhd.sum()
    c1.metric("Portfolio exposure (EAD)", fmt_bhd(tot_ead)); c2.metric("Expected loss", fmt_bhd(tot_el), f"{tot_el / tot_ead:.2%} of exposure", delta_color="off")
    c3.metric("Realised loss in sample", fmt_bhd(bt.realised_loss_bhd.sum()), help="Actual defaults x LGD x EAD: a back-test of the expected loss.", delta_color="off")
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6)); x = np.arange(len(bt)); w = .38
    axs[0].bar(x - w / 2, bt.share_of_ead * 100, w, color=BLUE, label="Share of exposure"); axs[0].bar(x + w / 2, bt.share_of_el * 100, w, color=RED, label="Share of expected loss")
    axs[0].set(ylabel="% of portfolio", title="Where the exposure is vs where the loss is"); axs[0].legend(fontsize=8)
    axs[1].bar(x - w / 2, bt.expected_loss_bhd / 1e3, w, color=RED, label="Expected"); axs[1].bar(x + w / 2, bt.realised_loss_bhd / 1e3, w, color=GREY, label="Realised")
    axs[1].set(ylabel="BHD thousand", title="Expected vs realised loss"); axs[1].legend(fontsize=8)
    for a in axs:
        a.set_xticks(x, [b.split("(")[0].strip() for b in bt.index], fontsize=8); a.spines[["top", "right"]].set_visible(False); a.grid(axis="y", alpha=.3)
    fig.tight_layout(); st.pyplot(fig, width="stretch"); plt.close(fig)
    tbl = bt.reset_index().rename(columns={"band": "Band"})
    tbl = pd.DataFrame({"Band": tbl["Band"].astype(str), "Action": tbl["decision"], "Borrowers": tbl["borrowers"], "Avg PD": tbl["avg_pd"].map(pct),
                        "Actual default rate": tbl["actual_default_rate"].map(pct), "Exposure (BHD)": tbl["ead_bhd"].map("{:,.0f}".format),
                        "Expected loss (BHD)": tbl["expected_loss_bhd"].map("{:,.0f}".format), "% of loss": tbl["share_of_el"].map(pct),
                        "EL rate": tbl["el_rate"].map(pct)})
    st.dataframe(tbl, hide_index=True, width="stretch")

    st.markdown("##### Approval-rate vs default-rate trade-off")
    cutoff_pct = st.slider("Decline applicants with PD at or above (%):", 2.0, 40.0, round(CUTOFF_DECLINE * 100, 1), 0.5, format="%.1f%%",
                           help=f"Policy cutoff is {CUTOFF_DECLINE:.0%}: the PD where expected loss from approving (PD x LGD) equals the margin lost by declining ({MARGIN:.0%}).")
    ap = p < cutoff_pct / 100
    ela, eadA = expected_loss(p[ap], ead_s[ap]).sum(), ead_s[ap].sum()
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Approval rate", f"{ap.mean():.1%}"); d2.metric("Default rate of approved book", f"{y[ap].mean():.2%}", f"{(y[ap].mean() - y.mean()) * 100:+.2f} pts vs approve-all", delta_color="inverse")
    d3.metric("Defaults avoided", f"{(~ap & (y == 1)).sum() / max(y.sum(), 1):.0%}", help="Share of all defaulters that this cutoff declines.")
    d4.metric("EL rate of approved book", f"{ela / eadA:.2%}" if eadA else "n/a")
    tt = tradeoff_table(p, y, ead_s)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot(tt.approval_rate * 100, tt.default_rate_approved * 100, color=BLUE, lw=2)
    ax.plot(ap.mean() * 100, y[ap].mean() * 100, "o", color=RED, ms=9, label="Your cutoff")
    ax.axhline(y.mean() * 100, color=GREY, ls="--", lw=1, label="Approve everyone")
    ax.set(xlabel="Approval rate %", ylabel="Default rate of approved %"); ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=.3)
    fig.tight_layout(); st.pyplot(fig, width="stretch"); plt.close(fig)
    st.caption("Moving right approves more people and the approved book gets riskier. Where to stop is a business decision, not a statistical one.")

# =============================================================================================
# TAB 3: model card
# =============================================================================================
with tab3:
    R = load_reports(); met, cmp_, el = R["metrics"], R["comparison"], R["el"]
    ci, cut, cal = met["test_auc_ci95"], met["test_at_cutoff"], met["calibration_mean_abs_gap"]
    st.subheader("Model card")
    st.markdown(f"""
**Purpose.** Rank loan applicants by default risk and convert risk into an approve / review / decline decision and an expected loss in BHD.
A portfolio demonstration of credit-risk modelling; **not validated for real lending**.

**Data.** *Give Me Some Credit*: 150,000 borrowers, 10 raw features, default = 90+ days past due within 2 years (6.68% default rate).
Public data from OpenML. **The publisher does not state the country or currency; it is not Bahraini data.** Split 60/20/20 train/validation/test, stratified, seed 42.

**Models.**
- **A: WoE scorecard** (primary). Optimal monotonic binning, features with IV > 0.02 and low collinearity, logistic regression, points scaled to {BASE_SCORE} at {BASE_ODDS}:1 odds and {PDO} points to double the odds. Fully transparent.
- **B: LightGBM** (challenger). Gradient-boosted trees, early stopping, light random-search tuning, explained with SHAP. No class weighting, to keep PDs calibrated.
""")
    t = cmp_[cmp_.split == "test"].set_index("model")
    st.markdown("**Test-set results (30,000 borrowers, scored once)**")
    st.dataframe(pd.DataFrame({"AUC": t.AUC.map("{:.4f}".format), "Gini": t.Gini.map("{:.4f}".format), "KS": t.KS.map("{:.4f}".format),
                               "Mean predicted PD": t.mean_pd.map("{:.2%}".format)}), width="stretch")
    st.caption(f"Actual default rate on test: {t.actual_rate.iloc[0]:.2%}. AUC 95% bootstrap CI: scorecard {ci['A'][0]:.3f}-{ci['A'][1]:.3f}, LightGBM {ci['B'][0]:.3f}-{ci['B'][1]:.3f}; "
               f"LightGBM minus scorecard {ci['B-A'][0]:+.3f} to {ci['B-A'][1]:+.3f}. Calibration: mean |predicted - actual| across deciles {cal['A'] * 100:.2f} pts (A), {cal['B'] * 100:.2f} pts (B).")
    st.markdown(f"""
**Decision policy.** Decline at PD >= {CUTOFF_DECLINE:.0%}, review at {CUTOFF_REVIEW:.0%}-{CUTOFF_DECLINE:.0%}, approve below {CUTOFF_REVIEW:.0%}.
Derived from economics: with LGD {LGD:.0%} and an assumed margin of {MARGIN:.0%} on good loans, declining is optimal once PD x LGD > (1 - PD) x margin, i.e. PD > {MARGIN:.0%} / ({MARGIN:.0%} + {LGD:.0%}) = {CUTOFF_DECLINE:.0%}.
On the test set at this cutoff the scorecard declines {cut['A']['decline_rate']:.1%} of applicants, catches {cut['A']['defaults_caught']:.1%} of defaults and the approved book defaults at {cut['A']['bad_rate_approved']:.2%} (vs 6.68% overall).

**Expected loss.** EL = PD x LGD x EAD. Portfolio EL on the test set = {fmt_bhd(el['A_scorecard']['expected_loss_bhd'])} ({el['A_scorecard']['el_rate']:.2%} of exposure) versus a realised
{fmt_bhd(el['A_scorecard']['realised_loss_bhd'])} (back-test using the same LGD/EAD assumptions).
- **LGD = {LGD:.0%}:** Basel foundation-IRB supervisory value for senior unsecured exposures (assumed; real LGD needs recovery data).
- **FX: 1 USD = {USD_TO_BHD} BHD:** the dinar's official peg. Dataset income currency is not stated; I treat it as USD (assumption), and the nominal conversion ignores purchasing-power differences.
- **EAD (assumed):** the data has no loan amount, so exposure = {EXPOSURE_MULTIPLE} x monthly income (floor USD {EAD_MIN_USD:,.0f}, cap USD {EAD_MAX_USD:,.0f}), fully drawn. The app lets you enter the real amount.
""")
    st.markdown("**Limitations**")
    st.markdown("""
1. **Not Bahraini data.** Population, economy, bureau and product mix differ; PDs would not transfer. Recalibration on local data is mandatory before any use.
2. **No reject inference.** The data only contains borrowers who were granted credit. Applicants who were rejected are missing, which biases the model towards the approved population.
3. **Random split, not out-of-time.** Performance over a later period (or a downturn) is untested; no drift monitoring is implemented.
4. **EAD and LGD are assumptions**, not data. BHD totals are illustrative; the PD ranking and calibration are the validated part.
5. **Default definition:** 90+ days past due within 2 years, not the Basel 12-month definition.
6. **Thin feature set:** 10 raw variables; no bureau history, employment, collateral or product information.
7. **Age is a model input** (scorecard IV 0.26). Many regulators and fair-lending regimes restrict age in credit decisions; this would need legal and fairness review. No fairness testing was done.
8. **Demo sample:** the portfolio tab uses a 2,000-row sample, so its numbers are noisy.
""")
    st.markdown("**Figures**")
    for pair in ((("roc_curves.png", "ROC curve (test set)"), ("calibration_curve.png", "Calibration (test set)")),
                 (("shap_summary.png", "SHAP summary: LightGBM"), ("woe_selected.png", "WoE by bin: scorecard features"))):
        for col, (fname, cap) in zip(st.columns(2, vertical_alignment="center"), pair):
            col.image(str(FIGS / fname), caption=cap, width="stretch")
