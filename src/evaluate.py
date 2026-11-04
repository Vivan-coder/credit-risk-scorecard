"""Evaluation: AUC / Gini / KS, calibration, confusion matrix at the policy cutoff, A-vs-B table, plots.

The decision policy is fixed HERE, derived from business economics and checked on the VALIDATION set
before the test set was ever scored:

  Declining a borrower who would have repaid costs the lender its margin (MARGIN, % of exposure).
  Approving a borrower who defaults costs LGD (% of exposure).
  With calibrated PDs, decline when  PD * LGD > (1 - PD) * MARGIN   <=>   PD > MARGIN / (MARGIN + LGD).
  MARGIN = 5%, LGD = 45%  ->  decline when PD > 10%.   (validation optimum was 9-10%: see STUDY_GUIDE)
  Review band 5%-10%: where the decision flips under a plausible 3%-8% margin assumption.
"""
from __future__ import annotations

import json

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve

from src.expected_loss import LGD

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MARGIN = 0.05                                    # assumed lender margin on a good loan, % of exposure
CUTOFF_DECLINE = MARGIN / (MARGIN + LGD)         # = 0.10
CUTOFF_REVIEW = 0.05                             # approve below this PD
BLUE, RED, GREY = "#4C78A8", "#E45756", "#888888"


# ---- core metrics --------------------------------------------------------------------------
def auc(y, p) -> float:
    return float(roc_auc_score(y, p))


def gini(y, p) -> float:
    """Gini = 2*AUC - 1."""
    return 2 * auc(y, p) - 1


def ks(y, p) -> float:
    """Kolmogorov-Smirnov: max gap between the cumulative bad and good score distributions."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def summary(y, p) -> dict:
    return {"AUC": auc(y, p), "Gini": gini(y, p), "KS": ks(y, p)}


def decide(pd_) -> np.ndarray:
    """Policy: APPROVE below 5% PD, REVIEW 5%-10%, DECLINE at/above 10%."""
    pd_ = np.asarray(pd_, dtype=float)
    return np.where(pd_ >= CUTOFF_DECLINE, "Decline", np.where(pd_ >= CUTOFF_REVIEW, "Review", "Approve"))


def cutoff_metrics(y, p, cutoff: float = CUTOFF_DECLINE) -> dict:
    y = np.asarray(y); dec = np.asarray(p) >= cutoff
    tp, fp = int((dec & (y == 1)).sum()), int((dec & (y == 0)).sum())
    fn, tn = int((~dec & (y == 1)).sum()), int((~dec & (y == 0)).sum())
    return dict(cutoff=cutoff, TN=tn, FP=fp, FN=fn, TP=tp,
                decline_rate=dec.mean(), bad_rate_approved=fn / max(fn + tn, 1),
                defaults_caught=tp / max(tp + fn, 1), precision=tp / max(tp + fp, 1),
                cost_pct_exposure=(fn * LGD + fp * MARGIN) / len(y))


def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["bin"] = pd.qcut(d["p"].rank(method="first"), bins, labels=False)
    t = d.groupby("bin").agg(mean_pd=("p", "mean"), actual=("y", "mean"), n=("y", "size"))
    return t


def bootstrap_auc(y, pa, pb=None, n: int = 1000, seed: int = 42):
    """95% bootstrap CIs for AUC of A (and B) and for the paired difference B - A."""
    rng = np.random.RandomState(seed); y = np.asarray(y)
    a, b = [], []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        a.append(roc_auc_score(y[i], np.asarray(pa)[i]))
        if pb is not None:
            b.append(roc_auc_score(y[i], np.asarray(pb)[i]))
    ci = lambda v: tuple(np.percentile(v, [2.5, 97.5]))
    out = {"A": ci(a)}
    if pb is not None:
        out["B"] = ci(b); out["B-A"] = ci(np.array(b) - np.array(a))
    return out


# ---- plots ---------------------------------------------------------------------------------
def _style(ax):
    ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=.3)


def plot_roc(y, preds: dict, path):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for (name, p), c in zip(preds.items(), (BLUE, RED)):
        fpr, tpr, _ = roc_curve(y, p); ax.plot(fpr, tpr, color=c, lw=2, label=f"{name}  AUC {auc(y, p):.3f} / Gini {gini(y, p):.3f}")
    ax.plot([0, 1], [0, 1], ls="--", color=GREY, label="Random (AUC 0.5)")
    ax.set(xlabel="False positive rate (good borrowers declined)", ylabel="True positive rate (defaulters caught)", title="ROC curve, test set")
    ax.legend(loc="lower right", fontsize=8); _style(ax); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_ks(y, preds: dict, path):
    fig, axs = plt.subplots(1, len(preds), figsize=(5.5 * len(preds), 4.5), sharey=True)
    y = np.asarray(y)
    for ax, (name, p) in zip(np.atleast_1d(axs), preds.items()):
        p = np.asarray(p); order = np.argsort(-p)                     # riskiest first
        ys = y[order]; cb = np.cumsum(ys) / ys.sum(); cg = np.cumsum(1 - ys) / (1 - ys).sum()
        x = np.arange(1, len(y) + 1) / len(y); i = int(np.argmax(cb - cg))
        ax.plot(x, cb, color=RED, label="Cumulative % of defaulters"); ax.plot(x, cg, color=BLUE, label="Cumulative % of good borrowers")
        ax.vlines(x[i], cg[i], cb[i], color="k", lw=2); ax.annotate(f"KS = {cb[i] - cg[i]:.3f}", (x[i], (cb[i] + cg[i]) / 2), xytext=(8, 0), textcoords="offset points")
        ax.set(xlabel="Borrowers ranked riskiest -> safest (cumulative share)", title=f"KS plot: {name}"); ax.legend(loc="lower right", fontsize=8); _style(ax)
    np.atleast_1d(axs)[0].set_ylabel("Cumulative share"); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_calibration(y, preds: dict, path):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    for (name, p), c in zip(preds.items(), (BLUE, RED)):
        t = calibration_table(y, p)
        axs[0].plot(t.mean_pd * 100, t.actual * 100, "o-", color=c, label=name)
        axs[1].plot(t.mean_pd * 100, t.actual * 100, "o-", color=c, label=name)
    m = max(axs[0].get_xlim()[1], axs[0].get_ylim()[1])
    axs[0].plot([0, m], [0, m], "--", color=GREY, label="Perfect calibration"); axs[0].set(title="Calibration, test set (deciles)", xlabel="Mean predicted PD %", ylabel="Actual default rate %"); axs[0].legend(fontsize=8)
    axs[1].plot([.1, 100], [.1, 100], "--", color=GREY); axs[1].set(xscale="log", yscale="log", title="Same, log scale (shows the low-risk end)", xlabel="Mean predicted PD % (log)", ylabel="Actual default rate % (log)")
    for a in axs: _style(a)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_confusion(y, preds: dict, path, cutoff=CUTOFF_DECLINE):
    fig, axs = plt.subplots(1, len(preds), figsize=(5 * len(preds), 4.3))
    for ax, (name, p) in zip(np.atleast_1d(axs), preds.items()):
        m = cutoff_metrics(y, p, cutoff); cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n({cm[i, j] / cm.sum():.1%})", ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["Not declined", f"Declined (PD>={cutoff:.0%})"]); ax.set_yticks([0, 1], ["Repaid (good)", "Defaulted (bad)"])
        ax.set_title(f"{name}\ncatches {m['defaults_caught']:.0%} of defaults, declines {m['decline_rate']:.0%}", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_score_dist(y, score, path, decline_score, review_score):
    y = np.asarray(y); fig, ax = plt.subplots(figsize=(7.5, 4.3))
    bins = np.linspace(np.min(score), np.max(score), 40)
    ax.hist(score[y == 0], bins, color=BLUE, alpha=.7, density=True, label="Repaid"); ax.hist(score[y == 1], bins, color=RED, alpha=.7, density=True, label="Defaulted")
    ax.axvline(decline_score, color="k", ls="--"); ax.text(decline_score - 1, ax.get_ylim()[1] * .92, "Decline below", ha="right", fontsize=8)
    ax.axvline(review_score, color="k", ls=":"); ax.text(review_score + 1, ax.get_ylim()[1] * .92, "Approve above", fontsize=8)
    ax.set(xlabel="Scorecard points (higher = safer)", ylabel="Density", title="Scorecard score distribution, test set"); ax.legend(); _style(ax)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


# ---- orchestration -------------------------------------------------------------------------
def evaluate_all(verbose: bool = True) -> dict:
    """Score both models on train / validation / TEST and write all reports. Run ONCE at the end."""
    from src.features import FEATURE_COLUMNS, ROOT, TARGET   # local imports avoid a circular import with train.py
    from src.train import load_lgbm, load_splits
    from src.woe import FACTOR, OFFSET, Scorecard

    reports, figs = ROOT / "reports", ROOT / "reports" / "figures"
    splits = load_splits(); sc = Scorecard.from_json(ROOT / "models" / "scorecard.json"); bst = load_lgbm()
    names = {"A": "A: WoE scorecard", "B": "B: LightGBM"}
    out, rows = {}, []
    for split, df in splits.items():
        X, y = df[FEATURE_COLUMNS], df[TARGET].to_numpy()
        pa, pb = sc.predict_pd(X), bst.predict(X)
        out[split] = (y, pa, pb, sc.score(X))
        for key, p in (("A", pa), ("B", pb)):
            m = summary(y, p)
            rows.append(dict(model=names[key], split=split, **m, brier=brier_score_loss(y, p), logloss=log_loss(y, p),
                             mean_pd=float(np.mean(p)), actual_rate=float(y.mean())))
    table = pd.DataFrame(rows)
    table.to_csv(reports / "model_comparison.csv", index=False)

    y, pa, pb, score = out["test"]
    preds = {names["A"]: pa, names["B"]: pb}
    ci = bootstrap_auc(y, pa, pb)
    cut = {k: {kk: (float(vv) if not isinstance(vv, int) else vv) for kk, vv in cutoff_metrics(y, p).items()} for k, p in (("A", pa), ("B", pb))}
    cal = {k: float((lambda t: (t.mean_pd - t.actual).abs().mean())(calibration_table(y, p))) for k, p in (("A", pa), ("B", pb))}
    to_score = lambda pdv: OFFSET + FACTOR * np.log((1 - pdv) / pdv)      # PD -> scorecard points
    decline_score, review_score = float(to_score(CUTOFF_DECLINE)), float(to_score(CUTOFF_REVIEW))

    plot_roc(y, preds, figs / "roc_curves.png"); plot_ks(y, preds, figs / "ks_plot.png")
    plot_calibration(y, preds, figs / "calibration_curve.png"); plot_confusion(y, preds, figs / "confusion_matrices.png")
    plot_score_dist(y, score, figs / "score_distribution.png", decline_score, review_score)

    md = ["| Model | Split | AUC | Gini | KS |", "|---|---|---|---|---|"]
    md += [f"| {r.model} | {r.split} | {r.AUC:.4f} | {r.Gini:.4f} | {r.KS:.4f} |" for r in table.itertuples()]
    (reports / "model_comparison.md").write_text("\n".join(md) + "\n")
    result = dict(policy=dict(lgd=LGD, margin=MARGIN, decline_pd=CUTOFF_DECLINE, review_pd=CUTOFF_REVIEW,
                              decline_score=decline_score, review_score=review_score),
                  test_at_cutoff=cut, test_auc_ci95=ci, calibration_mean_abs_gap=cal)
    (reports / "metrics.json").write_text(json.dumps(result, indent=1, default=float))
    if verbose:
        pd.set_option("display.width", 200)
        print(table.round(4).to_string(index=False))
        print("\nTEST AUC 95% bootstrap CI:", {k: tuple(round(x, 4) for x in v) for k, v in ci.items()})
        print(f"Policy: decline PD>={CUTOFF_DECLINE:.0%} (score <{decline_score:.0f}), approve PD<{CUTOFF_REVIEW:.0%} (score >={review_score:.0f})")
        for k in ("A", "B"):
            c = cut[k]; print(f"Test @ {c['cutoff']:.0%} cutoff [{k}]: TN {c['TN']:,} FP {c['FP']:,} FN {c['FN']:,} TP {c['TP']:,} | decline {c['decline_rate']:.1%} | "
                              f"defaults caught {c['defaults_caught']:.1%} | precision {c['precision']:.1%} | bad rate approved {c['bad_rate_approved']:.2%} | cost {c['cost_pct_exposure']*100:.3f}% of exposure")
        print("Mean |predicted - actual| across calibration deciles:", {k: round(v, 4) for k, v in cal.items()})
    return result


if __name__ == "__main__":
    evaluate_all()
