"""Train both models and save them to models/.

Model A: WoE + logistic-regression scorecard (Step 4).
Model B: LightGBM gradient-boosted trees (Step 5).
Everything is fitted/tuned on TRAIN and VALIDATION only; the test split is never touched here.
"""
from __future__ import annotations

import json
import warnings

import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluate import summary
from src.features import FEATURE_COLUMNS, PROCESSED_PATH, ROOT, TARGET
from src.woe import FACTOR, OFFSET, Scorecard, fit_all_binnings, iv_label, select_features, woe_frame

matplotlib.use("Agg")
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"


def load_splits():
    d = pd.read_parquet(PROCESSED_PATH)
    return {s: d[d["split"] == s] for s in ("train", "validation", "test")}


def plot_iv(binnings, kept, path):
    iv = pd.Series({k: v.iv for k, v in binnings.items()}).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(iv.index, iv.values, color=["#4C78A8" if k in kept else "#BBBBBB" for k in iv.index])
    ax.axvline(0.02, color="#E45756", ls="--", lw=1, label="IV = 0.02 threshold")
    ax.set_xscale("symlog", linthresh=0.01); ax.set_xlabel("Information Value (log-ish scale)")
    ax.set_title("Information Value by feature (blue = used in the scorecard)"); ax.legend()
    ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_woe(sc: Scorecard, path):
    feats = sc.features
    cols = 3; rows = int(np.ceil(len(feats) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(14, 3.3 * rows)); axs = np.atleast_1d(axs).ravel()
    for ax, f in zip(axs, feats):
        b = sc.binnings[f]; st = [s for s in b.stats if not (s["bin"] == "Missing" and s["count"] == 0)]
        ax.bar(range(len(st)), [s["woe"] for s in st], color=["#4C78A8" if s["woe"] >= 0 else "#E45756" for s in st])
        ax.set_xticks(range(len(st))); ax.set_xticklabels([s["bin"] for s in st], rotation=45, ha="right", fontsize=6)
        ax.axhline(0, color="k", lw=.6); ax.set_title(f"{f} (IV {b.iv:.2f})", fontsize=9); ax.set_ylabel("WoE")
    for ax in axs[len(feats):]: ax.axis("off")
    fig.suptitle("WoE per bin: blue = safer than average, red = riskier", y=1.0)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def train_scorecard(splits=None, verbose=True) -> Scorecard:
    splits = splits or load_splits()
    tr, va = splits["train"], splits["validation"]
    Xtr, ytr = tr[FEATURE_COLUMNS], tr[TARGET]

    binnings = fit_all_binnings(Xtr, ytr)
    W = woe_frame(Xtr, binnings)
    kept, log = select_features(binnings, W)
    sc = Scorecard.fit(Xtr, ytr, binnings, kept)

    # ---- reports ----
    iv_tbl = pd.DataFrame(log, columns=["feature", "iv", "decision"])
    iv_tbl["strength"] = iv_tbl["iv"].map(iv_label)
    iv_tbl.to_csv(REPORTS / "iv_table.csv", index=False)
    sc.table().to_csv(REPORTS / "scorecard_table.csv", index=False)
    sc.to_json(MODELS / "scorecard.json")
    plot_iv(binnings, sc.features, FIGS / "iv_ranking.png")
    plot_woe(sc, FIGS / "woe_selected.png")

    if verbose:
        pd.set_option("display.width", 200)
        print("\n=== IV table / feature selection ===")
        print(iv_tbl.assign(iv=iv_tbl.iv.round(4)).to_string(index=False))
        dropped_by_sign = [f for f in kept if f not in sc.features]
        print("\nDropped for wrong-signed coefficient:", dropped_by_sign or "none")
        print("\n=== Logistic regression (all coefficients must be negative) ===")
        print(pd.Series(sc.coef).round(4).to_string(), f"\nintercept = {sc.intercept:.4f}")
        print(f"\nScaling: score = {OFFSET:.2f} + {FACTOR:.3f} * ln(good:bad odds); base 600 @ 50:1, PDO 20")
        for name, part in (("train", tr), ("validation", va)):
            X, y = part[FEATURE_COLUMNS], part[TARGET]
            m = summary(y, sc.predict_pd(X)); s = summary(y, -sc.score(X))
            rnd = np.abs(sc.score(X) - sc.exact_score(X))
            print(f"{name:10s} AUC {m['AUC']:.4f} Gini {m['Gini']:.4f} KS {m['KS']:.4f} | "
                  f"points-table AUC {s['AUC']:.4f} | rounding error mean {rnd.mean():.2f} max {rnd.max():.1f} pts")
        sc_va = sc.score(va[FEATURE_COLUMNS])
        print(f"Validation score range {sc_va.min():.0f}..{sc_va.max():.0f}, median {np.median(sc_va):.0f}")
    return sc


# ---------------------------------------------------------------------------------------
# Model B: LightGBM
# ---------------------------------------------------------------------------------------
LGBM_BASE = dict(objective="binary", metric="auc", learning_rate=0.05, seed=42, deterministic=True,
                 force_row_wise=True, bagging_freq=1, verbose=-1, num_threads=4)
# NOTE: no class weighting on purpose. Weighting inflates predicted PDs (mean PD 30.6% vs a true 6.7%),
# which would wreck expected-loss maths. Imbalance is handled by stratified splits + AUC/KS/Gini.


def sample_params(rng: np.random.RandomState) -> dict:
    return dict(num_leaves=int(rng.choice([8, 15, 31, 48, 64])), min_data_in_leaf=int(rng.choice([50, 100, 200, 400])),
                feature_fraction=float(rng.choice([0.5, 0.7, 0.9, 1.0])), bagging_fraction=float(rng.choice([0.6, 0.8, 1.0])),
                lambda_l2=float(rng.choice([0, 1, 5, 20])), lambda_l1=float(rng.choice([0, 0.5, 2])),
                max_depth=int(rng.choice([-1, 4, 6, 8])))


def fit_lgbm(tr: pd.DataFrame, va: pd.DataFrame, params: dict, max_rounds: int = 3000):
    """Fit one LightGBM with early stopping on the validation AUC. Returns (booster, validation_auc)."""
    p = {**LGBM_BASE, **params}
    dtr = lgb.Dataset(tr[FEATURE_COLUMNS], tr[TARGET])
    dva = lgb.Dataset(va[FEATURE_COLUMNS], va[TARGET])
    b = lgb.train(p, dtr, max_rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    return b, float(b.best_score["valid_0"]["auc"])


def train_lgbm(splits=None, n_trials: int = 30, verbose=True) -> lgb.Booster:
    splits = splits or load_splits()
    tr, va = splits["train"], splits["validation"]
    rng = np.random.RandomState(42)
    trials = [dict()] + [sample_params(rng) for _ in range(n_trials)]   # trial 0 = untuned defaults (baseline)
    rows, best = [], (None, -1.0, None)
    for i, params in enumerate(trials):
        b, v_auc = fit_lgbm(tr, va, params)
        rows.append({"trial": i, **params, "best_iteration": b.best_iteration, "val_auc": v_auc})
        if v_auc > best[1]:
            best = (b, v_auc, params)
        if verbose:
            print(f"trial {i:2d} val AUC {v_auc:.4f} iters {b.best_iteration:4d} {params}")
    booster, _, params = best
    pd.DataFrame(rows).sort_values("val_auc", ascending=False).to_csv(REPORTS / "lgbm_tuning.csv", index=False)
    booster.save_model(str(MODELS / "lgbm.txt"), num_iteration=booster.best_iteration)
    meta = dict(params={**LGBM_BASE, **params}, best_iteration=int(booster.best_iteration), features=FEATURE_COLUMNS,
                baseline_val_auc=rows[0]["val_auc"], tuned_val_auc=best[1], n_trials=n_trials,
                lightgbm_version=lgb.__version__)
    (MODELS / "lgbm_meta.json").write_text(json.dumps(meta, indent=1))
    if verbose:
        X, y = va[FEATURE_COLUMNS], va[TARGET]
        p = booster.predict(X, num_iteration=booster.best_iteration)
        m = summary(y, p)
        print(f"\nBest params: {params}\nBest iteration: {booster.best_iteration}")
        print(f"Validation AUC {m['AUC']:.4f} Gini {m['Gini']:.4f} KS {m['KS']:.4f} | baseline (defaults) AUC {rows[0]['val_auc']:.4f}")
        print(f"Calibration check: mean predicted PD {p.mean():.4f} vs actual default rate {y.mean():.4f}")
        imp = pd.Series(booster.feature_importance("gain"), index=FEATURE_COLUMNS).sort_values(ascending=False)
        print("\nTop features by gain (% of total):")
        print((100 * imp / imp.sum()).round(1).head(8).to_string())
    return booster


def load_lgbm() -> lgb.Booster:
    return lgb.Booster(model_file=str(MODELS / "lgbm.txt"))


def main() -> None:
    warnings.filterwarnings("ignore")
    train_scorecard()
    train_lgbm()


if __name__ == "__main__":
    main()
