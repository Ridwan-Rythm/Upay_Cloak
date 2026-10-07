"""UpayShield ML v1: compare basic algorithms, pick the best, tune actions, report honestly.

Usage (from repo root):  python -m ml.train
Reads  data/train.csv + data/test.csv   (time-based split made by scripts/generate_data.py)
Writes models/risk_engine.joblib, data/cache/scored_cache.parquet, reports/metrics.json,
reports/model_comparison.md, reports/model_comparison.png, reports/shap_importance.json
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.anomaly import BehaviorAnomaly  # noqa: E402
from ml.decision import tune_thresholds, total_cost  # noqa: E402
from ml.evaluate import (  # noqa: E402
    business_impact,
    calibration,
    precision_by_action,
    ranking_metrics,
    scenario_recall,
)
from ml.explain import Explainer  # noqa: E402
from ml.features import FEATURES, build_features  # noqa: E402

warnings.filterwarnings("ignore")
SEED = 42
ARTIFACT_VERSION = "v1.2"  # stored inside models/risk_engine.joblib and returned by the API
FEEDBACK_LABELS = ROOT / "data" / "feedback_labels.csv"   # optional, written by ml/feedback.py


def make_models(pos_w):
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample",
                                                n_jobs=-1, random_state=SEED),
        "LightGBM": lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, subsample=0.8,
                                       subsample_freq=1, colsample_bytree=0.8, scale_pos_weight=min(pos_w, 10),
                                       random_state=SEED, verbose=-1),
    }


def cv_pr_auc(tr_all, final_valid_pr):
    """Expanding-window time-series CV on the TRAINING period only (the test set is never touched).

    A single validation slice holds only ~150 frauds, so picking a model on it is noisy. We average PR-AUC over
    three consecutive time windows instead: train on [0, 50%) -> score [50%, 65%); train on [0, 65%) -> score
    [65%, 80%); the third window is the regular train/validation split (`final_valid_pr`)."""
    cuts = [tr_all.ts.quantile(q) for q in (0.5, 0.65, 0.8)]
    out = {n: [v] for n, v in final_valid_pr.items()}
    for lo, hi in zip(cuts[:2], cuts[1:]):
        a, b = tr_all[tr_all.ts < lo], tr_all[(tr_all.ts >= lo) & (tr_all.ts < hi)]
        ya, yb = a.is_fraud.values, b.is_fraud.values
        for n, m in make_models((ya == 0).sum() / max((ya == 1).sum(), 1)).items():
            m.fit(a[FEATURES], ya)
            out[n].append(ranking_metrics(yb, m.predict_proba(b[FEATURES])[:, 1])["pr_auc"])
    return {n: round(float(np.mean(v)), 4) for n, v in out.items()}


def main():
    train_raw = pd.read_csv(ROOT / "data" / "train.csv", parse_dates=["ts"])
    test_raw = pd.read_csv(ROOT / "data" / "test.csv", parse_dates=["ts"])
    train_raw["_part"], test_raw["_part"] = "train", "test"
    # Features are built over the whole timeline in order (point-in-time), then split back by file.
    feats = build_features(pd.concat([train_raw, test_raw], ignore_index=True))
    tr_all, te = (feats[feats._part == p].reset_index(drop=True) for p in ("train", "test"))

    # Analyst feedback (confirmed / dismissed cases) overrides TRAINING labels only; test labels stay untouched.
    if FEEDBACK_LABELS.exists():
        fb = pd.read_csv(FEEDBACK_LABELS).drop_duplicates("txn_id", keep="last").set_index("txn_id").is_fraud
        hit = tr_all.txn_id.isin(fb.index)
        tr_all.loc[hit, "is_fraud"] = tr_all.loc[hit, "txn_id"].map(fb).astype(int)
        print(f"applied {int(hit.sum())} analyst-feedback labels to the training data")

    # carve a validation slice (last 20% of train, by time) for model selection + threshold tuning
    cut = tr_all.ts.quantile(0.8)
    tr, va = tr_all[tr_all.ts < cut].reset_index(drop=True), tr_all[tr_all.ts >= cut].reset_index(drop=True)
    print(f"train {len(tr):,} | valid {len(va):,} | test {len(te):,} | fraud rate "
          f"{tr.is_fraud.mean():.2%} / {va.is_fraud.mean():.2%} / {te.is_fraud.mean():.2%}")

    Xtr, ytr, Xva, yva, Xte, yte = tr[FEATURES], tr.is_fraud.values, va[FEATURES], va.is_fraud.values, te[FEATURES], te.is_fraud.values
    pos_w = (ytr == 0).sum() / max((ytr == 1).sum(), 1)

    models = make_models(pos_w)
    results, val_scores, test_scores = {}, {}, {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        val_scores[name], test_scores[name] = m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]
        results[name] = dict(valid=ranking_metrics(yva, val_scores[name]), test=ranking_metrics(yte, test_scores[name]))

    # unsupervised baseline: Isolation Forest on behaviour-deviation features only (never sees labels)
    anomaly = BehaviorAnomaly(SEED).fit(tr)
    results["Isolation Forest (unsupervised)"] = dict(
        valid=ranking_metrics(yva, anomaly.score(va)), test=ranking_metrics(yte, anomaly.score(te)))

    # pick the best supervised model by time-series CV PR-AUC (test is only used for the final report)
    cv = cv_pr_auc(tr_all, {n: results[n]["valid"]["pr_auc"] for n in models})
    for n in models:
        results[n]["cv_pr_auc"] = cv[n]
    best = max(models, key=lambda n: cv[n])
    print(f"time-series CV PR-AUC: {cv}  ->  best model: {best}")
    th = tune_thresholds(val_scores[best], va.amount.values, yva)
    impact = business_impact(te.amount.values, yte, test_scores[best], th)
    recall_any, recall_strong = scenario_recall(te.scenario.values, test_scores[best], th)

    # How good are the tuned thresholds? Compare with thresholds tuned on the TEST set itself (an oracle).
    th_oracle = tune_thresholds(test_scores[best], te.amount.values, yte)
    cost = lambda t: float(total_cost(test_scores[best], te.amount.values, yte, t))   # noqa: E731
    quality = dict(
        calibration=calibration(yte, test_scores[best]),
        precision_by_action=precision_by_action(yte, test_scores[best], th),
        threshold_cost_bdt=round(cost(th)), oracle_threshold_cost_bdt=round(cost(th_oracle)),
        oracle_thresholds=th_oracle,
        scenario_recall_hold_or_block=recall_strong,
    )

    # ---------- save artefacts ----------
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    joblib.dump(dict(model=models[best], anomaly=anomaly, name=best, features=FEATURES, thresholds=th,
                     thresholds_model=list(th), version=ARTIFACT_VERSION,
                     trained_on=dict(rows=len(tr), until=str(tr.ts.max()))), ROOT / "models" / "risk_engine.joblib")

    # global explainability: mean |contribution| per feature on a test sample
    shap_imp = Explainer(models[best], FEATURES).global_importance(Xte.sample(min(1500, len(Xte)), random_state=1))
    (ROOT / "reports" / "shap_importance.json").write_text(json.dumps(shap_imp.round(4).to_dict(), indent=2))
    metrics = dict(version=ARTIFACT_VERSION, best_model=best, split=dict(train=len(tr), valid=len(va), test=len(te)),
                   models=results, business_impact=impact, scenario_recall=recall_any, quality=quality,
                   notes="Synthetic data with ~6% label noise. Absolute numbers are optimistic; use for model comparison. "
                         "Scores are class-weighted ranking scores, not calibrated probabilities (see quality.calibration).")
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2))

    lines = ["| Model | ROC-AUC | PR-AUC | Precision@3% | Recall@3% |", "|---|---|---|---|---|"]
    for n, r in results.items():
        t = r["test"]
        lines.append(f"| {n} | {t['roc_auc']} | {t['pr_auc']} | {t['precision@3%']} | {t['recall@3%']} |")
    (ROOT / "reports" / "model_comparison.md").write_text("\n".join(lines) + "\n")

    # plots: precision-recall curves + feature importance of the best model
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for n in models:
        p, r, _ = precision_recall_curve(yte, test_scores[n])
        ax[0].plot(r, p, label=f"{n} (AP={results[n]['test']['pr_auc']:.2f})")
    ax[0].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall on test set")
    ax[0].legend(loc="lower left")
    bm = models[best]
    imp = getattr(bm, "feature_importances_", None)
    if imp is None:
        imp = np.abs(bm[-1].coef_[0])
    s = pd.Series(imp, index=FEATURES).sort_values().tail(12)
    ax[1].barh(s.index, s.values)
    ax[1].set(title=f"Top features ({best})")
    plt.tight_layout()
    plt.savefig(ROOT / "reports" / "model_comparison.png", dpi=130)

    print("\n" + "\n".join(lines))
    print(f"\nThresholds (otp, hold, block): {th}")
    print(f"Fraud value prevented: {impact['pct_fraud_value_prevented']:.1%} | legit txns with friction: "
          f"{impact['legit_txns_with_friction_pct']:.2%} | net benefit BDT {impact['net_benefit_bdt']:,.0f}")
    print("Recall by scenario - any action:", recall_any)
    print("Recall by scenario - hold/block:", recall_strong)
    print(f"Precision by action: {quality['precision_by_action']} | calibration: {quality['calibration']}")
    print(f"Test cost with validation-tuned thresholds BDT {quality['threshold_cost_bdt']:,} "
          f"vs. oracle thresholds {th_oracle}: BDT {quality['oracle_threshold_cost_bdt']:,}")
    print("Top global factors:", ", ".join(f"{k} ({v:.3f})" for k, v in shap_imp.head(5).items()))

    from ml.cache import build_cache
    build_cache()          # so the backend boots instantly from data/cache/scored_cache.parquet


if __name__ == "__main__":
    main()
