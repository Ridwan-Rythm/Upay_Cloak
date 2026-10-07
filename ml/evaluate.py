"""Metrics: ML quality + business impact (not just AUC). Used by ml/train.py."""
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from ml.decision import ACTIONS, FRICTION, STOP_RATE, action_idx


def ranking_metrics(y, s, ks=(0.01, 0.03)):
    """ROC/PR-AUC plus precision/recall when only the top k% highest scores are flagged."""
    m = dict(roc_auc=roc_auc_score(y, s), pr_auc=average_precision_score(y, s))
    order = np.argsort(-s)
    for k in ks:
        top = y[order[: max(1, int(len(y) * k))]]
        m[f"precision@{k:.0%}"] = float(top.mean())
        m[f"recall@{k:.0%}"] = float(top.sum() / y.sum())
    return {k: round(float(v), 4) for k, v in m.items()}


def business_impact(amount, y, score, th):
    a = action_idx(score, th)
    legit = y == 0
    fraud_total = amount[y == 1].sum()
    prevented = (amount * STOP_RATE[a])[y == 1].sum()
    friction = FRICTION[a][legit].sum()
    return dict(
        thresholds=dict(zip(ACTIONS[1:], th)),
        fraud_value_total_bdt=float(fraud_total),
        fraud_value_prevented_bdt=float(prevented),
        pct_fraud_value_prevented=round(float(prevented / fraud_total), 4),
        legit_txns_with_friction_pct=round(float((a[legit] > 0).mean()), 4),
        alert_rate_pct=round(float((a > 0).mean()), 4),
        friction_cost_bdt=float(friction),
        net_benefit_bdt=float(prevented - friction),
        analyst_queue_size=int((a >= 2).sum()),
        actions={ACTIONS[i]: int((a == i).sum()) for i in range(4)},
    )


def precision_by_action(y, score, th):
    """Share of transactions that are labelled fraud, per recommended action (how trustworthy each tier is)."""
    a = action_idx(score, th)
    return {ACTIONS[i]: (round(float(y[a == i].mean()), 4) if (a == i).any() else None) for i in range(4)}


def scenario_recall(scenarios, score, th):
    """Share of each injected scenario that is caught. Uses the TRUE scenario (ignores the ~6% label noise).
    Returns (any action, hold-or-block only)."""
    a = action_idx(score, th)
    scen = np.asarray(scenarios)
    any_, strong = {}, {}
    for s in sorted(set(scen) - {"normal"}):
        m = scen == s
        any_[s] = round(float((a[m] > 0).mean()), 3)
        strong[s] = round(float((a[m] >= 2).mean()), 3)
    return any_, strong


def calibration(y, s, bins=10):
    """Brier score + expected calibration error. Class-weighted models are NOT calibrated probabilities."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(s, edges) - 1, 0, bins - 1)
    ece = sum((idx == b).mean() * abs(y[idx == b].mean() - s[idx == b].mean()) for b in range(bins) if (idx == b).any())
    return dict(brier=round(float(brier_score_loss(y, np.clip(s, 0, 1))), 4), ece=round(float(ece), 4))
