"""ML layer: the streaming feature path must equal the batch path, the model must be good and the evaluation honest."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.features import FEATURES, FeatureState, as_txn, build_features
from ml.rules import purpose_of, rule_trace

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def raw():
    return pd.concat([pd.read_csv(ROOT / "data" / f"{s}.csv", parse_dates=["ts"]) for s in ("train", "test")],
                     ignore_index=True).sort_values("ts", kind="stable").reset_index(drop=True)


def test_streaming_features_equal_batch(raw):
    """What the live API computes for a new transaction must be exactly what training saw."""
    batch = build_features(raw.iloc[:3000])
    state, bad = FeatureState(), 0
    for i, r in enumerate(raw.iloc[:3000].to_dict("records")):
        f = state.features(as_txn(r))
        state.update(as_txn(r))
        for k, v in f.items():
            bad += not np.isclose(v, float(batch.loc[i, k]), equal_nan=True)
    assert bad == 0


def test_features_are_point_in_time(raw):
    """Adding FUTURE transactions must not change the features of earlier ones."""
    a = build_features(raw.iloc[:1500])
    b = build_features(raw.iloc[:2500]).iloc[:1500]
    pd.testing.assert_frame_equal(a[FEATURES], b[FEATURES])


def test_all_model_features_exist_in_cache(history):
    assert set(FEATURES) <= set(history.columns)


def test_rules_cover_new_signals():
    base = {k: 0 for k in FEATURES}
    base.update(type="TRANSFER", amount=100.0, concurrent_sessions=1)
    otp = rule_trace({**base, "otp_device_mismatch": 1, "concurrent_sessions": 2})
    assert [h["tag"] for h in otp] == ["otp_breach"]
    assert [h["tag"] for h in rule_trace({**base, "otp_failures_10m": 3})] == ["otp_breach"]
    assert "gambling" in [h["tag"] for h in rule_trace({**base, "is_gambling": 1, "gambling_txn_24h": 1})]
    rep = [h["tag"] for h in rule_trace({**base, "is_gambling": 1, "gambling_txn_24h": 3})]
    assert "gambling_repeat" in rep
    # repeated-betting rule must not fire on the bettor's unrelated (non-betting) transactions
    assert "gambling_repeat" not in [h["tag"] for h in rule_trace({**base, "is_gambling": 0, "gambling_txn_24h": 4})]
    assert [h["tag"] for h in rule_trace({**base, "sim_swap_recent": 1, "is_new_device": 1})] == ["sim_swap"]


def test_purpose_of():
    assert purpose_of("PAYMENT", "gambling") == "betting"
    assert purpose_of("PAYMENT", "utilities") == "utilities"
    assert purpose_of("TRANSFER", None) == "person_to_person"
    assert purpose_of("CASH_OUT", float("nan")) == "cash_withdrawal"


def test_evaluate_module_imports_and_works():
    """ml/evaluate.py used to import a function that does not exist."""
    from ml.evaluate import business_impact, ranking_metrics, scenario_recall
    y = np.array([0, 0, 1, 1, 0, 1])
    s = np.array([.01, .02, .9, .8, .03, .3])
    assert ranking_metrics(y, s)["roc_auc"] == 1.0
    imp = business_impact(np.full(6, 1000.0), y, s, [0.05, 0.1, 0.5])
    assert imp["fraud_value_total_bdt"] == 3000
    assert scenario_recall(np.array(["normal"] * 3 + ["ato"] * 3), s, [0.05, 0.1, 0.5])[0]["ato"] == pytest.approx(2 / 3, abs=1e-3)


def test_model_quality_on_heldout_period():
    m = json.loads((ROOT / "reports" / "metrics.json").read_text())
    best = m["models"][m["best_model"]]["test"]
    assert best["roc_auc"] > 0.98 and best["pr_auc"] > 0.85
    # the selected model must not be clearly worse than another supervised model on the untouched test set
    others = [v["test"]["pr_auc"] for k, v in m["models"].items() if k != "Isolation Forest (unsupervised)"]
    assert best["pr_auc"] >= max(others) - 0.03
    assert m["business_impact"]["pct_fraud_value_prevented"] > 0.85
    assert m["business_impact"]["legit_txns_with_friction_pct"] < 0.03
    th = m["business_impact"]["thresholds"]
    assert th["otp_step_up"] <= th["hold"] <= th["block"]


def test_new_scenarios_are_detected():
    m = json.loads((ROOT / "reports" / "metrics.json").read_text())
    rec = m["scenario_recall"]
    for s in ("otp_breach", "gambling", "ato", "mule_passthrough", "structuring", "rogue_agent"):
        assert rec[s] >= 0.85, (s, rec[s])
    assert rec["scam_victim"] >= 0.7          # the hardest scenario: looks like a normal user making a big one-off transfer


def test_action_tiers_are_ordered_by_precision():
    p = json.loads((ROOT / "reports" / "metrics.json").read_text())["quality"]["precision_by_action"]
    assert p["allow"] < 0.01 and p["block"] > p["hold"] > p["allow"]


def test_cache_matches_model(container):
    from ml.cache import is_stale
    assert not is_stale(container.scoring.history, container.scoring.engine)


def test_explanations_do_not_contradict_values(container):
    """'Recipient has few distinct senders' used to be shown for a recipient with 28 senders."""
    from ml.explain import describe
    assert "few" not in describe("recipient_unique_senders", 28, -0.5)
    assert "28" in describe("recipient_unique_senders", 28, 0.5)
