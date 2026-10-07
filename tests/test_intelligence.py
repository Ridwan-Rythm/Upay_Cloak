"""Graph (mule rings), agent risk and policy engine."""
import pandas as pd

from backend.app.contracts.schemas import Action, GraphSignals
from backend.app.intelligence.graph import NetworkXGraphService
from backend.app.services.decision_service import DecisionEngine


def test_rings_are_mules_not_busy_wallets(container, history):
    """The old detector flagged every agent / merchant / popular wallet (870 false positives)."""
    g = container.intel.graph
    flagged = g.ring_wallets()
    mules = set(history[history.user_id.str.startswith("MU")].user_id) | set(history[history.recipient_id.str.startswith("MU")].recipient_id)
    assert len(g.rings()) == 6                         # the generator injects 6 rings
    assert len(flagged & mules) >= 0.9 * len(mules)    # finds (nearly) every mule ...
    assert flagged - mules == set()                    # ... and nothing else (tightened from <= 2 in M2.1)
    import re
    assert not any(re.fullmatch(r"(A|M|G)\d{3}|BK\d{2}", w) for w in flagged)      # no agents / merchants / betting sites / bookies


def test_ring_membership_is_point_in_time(container):
    g = container.intel.graph
    hub = max(g.rings()[0].wallet_ids, key=lambda w: g.G.nodes[w]["total_volume_bdt"])
    assert g.wallet_signals(hub).ring_id is not None
    assert g.wallet_signals(hub, as_of=pd.Timestamp("2026-01-01")).ring_id is None      # not yet detectable


def test_freeze_simulation_is_real_and_zero_when_nothing_matches():
    g = NetworkXGraphService()
    impact = g.simulate_freeze(["nobody"])             # used to return invented numbers (14 txns, BDT 38,000 ...)
    assert impact.blocked_txn_count == 0 and impact.blocked_value_bdt == 0


def test_freeze_simulation_stops_fraud(container):
    g = container.intel.graph
    r = g.rings()[0]
    impact = g.simulate_freeze(r.wallet_ids)
    assert impact.blocked_txn_count > 0 and impact.fraud_value_stopped_bdt > impact.legit_value_blocked_bdt


def test_rogue_agents_rank_first(container, history):
    rogue = set(history[history.scenario == "rogue_agent"].agent_id)
    scores = {}
    for d in pd.date_range("2026-01-08", "2026-01-30", freq="12h"):
        for a in container.intel.agents.leaderboard(top_n=40, as_of=d):
            scores[a.agent_id] = max(scores.get(a.agent_id, 0), a.risk_score)
    hot = {a for a, s in scores.items() if s >= 0.85}
    assert hot == rogue                                # no fake seeded agents, no innocent agents


def test_no_fake_demo_agents(container):
    assert not any(a.agent_id.startswith("AGT-") for a in container.intel.agents.leaderboard(top_n=50))


# ---- policy engine: rules alone must not hold / block legitimate customers
def test_rule_alone_is_capped_at_step_up():
    d = DecisionEngine([0.05, 0.10, 0.20])
    out = d.evaluate(0.001, 5000, rule_tags=["account_takeover"])
    assert out.action == Action.OTP_STEP_UP and out.base_action == Action.ALLOW
    assert "capped" in out.policy_trace[0].text


def test_rule_with_model_support_escalates():
    d = DecisionEngine([0.05, 0.10, 0.20])
    assert d.evaluate(0.07, 5000, rule_tags=["otp_breach"]).action == Action.BLOCK
    assert d.evaluate(0.07, 5000, rule_tags=["gambling_repeat"]).action == Action.HOLD


def test_agent_risk_never_punishes_a_clean_transaction():
    d = DecisionEngine([0.05, 0.10, 0.20])
    assert d.evaluate(0.001, 5000, agent_risk_score=0.95).action == Action.ALLOW          # was: HOLD for every customer
    assert d.evaluate(0.15, 5000, agent_risk_score=0.95).action == Action.HOLD


def test_ring_member_is_frozen_and_policies_never_lower():
    d = DecisionEngine([0.05, 0.10, 0.20])
    sig = GraphSignals(wallet_id="w", ring_id="RING-x", ring_score=0.9)
    assert d.evaluate(0.001, 100, graph_signals=sig).action == Action.FREEZE_WALLET
    assert d.evaluate(0.9, 100, rule_tags=["gambling"]).action == Action.BLOCK            # step-up policy cannot lower a block


def test_betting_triggers_confirmation_and_alert_type():
    out = DecisionEngine([0.05, 0.10, 0.20]).evaluate(0.001, 800, rule_tags=["gambling"])
    assert out.action == Action.OTP_STEP_UP and out.alert_type.value == "gambling"
    assert DecisionEngine([0.05, 0.1, 0.2]).evaluate(0.001, 8, rule_tags=["otp_breach"]).alert_type.value == "otp_breach"
