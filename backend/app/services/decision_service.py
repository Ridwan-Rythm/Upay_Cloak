"""Policy and decision engine for UpayShield.

The ML model proposes an action from its risk score using the thresholds learned by `ml.train`
(optionally re-tuned from analyst feedback). Policy rules may only RAISE the severity, never lower it,
and every raise is recorded in `policy_trace` so it can be audited.
"""
from __future__ import annotations

from backend.app.contracts.schemas import (
    ACTION_SEVERITY,
    ALERT_TYPE_PRECEDENCE,
    SEVERITY_TO_LEVEL,
    TAG_TO_ALERT_TYPE,
    Action,
    AlertType,
    Decision,
    GraphSignals,
    PolicyHit,
)

DEFAULT_THRESHOLDS = [0.05, 0.10, 0.20]       # only used if no model artifact is available (tests / degraded mode)


class DecisionEngine:
    def __init__(self, thresholds: list[float] | None = None, version: str = "model") -> None:
        self.thresholds = list(thresholds) if thresholds else list(DEFAULT_THRESHOLDS)   # [otp, hold, block]
        self.version = version

    def base_recommend(self, score: float) -> Action:
        t_otp, t_hold, t_block = self.thresholds
        if score >= t_block:
            return Action.BLOCK
        if score >= t_hold:
            return Action.HOLD
        if score >= t_otp:
            return Action.OTP_STEP_UP
        return Action.ALLOW

    def evaluate(
        self,
        risk_score: float,
        amount_bdt: float,
        rule_tags: list[str] | None = None,
        graph_signals: GraphSignals | None = None,
        agent_risk_score: float = 0.0,
        is_ring_member: bool = False,
    ) -> Decision:
        rule_tags = rule_tags or []
        base_action = self.base_recommend(risk_score)
        current = base_action
        trace: list[PolicyHit] = []

        def raise_to(target: Action, policy_id: str, text: str) -> None:
            nonlocal current
            if ACTION_SEVERITY[target] > ACTION_SEVERITY[current]:       # policies never lower the action
                current = target
                trace.append(PolicyHit(policy_id=policy_id, text=text, raises_to=target))

        corroborated = ACTION_SEVERITY[base_action] >= ACTION_SEVERITY[Action.OTP_STEP_UP]   # model also sees risk

        def soft(target: Action, policy_id: str, text: str) -> None:
            """Rule-based escalation. A rule alone may only ask for step-up verification; HOLD/BLOCK need the model
            to corroborate. (Measured: rules alone were wrong for legitimate travellers / customers of a flagged agent.)"""
            if corroborated:
                raise_to(target, policy_id, text)
            else:
                raise_to(Action.OTP_STEP_UP, policy_id, text + " Model did not corroborate, so capped at step-up.")

        # Policy 1: mule-ring membership -> freeze wallet (network evidence is strong: 100% precise on held-out data)
        if is_ring_member or (graph_signals and graph_signals.ring_id):
            raise_to(Action.FREEZE_WALLET, "POL_RING_FREEZE",
                     "Wallet is a member of a detected money-mule ring; escalate to wallet freeze.")
        # Policy 2: rogue agent -> hold, but only when this transaction itself looks risky to the model
        if agent_risk_score >= 0.85 and corroborated:
            raise_to(Action.HOLD, "POL_AGENT_ANOMALY",
                     "Processing agent scores >= 0.85 against its peers and the transaction is flagged; hold it.")
        # Policy 3: account takeover signals -> block
        if "account_takeover" in rule_tags:
            soft(Action.BLOCK, "POL_ATO_BLOCK",
                 "Account-takeover signals (new device + new location, or night-time burst).")
        # Policy 4: scam-victim pattern -> step-up verification / warning
        if "scam_victim" in rule_tags:
            raise_to(Action.OTP_STEP_UP, "POL_SCAM_PROTECT",
                     "First-time recipient with an amount far above the user's norm; require step-up verification.")
        # Policy 5: OTP given to / used by someone else -> block and force re-authentication
        if "otp_breach" in rule_tags:
            soft(Action.BLOCK, "POL_OTP_BREACH",
                 "OTP confirmed from a different device while another session was active (or repeated OTP failures): "
                 "possible OTP sharing / remote takeover.")
        if "sim_swap" in rule_tags:
            soft(Action.HOLD, "POL_SIM_SWAP", "Recent SIM swap combined with a new device.")
        # Policy 6: betting payments -> confirmation with a responsible-use warning; repeated sessions -> hold
        if "gambling" in rule_tags:
            raise_to(Action.OTP_STEP_UP, "POL_GAMBLING",
                     "Payment to a betting / gambling service; require confirmation and show a responsible-use warning.")
        if "gambling_repeat" in rule_tags:
            soft(Action.HOLD, "POL_GAMBLING_REPEAT",
                 "Repeated betting payments within 24 hours (loss-chasing).")

        alert_type: AlertType | None = None
        for tag in rule_tags:
            mapped = TAG_TO_ALERT_TYPE.get(tag)
            if mapped and (alert_type is None or ALERT_TYPE_PRECEDENCE.index(mapped) < ALERT_TYPE_PRECEDENCE.index(alert_type)):
                alert_type = mapped
        if alert_type is None and current != Action.ALLOW:
            if is_ring_member or (graph_signals and graph_signals.ring_id):
                alert_type = AlertType.MULE_NETWORK
            elif agent_risk_score >= 0.85:
                alert_type = AlertType.AGENT_ANOMALY
            else:
                alert_type = AlertType.ANOMALY

        return Decision(
            action=current,
            risk_level=SEVERITY_TO_LEVEL[ACTION_SEVERITY[current]],
            base_action=base_action,
            alert_type=alert_type,
            policy_trace=trace,
            requires_analyst=current in (Action.HOLD, Action.ESCALATE),
            priority=round(risk_score * amount_bdt, 2),
            thresholds_version=self.version,
        )
