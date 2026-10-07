"""Frontend API (/api): thin adapters that turn the core ScoredTransaction / Case objects (all produced by the ML
model) into the JSON the static dashboard draws. No scoring logic lives here."""
from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from backend.app.contracts.interfaces import Container
from backend.app.contracts.schemas import Action, Case, Language, ScoredTransaction, Transaction
from backend.app.deps import get_container
from backend.app.services.reasons import (
    feature_key,
    frontend_decision,
    purpose_label,
    rule_key,
)
from backend.app.store.case_store import case_from_txn
from ml.rules import purpose_of

router = APIRouter(prefix="/api", tags=["frontend"])

TYPE_LABEL = {"TRANSFER": "Transfer", "CASH_OUT": "Cash-out", "CASH_IN": "Cash-in", "PAYMENT": "Payment"}
UI_TYPES = {"send_money": "TRANSFER", "transfer": "TRANSFER", "cash_out": "CASH_OUT", "cash_in": "CASH_IN",
            "payment": "PAYMENT", "pay_bill": "PAYMENT", "recharge": "PAYMENT"}
ACTION_LABEL = {
    "allow": ("Allow the transaction", "No policy or model signal asks for friction."),
    "otp_step_up": ("Ask the customer to verify", "Step-up verification keeps legitimate customers moving while stopping takeovers."),
    "hold": ("Hold for analyst review", "Money stays in the wallet until an analyst confirms or releases the transaction."),
    "escalate": ("Escalate to a senior analyst", "Needs compliance sign-off."),
    "block": ("Block and contact the customer", "Strong signals: the transaction is stopped before money leaves the wallet."),
    "freeze_wallet": ("Freeze the wallet", "The wallet is part of a detected mule network; freezing cuts off the cash-out route."),
}
TAG_ACTIONS = {
    "otp_breach": ("Force re-authentication and end other sessions",
                   "OTP was used from another device while a second session was open. Tell the customer never to share OTPs, even with staff."),
    "sim_swap": ("Verify the customer through a trusted channel", "The SIM was changed recently; confirm it was the owner."),
    "gambling": ("Show a responsible-use warning and offer spending limits", "The payment goes to a betting service."),
    "gambling_repeat": ("Review for loss-chasing", "Several betting payments within 24 hours, with growing stakes."),
    "agent_risk": ("Review the agent's recent cash-outs", "The agent shows a burst of late-night cash-outs compared with its peers."),
    "structuring": ("Prepare a threshold report", "Repeated cash-outs just under the 50,000 BDT reporting limit."),
}


def err(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"error": {"code": code, "message": message}})


def need_model(c: Container):
    if c.scoring is None:
        raise err(503, "model_not_loaded", "ML model not loaded. Run `python -m ml.train` and restart.")
    return c.scoring


# ------------------------------------------------------------------ formatting helpers
def _iso(t: str) -> str:
    return t.replace(" ", "T")


def _model_reasons(s: ScoredTransaction) -> list[dict]:
    rs = sorted(s.why_risky.reasons, key=lambda r: -r.shap)
    return [dict(feature=feature_key(r.feature), ml_feature=r.feature, label=r.text, contribution=round(r.shap, 2),
                 source="model") for r in rs]


def _customer_keys(s: ScoredTransaction) -> list[str]:
    if s.what_next.action == Action.ALLOW:
        return []
    keys = [feature_key(r.feature) for r in sorted(s.why_risky.reasons, key=lambda r: -r.shap) if r.shap > 0]
    keys += [rule_key(h.rule_id) for h in s.why_risky.rule_trace]
    g = s.why_risky.graph
    if g and g.ring_id:
        keys.append("mule")
    if s.why_risky.agent and s.why_risky.agent.flags and s.what_happened.agent:
        keys.append("agent")
    return [k for k in dict.fromkeys(keys) if k != "other"][:4]


def _trace(s: ScoredTransaction) -> list[dict]:
    out = [dict(source="rule", label=f"{h.rule_id}: {h.text}") for h in s.why_risky.rule_trace]
    out += [dict(source="policy", label=f"{p.policy_id}: {p.text}") for p in s.what_next.policy_trace]
    g = s.why_risky.graph
    if g and g.ring_id:
        out.append(dict(source="graph", label=f"Member of mule ring {g.ring_id} (ring score {g.ring_score:.2f})"))
    elif g and g.flags:
        out.append(dict(source="graph", label="Network signals: " + ", ".join(f.replace("_", " ") for f in g.flags)))
    a = s.why_risky.agent
    if a and a.flags:
        out.append(dict(source="agent", label=f"Agent {a.agent_id} (risk {a.risk_score:.2f}): " + ", ".join(f.replace("_", " ") for f in a.flags)))
    return out


def f_txn(s: ScoredTransaction) -> dict:
    wh, wr, wn = s.what_happened, s.why_risky, s.what_next
    return dict(
        id=s.txn_id, sender=s.user_id, receiver=wh.recipient, type=wh.type.lower(), amount_bdt=wh.amount_bdt,
        timestamp=_iso(wh.time), risk_score=wr.risk_score, risk_level=wn.risk_level.value,
        decision=frontend_decision(wn.action.value, wr.tags), action=wn.action.value, purpose=wh.purpose,
        purpose_label=purpose_label(wh.purpose), device_id=wh.device, location=wh.location, tags=wr.tags,
        reasons=[r for r in _model_reasons(s) if r["contribution"] > 0][:4], customer_keys=_customer_keys(s),
        case_id=s.case_id)


def _sentence(s: ScoredTransaction) -> str:
    wh = s.what_happened
    dest = f"{wh.recipient}" if wh.type != "PAYMENT" else f"{wh.recipient} ({purpose_label(wh.purpose).lower()})"
    extra = f" Purpose: {purpose_label(wh.purpose).lower()}." if wh.type != "PAYMENT" else ""
    return (f"{TYPE_LABEL.get(wh.type, wh.type)} of ৳{wh.amount_bdt:,.0f} from {s.user_id} to {dest} at {wh.time} "
            f"using device {wh.device} in {wh.location}.{extra}")


def _entity(c: Case) -> dict:
    wh = c.scored.what_happened
    if c.alert_type and c.alert_type.value == "agent_anomaly" and wh.agent:
        return dict(id=wh.agent, type="agent")
    return dict(id=c.scored.user_id, type="wallet")


def f_case_summary(c: Case) -> dict:
    s = c.scored
    return dict(
        case_id=c.case_id, severity=s.what_next.risk_level.value, status=c.status.value,
        alert_type=(c.alert_type.value if c.alert_type else "anomaly"), what_happened=_sentence(s), entity=_entity(c),
        amount_bdt=s.what_happened.amount_bdt, action=s.what_next.action.value,
        decision=frontend_decision(s.what_next.action.value, s.why_risky.tags), risk_score=s.why_risky.risk_score,
        purpose_label=purpose_label(s.what_happened.purpose), opened_at=_iso(s.what_happened.time), priority=c.priority)


# ------------------------------------------------------------------ user context for the case page
def _baseline(hist: pd.DataFrame) -> dict:
    if len(hist) < 5:
        none = "No history yet (new wallet)"
        return dict(usual_amount=none, usual_hours=none, usual_devices=none, usual_location=none, usual_spending=none)
    q = hist.amount.quantile([.25, .5, .75])
    hrs = hist.ts.dt.hour.quantile([.1, .9])
    loc = hist.location.value_counts()
    cats = hist.merchant_category.dropna().value_counts()
    bets = int((hist.merchant_category == "gambling").sum())
    spend = ", ".join(purpose_label(c).lower() for c in cats.index[:3]) or "mostly person-to-person and cash"
    return dict(
        usual_amount=f"৳{q[.25]:,.0f} to ৳{q[.75]:,.0f} (median ৳{q[.5]:,.0f})",
        usual_hours=f"{int(hrs[.1]):02d}:00 to {int(hrs[.9]):02d}:59",
        usual_devices=f"{hist.device_id.nunique()} known device{'s' if hist.device_id.nunique() > 1 else ''}",
        usual_location=f"{loc.index[0]} ({loc.iloc[0] / len(hist):.0%} of activity)",
        usual_spending=spend + (f"; {bets} betting payment(s) before" if bets else "; no betting before"))


def _row_txn(r) -> dict:
    cat = r.merchant_category if isinstance(r.merchant_category, str) else None
    return dict(id=r.txn_id, timestamp=_iso(str(r.ts)), amount_bdt=float(r.amount), type=str(r.type).lower(),
                receiver=r.recipient_id, device_id=r.device_id, location=r.location,
                purpose_label=purpose_label(purpose_of(r.type, cat)),
                risk_score=float(r.risk_score) if pd.notna(r.risk_score) else None,
                decision=frontend_decision(r.action, str(r.tags).split(",")) if isinstance(r.action, str) else "allow",
                f=dict(hr=pd.Timestamp(r.ts).hour))


def _recommended(c: Case) -> list[dict]:
    wn, wr = c.scored.what_next, c.scored.why_risky
    label, why = ACTION_LABEL[wn.action.value]
    if wn.policy_trace:
        why += " Policy: " + wn.policy_trace[-1].text
    out = [dict(label=label, rationale=why)]
    for tag in wr.tags:
        if tag in TAG_ACTIONS and (TAG_ACTIONS[tag][0], TAG_ACTIONS[tag][1]) not in [(o["label"], o["rationale"]) for o in out]:
            out.append(dict(label=TAG_ACTIONS[tag][0], rationale=TAG_ACTIONS[tag][1]))
    return out[:4]


def f_case_detail(c: Case, container: Container) -> dict:
    s, store = c.scored, container.cases
    wh, wr, wn = s.what_happened, s.why_risky, s.what_next
    ts = pd.Timestamp(wh.time)
    hist = store.user_history(s.user_id, ts)
    ent = _entity(c)

    # related transactions + timeline (the customer's own recent activity)
    prev = hist.sort_values("ts").tail(5)
    focal = dict(id=s.txn_id, timestamp=_iso(wh.time), amount_bdt=wh.amount_bdt, type=wh.type.lower(), receiver=wh.recipient,
                 device_id=wh.device, location=wh.location, purpose_label=purpose_label(wh.purpose),
                 risk_score=wr.risk_score, decision=frontend_decision(wn.action.value, wr.tags), f=dict(hr=ts.hour))
    related = [focal] + [_row_txn(r) for r in prev.iloc[::-1].itertuples(index=False)]
    near = prev[prev.ts >= ts - pd.Timedelta(hours=6)]
    timeline = []
    for i, r in enumerate(near.itertuples(index=False), 1):
        t = _row_txn(r)
        lvl = {"allow": "low", "warn": "medium", "step_up": "medium", "hold": "high", "block": "critical"}.get(t["decision"], "low")
        timeline.append(dict(id=f"E{i}", severity=lvl, title=f"{TYPE_LABEL.get(r.type, r.type)} ৳{r.amount:,.0f} to {r.recipient_id}",
                             timestamp=t["timestamp"].replace("T", " "), evidence_id=f"TXN-{r.txn_id.split('-', 1)[-1]}",
                             detail=f"{t['purpose_label']} · device {r.device_id} · {r.location}"))
    timeline.append(dict(id=f"E{len(timeline) + 1}", severity=wn.risk_level.value,
                         title=f"{TYPE_LABEL.get(wh.type, wh.type)} ৳{wh.amount_bdt:,.0f} to {wh.recipient} (this alert)",
                         timestamp=wh.time, evidence_id=f"TXN-{s.txn_id.split('-', 1)[-1]}",
                         detail=f"{purpose_label(wh.purpose)} · device {wh.device} · {wh.location} · action {wn.action.value}"))

    # graph neighbourhood, centre first
    gv = container.intel.graph.frontend_view(ent["id"], depth=1) if hasattr(container.intel.graph, "frontend_view") else \
        {"nodes": [], "edges": []}
    nodes = sorted(gv["nodes"], key=lambda n: n["id"] != ent["id"])

    bundle = container.intel.evidence.build(c)
    narr = container.intel.assistant.narrate(bundle)
    narrative = " ".join(x.text for x in narr.what_happened + narr.why_risky + narr.what_next)
    tier = container.scoring.tier_precision(wn.base_action.value if wn.base_action != Action.ALLOW else wn.action.value) \
        if container.scoring else None
    bars = _model_reasons(s)
    bars = sorted(bars, key=lambda r: -abs(r["contribution"]))[:8]
    return dict(
        **f_case_summary(c), created_at=c.created_at.isoformat(), timeline=timeline, why_risky=bars, trace=_trace(s),
        confidence=round(tier if tier is not None else wr.risk_score, 2),
        confidence_basis=("measured precision of this action tier on held-out data" if tier is not None else "model score"),
        baseline=_baseline(hist), recommended_actions=_recommended(c), related_transactions=related,
        purpose=dict(key=wh.purpose, label=purpose_label(wh.purpose), merchant_category=wh.merchant_category),
        subgraph=dict(nodes=nodes, edges=[e for e in gv["edges"]]), narrative=narrative, narrative_source=narr.source,
        evidence=[dict(id=i.id, kind=i.kind.value, label=i.label) for i in bundle.items][:40],
        policy_trace=[dict(id=p.policy_id, text=p.text) for p in wn.policy_trace],
        audit=container.cases.get_audit(c.case_id), model_score=wr.model_score, anomaly_score=wr.anomaly_score)


# ------------------------------------------------------------------ routes
@router.get("/overview")
def overview(container: Container = Depends(get_container)) -> dict:
    sc = need_model(container)
    k = container.cases.get_kpis()
    bi = (sc.metrics.get("business_impact") or {})
    held = sum(1 for c in container.cases._cases.values() if c.scored.what_next.action in (Action.HOLD, Action.BLOCK, Action.FREEZE_WALLET))
    lb = container.intel.agents.leaderboard(top_n=5)
    fp = bi.get("legit_txns_with_friction_pct")
    lat = sc.scorer.mean_latency_ms
    return dict(
        kpis=dict(scored=k.total_txns_scored, alerts=k.alerts, blocked=held, protected=k.blocked_value_bdt,
                  latency=f"{lat:.0f} ms" if lat else "n/a", fp=f"{fp:.2%}" if fp is not None else "n/a",
                  open_cases=k.open_cases),
        dist=k.risk_mix, trend=[b.count for b in k.alerts_over_time],
        trend_labels=[pd.Timestamp(b.ts).strftime("%d %b") for b in k.alerts_over_time],
        reasons=[[r.text, r.count] for r in k.top_reasons], by_type=k.alerts_by_type,
        agents=[dict(id=a.agent_id, score=a.risk_score, flags=a.flags) for a in lb],
        agent_median=container.intel.agents.peer_median_score() if hasattr(container.intel.agents, "peer_median_score") else 0.1,
        model=dict(name=sc.engine.name, version=sc.engine.version, thresholds=sc.thresholds_active))


@router.get("/stream")
def stream(seq: int = Query(0, ge=0), mode: str = Query("demo", pattern="^(demo|replay)$"),
           container: Container = Depends(get_container)) -> dict:
    """Event number `seq` of the simulated live feed (each one really scored by the model)."""
    need_model(container)
    s = container.cases.stream_pick(seq, mode)
    if s is None:
        raise err(404, "no_data", "No transactions to stream")
    return dict(seq=seq, txn=f_txn(s), interval_ms=container.cases.settings.stream_default_interval_ms)


@router.get("/transactions")
def transactions(n: int = Query(40, ge=1, le=200), mode: str = Query("demo", pattern="^(demo|replay)$"),
                 container: Container = Depends(get_container)) -> dict:
    """The first `n` events of the feed, newest first. A client continues the stream with seq=next_seq."""
    need_model(container)
    rows = [f_txn(container.cases.stream_pick(i, mode)) for i in range(n)][::-1]
    return dict(items=rows, next_seq=n)


@router.get("/cases")
def cases(alert_type: str | None = None, status: str | None = None, severity: str | None = None,
          limit: int = Query(100, ge=1, le=500), container: Container = Depends(get_container)) -> list[dict]:
    items = container.cases.list_cases(status=status or None, alert_type=alert_type or None, risk_level=severity or None)
    return [f_case_summary(c) for c in items[:limit]]


@router.get("/cases/count")
def case_count(container: Container = Depends(get_container)) -> dict:
    items = container.cases.list_cases(status="open")
    hi = sum(1 for c in items if c.scored.what_next.risk_level.value in ("high", "critical"))
    return dict(open=len(items), high_or_critical=hi)


@router.get("/cases/{case_id}")
def case_detail(case_id: str, container: Container = Depends(get_container)) -> dict:
    c = container.cases.get_case(case_id)
    if not c:
        raise err(404, "case_not_found", f"Case {case_id} not found")
    return f_case_detail(c, container)


class ActionBody(BaseModel):
    a: str


@router.post("/cases/{case_id}/action")
def case_action(case_id: str, body: ActionBody, container: Container = Depends(get_container)) -> dict:
    if body.a not in ("hold", "block", "kyc", "escalate", "false_positive"):
        raise err(422, "validation_error", f"Unknown action '{body.a}'")
    try:
        return container.cases.apply_action(case_id, body.a, by="Analyst")
    except KeyError:
        raise err(404, "case_not_found", f"Case {case_id} not found") from None


class AskBody(BaseModel):
    q: str
    language: Language = "en"


@router.post("/cases/{case_id}/ask")
def case_ask(case_id: str, body: AskBody, container: Container = Depends(get_container)) -> dict:
    c = container.cases.get_case(case_id)
    if not c:
        raise err(404, "case_not_found", f"Case {case_id} not found")
    if len(body.q.strip()) < 3:
        raise err(422, "validation_error", "Question is too short")
    bundle = container.intel.evidence.build(c)
    ans = container.intel.assistant.ask(bundle, body.q.strip()[:500], body.language)
    ids = list(dict.fromkeys(i for s in ans.sentences for i in s.evidence_ids))
    return dict(text=" ".join(s.text for s in ans.sentences), evidence_ids=ids, source=ans.source,
                answerable=ans.answerable, fallback_reason=ans.fallback_reason)


@router.get("/graph")
def graph(q: str | None = None, depth: int = Query(2, ge=1, le=4), container: Container = Depends(get_container)) -> dict:
    return container.intel.graph.frontend_view(q or None, depth)


@router.get("/graph/{entity_id}")
def graph_for(entity_id: str, depth: int = Query(2, ge=1, le=4), container: Container = Depends(get_container)) -> dict:
    return container.intel.graph.frontend_view(entity_id, depth)


# ------------------------------------------------------------------ warning demo
@router.get("/demo/scenarios")
def demo_scenarios(container: Container = Depends(get_container)) -> list[dict]:
    sc = need_model(container)
    out = []
    for key, p in sc.presets.items():
        row = sc.preset_row(key)
        out.append(dict(
            key=key, title_en=p["title_en"], title_bn=p["title_bn"], blurb_en=p["blurb_en"], blurb_bn=p["blurb_bn"],
            case_id=case_from_txn(p["txn_id"]) if container.cases.get_case(case_from_txn(p["txn_id"])) else None,
            txn=dict(sender=row["user_id"], receiver=row["recipient_id"], amount_bdt=float(row["amount"]), type=row["type"].lower(),
                     device_id=row["device_id"], location=row["location"],
                     purpose_label=purpose_label(purpose_of(row["type"], row["merchant_category"])))))
    return out


class FrontScore(BaseModel):
    preset: str | None = None               # score an edited copy of a real transaction (point-in-time)
    amount_bdt: float | None = None
    receiver: str | None = None
    sender: str | None = None
    type: str | None = None
    device_id: str | None = None
    location: str | None = None
    merchant_category: str | None = None
    balance_before: float | None = None


@router.post("/score")
def front_score(body: FrontScore, container: Container = Depends(get_container)) -> dict:
    sc = need_model(container)
    snapshot = None
    if body.preset:
        row = sc.preset_row(body.preset)
        if row is None:
            raise err(404, "preset_not_found", f"Unknown scenario '{body.preset}'")
        base = {k: (None if isinstance(row[k], float) and row[k] != row[k] else row[k]) for k in
                ("ts", "user_id", "type", "amount", "recipient_id", "agent_id", "device_id", "location", "balance_before",
                 "merchant_category", "otp_requests_10m", "otp_failures_10m", "otp_device_mismatch", "concurrent_sessions",
                 "sim_swap_recent")}
        snapshot = row["txn_id"]
        base["txn_id"] = f"DEMO-{row['txn_id']}"
    else:
        if not (body.sender and body.receiver and body.amount_bdt):
            raise err(422, "validation_error", "Provide either a preset or sender, receiver and amount_bdt")
        base = dict(ts=pd.Timestamp.now(), user_id=body.sender, type="TRANSFER", amount=body.amount_bdt,
                    recipient_id=body.receiver, device_id=body.device_id or "unknown", location=body.location or "unknown",
                    balance_before=body.balance_before or max(body.amount_bdt * 3, 1.0), txn_id=None)
    if body.amount_bdt:
        base["amount"] = body.amount_bdt
    if body.receiver:
        base["recipient_id"] = body.receiver
    if body.type:
        base["type"] = UI_TYPES.get(body.type.lower(), body.type.upper())
    for f in ("device_id", "location", "merchant_category"):
        if getattr(body, f):
            base[f] = getattr(body, f)
    if body.balance_before:
        base["balance_before"] = body.balance_before
    try:
        txn = Transaction(**base)
        scored, ms, warnings = sc.score_live(txn, snapshot_id=snapshot)
    except (ValueError, KeyError) as e:
        raise err(422, "validation_error", str(e)) from e
    t = f_txn(scored)
    return dict(risk_score=t["risk_score"], risk_level=t["risk_level"], decision=t["decision"], action=t["action"],
                latency_ms=ms, reasons=_model_reasons(scored)[:6], customer_keys=t["customer_keys"], tags=t["tags"],
                purpose_label=t["purpose_label"], warnings=warnings, trace=_trace(scored),
                case_id=case_from_txn(body.preset and sc.presets[body.preset]["txn_id"] or "") if body.preset and
                container.cases.get_case(case_from_txn(sc.presets[body.preset]["txn_id"])) else None)
