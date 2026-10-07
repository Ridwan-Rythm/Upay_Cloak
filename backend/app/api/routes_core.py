"""Core API (/api/v1): config, scoring through the ML model, transactions, case queue, feedback, KPIs, metrics,
and the customer pre-send warning check. Every score here comes from the trained model via ScoringService."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from backend.app.contracts.interfaces import Container
from backend.app.contracts.schemas import (
    ACTION_SEVERITY,
    Action,
    Case,
    Page,
    ScoredTransaction,
    Transaction,
    Verdict,
)
from backend.app.contracts.schemas_core import (
    ConfigInfo,
    ImpactMetrics,
    KpiSummary,
    ScoreRequest,
    ScoreResponse,
    WarningRequest,
    WarningResult,
)
from backend.app.deps import get_container
from backend.app.services.reasons import CUSTOMER_TEXT, feature_key, rule_key

router = APIRouter(prefix="/api/v1", tags=["core"])

COLD_START_NUDGE_BDT = 15_000     # wallets with no history: nudge on large amounts (the model has no baseline)


def err(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"error": {"code": code, "message": message}})


def require_model(container: Container):
    if container.scoring is None:
        raise err(503, "model_not_loaded", "ML model artifact not found. Run `python -m ml.train` and restart the server.")
    return container.scoring


# ---------------------------------------------------------------- config
@router.get("/config", response_model=ConfigInfo)
def get_config(container: Container = Depends(get_container)) -> ConfigInfo:
    from backend.app.config import get_settings
    st = get_settings()
    sc = container.scoring
    th_model = sc.thresholds_model if sc else []
    th = sc.thresholds_active if sc else []
    hist = sc.history if sc else None
    return ConfigInfo(
        contract_version="1.0", app_version="1.1.0", model_loaded=sc is not None,
        intelligence="live" if container.intel.live else "stubs",
        thresholds_model=th_model, thresholds_active=th, thresholds_version=sc.decision.version if sc else "none",
        risk_levels=({
            "low": f"Allow (model score < {th[0]:.2f})",
            "medium": f"Step-up verification (score {th[0]:.2f} to {th[1]:.2f}) or a policy rule",
            "high": f"Hold for analyst review (score {th[1]:.2f} to {th[2]:.2f}) or a corroborated policy rule",
            "critical": f"Block or freeze (score >= {th[2]:.2f}), ring membership or OTP-breach pattern",
        } if sc else {}),
        actions=list(Action), languages=["en", "bn"], llm_provider=st.llm_provider,
        llm_enabled=bool(st.llm_api_key) and st.llm_provider != "none",
        dataset=({"rows": int(len(hist)), "start": str(hist.ts.min()), "end": str(hist.ts.max()),
                  "model": sc.engine.name, "model_version": sc.engine.version,
                  "features": len(sc.engine.features)} if sc is not None else {}))


# ---------------------------------------------------------------- scoring
@router.post("/score", response_model=ScoreResponse)
def score_transaction(req: ScoreRequest, container: Container = Depends(get_container)) -> ScoreResponse:
    sc = require_model(container)
    if req.txn_id:
        scored = container.cases.get_scored(req.txn_id)
        if scored is None:
            raise err(404, "txn_not_found", f"Transaction {req.txn_id} not found")
        return ScoreResponse(scored=scored, latency_ms=0.0, feature_mode="replay")
    if req.transaction is None:
        raise err(422, "validation_error", "Either txn_id or transaction must be provided")
    txn = req.transaction
    if txn.txn_id:
        existing = container.cases.get_scored(txn.txn_id)
        if existing is not None:                                   # idempotent: never duplicates a case
            return ScoreResponse(scored=existing, latency_ms=0.0, feature_mode="replay")
    else:
        txn = txn.model_copy(update={"txn_id": f"LIVE-{uuid.uuid4().hex[:8].upper()}"})
    try:
        scored, ms, warnings = sc.score_live(txn, commit=req.commit)
    except ValueError as e:
        raise err(422, "validation_error", str(e)) from e
    if req.commit:
        sc.ingest_committed(txn)
        container.cases.register_live(scored, txn.ts)
    return ScoreResponse(scored=scored, latency_ms=ms, feature_mode="live", warnings=warnings)


# ---------------------------------------------------------------- transactions / cases
def _page(items: list, total: int, offset: int, limit: int) -> Page:
    nxt = str(offset + limit) if offset + limit < total else None
    return Page(items=items, total=total, next_cursor=nxt)


@router.get("/transactions", response_model=Page[ScoredTransaction])
def list_transactions(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0), risk_level: str | None = None,
                      user_id: str | None = None, since: str | None = None,
                      container: Container = Depends(get_container)) -> Page[ScoredTransaction]:
    items, total = container.cases.list_transactions(limit=limit, offset=offset, risk_level=risk_level,
                                                     user_id=user_id, since=since)
    return _page(items, total, offset, limit)


@router.get("/cases", response_model=Page[Case])
def list_cases(status: str | None = None, alert_type: str | None = None, risk_level: str | None = None,
               min_amount: float | None = None, sort: str = Query("priority", pattern="^(priority|time)$"),
               limit: int = Query(50, ge=1, le=500), cursor: str | None = None,
               container: Container = Depends(get_container)) -> Page[Case]:
    items = container.cases.list_cases(status=status, alert_type=alert_type, risk_level=risk_level,
                                       min_amount=min_amount, sort=sort)
    offset = int(cursor) if cursor and cursor.isdigit() else 0
    return _page(items[offset: offset + limit], len(items), offset, limit)


@router.get("/cases/{case_id}", response_model=Case)
def get_case_by_id(case_id: str, container: Container = Depends(get_container)) -> Case:
    case = container.cases.get_case(case_id)
    if not case:
        raise err(404, "case_not_found", f"Case {case_id} not found")
    return case


@router.post("/cases/{case_id}/feedback", response_model=Case)
def submit_feedback(case_id: str, verdict: Verdict = Query(...), analyst: str = Query("analyst"), note: str | None = None,
                    container: Container = Depends(get_container)) -> Case:
    try:
        return container.cases.record_feedback(case_id=case_id, verdict=verdict, analyst=analyst, note=note)
    except KeyError:
        raise err(404, "case_not_found", f"Case {case_id} not found") from None


@router.get("/feedback/export")
def export_feedback(container: Container = Depends(get_container)) -> Response:
    """CSV (txn_id, decision, ...) that `python -m ml.feedback --labels <file>` can consume."""
    return Response(content=container.cases.export_feedback_csv(), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=analyst_feedback.csv"})


# ---------------------------------------------------------------- dashboard numbers
@router.get("/kpis", response_model=KpiSummary)
def get_kpis(container: Container = Depends(get_container)) -> KpiSummary:
    return container.cases.get_kpis()


@router.get("/metrics", response_model=ImpactMetrics)
def get_metrics(container: Container = Depends(get_container)) -> ImpactMetrics:
    sc = require_model(container)
    m = sc.metrics
    if not m:
        raise err(503, "report_missing", "reports/metrics.json not found. Run `python -m ml.train`.")
    best = m["models"][m["best_model"]]["test"]
    bi = m["business_impact"]
    return ImpactMetrics(
        n_test=m["split"]["test"], roc_auc=best["roc_auc"], pr_auc=best["pr_auc"],
        precision_at_1pct=best["precision@1%"], recall_at_1pct=best["recall@1%"],
        fraud_value_total_bdt=bi["fraud_value_total_bdt"], fraud_value_prevented_bdt=bi["fraud_value_prevented_bdt"],
        pct_fraud_value_prevented=bi["pct_fraud_value_prevented"],
        legit_txns_with_friction_pct=bi["legit_txns_with_friction_pct"], alert_rate_pct=bi["alert_rate_pct"],
        net_benefit_bdt=bi["net_benefit_bdt"], analyst_queue_size=bi["analyst_queue_size"], actions=bi["actions"],
        scenario_recall=m.get("scenario_recall", {}), notes=m.get("notes", ""),
        analyst_feedback=container.cases.feedback_stats())


# ---------------------------------------------------------------- customer pre-send warning
HEAD = {
    "en": {"caution": ("Caution: please double-check this transfer", "Take a moment before you continue."),
           "danger": ("Stop: this transfer looks risky", "We recommend you do not continue. No money moves until you verify.")},
    "bn": {"caution": ("সতর্কতা: লেনদেনটি আবার যাচাই করুন", "চালিয়ে যাওয়ার আগে একটু ভেবে দেখুন।"),
           "danger": ("থামুন: এই লেনদেনটি ঝুঁকিপূর্ণ মনে হচ্ছে", "আমরা চালিয়ে না যাওয়ার পরামর্শ দিচ্ছি। যাচাই না করা পর্যন্ত টাকা যাবে না।")},
}
NONE_TEXT = {"en": ("Looks normal", "This transfer matches your usual activity."),
             "bn": ("স্বাভাবিক মনে হচ্ছে", "এই লেনদেনটি আপনার স্বাভাবিক লেনদেনের সাথে মিলে যায়।")}


@router.post("/warning/check", response_model=WarningResult)
def warning_check(req: WarningRequest, container: Container = Depends(get_container)) -> WarningResult:
    """Score the transfer BEFORE it is sent and explain it in plain language (never raw scores)."""
    sc = require_model(container)
    past = container.cases.user_history(req.user_id, datetime(2100, 1, 1)) if hasattr(container.cases, "user_history") else None
    last = past.iloc[-1] if past is not None and len(past) else None
    cat = req.merchant_category
    txn = Transaction(
        ts=datetime.now(timezone.utc), user_id=req.user_id, type=req.type, amount=req.amount,
        recipient_id=req.recipient_id, device_id=req.device_id or (str(last.device_id) if last is not None else "unknown"),
        location=req.location or (str(last.location) if last is not None else "unknown"),
        balance_before=float(past.balance_before.median()) if last is not None else max(req.amount * 3, 1.0),
        merchant_category=cat)
    scored, _, _ = sc.score_live(txn)
    action = scored.what_next.action
    cold = not sc.has_baseline(req.user_id)
    if cold and action == Action.ALLOW and req.amount >= COLD_START_NUDGE_BDT and req.type in ("TRANSFER", "CASH_OUT"):
        action = Action.OTP_STEP_UP                                   # no baseline: be careful with large amounts
    sev = ACTION_SEVERITY[action]
    severity = "none" if sev == 0 else "caution" if sev == 1 else "danger"
    keys: list[str] = []
    wr = scored.why_risky
    keys += [feature_key(r.feature) for r in sorted(wr.reasons, key=lambda r: -r.shap) if r.shap > 0]
    keys += [rule_key(h.rule_id) for h in wr.rule_trace]
    if wr.graph and wr.graph.ring_id:
        keys.append("mule")
    keys = [k for k in dict.fromkeys(keys) if k != "other"][:3] or (["amt"] if cold and severity != "none" else [])
    lang = req.language
    head, body = NONE_TEXT[lang] if severity == "none" else HEAD[lang][severity]
    evidence = [f"FACTOR-{r.feature}" for r in wr.reasons if r.shap > 0][:3] + [f"RULE-{h.rule_id}" for h in wr.rule_trace]
    if cold:
        evidence.append("METRIC-cold_start")
    return WarningResult(show_warning=severity != "none", severity=severity, language=lang, headline=head, body=body,
                         reasons=[CUSTOMER_TEXT[lang][k] for k in keys], suggested_action=action, evidence_ids=evidence)
