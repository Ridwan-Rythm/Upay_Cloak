"""Operations API (/api/v1): SSE stream + stream control, and the analyst-feedback -> threshold learning loop."""
from __future__ import annotations

import asyncio

import pandas as pd
from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from backend.app.api.routes_core import err, get_config, require_model
from backend.app.contracts.interfaces import Container
from backend.app.contracts.schemas import Action, Verdict
from backend.app.contracts.schemas_core import ConfigInfo, StreamControl, StreamEvent, StreamState
from backend.app.deps import get_container
from backend.app.services.decision_service import DecisionEngine

router = APIRouter(prefix="/api/v1", tags=["ops"])
_STATE: dict[int, StreamState] = {}          # one simulated stream per running app (keyed by container identity)


def _state(c: Container) -> StreamState:
    return _STATE.setdefault(id(c), StreamState(running=False, mode="demo",
                                                interval_ms=c.cases.settings.stream_default_interval_ms, seq=0))


@router.get("/stream/state", response_model=StreamState)
def stream_state(container: Container = Depends(get_container)) -> StreamState:
    st = _state(container)
    s = container.cases.stream_pick(max(st.seq - 1, 0), st.mode)
    return st.model_copy(update={"sim_time": s.what_happened.time if s else None})


@router.post("/stream/control", response_model=StreamState)
def stream_control(ctl: StreamControl, container: Container = Depends(get_container)) -> StreamState:
    require_model(container)
    st = _state(container)
    if ctl.mode:
        st.mode = ctl.mode
    if ctl.interval_ms:
        st.interval_ms = ctl.interval_ms
    if ctl.action == "reset":
        st.seq, st.running = 0, False
    else:
        st.running = ctl.action in ("start", "resume")
    return stream_state(container)


@router.get("/stream")
async def stream(mode: str | None = Query(None, pattern="^(demo|replay)$"), interval_ms: int | None = Query(None, ge=0, le=60000),
                 limit: int = Query(50, ge=1, le=1000), container: Container = Depends(get_container)) -> StreamingResponse:
    """Server-sent events: each event is a real transaction scored by the model (`alert` when it is not an allow)."""
    require_model(container)
    st = _state(container)
    mode = mode or st.mode
    wait = (st.interval_ms if interval_ms is None else interval_ms) / 1000

    async def gen():
        for _ in range(limit):
            s = container.cases.stream_pick(st.seq, mode)
            ev = StreamEvent(seq=st.seq, sim_time=s.what_happened.time, kind="txn" if s.what_next.action == Action.ALLOW else "alert",
                             scored=s, alert_type=s.what_next.alert_type)
            st.seq += 1
            yield f"event: {ev.kind}\ndata: {ev.model_dump_json()}\n\n"
            if wait:
                await asyncio.sleep(wait)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


# ---------------------------------------------------------------- learning loop
def _apply_thresholds(c: Container, th: list[float], version: str) -> None:
    sc = c.scoring
    sc.engine.th = list(th)                         # the live scorer shares this engine object
    sc.decision = DecisionEngine(th, version=version)


@router.post("/admin/thresholds/adapt")
def adapt_thresholds(container: Container = Depends(get_container)) -> dict:
    """Re-tune [otp, hold, block] from analyst verdicts (bounded: each threshold moves at most 0.10 from the model's).
    Applies to transactions scored from now on; already-replayed cases keep the decision they were opened with."""
    from ml.feedback import adapt_thresholds as adapt
    sc = require_model(container)
    latest = {}
    for f in container.cases._feedback:
        case = container.cases.get_case(f.case_id)
        if case:
            latest[case.scored.txn_id] = int(f.verdict == Verdict.CONFIRM)
    fb = pd.DataFrame(dict(txn_id=list(latest), is_fraud=list(latest.values())))
    if fb.empty:
        raise err(422, "no_feedback", "No analyst verdicts yet. Confirm or dismiss some cases first.")
    new, info = adapt(sc.history, fb, sc.thresholds_model)
    if info.get("changed"):
        _apply_thresholds(container, new, "adaptive")
    return dict(thresholds=new, info=info, config=get_config(container).model_dump())


@router.post("/admin/thresholds/reset", response_model=ConfigInfo)
def reset_thresholds(container: Container = Depends(get_container)) -> ConfigInfo:
    sc = require_model(container)
    _apply_thresholds(container, sc.thresholds_model, f"model-{sc.engine.version}")
    return get_config(container)
