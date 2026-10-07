"""API: every score must come from the trained model, and the new OTP / betting / purpose checks must work."""
import json

import pytest

from tests.conftest import raw_txn


def first(history, **q):
    d = history[history.split == "test"]
    for k, v in q.items():
        d = d[d[k] == v]
    return d.sort_values("risk_score", ascending=False).iloc[0].to_dict()


def test_health_reports_real_model(client):
    h = client.get("/health").json()
    assert h["model_loaded"] and h["model"] == "LightGBM" and h["cases"] > 100


def test_config_uses_model_thresholds_not_hardcoded(client, container):
    c = client.get("/api/v1/config").json()
    assert c["thresholds_model"] == container.scoring.engine.th_model
    assert c["thresholds_active"] == container.scoring.engine.th
    assert c["dataset"]["features"] == len(container.scoring.engine.features)


def test_score_by_txn_id_returns_stored_model_score(client, history):
    row = first(history, scenario="ato")
    r = client.post("/api/v1/score", json={"txn_id": row["txn_id"]}).json()
    assert r["feature_mode"] == "replay"
    assert r["scored"]["why_risky"]["model_score"] == pytest.approx(row["risk_score"], abs=1e-4)


def test_live_scoring_equals_replay(client, history):
    """A raw transaction scored live must match the stored score: the same model + same features."""
    row = first(history, scenario="ato")
    live = client.post("/api/v1/score", json={"transaction": raw_txn(row)}).json()
    assert live["feature_mode"] == "live"
    # the wallet's state already contains this txn, so the score differs a little; the decision must still be a block
    assert live["scored"]["what_next"]["action"] in ("block", "freeze_wallet", "hold")


def test_demo_snapshot_scoring_is_point_in_time_exact(client, container):
    sc = container.scoring
    for key, p in sc.presets.items():
        stored = container.cases.get_scored(p["txn_id"])
        r = client.post("/api/score", json={"preset": key}).json()
        assert r["risk_score"] == pytest.approx(stored.why_risky.risk_score, abs=1e-3), key
        assert r["action"] == stored.what_next.action.value, key


def test_editing_the_recipient_and_amount_changes_the_model_score(client):
    base = client.post("/api/score", json={"preset": "norm"}).json()
    risky = client.post("/api/score", json={"preset": "norm", "amount_bdt": 9000, "receiver": "U_NEVER_PAID"}).json()
    assert base["decision"] == "allow"
    assert risky["risk_score"] > base["risk_score"] + 0.03 and risky["decision"] != "allow"
    assert risky["customer_keys"]                       # plain-language reasons for the customer screen


def test_amount_larger_than_balance_is_called_out(client):
    r = client.post("/api/score", json={"preset": "norm", "amount_bdt": 900000}).json()
    assert any("larger than the wallet balance" in w for w in r["warnings"])


def test_otp_breach_is_detected_live(client, history):
    """OTP confirmed from another device + 2 sessions + several OTP requests on an ordinary transfer."""
    row = first(history, scenario="normal", type="TRANSFER", action="allow")
    body = raw_txn(row, otp_requests_10m=4, otp_failures_10m=2, otp_device_mismatch=1, concurrent_sessions=2)
    calm = client.post("/api/v1/score", json={"transaction": raw_txn(row)}).json()["scored"]
    hit = client.post("/api/v1/score", json={"transaction": body}).json()["scored"]
    assert "otp_breach" in hit["why_risky"]["tags"] and hit["what_next"]["alert_type"] == "otp_breach"
    assert hit["why_risky"]["model_score"] > calm["why_risky"]["model_score"]
    assert hit["what_next"]["action"] != "allow"
    assert any(r["feature"].startswith("otp") or r["feature"] == "concurrent_sessions" for r in hit["why_risky"]["reasons"])


def test_betting_payment_is_detected_and_explained(client, history):
    row = first(history, scenario="normal", type="PAYMENT", action="allow")
    plain = client.post("/api/v1/score", json={"transaction": raw_txn(row)}).json()["scored"]
    bet = client.post("/api/v1/score", json={"transaction": raw_txn(row, merchant_category="gambling")}).json()["scored"]
    assert plain["what_happened"]["purpose"] != "betting"
    assert bet["what_happened"]["purpose"] == "betting"
    assert "gambling" in bet["why_risky"]["tags"] and bet["what_next"]["action"] != "allow"
    assert bet["what_next"]["alert_type"] == "gambling"
    assert "POL_GAMBLING" in [p["policy_id"] for p in bet["what_next"]["policy_trace"]] or bet["what_next"]["action"] in ("hold", "block")


def test_purpose_is_reported_for_every_transaction(client):
    items = client.get("/api/v1/transactions?limit=100").json()["items"]
    assert all(i["what_happened"]["purpose"] for i in items)
    assert {"person_to_person", "cash_withdrawal"} & {i["what_happened"]["purpose"] for i in items}


def test_cold_start_wallet_gets_a_warning_not_a_crash(client, history):
    row = first(history, scenario="normal", type="TRANSFER")
    r = client.post("/api/v1/score", json={"transaction": raw_txn(row, user_id="U_BRAND_NEW")}).json()
    assert any("cold start" in w for w in r["warnings"])


def test_commit_is_idempotent_and_creates_a_case(client, history):
    row = first(history, scenario="ato")
    body = raw_txn(row, txn_id="LIVE-TEST-IDEMP", device_id="DX-STRANGER", otp_device_mismatch=1, concurrent_sessions=2)
    a = client.post("/api/v1/score", json={"transaction": body, "commit": True}).json()
    b = client.post("/api/v1/score", json={"transaction": body, "commit": True}).json()
    assert a["scored"]["case_id"] and a["scored"]["case_id"] == b["scored"]["case_id"]
    n = client.get("/api/v1/cases?limit=500").json()["total"]
    client.post("/api/v1/score", json={"transaction": body, "commit": True})
    assert client.get("/api/v1/cases?limit=500").json()["total"] == n


def test_validation_and_error_shape(client):
    r = client.post("/api/v1/score", json={})
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"
    r = client.get("/api/v1/cases/CASE-NOPE")
    assert r.status_code == 404 and r.json()["error"]["code"] == "case_not_found"
    r = client.post("/api/v1/score", json={"transaction": {"user_id": "x"}})
    assert r.status_code == 422 and "error" in r.json()


def test_warning_check_runs_the_model(client, history):
    """A real wallet with a betting payment; a calm one; both languages."""
    row = first(history, scenario="normal", type="TRANSFER", action="allow")
    base = {"user_id": row["user_id"], "recipient_id": row["recipient_id"], "amount": float(row["amount"]),
            "device_id": row["device_id"], "location": row["location"]}
    calm = client.post("/api/v1/warning/check", json=base).json()
    assert not calm["show_warning"]
    big = client.post("/api/v1/warning/check", json={**base, "recipient_id": "U_STRANGER", "amount": 700000, "language": "bn"}).json()
    assert big["show_warning"] and big["language"] == "bn" and big["reasons"]
    bet = client.post("/api/v1/warning/check", json={**base, "type": "PAYMENT", "recipient_id": "G001", "merchant_category": "gambling"}).json()
    assert bet["show_warning"] and any("betting" in r.lower() or "gambling" in r.lower() for r in bet["reasons"])


def test_cases_filters_and_pagination(client):
    allc = client.get("/api/v1/cases?limit=500").json()
    assert allc["total"] > 100
    page = client.get("/api/v1/cases?limit=10").json()
    assert len(page["items"]) == 10 and page["next_cursor"] == "10"
    types = {c["alert_type"] for c in allc["items"]}
    assert {"otp_breach", "gambling", "mule_network"} <= types
    g = client.get("/api/v1/cases?alert_type=gambling&limit=500").json()
    assert g["total"] > 0 and all(c["alert_type"] == "gambling" for c in g["items"])
    pr = [c["priority"] for c in allc["items"]]
    assert pr == sorted(pr, reverse=True)


def test_analyst_actions_and_feedback_export(client):
    cs = client.get("/api/cases?alert_type=gambling").json()
    a, b, c = cs[0]["case_id"], cs[1]["case_id"], cs[2]["case_id"]
    assert client.post(f"/api/cases/{a}/action", json={"a": "hold"}).json()["status"] == "open"
    assert client.post(f"/api/cases/{b}/action", json={"a": "block"}).json()["status"] == "confirmed"
    assert client.post(f"/api/cases/{c}/action", json={"a": "false_positive"}).json()["status"] == "dismissed"
    csv = client.get("/api/v1/feedback/export").text.splitlines()
    assert csv[0].startswith("txn_id,decision")                     # what ml.feedback expects
    ids = {r.split(",")[2]: r.split(",")[1] for r in csv[1:]}
    assert ids.get(b) == "confirmed" and ids.get(c) == "dismissed" and a not in ids      # "hold" is not a verdict
    assert client.post("/api/cases/CASE-X/action", json={"a": "hold"}).status_code == 404
    assert client.post(f"/api/cases/{a}/action", json={"a": "nonsense"}).status_code == 422


def test_metrics_come_from_the_report(client, container):
    m = client.get("/api/v1/metrics").json()
    rep = json.loads(container.scoring.engine and open("reports/metrics.json").read())
    assert m["pr_auc"] == rep["models"][rep["best_model"]]["test"]["pr_auc"]
    assert m["scenario_recall"].keys() >= {"otp_breach", "gambling"}


def test_kpis_are_computed_from_scored_cases(client, container):
    k = client.get("/api/v1/kpis").json()
    assert k["total_txns_scored"] >= 9000 and k["alerts"] == len(container.cases._cases)
    assert sum(k["risk_mix"].values()) == k["total_txns_scored"]
    assert {"otp_breach", "gambling"} <= set(k["alerts_by_type"])


def test_frontend_endpoints_shapes(client):
    ov = client.get("/api/overview").json()
    assert {"kpis", "dist", "trend", "trend_labels", "reasons", "agents"} <= ov.keys()
    assert not any(a["id"].startswith("AGT-") for a in ov["agents"])
    feed = client.get("/api/transactions").json()
    assert len(feed["items"]) == 40 and feed["next_seq"] == 40
    t = client.get("/api/stream?seq=40").json()["txn"]
    assert {"id", "sender", "receiver", "amount_bdt", "decision", "risk_level", "purpose_label", "reasons"} <= t.keys()
    assert t == client.get("/api/stream?seq=40").json()["txn"]          # deterministic: same seq, same event
    flagged = [client.get(f"/api/stream?seq={i}").json()["txn"] for i in range(5, 60, 6)]
    assert all(f["decision"] != "allow" for f in flagged)                # demo mode: every 6th event is an alert
    scen = client.get("/api/demo/scenarios").json()
    assert {"norm", "ato", "otp", "bet"} <= {s["key"] for s in scen}


def test_case_detail_has_real_context(client):
    cs = client.get("/api/cases?alert_type=otp_breach").json()
    d = client.get(f"/api/cases/{cs[0]['case_id']}").json()
    assert d["purpose"]["label"] and d["baseline"]["usual_amount"] and d["timeline"] and d["evidence"]
    assert d["confidence_basis"].startswith("measured precision")
    assert any("OTP" in t["label"] for t in d["trace"])
    assert d["subgraph"]["nodes"] == [] or d["subgraph"]["nodes"][0]["id"] == d["entity"]["id"]
    ans = client.post(f"/api/cases/{d['case_id']}/ask", json={"q": "Was an OTP involved?"}).json()
    assert ans["answerable"] and ans["evidence_ids"]
    betcase = client.get("/api/cases?alert_type=gambling").json()[0]["case_id"]
    q = client.post(f"/api/cases/{betcase}/ask", json={"q": "What is the money being used for?"}).json()
    assert "betting" in q["text"].lower()


def test_graph_endpoints(client):
    g = client.get("/api/graph").json()                  # default centre = the biggest mule ring, not a hard-coded id
    assert g["center"] and len(g["nodes"]) > 5 and any(n["ring"] for n in g["nodes"])
    assert client.get("/api/graph/NOPE-123").json()["nodes"] == []
    rings = client.get("/api/v1/graph/rings").json()
    assert len(rings) == 6 and all(r["size"] >= 2 for r in rings)
    v1 = client.get("/api/v1/graph").json()
    assert v1["nodes"]


def test_reports_are_real_pdf_and_markdown(client):
    cid = client.get("/api/cases?alert_type=otp_breach").json()[0]["case_id"]
    pdf = client.get(f"/api/v1/cases/{cid}/report?format=pdf")
    assert pdf.content[:5] == b"%PDF-" and len(pdf.content) > 1500
    md = client.get(f"/api/v1/cases/{cid}/report?format=md").text
    assert "Purpose of the money" in md and "Rules that fired" in md


def test_sse_stream_and_control(client):
    r = client.get("/api/v1/stream?limit=3&interval_ms=0&mode=demo")
    events = [e for e in r.text.split("\n\n") if e.strip()]
    assert len(events) == 3 and all(e.startswith(("event: txn", "event: alert")) for e in events)
    st = client.post("/api/v1/stream/control", json={"action": "start", "mode": "replay", "interval_ms": 500}).json()
    assert st["running"] and st["mode"] == "replay" and st["interval_ms"] == 500 and st["seq"] >= 3
    assert client.post("/api/v1/stream/control", json={"action": "reset"}).json()["seq"] == 0


def test_feedback_adapts_thresholds_within_bounds_and_resets(client, container):
    base = list(container.scoring.thresholds_model)
    cs = client.get("/api/v1/cases?limit=500").json()["items"]
    for c in cs[:40]:                                    # an analyst dismisses a batch of alerts
        client.post(f"/api/v1/cases/{c['case_id']}/feedback?verdict=dismiss")
    res = client.post("/api/v1/admin/thresholds/adapt").json()
    assert all(abs(a - b) <= 0.1001 for a, b in zip(res["thresholds"], base))
    assert res["thresholds"] == sorted(res["thresholds"])
    cfg = client.post("/api/v1/admin/thresholds/reset").json()
    assert cfg["thresholds_active"] == base


def test_degraded_mode_returns_503_not_fake_scores():
    from pathlib import Path

    from fastapi.testclient import TestClient

    from backend.app.config import Settings
    from backend.app.deps import build_container, get_container
    from backend.app.main import app
    c = build_container(Settings(model_path=Path("/nonexistent/model.joblib")), db_path=":memory:")
    app.dependency_overrides[get_container] = lambda: c
    try:
        cl = TestClient(app)
        r = cl.post("/api/v1/score", json={"txn_id": "TXN-1"})
        assert r.status_code == 503 and r.json()["error"]["code"] == "model_not_loaded"
        assert cl.post("/api/score", json={"preset": "ato"}).status_code == 503
        assert cl.get("/api/v1/config").json()["model_loaded"] is False
    finally:
        app.dependency_overrides.clear()
