"""Case store: scored transactions, the analyst case queue, KPIs, audit trail and feedback.

Everything here is derived from the ML: the held-out (test) period of the scored cache is replayed as the
"live" timeline. Each replayed transaction is converted by ScoringService (model score + SHAP reasons + rule trace
+ point-in-time graph / agent evidence + policy engine); any transaction whose final action is not ALLOW
becomes a case. Transactions scored live with `commit=true` join the same queue.
"""
from __future__ import annotations

import threading
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from backend.app.config import Settings
from backend.app.contracts.schemas import (
    ACTION_SEVERITY,
    Action,
    Case,
    CaseStatus,
    Feedback,
    RiskLevel,
    ScoredTransaction,
    Transaction,
    Verdict,
)
from backend.app.contracts.schemas_core import (
    FeedbackStats,
    KpiSummary,
    ReasonCount,
    TimeBucket,
)
from backend.app.services.reasons import KEY_LABEL, feature_key, rule_key
from backend.app.services.scoring_service import ScoringService
from backend.app.store.db import Db


def case_from_txn(txn_id: str) -> str:
    """TXN-0062110 -> CASE-0062110 (idempotent: re-scoring a txn never creates a second case)."""
    return "CASE-" + txn_id.split("-", 1)[-1]


def _utc(ts) -> datetime:
    t = pd.Timestamp(ts)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return t.to_pydatetime()


# how each analyst button changes the case; only block / false_positive are real fraud verdicts
ACTION_STATUS = {"hold": CaseStatus.OPEN, "kyc": CaseStatus.OPEN, "block": CaseStatus.CONFIRMED,
                 "escalate": CaseStatus.ESCALATED, "false_positive": CaseStatus.DISMISSED}
ACTION_VERDICT = {"block": Verdict.CONFIRM, "false_positive": Verdict.DISMISS}


class CaseStore:
    def __init__(self, scoring: ScoringService | None, settings: Settings, db_path: str | None = None) -> None:
        self.scoring = scoring
        self.settings = settings
        self.db = Db(db_path or settings.db_path)
        self._lock = threading.RLock()
        self._cases: dict[str, Case] = {}
        self._scored: dict[str, ScoredTransaction] = {}            # replayed + live scored transactions
        self._replay: list[str] = []                               # txn ids of the replay timeline, in time order
        self._live: list[str] = []
        self._audit: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._feedback: list[Feedback] = []
        self.history = scoring.history if scoring else pd.DataFrame()
        self._pos: dict[str, int] = {}
        self._by_user: dict[str, np.ndarray] = {}
        self._by_recipient: dict[str, np.ndarray] = {}
        self._rows: dict[str, dict] = {}                           # txn_id -> raw cache row (replay timeline)
        self._flagged_idx: list[int] = []
        self._normal_idx: list[int] = []
        if scoring is not None:
            self._load_replay()
        self._restore_from_db()

    # ------------------------------------------------------------------ loading
    def _load_replay(self) -> None:
        h = self.history
        self._pos = {t: i for i, t in enumerate(h.txn_id)}
        self._by_user = {k: np.asarray(v) for k, v in h.groupby("user_id").indices.items()}
        self._by_recipient = {k: np.asarray(v) for k, v in h.groupby("recipient_id").indices.items()}
        test = h[h.split == "test"]
        for row in test.to_dict("records"):
            scored = self.scoring.replay(row)
            tid = row["txn_id"]
            self._scored[tid] = scored
            self._rows[tid] = row
            self._replay.append(tid)
            if scored.what_next.action != Action.ALLOW:
                self._open_case(scored, row["ts"])
        for i, tid in enumerate(self._replay):
            (self._flagged_idx if self._scored[tid].what_next.action != Action.ALLOW else self._normal_idx).append(i)

    def _open_case(self, scored: ScoredTransaction, ts) -> Case:
        cid = case_from_txn(scored.txn_id)
        if cid in self._cases:
            return self._cases[cid]
        scored.case_id = cid
        when = _utc(ts)
        case = Case(case_id=cid, status=CaseStatus.OPEN, created_at=when, updated_at=when,
                    alert_type=scored.what_next.alert_type, priority=scored.what_next.priority, scored=scored)
        self._cases[cid] = case
        return case

    def _restore_from_db(self) -> None:
        for r in self.db.query("SELECT * FROM audit ORDER BY id"):
            self._audit[r["case_id"]].append(dict(ts=r["ts"], action=r["action"], by=r["actor"]))
            case = self._cases.get(r["case_id"])
            if case and r["status"]:
                case.status = CaseStatus(r["status"])
                case.updated_at = datetime.fromisoformat(r["ts"])
        for r in self.db.query("SELECT * FROM feedback ORDER BY id"):
            fb = Feedback(case_id=r["case_id"], verdict=Verdict(r["verdict"]), analyst=r["analyst"], note=r["note"],
                          created_at=datetime.fromisoformat(r["created_at"]))
            self._feedback.append(fb)
            if r["case_id"] in self._cases:
                self._cases[r["case_id"]].feedback.append(fb)

    # ------------------------------------------------------------------ CaseProvider protocol
    def get_case(self, case_id: str) -> Case | None:
        return self._cases.get(case_id)

    def get_scored(self, txn_id: str) -> ScoredTransaction | None:
        return self._scored.get(txn_id)

    def history_frame(self) -> pd.DataFrame:
        return self.history

    def recent_transactions(self, wallet_id: str, until: datetime, limit: int = 20) -> list[Transaction]:
        idx = np.concatenate([self._by_user.get(wallet_id, np.array([], dtype=int)),
                              self._by_recipient.get(wallet_id, np.array([], dtype=int))])
        if not len(idx):
            return []
        rows = self.history.iloc[np.unique(idx)]
        cutoff = pd.Timestamp(until).tz_localize(None) if pd.Timestamp(until).tzinfo is None else \
            pd.Timestamp(until).tz_convert("UTC").tz_localize(None)
        rows = rows[rows.ts < cutoff].sort_values("ts", ascending=False).head(limit)
        cols = ["txn_id", "ts", "user_id", "type", "amount", "recipient_id", "agent_id", "device_id", "location",
                "balance_before", "merchant_category", "otp_requests_10m", "otp_failures_10m", "otp_device_mismatch",
                "concurrent_sessions", "sim_swap_recent"]
        out = []
        for r in rows[cols].to_dict("records"):
            out.append(Transaction(**{k: (None if (isinstance(v, float) and v != v) else v) for k, v in r.items()}))
        return out

    # ------------------------------------------------------------------ listing
    def list_cases(self, status: str | None = None, alert_type: str | None = None, risk_level: str | None = None,
                   min_amount: float | None = None, sort: str = "priority") -> list[Case]:
        items = list(self._cases.values())
        if status:
            items = [c for c in items if c.status.value == status]
        if alert_type:
            items = [c for c in items if c.alert_type and c.alert_type.value == alert_type]
        if risk_level:
            items = [c for c in items if c.scored.what_next.risk_level.value == risk_level]
        if min_amount:
            items = [c for c in items if c.scored.what_happened.amount_bdt >= min_amount]
        key = (lambda c: c.scored.what_happened.time) if sort == "time" else (lambda c: c.priority)
        return sorted(items, key=key, reverse=True)

    def list_transactions(self, limit: int = 50, offset: int = 0, risk_level: str | None = None,
                          user_id: str | None = None, since: str | None = None) -> tuple[list[ScoredTransaction], int]:
        ids = self._live[::-1] + self._replay[::-1]                       # newest first
        items = (self._scored[t] for t in ids)
        if risk_level:
            items = (s for s in items if s.what_next.risk_level.value == risk_level)
        if user_id:
            items = (s for s in items if s.user_id == user_id)
        if since:
            items = (s for s in items if s.what_happened.time >= since.replace("T", " "))
        rows = list(items)
        return rows[offset: offset + limit], len(rows)

    # ------------------------------------------------------------------ live (committed) transactions
    def register_live(self, scored: ScoredTransaction, ts) -> Case | None:
        with self._lock:
            if scored.txn_id in self._scored:
                return self._cases.get(case_from_txn(scored.txn_id))
            self._scored[scored.txn_id] = scored
            self._live.append(scored.txn_id)
            if scored.what_next.action != Action.ALLOW:
                return self._open_case(scored, ts)
        return None

    # ------------------------------------------------------------------ analyst actions + feedback
    def _log(self, case_id: str, action: str, by: str, status: CaseStatus | None) -> dict[str, Any]:
        entry = dict(ts=datetime.now(timezone.utc).isoformat(), action=action, by=by)
        self._audit[case_id].append(entry)
        self.db.execute("INSERT INTO audit(case_id, ts, action, actor, status) VALUES (?,?,?,?,?)",
                        (case_id, entry["ts"], action, by, status.value if status else None))
        return entry

    def _add_feedback(self, case: Case, verdict: Verdict, analyst: str, note: str | None) -> Feedback:
        fb = Feedback(case_id=case.case_id, verdict=verdict, analyst=analyst, note=note, created_at=datetime.now(timezone.utc))
        case.feedback.append(fb)
        self._feedback.append(fb)
        self.db.execute("INSERT INTO feedback(case_id, txn_id, verdict, analyst, note, created_at) VALUES (?,?,?,?,?,?)",
                        (case.case_id, case.scored.txn_id, verdict.value, analyst, note, fb.created_at.isoformat()))
        return fb

    def apply_action(self, case_id: str, action_name: str, by: str = "Analyst") -> dict[str, Any]:
        with self._lock:
            case = self._cases.get(case_id)
            if not case:
                raise KeyError(case_id)
            status = ACTION_STATUS.get(action_name, CaseStatus.OPEN)
            case.status, case.updated_at = status, datetime.now(timezone.utc)
            self._log(case_id, action_name, by, status)
            verdict = ACTION_VERDICT.get(action_name)
            if verdict:                     # holding / escalating / KYC are NOT fraud verdicts, so they create no label
                self._add_feedback(case, verdict, by, None)
            return {"status": status.value, "audit": self._audit[case_id]}

    def record_feedback(self, case_id: str, verdict: Verdict, analyst: str = "analyst", note: str | None = None) -> Case:
        with self._lock:
            case = self._cases.get(case_id)
            if not case:
                raise KeyError(case_id)
            self._add_feedback(case, verdict, analyst, note)
            case.status = CaseStatus.CONFIRMED if verdict == Verdict.CONFIRM else CaseStatus.DISMISSED
            case.updated_at = datetime.now(timezone.utc)
            self._log(case_id, f"feedback:{verdict.value}", analyst, case.status)
            return case

    def export_feedback_csv(self) -> str:
        """Columns understood by `python -m ml.feedback --labels analyst_feedback.csv` (txn_id + decision)."""
        lines = ["txn_id,decision,case_id,analyst,created_at"]
        latest: dict[str, Feedback] = {}
        for f in self._feedback:
            latest[f.case_id] = f                                           # last verdict per case wins
        for f in latest.values():
            case = self._cases.get(f.case_id)
            if case:
                decision = "confirmed" if f.verdict == Verdict.CONFIRM else "dismissed"
                lines.append(f"{case.scored.txn_id},{decision},{f.case_id},{f.analyst},{f.created_at.isoformat()}")
        return "\n".join(lines) + "\n"

    def feedback_stats(self) -> FeedbackStats:
        latest: dict[str, Feedback] = {f.case_id: f for f in self._feedback}
        conf = sum(1 for f in latest.values() if f.verdict == Verdict.CONFIRM)
        dism = sum(1 for f in latest.values() if f.verdict == Verdict.DISMISS)
        by_type: dict[str, dict[str, int]] = {}
        for cid, f in latest.items():
            c = self._cases.get(cid)
            t = c.alert_type.value if c and c.alert_type else "anomaly"
            by_type.setdefault(t, {"confirmed": 0, "dismissed": 0})["confirmed" if f.verdict == Verdict.CONFIRM else "dismissed"] += 1
        return FeedbackStats(confirmed=conf, dismissed=dism,
                             precision_confirmed=round(conf / (conf + dism), 3) if conf + dism else None,
                             by_alert_type=by_type)

    def get_audit(self, case_id: str) -> list[dict[str, Any]]:
        return self._audit.get(case_id, [])

    # ------------------------------------------------------------------ KPIs
    def _all_scored(self) -> list[ScoredTransaction]:
        return [self._scored[t] for t in self._replay + self._live]

    def get_kpis(self) -> KpiSummary:
        scored = self._all_scored()
        cases = list(self._cases.values())
        mix = Counter(s.what_next.risk_level.value for s in scored)
        by_type = Counter((c.alert_type.value if c.alert_type else "anomaly") for c in cases)
        blocked = sum(c.scored.what_happened.amount_bdt for c in cases
                      if ACTION_SEVERITY[c.scored.what_next.action] >= ACTION_SEVERITY[Action.HOLD])
        days: Counter = Counter(c.scored.what_happened.time[:10] for c in cases)
        reasons: Counter = Counter()
        for c in cases:
            keys = []
            for r in c.scored.why_risky.reasons:
                if r.shap > 0:
                    keys.append(feature_key(r.feature))
            keys += [rule_key(h.rule_id) for h in c.scored.why_risky.rule_trace]
            for k in dict.fromkeys(keys):
                if k != "other":
                    reasons[KEY_LABEL[k]] += 1
        return KpiSummary(
            sim_time=self._replay and self._scored[self._replay[-1]].what_happened.time or None,
            total_txns_scored=len(scored), alerts=len(cases),
            open_cases=sum(1 for c in cases if c.status == CaseStatus.OPEN),
            blocked_value_bdt=round(blocked, 2), alerts_by_type=dict(by_type),
            risk_mix={lv.value: mix.get(lv.value, 0) for lv in RiskLevel},
            alerts_over_time=[TimeBucket(ts=d, count=n) for d, n in sorted(days.items())],
            top_reasons=[ReasonCount(text=t, count=n) for t, n in reasons.most_common(6)])

    # ------------------------------------------------------------------ stream (stateless: the client keeps `seq`)
    def stream_pick(self, seq: int, mode: str = "demo") -> ScoredTransaction | None:
        """Event number `seq` of the simulated live feed. In demo mode every Nth event is a real flagged transaction,
        so the feed shows alerts regularly; the others are ordinary traffic. Same seq -> same event."""
        if not self._replay:
            return None
        every = max(self.settings.stream_demo_alert_every, 2)
        if mode == "demo" and self._flagged_idx and (seq + 1) % every == 0:
            idx = self._flagged_idx[(seq // every) % len(self._flagged_idx)]
        elif mode == "demo" and self._normal_idx:
            idx = self._normal_idx[(seq - seq // every) % len(self._normal_idx)]
        else:
            idx = seq % len(self._replay)
        return self._scored[self._replay[idx]]

    # ------------------------------------------------------------------ per-user context for the case view
    def user_history(self, user_id: str, before) -> pd.DataFrame:
        idx = self._by_user.get(user_id)
        if idx is None:
            return self.history.iloc[0:0]
        rows = self.history.iloc[idx]
        return rows[rows.ts < pd.Timestamp(before)]
