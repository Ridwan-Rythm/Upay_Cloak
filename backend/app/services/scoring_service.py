"""ScoringService: the bridge between the ML package and the API.

It owns the trained RiskEngine and a LiveScorer (rolling per-user state replayed from history), and turns the
ML output into the API's ScoredTransaction (what happened / why risky / what next) by adding graph evidence,
agent risk and the policy engine. Every score the API returns comes from here, whether it is a replayed
historical transaction, a stream event or a brand-new transaction scored live.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import Any

import pandas as pd

from backend.app.config import Settings
from backend.app.contracts.schemas import (
    AgentRisk,
    GraphSignals,
    RiskReason,
    RuleHit,
    ScoredTransaction,
    Transaction,
    WhatHappened,
    WhyRisky,
)
from backend.app.services.decision_service import DecisionEngine
from ml.cache import build_cache, is_stale, load_cache
from ml.features import RAW_COLUMNS
from ml.live import LiveScorer
from ml.rules import purpose_of, rule_trace
from ml.score import RiskEngine

log = logging.getLogger("upayshield.scoring")


def _naive_ts(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_convert("UTC").tz_localize(None) if t.tzinfo is not None else t


class ScoringService:
    available = True

    def __init__(self, settings: Settings, engine: RiskEngine, history: pd.DataFrame) -> None:
        self.settings = settings
        self.engine = engine
        self.history = history.sort_values("ts", kind="stable").reset_index(drop=True)
        self.scorer = LiveScorer(engine)
        self.graph: Any = None
        self.agents: Any = None
        self.presets: dict[str, dict] = {}
        self.metrics = self._load_metrics()
        adaptive = list(engine.th) != list(engine.th_model)
        self.decision = DecisionEngine(engine.th, version=("adaptive" if adaptive else f"model-{engine.version}"))

    # ------------------------------------------------------------------ construction
    @classmethod
    def load(cls, settings: Settings) -> ScoringService | None:
        """Load model + cache. Returns None (degraded mode: scoring endpoints answer 503) if there is no model."""
        if not settings.model_path.exists():
            if settings.auto_train:
                log.warning("model artifact missing -> training (UPAY_AUTO_TRAIN=true)")
                from ml.train import main as train_main
                train_main()
            else:
                log.error("model artifact %s missing. Run `python -m ml.train` (or set UPAY_AUTO_TRAIN=true)",
                          settings.model_path)
                return None
        engine = RiskEngine(settings.model_path)
        cache = load_cache(settings.cache_path)
        if is_stale(cache, engine):
            log.warning("scored cache does not match the current model -> rebuilding")
            cache = build_cache(out=settings.cache_path, model_path=settings.model_path)
        return cls(settings, engine, cache)

    def _load_metrics(self) -> dict:
        try:
            return json.loads(self.settings.metrics_path.read_text())
        except (OSError, ValueError):
            return {}

    def attach(self, graph: Any, agents: Any) -> None:
        self.graph, self.agents = graph, agents

    def warm_up(self, snapshot_before: list[str] | None = None) -> None:
        """Replay all history into the live feature state (a second or two)."""
        self.scorer.warm(self.history, snapshot_before=snapshot_before or [])

    # ------------------------------------------------------------------ conversion ML -> API
    def to_scored(self, ml: dict, ts: datetime | pd.Timestamp) -> ScoredTransaction:
        wh, wr = ml["what_happened"], ml["why_risky"]
        fc = wr.get("feature_contributions") or {}
        reasons = [RiskReason(feature=i["feature"], value=float(i["value"]), shap=float(i["contribution"]), text=i["text"])
                   for i in fc.get("top_positive", []) + fc.get("top_negative", [])]
        rules = [RuleHit(**h) for h in wr["rule_trace"]]
        tags = list(wr["tags"])
        asof = _naive_ts(ts).to_pydatetime()
        gs: GraphSignals | None = self.graph.wallet_signals(ml["user_id"], as_of=asof) if self.graph else None
        agent_id = wh.get("agent")
        ag: AgentRisk | None = self.agents.agent_risk(agent_id, as_of=asof) if (self.agents and agent_id) else None
        decision = self.decision.evaluate(
            risk_score=wr["risk_score"], amount_bdt=wh["amount_bdt"], rule_tags=tags, graph_signals=gs,
            agent_risk_score=ag.risk_score if ag else 0.0, is_ring_member=bool(gs and gs.ring_id))
        return ScoredTransaction(
            txn_id=ml["txn_id"], user_id=ml["user_id"], what_happened=WhatHappened(**wh),
            why_risky=WhyRisky(risk_score=wr["risk_score"], model_score=wr["risk_score"], anomaly_score=wr["anomaly_score"],
                               reasons=reasons, rule_trace=rules, tags=tags, graph=gs, agent=ag),
            what_next=decision)

    def row_to_ml(self, row: dict) -> dict:
        """A cache row (raw columns + point-in-time features + stored score) in the shape ml.score produces."""
        r = pd.Series(row)
        reasons = json.loads(row["reasons"]) if isinstance(row.get("reasons"), str) and row["reasons"] else None
        trace = rule_trace(r)
        has = lambda x: x is not None and x == x and x != ""      # noqa: E731
        return dict(
            txn_id=row["txn_id"], user_id=row["user_id"],
            what_happened=dict(type=row["type"], amount_bdt=float(row["amount"]), recipient=row["recipient_id"],
                               device=row["device_id"], location=row["location"], time=str(row["ts"]),
                               agent=row["agent_id"] if has(row.get("agent_id")) else None,
                               merchant_category=row["merchant_category"] if has(row.get("merchant_category")) else None,
                               purpose=purpose_of(row["type"], row.get("merchant_category"))),
            why_risky=dict(risk_score=float(row["risk_score"]), anomaly_score=float(row["anomaly_score"]),
                           feature_contributions=reasons, rule_trace=trace, tags=sorted({h["tag"] for h in trace})),
            what_next=dict(action=row["action"]))

    def replay(self, row: dict) -> ScoredTransaction:
        return self.to_scored(self.row_to_ml(row), row["ts"])

    # ------------------------------------------------------------------ live scoring
    def has_baseline(self, user_id: str) -> bool:
        st = self.scorer.state.U.get(user_id)
        return bool(st and st["n"] >= 5)

    def score_live(self, txn: Transaction, snapshot_id: str | None = None, commit: bool = False
                   ) -> tuple[ScoredTransaction, float, list[str]]:
        """Run the model on a raw transaction (features are computed from the user's own history)."""
        t0 = time.perf_counter()
        raw = txn.model_dump()
        raw["ts"] = _naive_ts(raw["ts"])
        raw["txn_id"] = raw.get("txn_id") or f"LIVE-{int(time.time() * 1000)}"
        ml = self.scorer.score(raw, snapshot_id=snapshot_id, commit=commit, explain="all")
        scored = self.to_scored(ml, raw["ts"])
        warnings = []
        if txn.type != "CASH_IN" and txn.amount > txn.balance_before > 0:
            warnings.append(f"The amount (BDT {txn.amount:,.0f}) is larger than the wallet balance "
                            f"(BDT {txn.balance_before:,.0f}); a ledger would reject this transfer.")
        if snapshot_id is None and not self.has_baseline(txn.user_id) and not commit:
            warnings.append(f"No behavioural baseline for wallet {txn.user_id} yet (cold start): "
                            "behavioural signals are muted, so the score relies on transaction-level and network signals.")
        return scored, round((time.perf_counter() - t0) * 1000, 1), warnings

    def ingest_committed(self, txn: Transaction) -> None:
        if self.graph:
            self.graph.ingest(txn)
        if self.agents:
            self.agents.ingest(txn)

    # ------------------------------------------------------------------ model facts for the API
    @property
    def thresholds_model(self) -> list[float]:
        return list(self.engine.th_model)

    @property
    def thresholds_active(self) -> list[float]:
        return list(self.engine.th)

    def tier_precision(self, action: str) -> float | None:
        """Measured precision of an action tier on held-out data (used as the honest 'confidence' of a decision)."""
        p = (self.metrics.get("quality") or {}).get("precision_by_action") or {}
        key = "block" if action in ("freeze_wallet", "block") else action
        return p.get(key)

    def bench_latency(self, n: int = 25) -> float | None:
        """Mean end-to-end scoring latency measured on real transactions (not committed)."""
        sample = self.history[self.history.split == "test"].head(n)
        if sample.empty:
            return None
        for r in sample[RAW_COLUMNS].to_dict("records"):
            try:
                self.scorer.score(r, commit=False, explain="flagged")
            except Exception:                                                # noqa: BLE001
                continue
        return self.scorer.mean_latency_ms

    # ------------------------------------------------------------------ demo presets (real transactions)
    PRESET_TEXT = {
        "norm": ("Everyday transfer", "দৈনন্দিন লেনদেন", "A regular transfer to a known contact, at the usual time.",
                 "পরিচিত কাউকে স্বাভাবিক সময়ে সাধারণ পাঠানো।"),
        "verify": ("Unusual transfer", "অস্বাভাবিক লেনদেন", "Different from the usual pattern but probably fine: a quick PIN check is enough.",
                   "স্বাভাবিকের চেয়ে আলাদা কিন্তু সম্ভবত ঠিক আছে: একটি দ্রুত পিন যাচাই যথেষ্ট।"),
        "ato": ("Account takeover", "অ্যাকাউন্ট দখল", "New phone, new city, at night, draining the balance.",
                "নতুন ফোন, নতুন শহর, রাতে, ব্যালেন্স খালি করা।"),
        "otp": ("OTP shared with a stranger", "অপরিচিতকে OTP দেওয়া",
                "OTP requested several times and confirmed from another device while a second session is open.",
                "কয়েকবার OTP চাওয়া এবং অন্য ডিভাইস থেকে নিশ্চিত করা, একই সময়ে দ্বিতীয় সেশন খোলা।"),
        "scam": ("Scam nudge", "প্রতারণার সতর্কতা", "First-time recipient, amount far above the usual: the customer is nudged to double-check.",
                 "প্রথমবার প্রাপক, সাধারণের চেয়ে অনেক বেশি পরিমাণ।"),
        "mule": ("Mule pass-through", "মিউল পাস-থ্রু", "Money just arrived and is forwarded within minutes.",
                 "টাকা এসেছে এবং কয়েক মিনিটের মধ্যে সরিয়ে ফেলা হচ্ছে।"),
        "bet": ("Betting payment", "বেটিং পেমেন্ট", "Repeated, growing payments to a betting service late in the evening.",
                "সন্ধ্যার পর বেটিং সার্ভিসে বারবার, বাড়তে থাকা পেমেন্ট।"),
        "struct": ("Structuring", "স্ট্রাকচারিং", "Repeated cash-outs just under the 50,000 BDT limit.",
                   "৫০,০০০ টাকার সীমার ঠিক নিচে বারবার ক্যাশ-আউট।"),
        "agent": ("Rogue agent", "সন্দেহজনক এজেন্ট", "An agent processing a late-night burst of cash-outs.",
                  "রাতে একের পর এক ক্যাশ-আউট করা একজন এজেন্ট।"),
    }
    PRESET_SCENARIO = {"ato": "ato", "otp": "otp_breach", "scam": "scam_victim", "mule": "mule_passthrough",
                       "bet": "gambling", "struct": "structuring", "agent": "rogue_agent"}

    def build_presets(self) -> dict[str, dict]:
        """Pick one representative REAL test transaction per scenario (labels are used only to choose demo examples)."""
        t = self.history[self.history.split == "test"]
        picks: dict[str, pd.Series] = {}
        for key, scen in self.PRESET_SCENARIO.items():
            pool = t[t.scenario == scen]
            if key == "mule":
                pool = pool[pool.type == "TRANSFER"]
            if key == "scam":                      # prefer the soft case: the model is unsure, the scam rule asks for a check
                soft = pool[pool.action.isin(["allow", "otp_step_up"]) & pool.tags.str.contains("scam_victim")]
                pool = soft if len(soft) else pool
            if len(pool):
                picks[key] = pool.sort_values("risk_score", ascending=False).iloc[0]
        normal = t[(t.scenario == "normal") & (t.type == "TRANSFER") & (t.action == "allow") & (t.user_txn_count >= 15)]
        if len(normal):
            med = normal.amount.median()
            picks["norm"] = normal.iloc[(normal.amount - med).abs().argsort().iloc[0]]
        verify = t[(t.scenario == "normal") & (t.action == "otp_step_up") & (t.tags == "") & (t.type == "TRANSFER")]
        if len(verify):
            picks["verify"] = verify.sort_values("risk_score", ascending=False).iloc[0]
        order = ["norm", "verify", "ato", "otp", "scam", "mule", "bet", "struct", "agent"]
        self.presets = {}
        for k in order:
            if k in picks:
                row = picks[k]
                self.presets[k] = dict(key=k, txn_id=row["txn_id"], title_en=self.PRESET_TEXT[k][0],
                                       title_bn=self.PRESET_TEXT[k][1], blurb_en=self.PRESET_TEXT[k][2],
                                       blurb_bn=self.PRESET_TEXT[k][3])
        return self.presets

    def preset_row(self, key: str) -> dict | None:
        p = self.presets.get(key)
        if not p:
            return None
        return self.history[self.history.txn_id == p["txn_id"]].iloc[0].to_dict()
