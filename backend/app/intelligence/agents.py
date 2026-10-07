"""Agent risk intelligence: peer-relative profiling of cash-in / cash-out agents.

Every agent is compared with the other agents over the same rolling 7-day window ending at `as_of`
(so a replayed transaction only ever sees the past). Signals, all relative to the peer median unless noted:

    night_burst          unusually large share of cash-outs between 22:00 and 06:00
    velocity_burst       busiest 60 minutes far above what peers ever reach
    volume_spike         total cash volume far above peers
    structuring_cluster  >= 3 transactions just under the 50,000 BDT reporting threshold (peers have ~none)
    cashout_skew         almost only cash-outs, never cash-ins
"""
from __future__ import annotations

import bisect
import statistics
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from backend.app.contracts.evidence_ids import agent as eid_agent
from backend.app.contracts.interfaces import AgentRiskService
from backend.app.contracts.schemas import AgentRisk, Transaction

WINDOW = timedelta(days=7)
MIN_TXNS = 20                 # below this an agent has no meaningful profile yet
NEAR_LO, NEAR_HI = 45_000.0, 50_000.0
WEIGHTS = {"night_burst": 0.25, "velocity_burst": 0.25, "volume_spike": 0.25, "structuring_cluster": 0.20,
           "cashout_skew": 0.10}
BASE_RISK = 0.10


def _robust_z(x: float, values: list[float], floor: float) -> float:
    """(x - median) / robust spread of the peers, with a floor so near-constant peers do not explode the score."""
    med = statistics.median(values)
    mad = statistics.median([abs(v - med) for v in values]) * 1.4826
    return (x - med) / max(mad, floor)


def _is_night(hour: int) -> bool:
    return hour >= 22 or hour < 6


def _naive(dt) -> datetime:
    dt = pd.Timestamp(dt)
    if dt.tzinfo is not None:
        dt = dt.tz_convert("UTC").tz_localize(None)
    return dt.to_pydatetime()


class PeerAgentRiskService(AgentRiskService):
    def __init__(self) -> None:
        self._rows: dict[str, list[tuple]] = defaultdict(list)   # agent -> [(epoch_s, amount, is_cashout, hour)] sorted
        self._cache: dict[tuple, dict[str, AgentRisk]] = {}
        self._last_ts: datetime | None = None

    # ------------------------------------------------------------------ data
    def build(self, txns: pd.DataFrame) -> None:
        self._rows.clear()
        self._cache.clear()
        self._last_ts = None
        if txns is None or txns.empty:
            return
        df = txns[txns["type"].isin(["CASH_IN", "CASH_OUT"]) & txns["agent_id"].notna()].copy()
        df["ts"] = pd.to_datetime(df["ts"])
        for r in df.sort_values("ts", kind="stable").itertuples(index=False):
            self._rows[str(r.agent_id)].append((r.ts.timestamp(), float(r.amount), r.type == "CASH_OUT", r.ts.hour))
            self._last_ts = r.ts.to_pydatetime() if self._last_ts is None else max(self._last_ts, r.ts.to_pydatetime())

    def ingest(self, txn: Transaction) -> None:
        if txn.type not in ("CASH_IN", "CASH_OUT"):
            return
        aid = txn.agent_id or txn.recipient_id
        ts = _naive(txn.ts)
        self._rows[aid].append((ts.timestamp(), float(txn.amount), txn.type == "CASH_OUT", ts.hour))
        self._rows[aid].sort(key=lambda x: x[0])
        self._last_ts = ts if self._last_ts is None else max(self._last_ts, ts)
        self._cache.clear()

    # ------------------------------------------------------------------ profiling
    @staticmethod
    def _profile(rows: list[tuple]) -> dict[str, float]:
        n = len(rows)
        if not n:
            return dict(n=0, volume=0.0, cashout_ratio=0.0, night_share=0.0, n_night=0, n_near=0, burst=0)
        ts = np.array([r[0] for r in rows])
        j = burst = 0
        for i in range(n):
            while ts[i] - ts[j] > 3600:
                j += 1
            burst = max(burst, i - j + 1)
        cash_rows = [r for r in rows if r[2]]
        night = [r for r in cash_rows if _is_night(r[3])]
        return dict(n=n, volume=float(sum(r[1] for r in rows)), cashout_ratio=len(cash_rows) / n,
                    night_share=(len(night) / len(cash_rows)) if cash_rows else 0.0, n_night=len(night),
                    n_near=sum(1 for r in cash_rows if NEAR_LO <= r[1] < NEAR_HI), burst=burst)

    def _snapshot(self, as_of: datetime | None) -> dict[str, AgentRisk]:
        asof = as_of or self._last_ts or datetime.now()
        key = (asof.replace(minute=0, second=0, microsecond=0),)
        if key in self._cache:
            return self._cache[key]
        end, start = asof.timestamp(), (asof - WINDOW).timestamp()
        profiles = {}
        for aid, rows in self._rows.items():
            times = [r[0] for r in rows]
            lo, hi = bisect.bisect_left(times, start), bisect.bisect_right(times, end)
            profiles[aid] = self._profile(rows[lo:hi])
        active = {a: p for a, p in profiles.items() if p["n"] >= MIN_TXNS}
        out: dict[str, AgentRisk] = {}
        if len(active) >= 5:
            vols = [p["volume"] for p in active.values()]
            nights = [p["night_share"] for p in active.values()]
            bursts = [float(p["burst"]) for p in active.values()]
            med_vol = statistics.median(vols) or 1.0
        for aid, p in profiles.items():
            flags: list[str] = []
            metrics = {k: round(float(v), 3) for k, v in p.items()}
            if aid in active and len(active) >= 5:
                vz = _robust_z(p["volume"], vols, 0.15 * med_vol)
                nz = _robust_z(p["night_share"], nights, 0.03)
                bz = _robust_z(float(p["burst"]), bursts, 1.0)
                metrics.update(volume_z=round(vz, 2), night_z=round(nz, 2), burst_z=round(bz, 2))
                if nz >= 4 and p["n_night"] >= 5:
                    flags.append("night_burst")
                if bz >= 3 and p["burst"] >= 8:
                    flags.append("velocity_burst")
                if vz >= 2.5:
                    flags.append("volume_spike")
                if p["n_near"] >= 3:
                    flags.append("structuring_cluster")
                if p["cashout_ratio"] >= 0.9 and p["n"] >= 10:
                    flags.append("cashout_skew")
            score = min(0.99, BASE_RISK + sum(WEIGHTS[f] for f in flags))
            out[aid] = AgentRisk(agent_id=aid, as_of=asof, risk_score=round(score, 2), peer_group="all agents (7d)",
                                 metrics=metrics, flags=flags, evidence_ids=[eid_agent(aid)])
        order = sorted(out.values(), key=lambda a: a.risk_score)
        n = max(len(order) - 1, 1)
        for rank, a in enumerate(order):
            a.peer_percentile = round(rank / n, 2)
        if len(self._cache) > 400:
            self._cache.clear()
        self._cache[key] = out
        return out

    # ------------------------------------------------------------------ protocol
    def agent_risk(self, agent_id: str, as_of: datetime | None = None) -> AgentRisk:
        asof = _naive(as_of) if as_of is not None else None
        res = self._snapshot(asof).get(agent_id)
        if res is not None:
            return res
        return AgentRisk(agent_id=agent_id, as_of=asof, risk_score=BASE_RISK, peer_group="no history",
                         evidence_ids=[eid_agent(agent_id)])

    def leaderboard(self, top_n: int = 10, as_of: datetime | None = None) -> list[AgentRisk]:
        snap = self._snapshot(_naive(as_of) if as_of is not None else None)
        return sorted(snap.values(), key=lambda a: (-a.risk_score, -a.metrics.get("volume", 0)))[:top_n]

    def peer_median_score(self, as_of: datetime | None = None) -> float:
        snap = self._snapshot(_naive(as_of) if as_of is not None else None)
        return float(statistics.median([a.risk_score for a in snap.values()])) if snap else BASE_RISK
