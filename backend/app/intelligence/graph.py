"""Graph analytics for UpayShield (NetworkX): mule-ring discovery, ego sub-graphs, wallet signals, freeze what-if.

Mule detection is behavioural, not volume-based. A wallet shows RAPID PASS-THROUGH when it forwards
(TRANSFER or CASH_OUT) 70-110% of money it received within 30 minutes. A wallet with >= 2 such events is a
mule suspect; suspects that move money between each other form a ring (connected component, >= 2 wallets).
Busy agents, merchants and popular wallets are never flagged just for receiving a lot of money.

Everything is POINT-IN-TIME: a wallet only counts as a ring member from the moment it had shown its 2nd
pass-through event, so replaying history never uses information from the future.
"""
from __future__ import annotations

import hashlib
import statistics
import time
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Any

import networkx as nx
import pandas as pd

from backend.app.contracts.evidence_ids import (
    agent as eid_agent,
)
from backend.app.contracts.evidence_ids import (
    device as eid_device,
)
from backend.app.contracts.evidence_ids import (
    ring as eid_ring,
)
from backend.app.contracts.evidence_ids import (
    wallet as eid_wallet,
)
from backend.app.contracts.interfaces import GraphService
from backend.app.contracts.schemas import GraphSignals, Transaction
from backend.app.contracts.schemas_intel import FreezeImpact, GraphEdge, GraphNode, GraphView, RingSummary

PT_DWELL_S = 30 * 60          # forwarded within 30 minutes ...
PT_LO, PT_HI = 0.7, 1.1       # ... at 70-110% of what came in
PT_MIN_EVENTS = 2             # events needed before a wallet is a mule suspect
RING_MIN_CONF = 0.5           # share of a suspect's pass-through events that must be "ring-consistent" (see _member_confidence)
NEIGHBOUR_CAP = 25            # per node when drawing ego graphs (keeps hubs readable)
MAX_TXN_IDS = 10


def _naive(dt: datetime | pd.Timestamp | None) -> datetime | None:
    """Dataset timestamps are naive; treat any tz-aware input as UTC and drop the tzinfo."""
    if dt is None:
        return None
    dt = pd.Timestamp(dt)
    if dt.tzinfo is not None:
        dt = dt.tz_convert("UTC").tz_localize(None)
    return dt.to_pydatetime()


def _iso(dt: datetime) -> str:
    return dt.isoformat()


class NetworkXGraphService(GraphService):
    def __init__(self) -> None:
        self._reset()

    def _reset(self) -> None:
        self.G: nx.MultiDiGraph = nx.MultiDiGraph()
        self._rings: dict[str, RingSummary] = {}
        self._wallet_to_ring: dict[str, str] = {}
        self._wallet_since: dict[str, datetime] = {}       # wallet -> when it became a mule suspect
        self._ring_wallets: frozenset[str] = frozenset()
        self._txn_lookup: dict[str, dict[str, Any]] = {}
        self._in: dict[str, list[tuple]] = defaultdict(list)    # wallet -> [(ts, amount, sender, txn_id)]
        self._out: dict[str, list[tuple]] = defaultdict(list)   # wallet -> [(ts, amount, recipient, txn_id, type)]
        self._events: dict[str, list[tuple]] = defaultdict(list)  # wallet -> [(ts_out, dwell_seconds, out_txn)]
        self._device_users: dict[str, set[str]] = defaultdict(set)
        self._ring_assoc: set[str] = set()                 # ring wallets + their cash-out agents + shared devices
        self._member_conf: dict[str, float] = {}           # ring member -> behavioural confidence (0..1)

    # ------------------------------------------------------------------ construction
    def _node(self, nid: str, kind: str) -> None:
        if nid not in self.G:
            self.G.add_node(nid, kind=kind, label=nid, risk=0.05, total_volume_bdt=0.0, flags=[], ring_id=None)

    def _edge(self, u: str, v: str, kind: str, amt: float, ts: datetime, tid: str) -> None:
        if self.G.has_edge(u, v, key=kind):
            d = self.G.edges[u, v, kind]
            d["amount_bdt"] += amt
            d["count"] += 1
            d["last_ts"] = _iso(ts)
            if len(d["txn_ids"]) < MAX_TXN_IDS:
                d["txn_ids"].append(tid)
        else:
            self.G.add_edge(u, v, key=kind, kind=kind, amount_bdt=amt, count=1,
                            first_ts=_iso(ts), last_ts=_iso(ts), txn_ids=[tid])

    def _add_txn(self, tid: str, ts: datetime, sender: str, ttype: str, amt: float, rcpt: str, device: str | None,
                 is_fraud: int = 0, scenario: str = "normal") -> None:
        ttype = ttype.upper()
        self._node(sender, "wallet")
        self._node(rcpt, {"CASH_IN": "agent", "CASH_OUT": "agent", "PAYMENT": "merchant"}.get(ttype, "wallet"))
        self.G.nodes[sender]["total_volume_bdt"] += amt
        self.G.nodes[rcpt]["total_volume_bdt"] += amt
        if ttype == "CASH_IN":
            self._edge(rcpt, sender, "cash_in", amt, ts, tid)           # money flows agent -> wallet
        else:
            self._edge(sender, rcpt, "cash_out" if ttype == "CASH_OUT" else "transfer", amt, ts, tid)
        if device:
            self._node(device, "device")
            if not self.G.has_edge(sender, device, key="uses_device"):
                self.G.add_edge(sender, device, key="uses_device", kind="uses_device", amount_bdt=0.0, count=1,
                                first_ts=_iso(ts), last_ts=_iso(ts), txn_ids=[tid])
            self._device_users[device].add(sender)
        if ttype == "TRANSFER":
            self._in[rcpt].append((ts, amt, sender, tid))
        if ttype in ("TRANSFER", "CASH_OUT"):
            self._out[sender].append((ts, amt, rcpt, tid, ttype))
        self._txn_lookup[tid] = dict(sender=sender, rcpt=rcpt, amount=amt, ts=ts, type=ttype.lower(),
                                     device=device, is_fraud=int(is_fraud), scenario=scenario)

    def build(self, txns: pd.DataFrame) -> None:
        """Bulk build from all history. Idempotent; replaces previous state."""
        self._reset()
        if txns is None or txns.empty:
            return
        df = txns.copy()
        df["ts"] = pd.to_datetime(df["ts"])
        df = df.sort_values("ts", kind="stable")
        has_fraud, has_scen = "is_fraud" in df.columns, "scenario" in df.columns
        for r in df.itertuples(index=False):
            self._add_txn(
                str(r.txn_id), r.ts.to_pydatetime(), str(r.user_id), str(r.type), float(r.amount), str(r.recipient_id),
                str(r.device_id) if pd.notna(r.device_id) else None,
                int(r.is_fraud) if has_fraud else 0, str(r.scenario) if has_scen else "normal")
        self._detect_pass_through(df)
        self._recompute_rings()

    # ------------------------------------------------------------------ mule detection
    def _detect_pass_through(self, df: pd.DataFrame) -> None:
        inb = df[df.type == "TRANSFER"][["recipient_id", "ts", "amount"]].rename(
            columns={"recipient_id": "w", "ts": "ts_in", "amount": "a_in"})
        out = df[df.type.isin(["TRANSFER", "CASH_OUT"])][["user_id", "ts", "amount", "txn_id"]].rename(
            columns={"user_id": "w", "ts": "ts_out", "amount": "a_out", "txn_id": "out_txn"})
        m = out.merge(inb, on="w")
        m["dwell"] = (m.ts_out - m.ts_in).dt.total_seconds()
        m = m[(m.dwell > 0) & (m.dwell <= PT_DWELL_S) & (m.a_out >= PT_LO * m.a_in) & (m.a_out <= PT_HI * m.a_in)]
        # one event per outgoing transaction (keep the quickest matching inbound)
        m = m.sort_values("dwell").drop_duplicates("out_txn").sort_values("ts_out")
        for r in m.itertuples(index=False):
            self._events[r.w].append((r.ts_out.to_pydatetime(), float(r.dwell), r.out_txn))
        for w, evs in self._events.items():
            if len(evs) >= PT_MIN_EVENTS:
                self._wallet_since[w] = evs[PT_MIN_EVENTS - 1][0]

    def _check_pass_through(self, sender: str, ts: datetime, amt: float, tid: str) -> None:
        """Incremental version for live ingestion."""
        best = None
        for t_in, a_in, _src, _tid in self._in.get(sender, ()):
            dwell = (ts - t_in).total_seconds()
            if 0 < dwell <= PT_DWELL_S and PT_LO * a_in <= amt <= PT_HI * a_in and (best is None or dwell < best):
                best = dwell
        if best is None:
            return
        self._events[sender].append((ts, best, tid))
        if len(self._events[sender]) >= PT_MIN_EVENTS and sender not in self._wallet_since:
            self._wallet_since[sender] = self._events[sender][PT_MIN_EVENTS - 1][0]
            self._recompute_rings()

    def _member_confidence(self) -> dict[str, float]:
        """Per-wallet confidence = share of its pass-through events that are ring-consistent, i.e. the forwarded money
        went to a cash-out agent or to another wallet that itself shows pass-through behaviour. Range 0..1."""
        movers = {w for w, evs in self._events.items() if evs}
        out: dict[str, float] = {}
        for w in self._wallet_since:
            evs = self._events.get(w, ())
            if not evs:
                continue
            ok = 0
            for _t, _d, out_txn in evs:
                info = self._txn_lookup.get(out_txn)
                if info and (info["type"] == "cash_out" or info["rcpt"] in movers):
                    ok += 1
            out[w] = ok / len(evs)
        return out

    def _recompute_rings(self) -> None:
        for nid in list(self._wallet_to_ring) + list(self._ring_assoc):
            if nid in self.G:
                nd = self.G.nodes[nid]
                nd["ring_id"] = None
                nd["flags"] = [f for f in nd["flags"] if f not in ("Mule ring", "Cashout exit", "Shared device")]
        self._rings, self._wallet_to_ring, self._ring_assoc = {}, {}, set()

        # Membership needs BEHAVIOUR, not just one shared edge (M2.1): a suspect must mostly forward money onward into
        # other suspects or to a cash-out agent, and two suspects are linked only when one actually forwarded
        # received money to the other. A victim who happens to forward twice and also paid a hub is no longer a member.
        conf = self._member_confidence()
        suspects = {w for w in self._wallet_since if conf.get(w, 0.0) >= RING_MIN_CONF}
        self._member_conf = {w: round(conf[w], 2) for w in suspects}
        H = nx.Graph()
        H.add_nodes_from(suspects)
        for u in suspects:
            for _t, _dwell, out_txn in self._events.get(u, ()):
                info = self._txn_lookup.get(out_txn)
                if info and info["type"] == "transfer" and info["rcpt"] in suspects and info["rcpt"] != u:
                    H.add_edge(u, info["rcpt"])

        for comp in nx.connected_components(H):
            if len(comp) < 2:
                continue                                    # a lone suspect is not a ring
            members = sorted(comp)
            rid = "RING-" + hashlib.sha1(members[0].encode()).hexdigest()[:6]
            inflow, victims = 0.0, set()
            for w in members:
                for _t, a, src, _ in self._in.get(w, ()):
                    if src not in comp:
                        inflow += a
                        victims.add(src)
            outflow, agents = 0.0, set()
            for w in members:
                for _t, a, dst, _, tp in self._out.get(w, ()):
                    if tp == "CASH_OUT":
                        outflow += a
                        agents.add(dst)
            shared = sorted({d for d, users in self._device_users.items() if len(users & comp) >= 2})
            evs = [e for w in members for e in self._events.get(w, ())]
            firsts = [t for t, _, _ in evs]
            score = 0.6 + 0.05 * min(len(members), 5) + (0.1 if shared else 0.0) + (0.05 if agents else 0.0)
            flags = ["mule_network", "rapid_passthrough"]
            if len(victims) >= 3:
                flags.append("fan_in_hub")
            if shared:
                flags.append("shared_device")
            if agents:
                flags.append("cashout_exit")
            self._rings[rid] = RingSummary(
                ring_id=rid, wallet_ids=members, size=len(members), ring_score=round(min(0.98, score), 2),
                total_inflow_bdt=round(inflow, 2), total_outflow_bdt=round(outflow, 2), victim_wallets=len(victims),
                shared_devices=shared, cashout_agents=sorted(agents), first_seen=_iso(min(firsts)),
                last_seen=_iso(max(firsts)), flags=flags,
                member_confidence={w: self._member_conf.get(w, 0.0) for w in members},
                evidence_ids=[eid_ring(rid)] + [eid_wallet(w) for w in members[:3]] + [eid_device(d) for d in shared[:3]])
            for w in members:
                self._wallet_to_ring[w] = rid
                nd = self.G.nodes[w]
                nd.update(ring_id=rid, risk=max(nd["risk"], self._rings[rid].ring_score))
                nd["flags"].append("Mule ring")
            self._ring_assoc |= set(members)
            for a in agents:
                nd = self.G.nodes[a]
                nd["risk"] = max(nd["risk"], 0.6)
                nd["flags"].append("Cashout exit")
                self._ring_assoc.add(a)
            for d in shared:
                nd = self.G.nodes[d]
                nd["risk"] = max(nd["risk"], 0.5)
                nd["flags"].append("Shared device")
                self._ring_assoc.add(d)
        self._ring_wallets = frozenset(self._wallet_to_ring)

    # ------------------------------------------------------------------ live ingestion
    def ingest(self, txn: Transaction) -> None:
        tid = txn.txn_id or f"LIVE-{int(time.time() * 1000)}"
        ts = _naive(txn.ts)
        self._add_txn(tid, ts, txn.user_id, txn.type, float(txn.amount), txn.recipient_id, txn.device_id)
        if txn.type in ("TRANSFER", "CASH_OUT"):
            self._check_pass_through(txn.user_id, ts, float(txn.amount), tid)

    def attach_scores(self, risk_by_txn: Mapping[str, float]) -> None:
        """Nodes carry the maximum ML risk of the transactions they take part in (recipients at 80%)."""
        for tid, score in risk_by_txn.items():
            meta = self._txn_lookup.get(tid)
            if not meta:
                continue
            s, r = float(score), float(score) * 0.8
            for nid, val in ((meta["sender"], s), (meta["rcpt"], r)):
                if nid in self.G:
                    self.G.nodes[nid]["risk"] = max(self.G.nodes[nid]["risk"], min(0.99, val))

    # ------------------------------------------------------------------ signals
    def signals(self, txn_id: str) -> GraphSignals:
        meta = self._txn_lookup.get(txn_id)
        if not meta:
            return GraphSignals(wallet_id="unknown")
        return self.wallet_signals(meta["sender"], as_of=meta["ts"])

    def wallet_signals(self, wallet_id: str, as_of: datetime | None = None) -> GraphSignals:
        asof = _naive(as_of)
        if wallet_id not in self.G:
            return GraphSignals(wallet_id=wallet_id, as_of=asof)

        def upto(rows):
            return [x for x in rows if asof is None or x[0] <= asof]

        ins, outs = upto(self._in.get(wallet_id, ())), upto(self._out.get(wallet_id, ()))
        fan_in, fan_out = len({x[2] for x in ins}), len({x[2] for x in outs})
        in_vol, out_vol = sum(x[1] for x in ins), sum(x[1] for x in outs)
        passthrough = min(2.0, out_vol / in_vol) if in_vol > 0 else 0.0
        events = upto(self._events.get(wallet_id, ()))
        dwell = statistics.median([e[1] for e in events]) if events else None

        rid = self._wallet_to_ring.get(wallet_id)
        if rid and asof is not None and asof < self._wallet_since.get(wallet_id, asof):
            rid = None                                          # not yet detectable at that time
        shared = max((len(self._device_users[d]) for d in self.G.successors(wallet_id)
                      if self.G.nodes[d].get("kind") == "device"), default=0)

        flags, evidence = [], [eid_wallet(wallet_id)]
        if fan_in >= 5 and passthrough >= 0.5:
            flags.append("fan_in_hub")
        if events:
            flags.append("rapid_passthrough")
        if shared >= 3:
            flags.append("shared_device")
        if rid:
            flags.append("ring_member")
            evidence.append(eid_ring(rid))
        return GraphSignals(
            wallet_id=wallet_id, as_of=asof, ring_id=rid, ring_score=self._rings[rid].ring_score if rid else 0.0,
            fan_in=fan_in, fan_out=fan_out, passthrough_ratio=round(passthrough, 2), median_dwell_seconds=dwell,
            shared_device_accounts=shared, flags=flags, evidence_ids=evidence)

    # ------------------------------------------------------------------ views
    def default_center(self) -> str | None:
        """Most interesting entry point: the busiest ring wallet, else the riskiest / busiest wallet."""
        if self._rings:
            top = max(self._rings.values(), key=lambda r: r.total_inflow_bdt)
            return max(top.wallet_ids, key=lambda w: self.G.nodes[w]["total_volume_bdt"])
        wallets = [n for n, d in self.G.nodes(data=True) if d["kind"] == "wallet"]
        return max(wallets, key=lambda n: (self.G.nodes[n]["risk"], self.G.nodes[n]["total_volume_bdt"]), default=None)

    def _neighbours(self, node: str) -> list[str]:
        vol: dict[str, float] = defaultdict(float)
        for _, v, d in self.G.out_edges(node, data=True):
            vol[v] += d["amount_bdt"] + 1
        for u, _, d in self.G.in_edges(node, data=True):
            vol[u] += d["amount_bdt"] + 1
        return [n for n, _ in sorted(vol.items(), key=lambda kv: -kv[1])[:NEIGHBOUR_CAP]]

    def view(self, center: str, depth: int = 2, max_nodes: int = 120, as_of: datetime | None = None) -> GraphView:
        if center not in self.G:
            return GraphView(center=center, nodes=[], edges=[], truncated=False)
        visited, frontier, truncated = {center}, [center], False
        for _ in range(depth):
            nxt: list[str] = []
            for node in frontier:
                for n in self._neighbours(node):
                    if n in visited:
                        continue
                    if len(visited) >= max_nodes:
                        truncated = True
                        break
                    visited.add(n)
                    nxt.append(n)
            frontier = nxt
        nodes = []
        for n in visited:
            d = self.G.nodes[n]
            kind = d["kind"]
            eid = eid_agent(n) if kind == "agent" else eid_device(n) if kind == "device" else eid_wallet(n)
            nodes.append(GraphNode(id=n, kind=kind, label=d["label"], risk=min(1.0, float(d["risk"])),
                                   ring_id=d.get("ring_id"), flags=list(d["flags"]), evidence_id=eid))
        edges = [GraphEdge(source=u, target=v, kind=d["kind"], amount_bdt=round(float(d["amount_bdt"]), 2),
                           count=int(d["count"]), first_ts=d["first_ts"], last_ts=d["last_ts"], txn_ids=list(d["txn_ids"]))
                 for u, v, _k, d in self.G.subgraph(visited).edges(keys=True, data=True)]
        return GraphView(center=center, as_of=_naive(as_of), nodes=nodes, edges=edges, truncated=truncated,
                         stats={"node_count": len(nodes), "edge_count": len(edges)})

    def frontend_view(self, center: str | None = None, depth: int = 2) -> dict[str, Any]:
        """JSON for the vis-network dashboard (shape documented in Frontend/assets/js/data.js)."""
        if not center:
            center = self.default_center()
        if center is None or center not in self.G:                      # unknown entity -> empty graph, never a silent fallback
            return {"center": None, "nodes": [], "edges": []}
        gv = self.view(center=center, depth=depth, max_nodes=100)
        nodes = [{"id": n.id, "type": n.kind, "label": n.label, "risk_score": round(n.risk, 2),
                  "total_volume_bdt": round(float(self.G.nodes[n.id]["total_volume_bdt"]), 2),
                  "flags": n.flags, "ring": n.id in self._ring_assoc} for n in gv.nodes]
        edges = [{"source": e.source, "target": e.target, "amount_bdt": e.amount_bdt, "count": e.count,
                  "last_seen": e.last_ts or "today"} for e in gv.edges]
        return {"center": center, "nodes": nodes, "edges": edges}

    def rings(self, min_score: float = 0.5, as_of: datetime | None = None) -> list[RingSummary]:
        asof = _naive(as_of)
        rows = [r for r in self._rings.values() if r.ring_score >= min_score
                and (asof is None or datetime.fromisoformat(r.first_seen) <= asof)]
        return sorted(rows, key=lambda r: -r.total_inflow_bdt)

    def ring(self, ring_id: str) -> RingSummary:
        if ring_id not in self._rings:
            raise KeyError(ring_id)
        return self._rings[ring_id]

    def ring_wallets(self) -> frozenset[str]:
        return self._ring_wallets

    # ------------------------------------------------------------------ what-if
    def simulate_freeze(self, wallet_ids: list[str], freeze_at: datetime | None = None) -> FreezeImpact:
        """What if these wallets had been frozen? Counts every transaction they sent or received afterwards.
        Uses the dataset's fraud labels (evaluation only, `labels_used=True`)."""
        frozen = set(wallet_ids)
        when = _naive(freeze_at)
        if when is None:
            ring_starts = [datetime.fromisoformat(self._rings[self._wallet_to_ring[w]].first_seen)
                           for w in frozen if w in self._wallet_to_ring]
            when = (min(ring_starts) + timedelta(minutes=30)) if ring_starts else datetime.min
        n = cashouts = 0
        blocked = fraud = legit = 0.0
        downstream: set[str] = set()
        for meta in self._txn_lookup.values():
            if meta["ts"] < when or not (meta["sender"] in frozen or meta["rcpt"] in frozen):
                continue
            n += 1
            blocked += meta["amount"]
            if meta["is_fraud"]:
                fraud += meta["amount"]
            else:
                legit += meta["amount"]
            if meta["type"] == "cash_out":
                cashouts += 1
            if meta["sender"] in frozen and meta["rcpt"] not in frozen:
                downstream.add(meta["rcpt"])
        return FreezeImpact(
            frozen_wallets=wallet_ids, freeze_at=when.replace(tzinfo=timezone.utc) if when != datetime.min
            else datetime(2000, 1, 1, tzinfo=timezone.utc),
            blocked_txn_count=n, blocked_value_bdt=round(blocked, 2), fraud_value_stopped_bdt=round(fraud, 2),
            legit_value_blocked_bdt=round(legit, 2), cashouts_prevented=cashouts,
            downstream_wallets_cut_off=len(downstream), labels_used=True)
