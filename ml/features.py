"""Leak-free, point-in-time feature pipeline.

Transactions are replayed in time order. Every feature is computed from state
*before* the current transaction, then the state is updated. So a test-set row
only ever "knows" the past, which is exactly how it works in production.

Two entry points share ONE implementation (so training and live scoring can never drift apart):

    build_features(df)          batch: replay a whole frame (training, cache building)
    FeatureState                streaming: .features(txn) for a new transaction, .update(txn) to commit it
"""
from collections import defaultdict, deque
from types import SimpleNamespace

import numpy as np
import pandas as pd

TYPES = ["CASH_IN", "CASH_OUT", "TRANSFER", "PAYMENT"]
THRESHOLD = 50_000          # BDT regulatory reporting threshold used for the structuring signal
NIGHT_START, NIGHT_END = 22, 6   # "night" = 22:00 .. 05:59 (also used by the agent night-burst rule)

FEATURES = [
    "amount", "log_amount", "type_code", "hour", "is_night",
    # behavioural deviation (vs. this user's own history)
    # (user_txn_count is still computed in the frame but NOT used as a model input:
    #  mule accounts are brand new in the synthetic data, so it would be a data artifact)
    "amount_z", "hour_freq", "amount_to_balance", "log_secs_since_last",
    # account takeover signals
    "is_new_device", "is_new_location", "is_new_recipient", "txn_count_1h", "amount_sum_1h",
    # mule / network signals
    "recipient_unique_senders", "recipient_inbound_count", "recipient_age_txns",
    "device_users_count", "inbound_1h", "passthrough_ratio",
    # structuring / agent signals
    "near_thr_24h", "agent_txn_1h", "agent_cashout_ratio",
    # what the money is used for: betting / gambling and unusual spending categories
    "is_gambling", "gambling_txn_24h", "log_gambling_spend_7d", "is_new_category",
    # OTP / session telemetry: OTP shared with a stranger, remote takeover, SIM swap
    "otp_requests_10m", "otp_failures_10m", "otp_device_mismatch", "concurrent_sessions", "sim_swap_recent",
]

# raw columns every transaction must carry (labels are optional and never used as features)
RAW_COLUMNS = ["txn_id", "ts", "user_id", "type", "amount", "recipient_id", "agent_id",
               "device_id", "location", "balance_before", "merchant_category",
               "otp_requests_10m", "otp_failures_10m", "otp_device_mismatch", "concurrent_sessions", "sim_swap_recent"]
OTP_DEFAULTS = dict(otp_requests_10m=0, otp_failures_10m=0, otp_device_mismatch=0, concurrent_sessions=1,
                    sim_swap_recent=0)
GAMBLING = "gambling"


def _user_state():
    return dict(n=0, mean=0.0, m2=0.0, hours=np.zeros(24), devices=set(), locs=set(), recips=set(),
                recent=deque(), last=None, inbound=deque(), near=deque(),
                gamble=deque(), cats=set())


def _has(x) -> bool:
    """True for a real value (not None / NaN / pandas NA / empty string)."""
    if x is None or x is pd.NA:
        return False
    if isinstance(x, float) and x != x:
        return False
    return x != ""


def is_night_hour(h: int) -> bool:
    return h >= NIGHT_START or h < NIGHT_END


class FeatureState:
    """Rolling per-user / per-recipient / per-device / per-agent state."""

    def __init__(self):
        self.U = defaultdict(_user_state)
        self.R_senders, self.R_in = defaultdict(set), defaultdict(int)
        self.D_users = defaultdict(set)
        self.A = defaultdict(lambda: dict(out=0, inn=0, recent=deque()))
        self.last_ts = None

    # ------------------------------------------------------------------ features (read-only apart from window pruning)
    def features(self, r) -> dict:
        """Features for transaction `r` (any object with the RAW_COLUMNS as attributes), from the PAST only."""
        t = r.ts.timestamp()
        h = r.ts.hour
        u = self.U[r.user_id] if r.user_id in self.U else _user_state()
        warm = u["n"] >= 5                      # need some history before "deviation" means anything
        la = np.log1p(r.amount)
        for dq, win in ((u["recent"], 3600), (u["inbound"], 3600), (u["near"], 86400), (u["gamble"], 7 * 86400)):
            while dq and dq[0][0] < t - win:
                dq.popleft()
        std = max(np.sqrt(u["m2"] / (u["n"] - 1)), 0.3) if u["n"] > 1 else 1.0
        is_cash = r.type in ("CASH_IN", "CASH_OUT")
        near = r.type == "CASH_OUT" and THRESHOLD * .9 <= r.amount < THRESHOLD
        a = None
        if is_cash and _has(r.agent_id):
            a = self.A.get(r.agent_id) or dict(out=0, inn=0, recent=deque())   # first-ever txn of an agent
        if a:
            while a["recent"] and a["recent"][0] < t - 3600:
                a["recent"].popleft()
        in1h = sum(x[1] for x in u["inbound"])
        rec_n = self.U[r.recipient_id]["n"] if (r.type == "TRANSFER" and r.recipient_id in self.U) else \
            (0 if r.type == "TRANSFER" else 100)
        secs = max(t - u["last"], 0.0) if u["last"] is not None else None
        cat = r.merchant_category if _has(r.merchant_category) else None
        gamble = cat == GAMBLING
        g24 = sum(1 for x in u["gamble"] if x[0] >= t - 86400) + int(gamble)
        g7 = sum(x[1] for x in u["gamble"]) + (r.amount if gamble else 0.0)
        return dict(
            log_amount=la, type_code=TYPES.index(r.type), hour=h, is_night=int(is_night_hour(h)),
            amount_z=(la - u["mean"]) / std if warm else 0.0,
            hour_freq=(u["hours"][h] + u["hours"][(h - 1) % 24] + u["hours"][(h + 1) % 24]) / u["n"] if warm else .5,
            amount_to_balance=r.amount / max(r.balance_before, 1),
            user_txn_count=u["n"],
            log_secs_since_last=np.log1p(secs) if secs is not None else np.log1p(7 * 86400),
            is_new_device=int(warm and r.device_id not in u["devices"]),
            is_new_location=int(warm and r.location not in u["locs"]),
            is_new_recipient=int(warm and r.recipient_id not in u["recips"]),
            txn_count_1h=len(u["recent"]) + 1,
            amount_sum_1h=sum(x[1] for x in u["recent"]) + r.amount,
            recipient_unique_senders=len(self.R_senders.get(r.recipient_id, ())),
            recipient_inbound_count=self.R_in.get(r.recipient_id, 0),
            recipient_age_txns=rec_n,
            device_users_count=len(self.D_users.get(r.device_id, set()) | {r.user_id}),
            inbound_1h=in1h,
            passthrough_ratio=min(r.amount / in1h, 2.0) if in1h > 0 else 0.0,
            near_thr_24h=len(u["near"]) + int(near),
            agent_txn_1h=len(a["recent"]) + 1 if a else 0,
            agent_cashout_ratio=a["out"] / (a["out"] + a["inn"] + 1) if a else 0.0,
            is_gambling=int(gamble), gambling_txn_24h=g24, log_gambling_spend_7d=np.log1p(g7),
            is_new_category=int(warm and cat is not None and cat not in u["cats"]),
        )                       # (the 5 OTP / session features are raw columns already present in the frame)

    # ------------------------------------------------------------------ state update (commit a transaction)
    def update(self, r) -> None:
        t = r.ts.timestamp()
        h = r.ts.hour
        u = self.U[r.user_id]
        la = np.log1p(r.amount)
        is_cash = r.type in ("CASH_IN", "CASH_OUT")
        near = r.type == "CASH_OUT" and THRESHOLD * .9 <= r.amount < THRESHOLD
        u["n"] += 1
        d = la - u["mean"]
        u["mean"] += d / u["n"]
        u["m2"] += d * (la - u["mean"])
        u["hours"][h] += 1
        u["devices"].add(r.device_id)
        u["locs"].add(r.location)
        u["recips"].add(r.recipient_id)
        u["recent"].append((t, r.amount))
        u["last"] = t
        if near:
            u["near"].append((t, r.amount))
        if _has(r.merchant_category):
            u["cats"].add(r.merchant_category)
            if r.merchant_category == GAMBLING:
                u["gamble"].append((t, r.amount))
        self.R_senders[r.recipient_id].add(r.user_id)
        self.R_in[r.recipient_id] += 1
        if r.type == "TRANSFER":
            self.U[r.recipient_id]["inbound"].append((t, r.amount))
        self.D_users[r.device_id].add(r.user_id)
        if is_cash and _has(r.agent_id):
            a = self.A[r.agent_id]
            a["recent"].append(t)
            a["out" if r.type == "CASH_OUT" else "inn"] += 1
        self.last_ts = r.ts if self.last_ts is None else max(self.last_ts, r.ts)


def as_txn(row) -> SimpleNamespace:
    """dict / Series -> attribute-style transaction with a proper Timestamp."""
    d = dict(row)
    d["ts"] = pd.Timestamp(d["ts"])
    d.setdefault("agent_id", None)
    d.setdefault("merchant_category", None)
    for k, v in OTP_DEFAULTS.items():                    # clients that cannot send OTP telemetry get "normal" values
        d[k] = v if d.get(k) is None else d[k]
    d["amount"] = float(d["amount"])
    d["balance_before"] = float(d.get("balance_before") or 0.0)
    return SimpleNamespace(**d)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    state = FeatureState()
    out = []
    for r in df.itertuples(index=False):
        out.append(state.features(r))
        state.update(r)
    return pd.concat([df, pd.DataFrame(out)], axis=1)
