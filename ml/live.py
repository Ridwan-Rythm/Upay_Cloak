"""Live (single-transaction) scoring with the SAME features and model as training.

    scorer = LiveScorer()
    scorer.warm(history_df)                       # replay past transactions to build every user's baseline
    case = scorer.score({...raw txn...})          # what-happened / why-risky / what-next JSON (see ml.score)

* `commit=True` also adds the transaction to the rolling state (use for real incoming traffic).
* `snapshot_id` scores against the state as it was just BEFORE a given historical transaction
  (set `snapshot_before=[...]` in `warm`). Lets the demo re-score an edited copy of a historical
  transaction with correct point-in-time baselines.
"""
import copy
import threading
import time
from collections.abc import Iterable

import pandas as pd

from ml.features import RAW_COLUMNS, TYPES, FeatureState, as_txn
from ml.score import MODEL_PATH, RiskEngine


class LiveScorer:
    def __init__(self, engine: RiskEngine | None = None, path=MODEL_PATH):
        self.engine = engine or RiskEngine(path)
        for est in (self.engine.model, self.engine.anomaly.model):      # thread start-up costs more than 1 row of work
            if hasattr(est, "n_jobs"):
                est.n_jobs = 1
        self.state = FeatureState()
        self.snapshots: dict[str, FeatureState] = {}
        self._lock = threading.Lock()
        self.n_history = 0
        self.latencies_ms: list[float] = []

    # ------------------------------------------------------------------ warm-up
    def warm(self, history: pd.DataFrame, snapshot_before: Iterable[str] = ()) -> "LiveScorer":
        """Replay `history` (any order) in time order; snapshot the state just before the given txn ids."""
        want = set(snapshot_before)
        df = history.sort_values("ts", kind="stable")
        with self._lock:
            self.state, self.snapshots = FeatureState(), {}
            for r in df[RAW_COLUMNS].itertuples(index=False):
                if r.txn_id in want:
                    self.snapshots[r.txn_id] = copy.deepcopy(self.state)
                self.state.update(r)
            self.n_history = len(df)
        return self

    # ------------------------------------------------------------------ scoring
    def score(self, txn: dict, snapshot_id: str | None = None, commit: bool = False, explain: str = "flagged") -> dict:
        t0 = time.perf_counter()
        txn = dict(txn)
        if txn.get("type") not in TYPES:
            raise ValueError(f"type must be one of {TYPES}")
        txn.setdefault("txn_id", f"LIVE-{int(time.time() * 1000)}")
        ns = as_txn(txn)
        with self._lock:
            if snapshot_id is not None:
                if snapshot_id not in self.snapshots:
                    raise KeyError(f"no snapshot for {snapshot_id}")
                state = self.snapshots[snapshot_id]     # read-only use: features() only prunes windows at this txn's own time
            else:
                state = self.state
            feats = state.features(ns)
            if commit and snapshot_id is None:
                state.update(ns)
        row = {**{c: getattr(ns, c) for c in RAW_COLUMNS}, **feats}
        frame = pd.DataFrame([row])
        case = self.engine.score_frame(frame, explain=explain)[0]
        case["features"] = {k: (float(v) if not isinstance(v, str) else v) for k, v in feats.items()}
        case["_row"] = row                                              # full row (raw + features) for the backend
        ms = (time.perf_counter() - t0) * 1000
        self.latencies_ms = (self.latencies_ms + [ms])[-500:]
        case["latency_ms"] = round(ms, 2)
        return case

    @property
    def mean_latency_ms(self) -> float | None:
        return round(sum(self.latencies_ms) / len(self.latencies_ms), 1) if self.latencies_ms else None
