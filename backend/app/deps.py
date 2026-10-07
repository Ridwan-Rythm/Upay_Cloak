"""Dependency wiring. Routes use `Depends(get_container)`; tests use `set_container()` or
`app.dependency_overrides[get_container]`.

`build_container()` is the ONE place the system is assembled:
    model + scored cache -> ScoringService (live feature state replayed from history)
    graph + agent services built from the same history
    CaseStore replays the held-out period through the model and opens a case for every flagged transaction
"""
from __future__ import annotations

import logging
import time

from backend.app.config import Settings, get_settings
from backend.app.contracts.interfaces import Container
from backend.app.intelligence import build_intelligence
from backend.app.services.scoring_service import ScoringService
from backend.app.store.case_store import CaseStore

log = logging.getLogger("upayshield")
_container: Container | None = None


def build_container(settings: Settings | None = None, db_path: str | None = None) -> Container:
    settings = settings or get_settings()
    t0 = time.time()
    scoring = ScoringService.load(settings)
    intel = build_intelligence(settings, live=True)
    if scoring is not None:
        intel.graph.build(scoring.history)
        intel.agents.build(scoring.history)
        scoring.attach(intel.graph, intel.agents)
        presets = scoring.build_presets()
        scoring.warm_up([p["txn_id"] for p in presets.values()])
        test = scoring.history[scoring.history.split == "test"]
        intel.graph.attach_scores(dict(zip(test.txn_id, test.risk_score)))
        scoring.bench_latency()
    cases = CaseStore(scoring, settings, db_path)
    log.info("container ready in %.1fs (model=%s, cases=%d)", time.time() - t0,
             scoring.engine.name if scoring else "NOT LOADED", len(cases._cases))
    return Container(cases=cases, intel=intel, scoring=scoring)


def set_container(c: Container | None) -> None:
    global _container
    _container = c


def get_container() -> Container:
    global _container
    if _container is None:
        _container = build_container()
    return _container
