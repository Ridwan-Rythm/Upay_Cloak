"""Central settings (env-driven). Every setting has a safe default, so the app boots with zero config
(LLM disabled -> grounded template narratives). Copy `.env.example` to `.env` to override.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_prefix="UPAY_", extra="ignore",
                                      protected_namespaces=())

    # --- artifacts (produced by `python -m ml.train`)
    model_path: Path = ROOT / "models" / "risk_engine.joblib"
    reports_dir: Path = ROOT / "reports"
    cache_dir: Path = ROOT / "data" / "cache"
    db_path: str = str(ROOT / "data" / "upayshield.db")     # SQLite: audit trail + analyst feedback (":memory:" ok)
    auto_train: bool = False                  # run `python -m ml.train` on boot if the model artifact is missing

    # --- http
    # Explicit allow-list (J4). The dashboard is served by this same app, so it needs no CORS at all; list extra
    # front-end origins in UPAY_CORS_ORIGINS (JSON list). "*" is no longer the default.
    cors_origins: list[str] = ["http://localhost:8000", "http://127.0.0.1:8000"]

    # --- stream simulator (dashboard "Live transactions")
    stream_default_interval_ms: int = 3500
    stream_demo_alert_every: int = 6          # in demo mode every Nth event is a real flagged transaction

    # --- LLM (never crash when unset)
    llm_provider: Literal["none", "gemini", "anthropic", "openai"] = "none"
    llm_api_key: str | None = Field(
        None, validation_alias=AliasChoices("UPAY_LLM_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"))
    llm_model: str | None = None              # provider default is chosen in code when None
    llm_timeout_s: float = 8.0
    llm_max_retries: int = 1

    @property
    def cache_path(self) -> Path:
        return self.cache_dir / "scored_cache.parquet"

    @property
    def metrics_path(self) -> Path:
        return self.reports_dir / "metrics.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()
