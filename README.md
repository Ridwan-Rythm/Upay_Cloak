# UpayShield (Upay_Cloak): AI fraud protection for mobile money

> **Phase 2 status:** see `docs/AUDIT.md` (what was verified and fixed), `docs/RESULTS.md` (the only numbers you may quote, with labels) and `docs/LIMITATIONS.md`. Phase 2 is **partly done**: ML depth (ablation, unseen patterns, calibration), PostgreSQL, authentication/RBAC and load tests are **not built yet** (Docker: single-container setup added, see "Run with Docker"; not yet run end to end). The backend, tests and lint were not runnable in the audit sandbox.

A trained ML model scores every transaction; a FastAPI backend turns the score into a decision, a case and a plain-language explanation; a static dashboard shows it. All data is **synthetic**.

## What it detects
| Threat | Signals the model uses |
|---|---|
| Account takeover | new device / location, night burst, amount vs. the user's own history |
| **OTP breach** (OTP shared with a stranger) | OTP confirmed from a *different device*, 2 sessions at once, OTP request bursts, failed OTPs, recent SIM swap |
| **Betting / gambling** | payment to a betting merchant or bookie wallet, repeated payments in 24h (loss-chasing), 7-day betting spend, new spending category |
| **What the money is for** | purpose per transaction (betting, utilities, person-to-person, cash...) + the user's usual spending profile |
| Scam victim | first-time recipient with an amount far above normal |
| Mule rings | rapid pass-through (money forwarded within 30 min), rings found as connected mule wallets, frozen |
| Structuring, rogue agents | cash-outs just under 50,000 BDT; agent night bursts vs. peers |

## Run it
```bash
pip install -r requirements.txt
python scripts/generate_data.py      # optional: data/ is committed (seed 42, reproducible)
python -m ml.train                   # trains, tunes thresholds, writes models/, reports/, data/cache/
uvicorn backend.main:app --port 8000 # open http://localhost:8000
python -m pytest tests -q            # 52 tests collected (47 + 5 mule-ring tests); NOT run in the Phase 2 audit sandbox
```
If `models/risk_engine.joblib` is missing the API starts in degraded mode (scoring endpoints return `503 model_not_loaded`); set `UPAY_AUTO_TRAIN=true` to train on boot.

## Run with Docker
One container serves the API and the dashboard (no separate frontend container, no new services).

**Prerequisites:** Docker Engine/Desktop with Compose v2.24 or newer (needed for the optional `.env`). `make` is optional.

```bash
cp env.example .env            # optional: the app runs with its defaults without a .env
docker compose up --build      # or: make docker-up  (detached)
```
- Open http://localhost:8000 (dashboard); `http://localhost:8000/health` should show `"status": "ok"`. Start-up replays the held-out period, so allow roughly 10-15 s (the healthcheck waits up to 90 s).
- Stop: `docker compose down` (`make docker-down`). Logs: `make docker-logs`. Other targets: `make docker-build`.
- **SQLite data** (analyst actions and feedback) is in the named volume `upay_db`, mounted at `/app/dbdata` (`UPAY_DB_PATH=/app/dbdata/upayshield.db`). It survives `docker compose down` and restarts; `docker compose down -v` deletes it.
- **Model artifact:** at build time, `models/risk_engine.joblib` from your checkout is used if it exists and loads with the image's library versions; otherwise `python -m ml.train` runs during the build (the file is git-ignored, so a fresh clone trains once, which takes a while). The container then starts in seconds-to-tens-of-seconds with no training on boot.
- **Retrain:** `python -m ml.train` on the host and `docker compose up --build`, or delete `models/risk_engine.joblib` and rebuild with `docker compose build --no-cache api` to train inside the build.
- `.env` is git-ignored and never copied into the image; set `UPAY_LLM_*` there if you want the optional LLM assistant.

## How the backend uses the ML
`ScoringService` loads the model and replays history into a rolling feature state (same code as training, tested to match). The held-out test period is replayed through the model at start-up; every non-allow becomes a case. Pages: **Dashboard** (live model-scored feed), **Cases** (what happened / why risky / what next, baseline, purpose, evidence, grounded Q&A, analyst actions), **Graph**, **Warning demo** (real transactions, editable amount/recipient, scored live, EN/বাংলা).

- `POST /api/v1/score` `{txn_id}` or `{transaction, commit}`: stored or live model score (+ cold-start / over-balance warnings)
- `POST /api/v1/warning/check`: pre-send customer warning (EN/BN) from the model
- `GET /api/v1/cases | /kpis | /metrics | /config`, `POST /api/v1/cases/{id}/feedback`, `GET /api/v1/feedback/export` (CSV for `python -m ml.feedback`)
- `GET /api/v1/stream` (SSE), `/stream/control`; `POST /api/v1/admin/thresholds/adapt|reset` (learn thresholds from analyst verdicts, bounded)
- `GET /api/v1/graph/rings`, `/agents`, `/cases/{id}/evidence|narrative|ask|report` (Markdown / real PDF); `/api/*` serves the dashboard. Docs: `/docs`, contract: `docs/API_CONTRACT.md`.

Decision = model thresholds (learned by `ml.train`, never hard-coded) + policies. Policies only *raise* actions; a rule alone is capped at step-up verification, HOLD/BLOCK need the model to agree, ring members are frozen.

## Model results (held-out last 7 days, 9,850 txns, 3.9% illicit)
Model selected by time-series CV (not a single noisy split): **LightGBM**.
| Model | CV PR-AUC | Test ROC-AUC | Test PR-AUC |
|---|---|---|---|
| Logistic Regression | 0.938 | 0.998 | 0.915 |
| Random Forest | 0.936 | 0.998 | 0.896 |
| **LightGBM** | **0.949** | 0.998 | **0.920** |
| Isolation Forest | n/a | 0.975 | 0.675 |

Business impact (**PROJECTED**, synthetic data): **94.0%** of fraud *value* would be prevented **if** step-up / hold / block stop 50% / 80% / 95% of the money (assumed in `ml/decision.py`, not measured); **1.26%** of legitimate transactions get friction (reported by `ml.train`, not re-run in the audit). Precision by action: allow 0.04% fraud, step-up 2.8%, hold 10.7%, block 86%. Recall by scenario (any action): ATO, gambling, mule, structuring, rogue agent 100%; OTP breach 93%; scam victim 92% (hold/block only 88%). Brier 0.005 (not re-verified). Latency: NOT MEASURED (no load test yet).

**Be honest about limits:** data is synthetic with 6% label noise, so absolute numbers are optimistic. Scam victims are the hardest class (a legitimate large first-time transfer looks the same), and about 14% of "block"s are such legitimate one-offs. Betting detection relies on knowing the merchant category. Scores are class-weighted ranking scores, not calibrated probabilities (call them "risk score", never "probability").

## Bugs fixed in this version
- Backend never called the ML (`/score` was `if amount > 30000`, cases/KPIs/metrics hard-coded, thresholds 0.20/0.25/0.30 hard-coded). Now everything comes from the model.
- Stale `transactions.csv` with colliding txn ids was loaded instead of the model's data (removed).
- Model selection picked the worst model on test (Random Forest) from one tiny validation slice; now time-series CV. `ml/evaluate.py` imported a function that didn't exist; duplicated, diverging rule code merged.
- Mule detector flagged 870 innocent wallets/agents as 732 "rings"; now 6 rings, 24/24 mules, and (Phase 2) ring members must behave like mules, which removed stray ATO victims that appeared on 7 of 31 generator seeds (`reports/mule_rings.md`). Agent policy held 158 legitimate customers; ATO rule blocked travellers; thresholds missed the rogue agents. Fake seeded agents / fake freeze numbers removed.
- "PDF" report was Markdown bytes; LLM path always called Gemini and accepted uncited sentences; "hold" was stored as a fraud verdict and the feedback CSV couldn't be read by `ml.feedback`; errors now use the contract shape.
- Frontend: POSTs lacked `Content-Type`, errors were ignored, the feed was generated in the browser, graph searched `/graph/null`, evidence chips threw, root-level duplicate pages removed.

## Layout
`ml/` features, rules, train, live scorer · `backend/app/` api, services (scoring, decision), intelligence (graph, agents, assistant, reports), store · `Frontend/` dashboard · `scripts/generate_data.py` · `tests/` · `docs/`. Config: `env.example`. Optional LLM: `UPAY_LLM_PROVIDER=gemini|anthropic|openai` + `UPAY_LLM_API_KEY` (answers must cite evidence ids, else template fallback).
