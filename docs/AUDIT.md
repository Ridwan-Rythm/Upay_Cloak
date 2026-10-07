# Phase 2 audit (Step 0)

Scope note (read first): this audit ran in a sandbox with **no network and without FastAPI, LightGBM, pydantic, pytest or ruff installed**.
So nothing that needs the backend, the trained model or the test suite could be executed. Everything below says which way each item was
checked: **RAN** (executed here), **READ** (code reading only) or **NOT CHECKED**. The backend/ML changes made in this pass are compile-checked
(`py_compile`) and covered by new tests that **have not been run**; run `make test`-equivalent (`python -m pytest tests -q`) before relying on them.

## Detected stack
Frontend: plain HTML/CSS/vanilla JS (`Frontend/`, vendored Chart.js + vis-network). Backend: FastAPI + pydantic (`backend/app/`), NetworkX graph,
SQLite (audit + feedback only). ML: pandas, scikit-learn, LightGBM (best model by time-series CV), SHAP, Isolation Forest (`ml/`). Data: synthetic CSV (`data/`).
Entry points: `uvicorn backend.main:app`, `python -m ml.train`, `python scripts/generate_data.py`.

## How the demo scores today (READ)
`ScoringService` loads `models/risk_engine.joblib`. The **held-out test period is replayed** through the model at start-up and each non-ALLOW becomes a case, so the
dashboard's default feed is a **replay** of prepared data, not newly submitted transactions. `POST /api/v1/score` with `{transaction, commit}` does score a new transaction
live with the same feature code. There is no authentication, no RBAC, no rate limit, no hash-chained audit log, no PostgreSQL, no Docker files, no login page.
Where things live: CORS `backend/app/main.py` + `config.py`; storage `backend/app/store/` (in-memory cases + SQLite audit/feedback); assistant `backend/app/intelligence/assistant.py`.

## Repo problems found (RAN / READ)
| Finding | How | Action |
|---|---|---|
| `data/transactions.csv` (79,711 rows, old 12-column schema) shares all 22,981 txn_ids with `train.csv` but only 0.05% of rows agree; README said it was removed; only dead `ml/data.py` pointed at it | RAN | **Deleted** both |
| Root-level `index.html` / `settings.html` duplicates (one loaded Chart.js from a CDN) although README said they were removed | READ | **Deleted** |
| `env.example` misnamed (README says `.env.example`), listed a `UPAY_DATA_PATH` setting that does not exist, set CORS to `*` | READ | **Replaced** by `.env.example` |
| `Frontend updates/` folder = older mock-mode copies (`USE_MOCK=true`) of the live JS files, plus `backend_foundation/` = an older divergent copy of the backend | READ | **Left in place, not touched** (may hold your work); delete if obsolete |
| Settings page said the 0.30/0.65/0.80 risk bands "are the rules the scoring uses"; the model's real thresholds are 0.05/0.10/0.20 (`reports/metrics.json`) | READ | **Text corrected** |
| "94% of fraud value prevented" = amount x **assumed** stop rate per action (50% / 80% / 95%, `ml/decision.py`), and the thresholds were tuned with the same assumptions | READ | Relabelled **PROJECTED** everywhere (README, RESULTS.md) |
| "1.5% customer friction" (Phase 1 pitch) appears nowhere in the repo; recorded value is 1.26% (`legit_txns_with_friction_pct`) | RAN (grep) | README keeps 1.26%; the 1.5% must not be quoted |
| Dataset has **no held-out typology, no ablation, no per-typology table, no multi-seed runs, no calibrator**; the same 24 mule wallets appear in train and test | RAN (`data/DATA_CARD.md`) | Not built (WS-B); disclosed in `docs/LIMITATIONS.md` |
| Validation prevalence 5.9% vs test 3.9% (thresholds tuned on the former) | RAN | Disclosed |

## Table 1: carry-in claims from section 2A of the prompt
| # | Item | Claimed | Verified? | Evidence | Action / where |
|---|---|---|---|---|---|
| 1 | LLM mode untested | not tested | not checked | no key, no network | **Open.** Mock-provider tests not written (M1.1) |
| 2 | Settings / mobile never opened in a browser | not tested | **yes (RAN)** | Playwright, 6 pages x 3 sizes, `reports/ui/` | Phone had **no navigation at all** (sidebar hidden, no menu) and 43 px overflow (76 px on Graph): **FIXED** (drawer + CSS), re-run shows 0 px everywhere; Settings save / reload / reset verified. Not covered: pages with live data (needs backend), axe accessibility check |
| 3 | `ml.feedback --apply` retrain untested | not tested | not checked | needs LightGBM | **Open** (M1.3) |
| 4 | Stray wallet in mule ring | known problem | **partly** | not present on committed seed 42; reproduced on 7 of 31 generator seeds (8 strays, all ATO victims) via a replica of the detector | **FIXED**: membership needs behaviour + forwarded-transfer links; precision 0.989 -> 1.000, recall 1.000 kept (`reports/mule_rings.md`); `member_confidence` added to `RingSummary`; `tests/test_mule_rings.py` (not run) |
| 5 | Scam-victim false blocks ~14% | known problem | not re-measured | needs the model | **DISCLOSED** as reported (`docs/LIMITATIONS.md`) |
| 6 | Bangla missing in PDF | known problem | READ only | no Bengali font available offline | **DISCLOSED**; needs a bundled font + shaping renderer (M2.3) |
| 7 | Live (`commit=true`) cases lost on restart | known problem | **yes (READ)** | `case_store.py` writes only `audit` and `feedback` to SQLite; cases live in `self._cases` | **DISCLOSED**; needs the WS-D persistence work |
| 8 | Adapted thresholds not retroactive | known problem | READ | threshold adaptation only affects new scoring | **DISCLOSED** |
| 9 | Case times show Jan 2026 | known problem | READ | replay uses dataset `ts` | **DISCLOSED** |
| 10 | Stale docs | stale | **yes** | `ml/README` said 23 features (there are 32), `data/README` wrong row counts / removed file, API_CONTRACT said CORS `*` | **FIXED** those; `check_docs_numbers.py` not built |
| 11 | Relaxed lint | stale | READ | `ruff.toml` ignores E501, B904, UP042/46/47 and excludes `ml/` and `scripts/` | **Open** (ruff not installed here) |
| 12 | Google Fonts offline | rough | **yes (RAN)** | 0 third-party requests on all pages at all sizes | Google links removed, `@font-face` points to local files; **font files not bundled** (no network): `Frontend/assets/fonts/README.md`, `docs/HUMAN_TODO.md` |
| 13 | ~13 s start-up | rough | not checked | backend not runnable | **Open** (M2.10) |
| 14 | No auth / open CORS | rough | yes (READ) | `allow_origins=["*"]`, no auth code | **CORS fixed** (allow-list, GET/POST/OPTIONS, 3 headers; compile-checked, test not run). **Auth, RBAC, audit chain, rate limits NOT implemented** |
| 15 | Synthetic data optimism | limit | yes | `data/DATA_CARD.md` | Disclosed; difficulty variants not built (M3.1) |
| 16 | Betting relies on known merchant category | limit | not checked | | **Open** (M3.2) |
| 17 | Scores are not calibrated probabilities | limit | READ | no calibrator in `ml/train.py` | Keep calling it a **risk score (ranking)**; calibration not built (B5/M3.3) |

## Table 2: Phase 2 workstreams, state after this pass
| Workstream | State |
|---|---|
| WS-A problem docs | `docs/PROBLEM.md`, `docs/HUMAN_TODO.md`, interview template written; baselines needing Upay data are `TODO(human)` |
| WS-B ML depth (splits, typologies, ablation, unseen patterns, calibration, artifacts) | **Not done.** Only B1 dataset docs (auto-generated) |
| WS-C live inference API | Not done (a live scoring endpoint already existed; consistency test exists as `test_live_scoring_equals_replay`) |
| WS-D PostgreSQL / case model / graph ADR | Not done |
| WS-E UI wiring, login, admin, assistant panel | Not done (only mobile nav + overflow fixes) |
| WS-F assistant hardening + injection suite | Not done |
| WS-G impact / economics / A-B warning | Not done |
| WS-H reliability + load tests | Not done |
| WS-I differentiation study | Not done |
| WS-J security | CORS only |
| WS-K Docker | Not done (README no longer needs to claim it) |
| WS-L docs | README claims corrected; RESULTS.md, LIMITATIONS.md, AUDIT.md written; API.md / ARCHITECTURE.md / traceability matrix not written |
| WS-M | M1.2 done, M2.1 done, M2.7 partly, M2.9 partly (see Table 1) |
