# Known limitations (every one is DISCLOSED here or FIXED elsewhere; see docs/AUDIT.md)

1. **Synthetic data.** The generator injects the fraud the model later finds, with only ~6% label noise. Every accuracy figure is an upper bound. `data/DATA_CARD.md`.
2. **No unseen-pattern test.** All 7 typologies appear in train and test, and the same 24 mule wallets recur in both, so the test set does not show how the model handles a new fraud pattern or new rings.
3. **No ablation.** The contribution of anomaly detection, graph signals and rules to detection is not measured. Only model families were compared (LogReg / RF / LightGBM / Isolation Forest).
4. **Value-prevented numbers are assumptions.** "94% of fraud value prevented" multiplies fraud amounts by assumed stop rates (50% step-up, 80% hold, 95% block). It is **PROJECTED**, not measured prevention.
5. **Scores are ranking scores.** No calibrator is fitted; do not read a score as a probability. The Brier/ECE in `reports/metrics.json` were computed on the raw class-weighted score and were not re-verified.
6. **Scam victims are the hardest class.** As reported by the previous build (not re-measured here): about 14% of blocks are legitimate large first-time transfers; hold/block recall for scam victims is 88.5% vs 92.3% with any action (`reports/metrics.json`).
7. **Betting detection relies on a known merchant category** (not tested with missing or renamed categories).
8. **Bangla does not render in PDF reports** (no Bengali glyphs in the PDF font).
9. **Live (`commit=true`) cases are lost on restart**; only analyst actions and feedback are saved (SQLite).
10. **Adapted thresholds are not retroactive**; already-replayed cases keep their original decisions.
11. **Case times** on the dashboard are dataset time (January 2026), not wall-clock.
12. **Security gaps:** no authentication, RBAC, tamper-evident audit log or rate limiting. CORS is now an allow-list, but anything that can reach the API can call every endpoint.
13. **Mule-ring check:** rings are always 4 wallets in this generator; the membership fix is validated on that shape only. `member_confidence` is a heuristic, not a probability. (`reports/mule_rings.md`)
14. **Feedback loop selection bias:** analysts only label cases the model already flagged, so retraining on their labels is biased toward the model's own view.
15. **Fonts are not bundled**, so the UI uses system fonts until the two woff2 files are added (`Frontend/assets/fonts/README.md`).
16. **Not run in the audit sandbox:** backend, test suite, lint, LLM mode, retraining, start-up timing, accessibility (axe).
