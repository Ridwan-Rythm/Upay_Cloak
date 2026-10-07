# Results register

Only numbers listed here may be quoted in the README, UI or slides. Labels: MEASURED (run in this repo) / SIMULATED / PROJECTED / NOT RE-VERIFIED (recorded by `ml.train` but not re-run in the audit sandbox) / NOT MEASURED. **All figures are on synthetic data.**

| Metric | Value | Label | Source | Reproduce |
|---|---|---|---|---|
| Dataset rows / fraud prevalence | 32,831 rows, 4.62% overall (test 3.89%) | MEASURED (synthetic) | `data/DATA_CARD.md` | `python -m ml.dataset_stats` |
| Mule-ring member precision, before -> after fix | 0.989 -> 1.000 (recall 1.000 both) | MEASURED (synthetic, 31 seeds, replica of the detector) | `reports/mule_rings.md` | `python scripts/ring_replica_check.py` |
| Pages overflowing horizontally at 375/768/1280 px | 0 of 6 pages at each size (was 6 of 6 at 375 px) | MEASURED (static pages, no backend) | `reports/ui/ui_smoke.json` | `python tests/e2e/ui_smoke.py --serve` |
| Third-party browser requests | 0 | MEASURED (same run) | `reports/ui/ui_smoke.json` | same |
| LightGBM test ROC-AUC / PR-AUC | 0.998 / 0.9199 | NOT RE-VERIFIED | `reports/metrics.json` | `python -m ml.train` |
| Fraud value prevented | 94.0% | **PROJECTED** (uses assumed stop rates 50/80/95%) | `reports/metrics.json`, `ml/decision.py` | `python -m ml.train` |
| Legit transactions receiving friction | 1.26% | NOT RE-VERIFIED | `reports/metrics.json` | `python -m ml.train` |
| Alert rate | 5.06% | NOT RE-VERIFIED | `reports/metrics.json` | `python -m ml.train` |
| Scoring latency (~20 ms in old README) | NOT MEASURED in this audit | NOT MEASURED | n/a | no load test exists |
| Ablation, unseen-pattern, calibration, load test, startup time | NOT MEASURED | NOT MEASURED | n/a | not built |
