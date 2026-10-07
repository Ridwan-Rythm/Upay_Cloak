# Problem definition

**Primary problem (chosen target): coordinated mule-account networks and the scam-payment cash-out path, as seen by a upay fraud analyst.**
Secondary (ranked below it): account takeover, OTP breach, structuring, rogue agents, prohibited betting.

| User | Problem | Consequence | Metric |
|---|---|---|---|
| upay fraud analyst | slow identification of coordinated mule accounts and suspicious transactions | investigation delay, financial loss, unnecessary customer friction | investigation time, fraud amount prevented, false-positive rate, analyst workload |

## Baseline (current situation)
| Item | Value | Source | Label |
|---|---|---|---|
| Fraud loss | TODO(human) | upay finance / fraud ops | TODO(human) |
| % suspicious transactions | TODO(human) | upay | TODO(human) |
| Average investigation time | TODO(human) | analyst interviews | TODO(human) |
| Customer friction from wrong holds | TODO(human) | upay support data | TODO(human) |
| False-positive rate of today's process | TODO(human) | upay | TODO(human) |
| Analyst workload (cases/day) | TODO(human) | analyst interviews | TODO(human) |
| Fraud prevalence in our synthetic data | 4.62% overall, 3.89% in test | `data/DATA_CARD.md` | MEASURED (synthetic data, not a real-world rate) |
| Alert rate of our model on the test period | 5.06% (`alert_rate_pct`) | `reports/metrics.json` | reported by `ml.train`, not re-run in the audit; synthetic |

## Validation evidence (to be collected, none invented here)
- Analyst interviews: `docs/templates/analyst_interview_template.md`, 0 completed.
- Independent public sources on mule/scam fraud in Bangladesh MFS: TODO(human) (add URLs).
- Operational data from upay: TODO(human).

## Improvement hypotheses (targets are filled in only after experiments exist)
- Reduce average investigation time by X%: X = TODO (needs the case-centric vs score-only study, not run).
- Reduce false positives by X%: X = TODO (needs the ablation, not run).
- Increase mule-member precision: measured **0.989 -> 1.000** at unchanged recall 1.000 on synthetic rings (`reports/mule_rings.md`).

> Upay fraud analysts currently face **slow, manual identification of coordinated mule accounts and scam cash-outs (size of the delay: TODO(human))**, causing **investigation delays, losses and wrongly held customers (amounts: TODO(human))**. Our system targets this and aims to improve **time to a justified decision and the precision of ring membership**; only the second is measured so far, and only on synthetic data.
