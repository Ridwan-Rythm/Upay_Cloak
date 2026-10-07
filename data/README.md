# Sample dataset (SYNTHETIC)

Our own synthetic, Bangladesh-flavoured mobile-money dataset (BDT amounts, Dhaka/Chattogram-style locations, salary-day spikes).
Reproducible with `python scripts/generate_data.py --seed 42`. **No real customer data; accuracy measured on it is optimistic.**
Row counts, prevalence, scenario mix and leakage checks are generated, never typed: see **`DATA_CARD.md`** and `docs/DATASET.md`
(`python -m ml.dataset_stats`).

| File | Purpose |
|---|---|
| `train.csv` | first 70% of the timeline; `ml/train.py` carves the last 20% of it (by time) as validation |
| `test.csv` | last 30% of the timeline, used only for the final report |

`transactions.csv` (an older dataset whose txn ids collided with `train.csv`) was deleted in the Phase 2 audit.

## Columns
`txn_id, ts, user_id, type (CASH_IN/CASH_OUT/TRANSFER/PAYMENT), amount (BDT), recipient_id, agent_id, device_id, location, balance_before,
merchant_category, otp_requests_10m, otp_failures_10m, otp_device_mismatch, concurrent_sessions, sim_swap_recent, is_fraud, scenario`.
`is_fraud` is the label (about 6% of injected fraud is deliberately left 0 as unreported fraud); `scenario` is the ground-truth typology
(analysis only, never a model input).
