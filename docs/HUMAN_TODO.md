# Things only humans can do

1. **Run the test suite and the backend** (`pip install -r requirements.txt && python -m pytest tests -q`). The Phase 2 audit sandbox could not; new tests `tests/test_mule_rings.py` and the tightened ring assertion are unrun.
2. **Add the two font files** (Plus Jakarta Sans, Noto Sans Bengali, both OFL) to `Frontend/assets/fonts/`; see its README.
3. **Re-verify RESULTS.md numbers**: run `python -m ml.train` and compare with `reports/metrics.json`; relabel anything that changes.
4. **Collect real validation evidence:** analyst interviews (use `docs/templates/analyst_interview_template.md`) and independent public sources on mule / scam fraud in Bangladesh MFS.
5. **Real baselines from upay** (fraud loss, % suspicious transactions, investigation time, false-positive rate, analyst workload). Until then they stay `TODO(human)`.
6. **Decide the policy** for BLOCK vs step-up friction on large first-time transfers (customer friction vs scam prevention).
7. **Provide an LLM key** if you want LLM mode tested live; decide whether the demo uses LLM or template mode.
8. **Run any analyst study / bilingual warning pilot** with real volunteers if results should be labelled MEASURED.
9. **Check the UI on a real phone** (and in Bangla) with the backend running, and delete `Frontend updates/` and `backend_foundation/` if they are obsolete.
10. **Pitch/slides:** only quote numbers that appear in `docs/RESULTS.md` with their label. Do not say "1.5% friction".
