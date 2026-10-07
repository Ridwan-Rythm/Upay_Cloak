"""M2.1 regression: ring membership is judged against generator ground truth (cluster level), on several seeds.

Before the fix, innocent ATO *victims* were pulled into mule rings (seeds 10, 11, 19, 21, 23, 26, 30: 8 stray wallets in
31 seeds) because any transfer between two "suspects" linked them. See reports/mule_rings.md.
"""
import pytest

from backend.app.intelligence.graph import NetworkXGraphService
from scripts.generate_data import generate

SEEDS = [42, 10, 11, 19]          # 42 = committed dataset; the others reproduced the stray-wallet bug


def _truth(df):
    m = df[df.scenario == "mule_passthrough"]
    return {w for w in set(m.user_id) | set(m.recipient_id) if str(w).startswith("MU")}


@pytest.mark.parametrize("seed", SEEDS)
def test_rings_have_no_stray_wallets_and_find_every_active_mule(seed):
    df = generate(seed=seed)
    g = NetworkXGraphService()
    g.build(df)
    flagged = g.ring_wallets()
    truth = _truth(df)
    assert flagged - truth == set(), f"stray wallets in rings: {sorted(flagged - truth)}"
    assert truth - flagged == set(), f"missed mules: {sorted(truth - flagged)}"
    assert len(g.rings()) == 6


def test_every_ring_member_has_a_confidence():
    g = NetworkXGraphService()
    g.build(generate(seed=42))
    for ring in g.rings():
        assert set(ring.member_confidence) == set(ring.wallet_ids)
        assert all(0.5 <= c <= 1.0 for c in ring.member_confidence.values())
