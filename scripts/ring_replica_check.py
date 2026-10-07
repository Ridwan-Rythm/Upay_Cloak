"""Dependency-light check of the mule-ring membership rule (M2.1): runs BEFORE vs AFTER on many generator seeds.

It re-implements the pass-through + membership logic of backend/app/intelligence/graph.py with pandas + networkx only
(no FastAPI/pydantic), so it runs anywhere. The real service is covered by tests/test_mule_rings.py.
Usage: python scripts/ring_replica_check.py [--seeds 31] [--write reports/mule_rings.md]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import networkx as nx
import pandas as pd

from scripts.generate_data import generate  # noqa: E402


def events(df, dwell=1800, lo=.7, hi=1.1):
    inb=df[df.type=="TRANSFER"][["recipient_id","ts","amount"]].rename(columns={"recipient_id":"w","ts":"ts_in","amount":"a_in"})
    out=df[df.type.isin(["TRANSFER","CASH_OUT"])][["user_id","ts","amount","txn_id","recipient_id","type"]].rename(columns={"user_id":"w","ts":"ts_out","amount":"a_out","txn_id":"out_txn","recipient_id":"dst"})
    m=out.merge(inb,on="w"); m["dwell"]=(m.ts_out-m.ts_in).dt.total_seconds()
    return m[(m.dwell>0)&(m.dwell<=dwell)&(m.a_out>=lo*m.a_in)&(m.a_out<=hi*m.a_in)].sort_values("dwell").drop_duplicates("out_txn")

def before(df,minev=2):
    m=events(df); cnt=m.groupby("w").size(); sus=set(cnt[cnt>=minev].index)
    H=nx.Graph(); H.add_nodes_from(sus)
    for r in df[(df.type=="TRANSFER")&df.user_id.isin(sus)&df.recipient_id.isin(sus)].itertuples(): H.add_edge(r.user_id,r.recipient_id)
    return [c for c in nx.connected_components(H) if len(c)>=2]

def after(df,minev=2,minconf=.5):
    m=events(df); cnt=m.groupby("w").size(); cand=set(cnt[cnt>=minev].index); anyev=set(cnt.index)
    m["ok"]=(m.type=="CASH_OUT")|m.dst.isin(anyev)
    conf=m.groupby("w").ok.mean(); sus={w for w in cand if conf[w]>=minconf}
    H=nx.Graph(); H.add_nodes_from(sus)
    for r in m[(m.type=="TRANSFER")&m.ok&m.w.isin(sus)&m.dst.isin(sus)].itertuples(): H.add_edge(r.w,r.dst)
    return [c for c in nx.connected_components(H) if len(c)>=2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--write", default=None)
    a = ap.parse_args()
    rows = []
    for seed in [42] + list(range(1, a.seeds + 1)):
        df = generate(seed=seed)
        m = df[df.scenario == "mule_passthrough"]
        truth = {w for w in set(m.user_id) | set(m.recipient_id) if str(w).startswith("MU")}
        r = [seed, len(truth)]
        for f in (before, after):
            cs = f(df)
            fl = set().union(*cs) if cs else set()
            r += [len(fl & truth), len(fl - truth), len(truth - fl)]
        rows.append(r)
    d = pd.DataFrame(rows, columns=["seed", "true_active_mules", "before_tp", "before_stray", "before_missed",
                                    "after_tp", "after_stray", "after_missed"])
    tot = d.sum(numeric_only=True)
    n = len(d)
    prec = lambda tp, st: tp / max(tp + st, 1)  # noqa: E731
    rec = lambda tp, ms: tp / max(tp + ms, 1)   # noqa: E731
    md = [
        "# Mule-ring membership: before vs after (M2.1)", "",
        f"Label: **MEASURED** (SIMULATED data; generator ground truth; {n} seeds incl. committed seed 42). "
        "Produced by `python scripts/ring_replica_check.py` using a pandas/networkx replica of the detector "
        "(the real service is tested by `tests/test_mule_rings.py`; that test was NOT run in the audit sandbox, see docs/AUDIT.md).", "",
        "| Rule | member precision | member recall | stray wallets (total) | seeds with a stray |", "|---|---|---|---|---|",
        f"| before (any transfer between two suspects links them) | {prec(tot.before_tp, tot.before_stray):.3f} | "
        f"{rec(tot.before_tp, tot.before_missed):.3f} | {int(tot.before_stray)} | {int((d.before_stray > 0).sum())} of {n} |",
        f"| after (suspect must mostly forward into suspects/cash-out; link only on forwarded transfers) | "
        f"{prec(tot.after_tp, tot.after_stray):.3f} | {rec(tot.after_tp, tot.after_missed):.3f} | {int(tot.after_stray)} | "
        f"{int((d.after_stray > 0).sum())} of {n} |", "",
        "Strays seen before the fix were ATO *victims* who had also forwarded two small P2P transfers weeks apart and "
        "had paid a ring hub; the old rule linked them to the ring through that victim->hub transfer.", "",
        "Caveats: ground truth counts only the 24 mule wallets that transact (the generator names 30 but 6 never move money, so "
        "ring-level recall of the 'named' 30 is not measurable). Rings are always 4 wallets in this generator, so the result "
        "says little about larger or camouflaged rings. Per-member `member_confidence` is a heuristic (share of a wallet's "
        "pass-through events that forward into suspects or a cash-out agent), not a calibrated probability.", "",
        "```", d.to_string(index=False), "```", "",
    ]
    out = "\n".join(md)
    print(out)
    if a.write:
        Path(a.write).write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
