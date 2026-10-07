# Mule-ring membership: before vs after (M2.1)

Label: **MEASURED** (SIMULATED data; generator ground truth; 31 seeds incl. committed seed 42). Produced by `python scripts/ring_replica_check.py` using a pandas/networkx replica of the detector (the real service is tested by `tests/test_mule_rings.py`; that test was NOT run in the audit sandbox, see docs/AUDIT.md).

| Rule | member precision | member recall | stray wallets (total) | seeds with a stray |
|---|---|---|---|---|
| before (any transfer between two suspects links them) | 0.989 | 1.000 | 8 | 7 of 31 |
| after (suspect must mostly forward into suspects/cash-out; link only on forwarded transfers) | 1.000 | 1.000 | 0 | 0 of 31 |

Strays seen before the fix were ATO *victims* who had also forwarded two small P2P transfers weeks apart and had paid a ring hub; the old rule linked them to the ring through that victim->hub transfer.

Caveats: ground truth counts only the 24 mule wallets that transact (the generator names 30 but 6 never move money, so ring-level recall of the 'named' 30 is not measurable). Rings are always 4 wallets in this generator, so the result says little about larger or camouflaged rings. Per-member `member_confidence` is a heuristic (share of a wallet's pass-through events that forward into suspects or a cash-out agent), not a calibrated probability.

```
 seed  true_active_mules  before_tp  before_stray  before_missed  after_tp  after_stray  after_missed
   42                 24         24             0              0        24            0             0
    1                 24         24             0              0        24            0             0
    2                 24         24             0              0        24            0             0
    3                 24         24             0              0        24            0             0
    4                 24         24             0              0        24            0             0
    5                 24         24             0              0        24            0             0
    6                 24         24             0              0        24            0             0
    7                 24         24             0              0        24            0             0
    8                 24         24             0              0        24            0             0
    9                 24         24             0              0        24            0             0
   10                 24         24             2              0        24            0             0
   11                 24         24             1              0        24            0             0
   12                 24         24             0              0        24            0             0
   13                 24         24             0              0        24            0             0
   14                 24         24             0              0        24            0             0
   15                 24         24             0              0        24            0             0
   16                 24         24             0              0        24            0             0
   17                 24         24             0              0        24            0             0
   18                 24         24             0              0        24            0             0
   19                 24         24             1              0        24            0             0
   20                 24         24             0              0        24            0             0
   21                 24         24             1              0        24            0             0
   22                 24         24             0              0        24            0             0
   23                 24         24             1              0        24            0             0
   24                 24         24             0              0        24            0             0
   25                 24         24             0              0        24            0             0
   26                 24         24             1              0        24            0             0
   27                 24         24             0              0        24            0             0
   28                 24         24             0              0        24            0             0
   29                 24         24             0              0        24            0             0
   30                 24         24             1              0        24            0             0
```
