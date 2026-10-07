"""Human-readable rule trace + scam/ATO/mule tags (feeds the case view & LLM evidence JSON).

Single source of truth: ml/score.py, ml/cache.py and the backend all import `rule_trace` from here.
`r` is any mapping with the raw transaction columns and the 23 point-in-time features.
"""


def rule_trace(r):
    hits = []

    def hit(rid, tag, text):
        hits.append(dict(rule_id=rid, tag=tag, text=text))

    if r["is_new_device"] and r["is_new_location"]:
        hit("ATO_01", "account_takeover", "New device AND new location for this wallet")
    if r["is_new_device"] and r["is_night"] and r["txn_count_1h"] >= 2:
        hit("ATO_02", "account_takeover", "New device at night with repeated transactions (velocity burst)")
    if r["type"] == "TRANSFER" and r["is_new_recipient"] and r["amount_z"] > 2.5:
        hit("SCAM_01", "scam_victim", "First-time recipient with an amount far above this user's norm")
    if r["inbound_1h"] > 0 and r["amount"] >= 0.7 * r["inbound_1h"] and r["type"] in ("TRANSFER", "CASH_OUT"):
        hit("MULE_01", "mule_passthrough", "Wallet just received money and is forwarding most of it")
    if r["device_users_count"] >= 3:
        hit("MULE_02", "mule_network", f"Device shared by {int(r['device_users_count'])} accounts")
    if r["near_thr_24h"] >= 2:
        hit("STRUCT_01", "structuring", "Repeated cash-outs just under the 50,000 BDT threshold")
    if r["otp_device_mismatch"] and (r["concurrent_sessions"] >= 2 or r["otp_requests_10m"] >= 2):
        hit("OTP_01", "otp_breach", "OTP confirmed from a different device while another session was active "
                                    "(OTP may have been shared with someone else)")
    if r["otp_failures_10m"] >= 3:
        hit("OTP_02", "otp_breach", f"{int(r['otp_failures_10m'])} failed OTP attempts just before this transaction")
    if r["sim_swap_recent"] and r["is_new_device"]:
        hit("OTP_03", "sim_swap", "SIM card was changed recently and this is a new device")
    if r["is_gambling"]:
        hit("GAMBLE_01", "gambling", "Payment to a betting / gambling service")
    if r["is_gambling"] and r["gambling_txn_24h"] >= 3:
        hit("GAMBLE_02", "gambling_repeat",
            f"{int(r['gambling_txn_24h'])} betting payments in 24 hours (possible loss-chasing)")
    if r["agent_txn_1h"] >= 8 and r["is_night"]:
        hit("AGENT_01", "agent_risk", "Agent processing an unusual late-night burst")
    return hits


PURPOSE_BY_TYPE = {"CASH_OUT": "cash_withdrawal", "CASH_IN": "cash_deposit", "TRANSFER": "person_to_person"}


def purpose_of(txn_type, merchant_category=None):
    """What the money is being used for: merchant category if known, else the kind of movement."""
    if merchant_category and merchant_category == merchant_category:        # not NaN
        return "betting" if merchant_category == "gambling" else str(merchant_category)
    return PURPOSE_BY_TYPE.get(txn_type, "payment")
