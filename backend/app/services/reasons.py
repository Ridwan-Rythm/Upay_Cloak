"""Maps raw ML features / rules to the vocabulary the UI speaks, and holds customer-safe bilingual wording.

UI keys (also used by Frontend/assets/js/i18n.js):
    dev loc rcp amt hr drain vel mule agent struct scam other
"""
from __future__ import annotations

FEATURE_KEY = {
    "is_new_device": "dev", "is_new_location": "loc", "is_new_recipient": "rcp",
    "amount_z": "amt", "amount": "amt", "log_amount": "amt",
    "hour_freq": "hr", "is_night": "hr", "hour": "hr",
    "amount_to_balance": "drain", "amount_sum_1h": "drain",
    "txn_count_1h": "vel", "log_secs_since_last": "vel",
    "recipient_unique_senders": "mule", "recipient_inbound_count": "mule", "recipient_age_txns": "mule",
    "device_users_count": "mule", "inbound_1h": "mule", "passthrough_ratio": "mule",
    "near_thr_24h": "struct", "agent_txn_1h": "agent", "agent_cashout_ratio": "agent",
    "is_gambling": "bet", "gambling_txn_24h": "bet", "log_gambling_spend_7d": "bet", "is_new_category": "cat",
    "otp_requests_10m": "otp", "otp_failures_10m": "otp", "otp_device_mismatch": "otp", "concurrent_sessions": "otp",
    "sim_swap_recent": "sim",
}
RULE_PREFIX_KEY = {"ATO": "dev", "SCAM": "scam", "MULE": "mule", "STRUCT": "struct", "AGENT": "agent",
                   "OTP": "otp", "GAMBLE": "bet"}
TAG_KEY = {"account_takeover": "dev", "scam_victim": "scam", "mule_passthrough": "mule", "mule_network": "mule",
           "structuring": "struct", "agent_risk": "agent", "otp_breach": "otp", "sim_swap": "sim",
           "gambling": "bet", "gambling_repeat": "bet"}

KEY_LABEL = {
    "dev": "New device", "loc": "Unfamiliar location", "rcp": "First-time recipient", "amt": "Amount far above usual",
    "hr": "Unusual hour", "drain": "Large share of balance moved", "vel": "Burst of transactions",
    "mule": "Mule-like money flow", "agent": "Unusual agent activity", "struct": "Just under the 50,000 BDT threshold",
    "scam": "Scam pattern", "otp": "OTP used from another device / session", "sim": "Recent SIM swap",
    "bet": "Betting / gambling payments", "cat": "New kind of spending", "other": "Unusual pattern",
}

# customer-facing sentences (never raw scores), same wording as Frontend/assets/js/i18n.js
CUSTOMER_TEXT = {
    "en": {
        "dev": "It is from a new device", "rcp": "You have never sent money to this number",
        "amt": "The amount is much higher than usual", "scam": "It matches known scam signs",
        "loc": "It is from an unfamiliar place", "hr": "It is at an unusual hour",
        "drain": "Your balance would be emptied quickly", "vel": "Many transactions in a short time",
        "mule": "The account is linked to suspicious activity", "agent": "The cash-out point is behaving unusually",
        "struct": "The amount is just under a reporting limit", "other": "It looks different from your usual activity",
        "otp": "Someone else may be using your one-time code (OTP). Never share it, even with the bank",
        "sim": "Your SIM card was changed recently", "bet": "This is a betting / gambling payment",
        "cat": "You have not spent money on this before",
    },
    "bn": {
        "dev": "এটি একটি নতুন ডিভাইস থেকে", "rcp": "আপনি আগে এই নম্বরে টাকা পাঠাননি",
        "amt": "পরিমাণ সাধারণের চেয়ে অনেক বেশি", "scam": "এটি পরিচিত প্রতারণার লক্ষণের সাথে মেলে",
        "loc": "এটি অপরিচিত স্থান থেকে", "hr": "এটি অস্বাভাবিক সময়ে",
        "drain": "আপনার ব্যালেন্স দ্রুত খালি হয়ে যেত", "vel": "অল্প সময়ে অনেকগুলো লেনদেন",
        "mule": "অ্যাকাউন্টটি সন্দেহজনক কার্যকলাপের সাথে যুক্ত", "agent": "ক্যাশ-আউট পয়েন্টের আচরণ অস্বাভাবিক",
        "struct": "পরিমাণটি নির্ধারিত সীমার ঠিক নিচে", "other": "এটি আপনার স্বাভাবিক লেনদেন থেকে আলাদা",
        "otp": "অন্য কেউ আপনার ওয়ান-টাইম কোড (OTP) ব্যবহার করছে হতে পারে। কাউকে OTP দেবেন না, ব্যাংককেও নয়",
        "sim": "আপনার সিম কার্ড সম্প্রতি পরিবর্তন করা হয়েছে", "bet": "এটি একটি বেটিং / জুয়ার পেমেন্ট",
        "cat": "আপনি আগে এই খাতে খরচ করেননি",
    },
}


def feature_key(feature: str) -> str:
    return FEATURE_KEY.get(feature, "other")


def rule_key(rule_id: str) -> str:
    return RULE_PREFIX_KEY.get(rule_id.split("_")[0], "other")


# ML action -> the decision vocabulary the frontend uses (allow / warn / step_up / hold / block)
def frontend_decision(action: str, tags: list[str] | None = None) -> str:
    if action == "allow":
        return "allow"
    if action == "otp_step_up":
        return "warn" if "scam_victim" in (tags or []) else "step_up"
    if action in ("block", "freeze_wallet"):
        return "block"
    return "hold"                                  # hold, escalate


PURPOSE_LABEL = {
    "betting": "Betting / gambling", "person_to_person": "Sending to a person", "cash_withdrawal": "Cash withdrawal",
    "cash_deposit": "Cash deposit", "payment": "Payment", "grocery": "Groceries", "utilities": "Utility bills",
    "telecom": "Mobile / telecom", "food": "Food", "transport": "Transport", "education": "Education",
    "health": "Health", "shopping": "Shopping",
}


def purpose_label(purpose: str | None) -> str:
    return PURPOSE_LABEL.get(purpose or "", (purpose or "Unknown").replace("_", " ").capitalize())
