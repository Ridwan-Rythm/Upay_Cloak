"""Grounded investigation assistant.

Two modes, same output contract:
  1. LLM (Gemini / Anthropic / OpenAI, chosen by UPAY_LLM_PROVIDER). The model is shown ONLY the evidence items and must
     cite their ids. Every sentence is validated: unknown or missing citations -> the whole answer is rejected.
  2. Deterministic template (default, zero latency, always valid): composes sentences straight from the evidence.
The LLM path never raises: any failure falls back to the template and records `fallback_reason`.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

from backend.app.config import get_settings
from backend.app.contracts.evidence_ids import extract_ids
from backend.app.contracts.interfaces import InvestigationAssistant
from backend.app.contracts.schemas import Language
from backend.app.contracts.schemas_intel import Answer, EvidenceBundle, EvidenceItem, Narrative, NarrativeSentence

log = logging.getLogger("upayshield.assistant")

DEFAULT_MODELS = {"gemini": "gemini-2.0-flash", "anthropic": "claude-haiku-4-5-20251001", "openai": "gpt-4o-mini"}

ACTION_TEXT = {
    "en": {"allow": "allow the transaction", "otp_step_up": "ask the customer for step-up verification",
           "hold": "hold the transaction for analyst review", "escalate": "escalate to a senior analyst",
           "block": "block the transaction and contact the customer", "freeze_wallet": "freeze the wallet and escalate the network"},
    "bn": {"allow": "লেনদেনটি অনুমোদন করুন", "otp_step_up": "গ্রাহকের কাছ থেকে অতিরিক্ত যাচাই নিন",
           "hold": "বিশ্লেষকের পর্যালোচনার জন্য লেনদেনটি স্থগিত রাখুন", "escalate": "জ্যেষ্ঠ বিশ্লেষকের কাছে পাঠান",
           "block": "লেনদেনটি বন্ধ করুন এবং গ্রাহকের সাথে যোগাযোগ করুন", "freeze_wallet": "ওয়ালেট ফ্রিজ করুন এবং নেটওয়ার্কটি এস্কেলেট করুন"},
}
T = {
    "en": {"happened": "{label}.", "why": "It was flagged because: {reasons}.", "why_none": "The risk engine found behaviour that deviates from the norm.",
           "next": "Recommended action: {action}.", "purpose": "The money is being used for: {purpose}.",
           "ring": "The wallet belongs to a detected mule ring.", "unknown": "The evidence does not say.",
           "summary": "Case {case}: recommended action is to {action}."},
    "bn": {"happened": "{label}।", "why": "এটি চিহ্নিত হয়েছে কারণ: {reasons}।", "why_none": "ঝুঁকি ইঞ্জিন স্বাভাবিকের চেয়ে ভিন্ন আচরণ পেয়েছে।",
           "next": "প্রস্তাবিত পদক্ষেপ: {action}।", "purpose": "টাকাটি ব্যবহার হচ্ছে: {purpose}।",
           "ring": "ওয়ালেটটি একটি শনাক্তকৃত মিউল রিংয়ের অংশ।", "unknown": "প্রমাণে এ বিষয়ে কিছু বলা নেই।",
           "summary": "কেস {case}: প্রস্তাবিত পদক্ষেপ হলো {action}।"},
}


def _by_kind(bundle: EvidenceBundle, ids: list[str], *kinds: str) -> list[EvidenceItem]:
    idx = bundle.index()
    return [idx[i] for i in ids if i in idx and idx[i].kind.value in kinds]


class GroundedInvestigationAssistant(InvestigationAssistant):
    def __init__(self) -> None:
        self.settings = get_settings()

    # ------------------------------------------------------------------ template composition
    def _facts(self, bundle: EvidenceBundle, lang: str):
        t = T[lang]
        what = _by_kind(bundle, bundle.what_happened, "TXN")
        happened = [NarrativeSentence(text=t["happened"].format(label=what[0].label), evidence_ids=[what[0].id])] if what else []
        if what and what[0].facts.get("purpose"):
            happened.append(NarrativeSentence(
                text=t["purpose"].format(purpose=str(what[0].facts["purpose"]).replace("_", " ")), evidence_ids=[what[0].id]))
        if not happened:
            happened = [NarrativeSentence(text=t["happened"].format(label=f"Case {bundle.case_id}"), evidence_ids=bundle.what_happened[:1] or list(bundle.ids())[:1])]
        reasons = _by_kind(bundle, bundle.why_risky, "RULE", "FACTOR", "RING")
        why = []
        if reasons:
            top = reasons[:4]
            why.append(NarrativeSentence(text=t["why"].format(reasons="; ".join(r.label for r in top)), evidence_ids=[r.id for r in top]))
        else:
            why.append(NarrativeSentence(text=t["why_none"], evidence_ids=bundle.why_risky[:1] or list(bundle.ids())[:1]))
        pol = bundle.what_next[:1] or list(bundle.ids())[:1]
        nxt = [NarrativeSentence(text=t["next"].format(action=ACTION_TEXT[lang][bundle.recommended_action.value]), evidence_ids=pol)]
        return happened, why, nxt

    def narrate(self, bundle: EvidenceBundle, language: Language = "en", force_template: bool = False) -> Narrative:
        happened, why, nxt = self._facts(bundle, language)
        return Narrative(case_id=bundle.case_id, language=language, source="template", validated=True,
                         what_happened=happened, why_risky=why, what_next=nxt, recommended_action=bundle.recommended_action)

    # ------------------------------------------------------------------ Q&A
    def ask(self, bundle: EvidenceBundle, question: str, language: Language = "en") -> Answer:
        fallback_reason = None
        if self.settings.llm_provider != "none" and self.settings.llm_api_key:
            ans, fallback_reason = self._ask_llm(bundle, question, language)
            if ans:
                return ans
        ans = self._ask_template(bundle, question, language)
        ans.fallback_reason = fallback_reason
        return ans

    def _ask_template(self, bundle: EvidenceBundle, question: str, lang: str) -> Answer:
        q, t = question.lower(), T[lang]
        happened, why, nxt = self._facts(bundle, lang)
        idx = bundle.index()
        sentences: list[NarrativeSentence]
        if any(w in q for w in ("purpose", "use the money", "spend", "bet", "gambl", "what for", "what is the money")):
            sentences = [s for s in happened if "money" in s.text.lower() or "টাকা" in s.text] or happened
        elif any(w in q for w in ("otp", "session", "sim", "password", "pin", "breach")):
            items = [i for i in idx.values() if i.kind.value in ("RULE", "FACTOR") and
                     any(k in (i.id + i.label).lower() for k in ("otp", "sim", "session"))]
            sentences = [NarrativeSentence(text=i.label + ".", evidence_ids=[i.id]) for i in items[:4]] or \
                [NarrativeSentence(text=t["unknown"], evidence_ids=list(bundle.ids())[:1])]
        elif any(w in q for w in ("who", "connect", "link", "network", "ring", "wallet")):
            w = [i for i in idx.values() if i.kind.value in ("WALLET", "RING", "DEVICE", "AGENT")]
            sentences = [NarrativeSentence(text=("; ".join(i.label for i in w[:5]) + "."), evidence_ids=[i.id for i in w[:5]])] if w else \
                [NarrativeSentence(text=t["unknown"], evidence_ids=list(bundle.ids())[:1])]
        elif any(w in q for w in ("summar", "compliance", "report", "overview")):
            sentences = happened + why + nxt
        elif any(w in q for w in ("why", "flag", "risk", "reason", "score", "suspicio")):
            sentences = why + nxt
        elif any(w in q for w in ("next", "do", "action", "recommend", "should")):
            sentences = nxt
        else:
            sentences = happened + why
        return Answer(case_id=bundle.case_id, question=question, language=lang, source="template", validated=True,
                      answerable=not any(s.text == t["unknown"] for s in sentences), sentences=sentences)

    # ------------------------------------------------------------------ LLM path (validated, never raises)
    def _ask_llm(self, bundle: EvidenceBundle, question: str, lang: str) -> tuple[Answer | None, str | None]:
        facts = [{"id": i.id, "kind": i.kind.value, "label": i.label, "facts": i.facts} for i in bundle.items]
        prompt = (
            "You are the fraud-investigation assistant of a mobile-money platform.\n"
            "Answer ONLY from the evidence items below. Put the evidence id(s) in square brackets after EVERY sentence, "
            "e.g. [TXN-0001234]. If the evidence does not contain the answer, say so. Never invent numbers.\n"
            f"Write one short sentence per line, in language code '{lang}'.\n\nQuestion: {question}\n\n"
            f"Evidence (JSON):\n{json.dumps(facts, ensure_ascii=False)}")
        try:
            text = self._complete(prompt)
        except Exception as e:                                                      # noqa: BLE001
            log.warning("LLM call failed, using template: %s", e)
            return None, f"llm_error: {type(e).__name__}"
        valid_ids = bundle.ids()
        sentences = []
        for line in (x.strip(" -•\t") for x in text.splitlines()):
            if not line:
                continue
            cited = extract_ids(line)
            if not cited or any(c not in valid_ids for c in cited):
                return None, "llm_ungrounded: a sentence cited no evidence or an unknown id"
            sentences.append(NarrativeSentence(text=line, evidence_ids=cited))
        if not sentences:
            return None, "llm_empty"
        return Answer(case_id=bundle.case_id, question=question, language=lang, source="llm", validated=True,
                      sentences=sentences), None

    def _complete(self, prompt: str) -> str:
        s = self.settings
        provider, key = s.llm_provider, s.llm_api_key
        model = s.llm_model or DEFAULT_MODELS[provider]

        def post(url: str, body: dict, headers: dict) -> dict:
            req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **headers})
            with urllib.request.urlopen(req, timeout=s.llm_timeout_s) as resp:           # noqa: S310 (fixed https endpoints)
                return json.loads(resp.read().decode())

        if provider == "gemini":
            d = post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                     {"contents": [{"parts": [{"text": prompt}]}]}, {"x-goog-api-key": key})
            return d["candidates"][0]["content"]["parts"][0]["text"].strip()
        if provider == "anthropic":
            d = post("https://api.anthropic.com/v1/messages",
                     {"model": model, "max_tokens": 500, "messages": [{"role": "user", "content": prompt}]},
                     {"x-api-key": key, "anthropic-version": "2023-06-01"})
            return "".join(b.get("text", "") for b in d["content"]).strip()
        if provider == "openai":
            d = post("https://api.openai.com/v1/chat/completions",
                     {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 500},
                     {"Authorization": f"Bearer {key}"})
            return d["choices"][0]["message"]["content"].strip()
        raise ValueError(f"unknown provider {provider}")
