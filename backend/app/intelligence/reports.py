"""Compliance report exporter: Markdown and a real PDF (reportlab)."""
from __future__ import annotations

import io

from backend.app.contracts.interfaces import ReportExporter
from backend.app.contracts.schemas import Case
from backend.app.contracts.schemas_intel import EvidenceBundle, Narrative


class MarkdownReportExporter(ReportExporter):
    def to_markdown(self, case: Case, bundle: EvidenceBundle, narrative: Narrative) -> str:
        wh, wn, wr = case.scored.what_happened, case.scored.what_next, case.scored.why_risky
        L = [
            f"# UpayShield Investigation Report: {case.case_id}",
            f"**Generated:** {bundle.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"**Alert type:** {case.alert_type.value if case.alert_type else 'anomaly'}",
            f"**Risk level:** {wn.risk_level.value.upper()} (model score {wr.risk_score:.2f})",
            f"**Recommended action:** {wn.action.value.upper()}  (model alone: {wn.base_action.value})", "",
            "## 1. What happened?",
            f"- **Transaction:** {wh.type} of ৳{wh.amount_bdt:,.2f}",
            f"- **Purpose of the money:** {(wh.purpose or 'unknown').replace('_', ' ')}"
            + (f" ({wh.merchant_category})" if wh.merchant_category else ""),
            f"- **Sender:** {case.scored.user_id}", f"- **Recipient:** {wh.recipient}",
            f"- **Device / location:** {wh.device} ({wh.location})", f"- **Time:** {wh.time}", "",
            "## 2. Why is it risky?",
            f"- **Model score:** {wr.model_score:.4f}   **Anomaly score:** {wr.anomaly_score:.4f}",
        ]
        if wr.rule_trace:
            L.append("### Rules that fired")
            L += [f"- `[{r.rule_id}]` {r.text}" for r in wr.rule_trace]
        pos = [f for f in wr.reasons if f.shap > 0]
        if pos:
            L.append("### Model factors that raised the risk")
            L += [f"- {f.text} (contribution {f.shap:+.2f})" for f in pos]
        if wr.graph and wr.graph.ring_id:
            L.append(f"### Network\n- Wallet belongs to mule ring **{wr.graph.ring_id}** (ring score {wr.graph.ring_score:.2f})")
        L += ["", "## 3. What should be done next?", f"1. **{wn.action.value.upper()}** (priority ৳{wn.priority:,.0f})."]
        L += [f"   - Policy `{p.policy_id}`: {p.text}" for p in wn.policy_trace]
        L += ["", "## 4. Grounded narrative"]
        for title, sents in (("Facts", narrative.what_happened), ("Risk", narrative.why_risky), ("Next steps", narrative.what_next)):
            L.append(f"### {title}")
            L += [f"> {s.text} *[{', '.join(s.evidence_ids)}]*" for s in sents]
        L += ["", f"_Narrative source: {narrative.source}; evidence items: {len(bundle.items)}._"]
        return "\n".join(L) + "\n"

    def to_pdf(self, case: Case, bundle: EvidenceBundle, narrative: Narrative) -> bytes:
        from xml.sax.saxutils import escape

        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

        styles = getSampleStyleSheet()
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, title=f"UpayShield {case.case_id}")
        story = []
        for line in self.to_markdown(case, bundle, narrative).splitlines():
            if not line.strip():
                story.append(Spacer(1, 6))
                continue
            text = escape(line.replace("**", "").replace("`", "").replace("*", "").replace("_", " "))
            # the built-in PDF fonts have no Bangla / ৳ glyphs: keep the report readable with ASCII currency
            text = text.replace("৳", "BDT ")
            text = text.encode("latin-1", "replace").decode("latin-1")
            style = styles["Heading1"] if line.startswith("# ") else styles["Heading2"] if line.startswith("## ") else \
                styles["Heading3"] if line.startswith("### ") else styles["BodyText"]
            story.append(Paragraph(text.lstrip("#> ").strip(), style))
        doc.build(story)
        return buf.getvalue()
