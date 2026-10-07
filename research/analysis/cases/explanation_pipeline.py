"""
Explanation-generation layer for the loan-dispute prediction pipeline.

Takes the structured extraction record + model prediction for a case,
builds an LLM prompt grounded ONLY in fields that were actually extracted
(nothing invented), and runs a cheap consistency check between the
generated explanation and the predicted label.

LLM-agnostic: plug in Claude / OpenAI / a local model by implementing
`llm_call(prompt: str) -> str` and passing it in.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class CaseRecord:
    file: str
    case_id: Optional[str] = None
    court: Optional[str] = None
    judge: Optional[str] = None
    parties: Optional[str] = None
    claim_amount: Optional[str] = None
    narrative_present: bool = False
    dispute_type: Optional[str] = None
    law_articles: Optional[str] = None
    decision_date: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class Prediction:
    label: Optional[str] = None          # e.g. "claim_upheld" / "claim_rejected"
    confidence: Optional[float] = None
    skipped: bool = False
    skip_reason: Optional[str] = None


FIELD_LABELS: Dict[str, str] = {
    "court": "Court",
    "judge": "Judge",
    "parties": "Parties",
    "claim_amount": "Claim amount",
    "dispute_type": "Dispute type",
    "law_articles": "Law articles cited",
    "decision_date": "Decision date",
}


def _known_facts(case: CaseRecord) -> List[str]:
    """Only fields that were actually extracted -- nothing invented later."""
    facts = []
    for attr, label in FIELD_LABELS.items():
        value = getattr(case, attr)
        if value:
            facts.append(f"{label}: {value}")
    return facts


def build_explanation_prompt(case: CaseRecord, prediction: Prediction, language: str = "en") -> str:
    facts = _known_facts(case)
    facts_block = "\n".join(f"- {f}" for f in facts) if facts else "(no structured fields were reliably extracted)"

    reliability_note = ""
    if not case.narrative_present:
        reliability_note = (
            "Note: the case narrative text was not available, so base the explanation "
            "only on the fields below and say so explicitly.\n"
        )

    lang_name = "Mongolian" if language == "mn" else "English"

    return f"""You are writing a short explanation for a legal-outcome prediction, for a non-technical reader.

Predicted outcome: {prediction.label}
Confidence: {prediction.confidence}

Known facts about the case (use ONLY these -- do not invent amounts, articles, dates, or parties not listed):
{facts_block}

{reliability_note}Write 2-4 sentences in plain {lang_name}, referencing 2-3 of the facts above, explaining why the
model likely reached this prediction. Do not state anything not listed above. End with one sentence
naming the single weakest point in this evidence."""


def build_consistency_check_prompt(explanation: str, prediction: Prediction) -> str:
    return f"""Predicted label: {prediction.label}

Explanation:
\"\"\"{explanation}\"\"\"

Does the explanation's stated conclusion match the predicted label above?
Answer with exactly one word: MATCH or MISMATCH."""


def generate_explanation(
    case: CaseRecord,
    prediction: Prediction,
    llm_call: Callable[[str], str],
    language: str = "en",
) -> dict:
    if prediction.skipped:
        return {
            "explanation": None,
            "consistency": None,
            "status": "skipped",
            "reason": prediction.skip_reason,
        }

    prompt = build_explanation_prompt(case, prediction, language)
    explanation = llm_call(prompt).strip()

    check_prompt = build_consistency_check_prompt(explanation, prediction)
    check = llm_call(check_prompt).strip().upper()

    return {
        "explanation": explanation,
        "consistency": "match" if "MATCH" in check and "MISMATCH" not in check else "mismatch",
        "status": "ok",
        "reason": None,
    }


if __name__ == "__main__":
    # The exact skipped case from your pipeline output, as a smoke test.
    case = CaseRecord(
        file="139_\u0428\u04282020_00529.txt",
        parties="individual vs individual",
        narrative_present=False,
        warnings=[
            "meta_court_missing", "meta_judge_missing", "meta_case_index_missing",
            "meta_case_id_missing", "meta_decision_date_missing", "meta_dispute_type_missing",
            "case_level_no_court_field", "outcome_no_point1",
            "claim_amount_requested_not_found", "law_articles_lead_in_not_found",
        ],
    )
    prediction = Prediction(
        skipped=True,
        skip_reason="dispute_type could not be extracted - can't confirm this is even a loan dispute",
    )

    result = generate_explanation(case, prediction, llm_call=lambda p: "")
    print(result)
    # Expect: status "skipped", no LLM ever called -- matches the pipeline's own behavior.
