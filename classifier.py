"""Small, transparent query classifier used by the LangGraph router."""

from __future__ import annotations

import re
from typing import Literal


Category = Literal["academic", "fee", "general"]

ACADEMIC_TERMS = (
    "academic",
    "attendance",
    "backlog",
    "cgpa",
    "course",
    "credit",
    "exam",
    "examination",
    "grade",
    "internship",
    "prerequisite",
    "project",
    "registration",
    "semester",
    "subject",
    "syllabus",
)

FEE_TERMS = (
    "dues",
    "fee",
    "fine",
    "installment",
    "late payment",
    "payment",
    "receipt",
    "refund",
    "scholarship",
    "tuition",
)


def _term_hits(text: str, terms: tuple[str, ...]) -> int:
    """Count whole-word or whole-phrase matches."""

    return sum(bool(re.search(rf"\b{re.escape(term)}\b", text)) for term in terms)


def classify_query(query: str) -> Category:
    """Classify a question using easy-to-explain keyword evidence.

    A tie goes to the academic route because examination and registration
    questions often contain both academic and payment vocabulary.
    """

    text = " ".join(query.lower().split())
    academic_hits = _term_hits(text, ACADEMIC_TERMS)
    fee_hits = _term_hits(text, FEE_TERMS)

    if fee_hits > academic_hits:
        return "fee"
    if academic_hits:
        return "academic"
    return "general"

