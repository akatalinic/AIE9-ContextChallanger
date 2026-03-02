from dataclasses import dataclass
from typing import Any


ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

PROBLEM_TYPES = {"ok", "missing_info", "contradiction", "ambiguous", "formatting_issue"}


@dataclass
class QuestionResult:
    answer_text: str
    answer_confidence: float
    classification_confidence: float
    problem_type: str
    reasons: str
    suggested_fix: str
    evidence: list[dict[str, Any]]

    @property
    def confidence(self) -> float:
        return self.answer_confidence
