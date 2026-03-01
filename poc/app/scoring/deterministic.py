from __future__ import annotations

from typing import Any, Callable, Protocol, cast

import numpy as np

from app.config import READINESS_THRESHOLD
from app.state import AnswerRecord, ChunkRecord, QuestionRecord


class QueryEmbedder(Protocol):
    def embed_query(self, text: str) -> list[float]: ...


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _as_vector(value: Any) -> np.ndarray:
    arr = np.array(value, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("Embedding vector must be a non-empty 1D array.")
    return arr


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _build_embed_fn(
    embeddings: QueryEmbedder | None,
    embed_fn: Callable[[str], list[float] | np.ndarray] | None,
) -> Callable[[str], np.ndarray] | None:
    if embed_fn is not None:
        return lambda text: _as_vector(embed_fn(text))
    if embeddings is not None:
        return lambda text: _as_vector(embeddings.embed_query(text))
    return None


def _citation_ids(answer: AnswerRecord) -> set[str]:
    ids: set[str] = set()
    for item in answer.get("used_citations", []):
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("chunk_id", "")).strip()
        if chunk_id:
            ids.add(chunk_id)
    return ids


def compute_score(
    question: QuestionRecord,
    answer: AnswerRecord,
    retrieved_chunks: list[ChunkRecord],
    embeddings: QueryEmbedder | None = None,
    embed_fn: Callable[[str], list[float] | np.ndarray] | None = None,
) -> float:
    """
    Deterministic score in [0, 1] using retrieval similarity, citations, risk bonus,
    and refusal penalties. Embeddings can be provided via `embeddings` or `embed_fn`.
    """
    resolved_embed = _build_embed_fn(embeddings=embeddings, embed_fn=embed_fn)

    retrieval_similarity = 0.0
    if resolved_embed is not None and retrieved_chunks:
        q_emb = resolved_embed(question["question"])
        chunk_embs = [resolved_embed(chunk["text"]) for chunk in retrieved_chunks if chunk.get("text", "").strip()]
        if chunk_embs:
            mean_chunk_emb = cast(np.ndarray, np.mean(np.vstack(chunk_embs), axis=0))
            cosine_sim = _cosine_similarity(q_emb, mean_chunk_emb)
            retrieval_similarity = _clamp(cosine_sim, -1.0, 1.0) * 0.5
            retrieval_similarity = max(0.0, retrieval_similarity)

    cited_ids = _citation_ids(answer)
    retrieved_ids = {chunk["chunk_id"] for chunk in retrieved_chunks if chunk.get("chunk_id")}
    citation_factor = (len(cited_ids & retrieved_ids) / max(len(retrieved_ids), 1)) * 0.3

    risk_bonus = 0.0
    if question.get("risk") == "high":
        caveats = ("caveat", "note", "warning", "consult", "disclaimer", "verify")
        if any(word in answer.get("answer", "").lower() for word in caveats):
            risk_bonus = 0.1

    refusal_phrases = ("i don't know", "not enough information", "cannot answer")
    penalties = 0.0
    if retrieved_chunks and any(phrase in answer.get("answer", "").lower() for phrase in refusal_phrases):
        penalties += 0.1

    raw = retrieval_similarity + citation_factor + risk_bonus - penalties
    return _clamp(raw)


def compute_readiness(
    answers_parent: list[AnswerRecord],
    threshold: float = READINESS_THRESHOLD,
) -> bool:
    if not answers_parent:
        return False
    return min(float(answer.get("score", 0.0)) for answer in answers_parent) >= float(threshold)


def find_failing_answers(
    answers_parent: list[AnswerRecord],
    threshold: float = READINESS_THRESHOLD,
) -> list[AnswerRecord]:
    return [answer for answer in answers_parent if float(answer.get("score", 0.0)) < float(threshold)]


_DASHBOARD_PROBLEM_WEIGHTS: dict[str, float] = {
    "missing_info": 1.35,
    "ambiguous": 1.00,
    "contradiction": 1.80,
    "formatting_issue": 0.60,
}
_READINESS_OK_RATIO_WEIGHT = 0.70
_READINESS_OK_CONF_WEIGHT = 0.30
_READINESS_WEIGHTED_PENALTY_MULTIPLIER = 1.15
_READINESS_PROBLEMATIC_RATIO_MULTIPLIER = 0.35


_ANSWER_PROBLEM_PENALTIES: dict[str, float] = {
    "ok": 0.00,
    "formatting_issue": 0.15,
    "ambiguous": 0.30,
    "missing_info": 0.40,
    "contradiction": 0.50,
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _readiness_label(score: float) -> str:
    if score >= 0.94:
        return "READY"
    if score >= 0.80:
        return "REVIEW"
    return "NOT OK"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "on"}


def compute_answer_quality_score(
    *,
    answer_confidence: float,
    classification_confidence: float | None,
    problem_type: str,
    evidence_count: int,
    context_count: int,
) -> float:
    """
    Deterministic per-answer score in [0, 1] used in the main dashboard.

    It combines:
    - answer confidence (base)
    - citation coverage bonus (evidence/context ratio)
    - problem-type penalty scaled by classification confidence
    """
    conf = _clamp(_safe_float(answer_confidence, 0.0))
    class_conf = _clamp(_safe_float(classification_confidence, conf))
    normalized_problem = str(problem_type or "").strip().lower()
    penalty_weight = _ANSWER_PROBLEM_PENALTIES.get(normalized_problem, 0.35)

    safe_context = max(1, int(context_count or 0))
    safe_evidence = max(0, int(evidence_count or 0))
    coverage = min(1.0, safe_evidence / safe_context)
    coverage_bonus = 0.10 * coverage

    penalty = penalty_weight * class_conf
    raw = conf + coverage_bonus - penalty
    return _clamp(raw)


def compute_dashboard_readiness(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Deterministic dashboard KPI (0..1) estimating documentation readiness.

    Formula (stricter):
      positive = 0.70 * ok_ratio + 0.30 * avg_ok_answer_confidence
      penalty  = 1.15 * weighted_problem_penalty + 0.35 * problematic_ratio
      readiness_score = clamp(positive - penalty)

    weighted_problem_penalty is normalized by answered rows and uses
    problem-type weights scaled by classification confidence.
    """
    # Reviewer-excluded rows are intentionally omitted from readiness.
    included_rows = [row for row in rows if not _as_bool(row.get("reviewer_excluded"))]
    answered_rows = [row for row in included_rows if row.get("answer_text")]
    answered_count = len(answered_rows)
    if answered_count == 0:
        return {
            "readiness_score": 0.0,
            "readiness_label": "NOT OK",
            "readiness_answered_count": 0,
            "readiness_ok_count": 0,
            "readiness_problematic_count": 0,
            "readiness_ok_ratio": 0.0,
            "avg_ok_answer_confidence": 0.0,
            "avg_answer_confidence": 0.0,
            "weighted_problem_penalty": 0.0,
            "readiness_hard_blocked": False,
            "readiness_hard_block_reason": None,
        }

    ok_rows = [row for row in answered_rows if str(row.get("problem_type") or "").strip().lower() == "ok"]
    problematic_rows = [row for row in answered_rows if str(row.get("problem_type") or "").strip().lower() != "ok"]
    contradiction_rows = [
        row for row in answered_rows if str(row.get("problem_type") or "").strip().lower() == "contradiction"
    ]
    ok_count = len(ok_rows)
    problematic_count = len(problematic_rows)
    contradiction_count = len(contradiction_rows)
    problematic_ratio = problematic_count / answered_count

    ok_ratio = ok_count / answered_count
    avg_ok_answer_confidence = (
        sum(_clamp(_safe_float(row.get("confidence"), 0.0)) for row in ok_rows) / ok_count if ok_count else 0.0
    )
    avg_answer_confidence = (
        sum(_clamp(_safe_float(row.get("confidence"), 0.0)) for row in answered_rows) / answered_count
    )

    penalty_sum = 0.0
    for row in problematic_rows:
        problem_type = str(row.get("problem_type") or "").strip().lower()
        weight = _DASHBOARD_PROBLEM_WEIGHTS.get(problem_type, 0.8)
        # If the column is missing/NULL (old runs), treat classification certainty as high enough to count.
        class_conf = _clamp(_safe_float(row.get("classification_confidence"), 0.8))
        penalty_sum += weight * max(class_conf, 0.5)
    weighted_problem_penalty = penalty_sum / answered_count

    positive = (_READINESS_OK_RATIO_WEIGHT * ok_ratio) + (_READINESS_OK_CONF_WEIGHT * avg_ok_answer_confidence)
    penalty = (
        _READINESS_WEIGHTED_PENALTY_MULTIPLIER * weighted_problem_penalty
        + (_READINESS_PROBLEMATIC_RATIO_MULTIPLIER * problematic_ratio)
    )
    readiness_score = _clamp(positive - penalty)
    hard_blocked = contradiction_count > 0
    hard_block_reason: str | None = None
    if hard_blocked:
        # Safety gate: contradictory source facts are not chatbot-ready.
        readiness_score = min(readiness_score, 0.64)
        hard_block_reason = "contradiction_detected"

    return {
        "readiness_score": readiness_score,
        "readiness_label": _readiness_label(readiness_score),
        "readiness_answered_count": answered_count,
        "readiness_ok_count": ok_count,
        "readiness_problematic_count": problematic_count,
        "readiness_ok_ratio": ok_ratio,
        "avg_ok_answer_confidence": avg_ok_answer_confidence,
        "avg_answer_confidence": avg_answer_confidence,
        "weighted_problem_penalty": weighted_problem_penalty,
        "readiness_hard_blocked": hard_blocked,
        "readiness_hard_block_reason": hard_block_reason,
    }
