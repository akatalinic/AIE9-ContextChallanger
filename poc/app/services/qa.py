import logging
import os
import re
from typing import Any

from app.models import PROBLEM_TYPES, QuestionResult
from .llm_client import get_client, get_model_name
from .llm_helpers import count_tokens, create_chat_completion_with_fallback, safe_json_parse

logger = logging.getLogger(__name__)

MAX_CONTEXT_TOKENS = int(os.getenv("QA_MAX_CONTEXT_TOKENS", "12000"))
QA_TIMEOUT = float(os.getenv("QA_TIMEOUT", "60"))
try:
    QA_TEMPERATURE = float(os.getenv("QA_TEMPERATURE", "0.0"))
except (TypeError, ValueError):
    QA_TEMPERATURE = 0.0
GLOSSARY_MAX_TERMS = int(os.getenv("QA_GLOSSARY_MAX_TERMS", "8"))
GLOSSARY_SNIPPETS_PER_TERM = int(os.getenv("QA_GLOSSARY_SNIPPETS_PER_TERM", "2"))

CODE_TOKEN_PATTERN = re.compile(r"\b[A-Z]{2,}\d{0,4}\b")
YES_NO_QUESTION_START_PATTERN = re.compile(
    r"^\s*(is|are|am|was|were|do|does|did|has|have|had|can|could|should|would|will|may|might|must)\b",
    re.IGNORECASE,
)
YES_NO_ANSWER_START_PATTERN = re.compile(r"^\s*(yes|no)\b", re.IGNORECASE)

# Markers that indicate a yes/no answer is hedged or not explicitly supported.
YES_NO_HEDGING_PATTERNS = [
    re.compile(r"\bnot\s+explicitly\s+stated\b", re.IGNORECASE),
    re.compile(r"\bnot\s+explicitly\s+specified\b", re.IGNORECASE),
    re.compile(r"\bnot\s+specified\b", re.IGNORECASE),
    re.compile(r"\bnot\s+clear\b", re.IGNORECASE),
    re.compile(r"\bunclear\b", re.IGNORECASE),
    re.compile(r"\bdepends\b", re.IGNORECASE),
    re.compile(r"\b(?:may|might|could)\s+(?:apply|require|vary|change|depend|be)\b", re.IGNORECASE),
    re.compile(r"\bsome\s+[^.]{0,60}\b(?:may|might|could)\b", re.IGNORECASE),
]

START_DATE_QUESTION_PATTERN = re.compile(
    r"\b(since\s+when|from\s+when|when\s+has|when\s+was|included\s+since|available\s+since)\b",
    re.IGNORECASE,
)
START_YEAR_PATTERN = re.compile(
    r"\b(?:since|included\s+since|available\s+since|introduced\s+since)\s+(?:\d{1,2}\s+[A-Za-z]+\s+)?(20\d{2})\b",
    re.IGNORECASE,
)
YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")
TIMELINE_EVENT_KEYWORDS = (
    "price",
    "invoice",
    "billing",
    "activation",
    "notification",
    "effective",
    "increase",
)
NO_FIX_NEEDED_PATTERN = re.compile(
    r"\b(no\s+fix\s+needed|no\s+change\s+needed|none|n/?a|not\s+needed)\b",
    re.IGNORECASE,
)
METHOD_OR_DEVICE_QUESTION_PATTERN = re.compile(
    r"\b(which|what)\s+(?:self[-\s]?service\s+)?(?:channel|channels|method|methods|way|ways|device|devices)\b",
    re.IGNORECASE,
)
EXHAUSTIVE_SCOPE_PATTERN = re.compile(
    r"\b(all|every|complete\s+list|full\s+list|exactly\s+which|specific\s+models?|model\s+numbers?)\b",
    re.IGNORECASE,
)
USAGE_PREPOSITION_PATTERN = re.compile(r"\b(via|through|using)\b", re.IGNORECASE)

SYSTEM_PROMPT = """You are a precise QA assistant for documentation validation.
Answer strictly based on the provided context.

Return ONLY a JSON object:
{
  "answer_text": "...",
  "answer_confidence": 0.0,
  "classification_confidence": 0.0,
  "problem_type": "ok|missing_info|contradiction|ambiguous|formatting_issue",
  "reasons": "...",
  "suggested_fix": "...",
  "evidence": [{"chunk_id": "...", "quote": "..."}],
  "flags": {
    "conflict": false,
    "ambiguous": false,
    "missing_info": false
  },
  "risk_assessment": "low|medium|high"
}

Rules:
- Use only provided context; do not use outside knowledge.
- Every factual claim in answer_text must be supported by at least one evidence quote.
- If two chunks provide different values for the same field (date, price, status, scope), classify as contradiction.
- For contradiction, cite BOTH conflicting chunks and explicitly name the conflicting values.
- Do not choose one side when conflict exists unless the context explicitly states supersession/replacement.
- If required detail is missing (exact date, price, scope, entity), classify as missing_info.
- If the answer is inferable from structured time expressions in context (range/season/period notation), derive it and answer explicitly.
- Do not classify as missing_info when the required value can be obtained by straightforward normalization of context facts.
- If context is unclear or partially scoped, classify as ambiguous.
- If question asks universal scope (all/always/every) but context is partial (some/may/might), classify as ambiguous.
- For definite yes/no questions, if support is hedged (e.g. may/might/could) or not explicitly stated, use ambiguous instead of ok.
- Map abbreviations/codes (e.g. MCD1, MCD24) to their meaning from context before concluding.
- If evidence is weak or absent, do not return problem_type=ok.
- If answer cannot be fully grounded, answer_text must start with "Not enough information in the provided context."
- answer_confidence = confidence in the answer content.
- classification_confidence = confidence that the chosen problem_type is correct.
- Evidence should include at least one chunk_id whenever possible.
- Write answer_text, reasons, and suggested_fix in English.
"""

USER_TEMPLATE = """QUESTION:
{question}

GLOSSARY OF CODES/ABBREVIATIONS (if available):
{glossary}

CONTEXT (each block includes chunk_id):
{context}

Return only a JSON object with no extra text.
"""


def answer_with_analysis(question: str, context_chunks: list[dict[str, str]]) -> QuestionResult:
    logger.info(
        "QA started | question_chars=%s context_chunk_count=%s",
        len(question),
        len(context_chunks),
    )

    client = get_client()
    deployment = get_model_name("OPENAI_QA_MODEL")
    context = _format_context(context_chunks)
    glossary = _build_glossary(question, context_chunks)

    request_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(question=question, glossary=glossary, context=context),
        },
    ]
    response = create_chat_completion_with_fallback(
        client=client,
        model=deployment,
        messages=request_messages,
        temperature=QA_TEMPERATURE,
        max_output_tokens=1800,
        timeout=QA_TIMEOUT,
        response_format={"type": "json_object"},
        log_label="QA",
    )

    content = (response.choices[0].message.content or "").strip()
    result = _parse_result(content, question, context_chunks)

    logger.info(
        "QA finished | answer_confidence=%.2f classification_confidence=%.2f problem_type=%s evidence_count=%s",
        result.answer_confidence,
        result.classification_confidence,
        result.problem_type,
        len(result.evidence),
    )
    return result


def _format_context(context_chunks: list[dict[str, str]]) -> str:
    parts = []
    total = 0
    for item in context_chunks:
        chunk_id = item.get("chunk_id", "")
        text = item.get("text", "")
        if not text:
            continue
        block = f"[chunk_id={chunk_id}]\n{text}"
        t = count_tokens(block)
        if total + t > MAX_CONTEXT_TOKENS:
            break
        parts.append(block)
        total += t
    return "\n\n---\n\n".join(parts)


def _extract_candidate_codes(question: str, context_chunks: list[dict[str, str]]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []

    def add_from_text(text: str) -> None:
        for token in CODE_TOKEN_PATTERN.findall(text or ""):
            normalized = token.strip().upper()
            # Prefer business-style codes (contains a digit) and avoid generic short words.
            if not normalized or (not any(ch.isdigit() for ch in normalized) and len(normalized) < 4):
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(normalized)

    add_from_text(question)
    for item in context_chunks:
        add_from_text(str(item.get("text", "")))

    return ordered[:GLOSSARY_MAX_TERMS]


def _context_snippets_for_token(token: str, context_chunks: list[dict[str, str]]) -> list[str]:
    snippets: list[str] = []
    for item in context_chunks:
        chunk_id = str(item.get("chunk_id", "")).strip()
        text = str(item.get("text", ""))
        if not chunk_id or not text or token not in text.upper():
            continue
        for line in [part.strip() for part in text.splitlines() if part.strip()]:
            if token not in line.upper():
                continue
            snippet = line
            if len(snippet) > 180:
                snippet = snippet[:177] + "..."
            snippets.append(f"[{chunk_id}] {snippet}")
            if len(snippets) >= GLOSSARY_SNIPPETS_PER_TERM:
                return snippets
    return snippets


def _build_glossary(question: str, context_chunks: list[dict[str, str]]) -> str:
    terms = _extract_candidate_codes(question, context_chunks)
    if not terms:
        return "(no recognized abbreviations/codes)"

    lines: list[str] = []
    for token in terms:
        snippets = _context_snippets_for_token(token, context_chunks)
        if snippets:
            lines.append(f"- {token}: " + " | ".join(snippets))
        else:
            lines.append(f"- {token}: (appears in the question/context, explicit definition not found)")
    return "\n".join(lines)


def _confidence_caps(problem_type: str) -> tuple[float, float]:
    # (max_answer_confidence, min_classification_confidence_hint)
    if problem_type == "missing_info":
        return (0.45, 0.60)
    if problem_type == "ambiguous":
        return (0.60, 0.55)
    if problem_type == "contradiction":
        return (0.65, 0.60)
    if problem_type == "formatting_issue":
        return (0.30, 0.70)
    return (1.00, 0.00)


def _apply_confidence_consistency(
    problem_type: str,
    answer_confidence: float,
    classification_confidence: float,
) -> tuple[float, float]:
    max_answer_conf, min_class_conf = _confidence_caps(problem_type)
    adjusted_answer = min(answer_confidence, max_answer_conf)
    adjusted_class = classification_confidence
    if problem_type != "ok" and classification_confidence < min_class_conf:
        adjusted_class = min_class_conf
    return (_normalize_confidence(adjusted_answer), _normalize_confidence(adjusted_class))


def _default_suggested_fix(problem_type: str) -> str:
    if problem_type == "contradiction":
        return "Resolve conflicting statements and keep one authoritative value for the same fact."
    if problem_type == "ambiguous":
        return "Clarify scope and conditions explicitly (who, when, and which cases apply)."
    if problem_type == "missing_info":
        return "Add the missing factual detail (exact date, value, scope, or entity) to the source text."
    if problem_type == "formatting_issue":
        return "Return valid structured JSON and include grounded evidence with chunk IDs."
    return "No fix needed."


def _normalize_suggested_fix(problem_type: str, suggested_fix: str) -> str:
    compact = " ".join(str(suggested_fix or "").split()).strip()
    if problem_type == "ok":
        return compact or "No fix needed."
    if not compact or NO_FIX_NEEDED_PATTERN.search(compact):
        return _default_suggested_fix(problem_type)
    return compact


def _parse_result(content: str, question: str, context_chunks: list[dict[str, str]]) -> QuestionResult:
    payload = safe_json_parse(content)
    if not isinstance(payload, dict):
        logger.warning("QA returned non-object JSON payload, using fallback result.")
        return _fallback_result()

    answer_text = str(payload.get("answer_text", "")).strip() or "No reliable answer can be determined from the provided context."
    raw_legacy_conf = payload.get("confidence", 0.0)
    answer_confidence = _normalize_confidence(payload.get("answer_confidence", raw_legacy_conf))

    raw_problem_type = str(payload.get("problem_type", "")).strip().lower()
    if raw_problem_type not in PROBLEM_TYPES:
        raw_problem_type = _infer_problem_type(payload)
    if raw_problem_type not in PROBLEM_TYPES:
        raw_problem_type = "missing_info"
    classification_confidence = _normalize_confidence(
        payload.get("classification_confidence", raw_legacy_conf)
    )

    reasons = str(payload.get("reasons", "")).strip()
    suggested_fix = str(payload.get("suggested_fix", "")).strip()

    evidence = _normalize_evidence(payload.get("evidence", []))
    if not evidence:
        evidence = _evidence_from_citations(payload.get("citations", []), context_chunks)

    (
        raw_problem_type,
        answer_confidence,
        classification_confidence,
        reasons,
        suggested_fix,
    ) = _apply_yes_no_uncertainty_guard(
        question=question,
        answer_text=answer_text,
        reasons=reasons,
        suggested_fix=suggested_fix,
        evidence=evidence,
        problem_type=raw_problem_type,
        answer_confidence=answer_confidence,
        classification_confidence=classification_confidence,
    )
    (
        raw_problem_type,
        answer_confidence,
        classification_confidence,
        reasons,
        suggested_fix,
    ) = _apply_timeline_consistency_guard(
        question=question,
        answer_text=answer_text,
        reasons=reasons,
        suggested_fix=suggested_fix,
        evidence=evidence,
        context_chunks=context_chunks,
        problem_type=raw_problem_type,
        answer_confidence=answer_confidence,
        classification_confidence=classification_confidence,
    )
    (
        raw_problem_type,
        answer_confidence,
        classification_confidence,
        reasons,
        suggested_fix,
    ) = _apply_method_or_device_presence_guard(
        question=question,
        answer_text=answer_text,
        reasons=reasons,
        suggested_fix=suggested_fix,
        evidence=evidence,
        context_chunks=context_chunks,
        problem_type=raw_problem_type,
        answer_confidence=answer_confidence,
        classification_confidence=classification_confidence,
    )
    answer_confidence, classification_confidence = _apply_confidence_consistency(
        problem_type=raw_problem_type,
        answer_confidence=answer_confidence,
        classification_confidence=classification_confidence,
    )
    suggested_fix = _normalize_suggested_fix(raw_problem_type, suggested_fix)

    return QuestionResult(
        answer_text=answer_text,
        answer_confidence=answer_confidence,
        classification_confidence=classification_confidence,
        problem_type=raw_problem_type,
        reasons=reasons,
        suggested_fix=suggested_fix,
        evidence=evidence,
    )


def _is_definite_yes_no_question(question: str) -> bool:
    return bool(YES_NO_QUESTION_START_PATTERN.search(str(question or "")))


def _is_explicit_yes_no_answer(answer_text: str) -> bool:
    return bool(YES_NO_ANSWER_START_PATTERN.search(str(answer_text or "")))


def _has_yes_no_hedged_support(
    answer_text: str,
    reasons: str,
    evidence: list[dict[str, str]],
) -> bool:
    evidence_quotes = " ".join(str(item.get("quote", "")) for item in evidence if isinstance(item, dict))
    text = " ".join(
        [
            str(answer_text or ""),
            str(reasons or ""),
            evidence_quotes,
        ]
    )
    for pattern in YES_NO_HEDGING_PATTERNS:
        if pattern.search(text):
            return True
    return False


def _apply_yes_no_uncertainty_guard(
    *,
    question: str,
    answer_text: str,
    reasons: str,
    suggested_fix: str,
    evidence: list[dict[str, str]],
    problem_type: str,
    answer_confidence: float,
    classification_confidence: float,
) -> tuple[str, float, float, str, str]:
    if problem_type != "ok":
        return (problem_type, answer_confidence, classification_confidence, reasons, suggested_fix)
    if not _is_definite_yes_no_question(question):
        return (problem_type, answer_confidence, classification_confidence, reasons, suggested_fix)
    if not _is_explicit_yes_no_answer(answer_text):
        return (problem_type, answer_confidence, classification_confidence, reasons, suggested_fix)
    if not _has_yes_no_hedged_support(answer_text, reasons, evidence):
        return (problem_type, answer_confidence, classification_confidence, reasons, suggested_fix)

    guard_reason = (
        "Deterministic QA guard: definite yes/no answer uses hedged or not-explicit support "
        "(e.g. may/might/could/not specified), so it was downgraded to ambiguous."
    )
    updated_reasons = reasons
    if guard_reason not in updated_reasons:
        updated_reasons = f"{updated_reasons} {guard_reason}".strip()
    updated_fix = suggested_fix or (
        "Make the requirement explicit in the source (clear yes/no wording and exact scope/conditions)."
    )
    updated_problem_type = "ambiguous"
    updated_classification_confidence = max(classification_confidence, 0.70)
    logger.info(
        "QA deterministic guard applied | question_is_yes_no=true downgraded_to=ambiguous answer_confidence=%.2f classification_confidence=%.2f",
        answer_confidence,
        updated_classification_confidence,
    )
    return (
        updated_problem_type,
        answer_confidence,
        updated_classification_confidence,
        updated_reasons,
        updated_fix,
    )


def _is_start_date_question(question: str) -> bool:
    return bool(START_DATE_QUESTION_PATTERN.search(str(question or "")))


def _extract_start_years(text: str) -> list[int]:
    years: list[int] = []
    for match in START_YEAR_PATTERN.findall(str(text or "")):
        try:
            years.append(int(match))
        except (TypeError, ValueError):
            continue
    return years


def _extract_event_years(text: str) -> list[int]:
    years: list[int] = []
    content = str(text or "")
    # Sentence-like splits are enough for deterministic heuristics.
    units = [part.strip() for part in re.split(r"[.\n]+", content) if part.strip()]
    for unit in units:
        lowered = unit.lower()
        if not any(keyword in lowered for keyword in TIMELINE_EVENT_KEYWORDS):
            continue
        for match in YEAR_PATTERN.findall(unit):
            try:
                years.append(int(match))
            except (TypeError, ValueError):
                continue
    return years


def _apply_timeline_consistency_guard(
    *,
    question: str,
    answer_text: str,
    reasons: str,
    suggested_fix: str,
    evidence: list[dict[str, str]],
    context_chunks: list[dict[str, str]],
    problem_type: str,
    answer_confidence: float,
    classification_confidence: float,
) -> tuple[str, float, float, str, str]:
    if problem_type == "contradiction":
        return (problem_type, answer_confidence, classification_confidence, reasons, suggested_fix)
    if not _is_start_date_question(question):
        return (problem_type, answer_confidence, classification_confidence, reasons, suggested_fix)

    evidence_quotes = " ".join(str(item.get("quote", "")) for item in evidence if isinstance(item, dict))
    context_text = " ".join(str(item.get("text", "")) for item in context_chunks if isinstance(item, dict))
    combined = " ".join([str(answer_text or ""), str(reasons or ""), evidence_quotes, context_text])

    start_years = _extract_start_years(combined)
    if not start_years:
        return (problem_type, answer_confidence, classification_confidence, reasons, suggested_fix)

    event_years = _extract_event_years(combined)
    unique_start_years = sorted(set(start_years))
    has_multiple_start_years = len(unique_start_years) > 1
    has_event_before_start = bool(event_years) and min(event_years) < min(unique_start_years)
    if not has_multiple_start_years and not has_event_before_start:
        return (problem_type, answer_confidence, classification_confidence, reasons, suggested_fix)

    guard_reason = (
        "Deterministic timeline guard: inconsistent chronology detected for start date "
        f"(start_years={unique_start_years}, event_years={sorted(set(event_years))})."
    )
    updated_reasons = reasons
    if guard_reason not in updated_reasons:
        updated_reasons = f"{updated_reasons} {guard_reason}".strip()
    updated_fix = suggested_fix or (
        "Align chronology in source text (start date, effective dates, and invoice/activation timelines)."
    )
    updated_problem_type = "contradiction"
    updated_classification_confidence = max(classification_confidence, 0.75)
    logger.info(
        "QA timeline guard applied | downgraded_to=contradiction answer_confidence=%.2f classification_confidence=%.2f start_years=%s event_years=%s",
        answer_confidence,
        updated_classification_confidence,
        unique_start_years,
        sorted(set(event_years)),
    )
    return (
        updated_problem_type,
        answer_confidence,
        updated_classification_confidence,
        updated_reasons,
        updated_fix,
    )


def _normalize_text_for_contains(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _extract_usage_target(question: str) -> str:
    text = str(question or "")
    match = re.search(r"\b(?:via|through|using)\s+([^?.;,:]+)", text, re.IGNORECASE)
    if not match:
        return ""
    target = _normalize_text_for_contains(match.group(1))
    return target if len(target) >= 4 else ""


def _apply_method_or_device_presence_guard(
    *,
    question: str,
    answer_text: str,
    reasons: str,
    suggested_fix: str,
    evidence: list[dict[str, str]],
    context_chunks: list[dict[str, str]],
    problem_type: str,
    answer_confidence: float,
    classification_confidence: float,
) -> tuple[str, float, float, str, str]:
    if problem_type not in {"missing_info", "ambiguous"}:
        return (problem_type, answer_confidence, classification_confidence, reasons, suggested_fix)
    question_text = str(question or "")
    if not METHOD_OR_DEVICE_QUESTION_PATTERN.search(question_text):
        return (problem_type, answer_confidence, classification_confidence, reasons, suggested_fix)
    if EXHAUSTIVE_SCOPE_PATTERN.search(question_text):
        return (problem_type, answer_confidence, classification_confidence, reasons, suggested_fix)
    if not USAGE_PREPOSITION_PATTERN.search(question_text):
        return (problem_type, answer_confidence, classification_confidence, reasons, suggested_fix)

    usage_target = _extract_usage_target(question_text)
    if not usage_target:
        return (problem_type, answer_confidence, classification_confidence, reasons, suggested_fix)

    evidence_quotes = " ".join(str(item.get("quote", "")) for item in evidence if isinstance(item, dict))
    context_text = " ".join(str(item.get("text", "")) for item in context_chunks if isinstance(item, dict))
    combined = _normalize_text_for_contains(" ".join([answer_text, reasons, evidence_quotes, context_text]))
    if usage_target not in combined:
        return (problem_type, answer_confidence, classification_confidence, reasons, suggested_fix)

    guard_reason = (
        "Deterministic QA guard: context explicitly contains the asked usage target "
        f"('{usage_target}'), so classification changed to ok."
    )
    updated_reasons = reasons
    if guard_reason not in updated_reasons:
        updated_reasons = f"{updated_reasons} {guard_reason}".strip()

    return (
        "ok",
        max(answer_confidence, 0.65),
        max(classification_confidence, 0.70),
        updated_reasons,
        "No fix needed.",
    )


def _normalize_confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _infer_problem_type(payload: dict[str, Any]) -> str:
    flags = payload.get("flags", {})
    risk = str(payload.get("risk_assessment", "")).strip().lower()

    if isinstance(flags, dict):
        if bool(flags.get("conflict")):
            return "contradiction"
        if bool(flags.get("ambiguous")):
            return "ambiguous"
        if bool(flags.get("missing_info")):
            return "missing_info"

    if risk == "high":
        return "missing_info"
    if risk == "medium":
        return "ambiguous"
    # Conservative fallback: if model does not provide a valid label/flags,
    # avoid silently marking answers as "ok".
    return "missing_info"


def _normalize_evidence(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []

    normalized = []
    for item in value:
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("chunk_id", "")).strip()
        quote = str(item.get("quote", "")).strip()
        if chunk_id and quote:
            normalized.append({"chunk_id": chunk_id, "quote": quote[:240]})
    return normalized


def _evidence_from_citations(citations: Any, context_chunks: list[dict[str, str]]) -> list[dict[str, str]]:
    if not isinstance(citations, list):
        return []

    by_id = {item.get("chunk_id", ""): item.get("text", "") for item in context_chunks}
    result = []
    for cid in citations:
        chunk_id = str(cid).strip()
        if not chunk_id:
            continue
        text = by_id.get(chunk_id, "")
        quote = (text[:220] + "...") if len(text) > 220 else text
        if quote:
            result.append({"chunk_id": chunk_id, "quote": quote})
    return result


def _fallback_result() -> QuestionResult:
    return QuestionResult(
        answer_text="Failed to parse the model response.",
        answer_confidence=0.0,
        classification_confidence=0.95,
        problem_type="formatting_issue",
        reasons="The model did not return valid JSON.",
        suggested_fix="Retry the request or adjust the prompt.",
        evidence=[],
    )
