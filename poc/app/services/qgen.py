import logging
import os
import re
from typing import Any

from .llm_client import get_client, get_model_name
from .llm_helpers import count_tokens, create_chat_completion_with_fallback, safe_json_parse

logger = logging.getLogger(__name__)

MAX_CONTEXT_TOKENS = int(os.getenv("QGEN_MAX_CONTEXT_TOKENS", "6000"))
MAX_QUESTIONS_PER_DOC = int(os.getenv("QGEN_MAX_QUESTIONS_PER_DOC", "10"))
MIN_QUESTIONS_PER_DOC = int(os.getenv("QGEN_MIN_QUESTIONS_PER_DOC", "3"))
GROUNDING_MIN_OVERLAP_TOKENS = int(os.getenv("QGEN_GROUNDING_MIN_OVERLAP_TOKENS", "2"))
QGEN_STRICT_ANSWERABLE_ONLY = str(os.getenv("QGEN_STRICT_ANSWERABLE_ONLY", "1")).strip().lower() in {
    "1",
    "true",
    "yes",
    "no"
    "on",
    "false"
}
try:
    QGEN_TEMPERATURE = float(os.getenv("QGEN_TEMPERATURE", "0.0"))
except (TypeError, ValueError):
    QGEN_TEMPERATURE = 0.0
try:
    QGEN_SEED = int(os.getenv("QGEN_SEED", "13"))
except (TypeError, ValueError):
    QGEN_SEED = None

TOKEN_PATTERN = re.compile(r"\b[a-zA-Z0-9]{3,}\b")
YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")
NUMBER_PATTERN = re.compile(r"\b\d+(?:[.,]\d+)?\b")
PRICING_QUESTION_PATTERN = re.compile(
    r"\b(price|pricing|cost|fee|difference|upgrade|downgrade|amount|tariff)\b",
    re.IGNORECASE,
)
PRICE_CHANGE_OPERATION_PATTERN = re.compile(
    r"\b(increase|increased|difference|changed|change|delta|old|new|before|after)\b",
    re.IGNORECASE,
)
TEMPORAL_HINT_PATTERN = re.compile(
    r"\b("
    r"old|new|previous|current|legacy|before|after|from|since|until|effective|as of|"
    r"january|february|march|april|may|june|july|august|september|october|november|december|"
    r"q[1-4]|20\d{2}"
    r")\b",
    re.IGNORECASE,
)
LOW_SIGNAL_META_QUESTION_PATTERNS = [
    re.compile(r"^\s*what\s+does\s+(?:the\s+)?(?:document|text|context)\s+state\s+about\b", re.IGNORECASE),
    re.compile(r"^\s*what\s+is\s+mentioned\s+about\b", re.IGNORECASE),
    re.compile(r"^\s*what\s+can\s+you\s+tell\s+me\s+about\b", re.IGNORECASE),
    re.compile(r"^\s*describe\s+(?:the\s+)?(?:document|text|context)\b", re.IGNORECASE),
    re.compile(r"^\s*what\s+exact\s+detail\s+is\s+stated\s+for\b", re.IGNORECASE),
    re.compile(r"\binclude\s+value,\s*date,\s*and\s*scope\s+if\s+present\b", re.IGNORECASE),
]
CARDINALITY_LEAD_PATTERN = re.compile(
    r"^\s*(?:what\s+are|which\s+are|which)\s+the\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b",
    re.IGNORECASE,
)
CARDINALITY_COUNT_PHRASE_PATTERN_TEMPLATE = r"\b{count}\s+(?:\w+\s+){{0,3}}(?:packages?|plans?|tariffs?)\b"
PACKAGE_SCOPE_PATTERN = re.compile(r"\b(packages?|plans?|tariffs?)\b", re.IGNORECASE)
PACKAGE_NAME_PATTERN = re.compile(
    r"\b(basic|standard|premium|mini|smart|pro|max|unlimited)\b",
    re.IGNORECASE,
)
BEFORE_TEMPORAL_PATTERN = re.compile(
    r"\b(before|old|previous|prior|pre[-\s]?change)\b",
    re.IGNORECASE,
)
AFTER_TEMPORAL_PATTERN = re.compile(
    r"\b(after|new|current|post[-\s]?change|effective|from)\b",
    re.IGNORECASE,
)
SENTENCE_SPLIT_PATTERN = re.compile(r"[.\n;]+")
PACKAGE_CONTEXT_PATTERN = re.compile(r"\b(packages?|plans?|tariffs?)\b", re.IGNORECASE)
RECOVERY_CONTEXT_PATTERN = re.compile(r"\b(recovery|re-register|registration|activation link)\b", re.IGNORECASE)
INVOICE_CONTEXT_PATTERN = re.compile(r"\b(invoice|invoices|billed|billing)\b", re.IGNORECASE)
DATE_CONTEXT_PATTERN = re.compile(r"\b(effective|from|since|until|as of|date)\b", re.IGNORECASE)
GENERIC_OLD_NEW_PRICING_PATTERN = re.compile(
    r"\b(which|what)\s+(?:old\s+vs\s+new|old\s+and\s+new)\s+prices?\b",
    re.IGNORECASE,
)
DEVICE_SUPPORT_VIA_PATTERN = re.compile(
    r"\b(which|what)\s+devices?\s+support\b.*\bvia\b",
    re.IGNORECASE,
)
RECEIVER_PATTERN = re.compile(r"\breceiver\b", re.IGNORECASE)
DEVICE_CATEGORY_PATTERN = re.compile(
    r"\b(smart\s*tvs?|mobile\s*phones?|tablets?|laptops?|desktop\s*computers?|gaming\s*consoles?)\b",
    re.IGNORECASE,
)
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "about",
    "what",
    "when",
    "where",
    "which",
    "who",
    "how",
    "into",
    "your",
    "their",
    "have",
    "will",
    "would",
    "could",
    "should",
    "does",
    "did",
    "are",
    "was",
    "were",
    "can",
    "all",
    "any",
    "some",
    "there",
    "been",
    "include",
    "included",
    "available",
    "service",
    "services",
    "document",
    "text",
    "context",
    "eur",
    "usd",
    "gbp",
    "streamplus",
    "globalconnect",
    "connecttv",
    "eurotel",
    "netflix",
    "max",
}
EXCLUSION_SIMILARITY_MIN_TOKENS = int(os.getenv("QGEN_EXCLUSION_MIN_TOKENS", "4"))
try:
    EXCLUSION_SIMILARITY_THRESHOLD = float(os.getenv("QGEN_EXCLUSION_SIMILARITY_THRESHOLD", "0.78"))
except (TypeError, ValueError):
    EXCLUSION_SIMILARITY_THRESHOLD = 0.78

SYSTEM_PROMPT = """You are an expert question generator for documentation validation.
Generate specific, grounded, and non-duplicate questions based only on the provided text.

Return ONLY a valid JSON object in this format:
{
  "questions": [
    {
      "question": "...",
      "category": "pricing|terms|availability|procedure|technical|other",
      "risk": "low|medium|high",
      "content_type": "pricing|instructions|terms|dates|technical|other",
      "hints": ["...", "..."]
    }
  ]
}

Rules:
- Generate exactly the requested number of questions.
- Questions must be answerable solely from the provided text.
- Each question must target one concrete fact or decision point (avoid broad/open-ended wording).
- Include explicit entity/scope in the question (e.g. package/tariff/channel/user segment/date window).
- For numeric/pricing questions, include unit and context if present (currency, monthly/annual, upgrade/add-on).
- If context has multiple price periods (old/new/effective dates), pricing questions must include time scope.
- Do not ask single-value pricing questions when answer depends on date/version; ask old vs new explicitly or include effective date.
- If you generate a package-specific old/new pricing question, do not also generate a generic "old vs new prices" summary question.
- For comparison/difference questions, explicitly name both compared items and the applicable time scope.
- Do not ask "which devices support ... via <receiver>" unless the same statement explicitly maps receiver method to concrete device categories/models.
- If the text references an external source (e.g. "see Tariff A", "as defined in Annex 3"), you may ask about the reference itself (e.g. "Which tariff is referenced for pricing details?") but NOT about the contents of that external source.
- Prefer concrete entity/value/scope/date questions over vague paraphrases.
- Avoid near-duplicate wording.
- Questions must be in English.
- Do not output any text outside JSON.
"""

USER_TEMPLATE = """TEXT INFORMATION:
- text_chars: {total_chars}
- target_questions: {n_questions}

CONTENT TO ANALYZE:
{content}

TASK:
1) Analyze the content.
2) Generate exactly {n_questions} grounded, specific questions.
3) Return only JSON in the required format.
"""


def generate_questions(
    chunk_text: str,
    count: int = 5,
    excluded_question_keys: list[str] | None = None,
) -> list[dict[str, Any]]:
    target_count = _resolve_question_count(chunk_text, count)
    limited_context = _limit_context(chunk_text, MAX_CONTEXT_TOKENS)
    blocked_keys = {
        _normalize_question_key(str(value))
        for value in (excluded_question_keys or [])
        if _normalize_question_key(str(value))
    }
    blocked_signatures = _build_exclusion_signatures(blocked_keys)

    logger.info(
        "Question generation started | chunk_chars=%s target_count=%s context_tokens=%s blocked_keys=%s",
        len(chunk_text),
        target_count,
        count_tokens(limited_context),
        len(blocked_keys),
    )

    client = get_client()
    deployment = get_model_name("OPENAI_QGEN_MODEL")

    user_prompt = USER_TEMPLATE.format(
        total_chars=len(chunk_text),
        n_questions=target_count,
        content=limited_context,
    )
    estimated_tokens = min(max(target_count * 120, 1500), 5000)
    response = create_chat_completion_with_fallback(
        client=client,
        model=deployment,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=QGEN_TEMPERATURE,
        max_output_tokens=estimated_tokens,
        response_format={"type": "json_object"},
        seed=QGEN_SEED,
        log_label="QGen",
    )

    content = (response.choices[0].message.content or "").strip()
    questions = _parse_questions(content, target_count, chunk_text)
    questions = _filter_excluded_questions(questions, blocked_keys, blocked_signatures)
    if len(questions) < target_count:
        fallback_rows = _fallback_questions(chunk_text, target_count - len(questions))
        fallback_rows = _filter_excluded_questions(fallback_rows, blocked_keys, blocked_signatures)
        questions = _merge_questions(questions, fallback_rows)
        questions = _filter_excluded_questions(questions, blocked_keys, blocked_signatures)

    questions = sorted(
        questions,
        key=lambda row: _normalize_question_key(str(row.get("question", ""))),
    )[:target_count]

    logger.info(
        "Question generation finished | requested=%s generated=%s",
        target_count,
        len(questions),
    )
    return questions


def _resolve_question_count(chunk_text: str, requested: int) -> int:
    if requested and requested > 0:
        return max(MIN_QUESTIONS_PER_DOC, min(requested, MAX_QUESTIONS_PER_DOC))

    chars = len(chunk_text)
    base = chars // 700
    return max(MIN_QUESTIONS_PER_DOC, min(base, MAX_QUESTIONS_PER_DOC))


def _limit_context(text: str, max_tokens: int) -> str:
    if count_tokens(text) <= max_tokens:
        return text

    chunks = []
    total = 0
    for para in [p.strip() for p in text.split("\n") if p.strip()]:
        t = count_tokens(para)
        if total + t > max_tokens:
            break
        chunks.append(para)
        total += t

    if chunks:
        return "\n\n".join(chunks)

    max_chars = max_tokens * 4
    return text[:max_chars]


def _tokenize_grounding(text: str) -> set[str]:
    return {
        token.lower()
        for token in TOKEN_PATTERN.findall(str(text or ""))
        if token and token.lower() not in STOPWORDS
    }


def _normalize_question_key(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _normalize_similarity_token(token: str) -> str:
    value = str(token or "").strip().lower()
    if not value:
        return ""
    if value.endswith("ies") and len(value) > 4:
        value = value[:-3] + "y"
    elif value.endswith("ing") and len(value) > 5:
        value = value[:-3]
    elif value.endswith("es") and len(value) > 4:
        value = value[:-2]
    elif value.endswith("s") and len(value) > 3:
        value = value[:-1]
    return value


def _question_similarity_signature(text: str) -> set[str]:
    tokens = {
        _normalize_similarity_token(raw)
        for raw in TOKEN_PATTERN.findall(str(text or ""))
    }
    return {
        token
        for token in tokens
        if token
        and token not in STOPWORDS
        and not token.isdigit()
        and not YEAR_PATTERN.fullmatch(token)
    }


def _build_exclusion_signatures(blocked_keys: set[str]) -> list[set[str]]:
    signatures: list[set[str]] = []
    for key in blocked_keys:
        signature = _question_similarity_signature(key)
        if len(signature) >= EXCLUSION_SIMILARITY_MIN_TOKENS:
            signatures.append(signature)
    return signatures


def _is_similar_to_excluded_question(question_text: str, blocked_signatures: list[set[str]]) -> bool:
    if not blocked_signatures:
        return False
    candidate = _question_similarity_signature(question_text)
    if len(candidate) < EXCLUSION_SIMILARITY_MIN_TOKENS:
        return False
    for blocked in blocked_signatures:
        if not blocked:
            continue
        intersection = len(candidate & blocked)
        if intersection < EXCLUSION_SIMILARITY_MIN_TOKENS:
            continue
        union = len(candidate | blocked)
        if union <= 0:
            continue
        similarity = intersection / float(union)
        if similarity >= EXCLUSION_SIMILARITY_THRESHOLD:
            return True
    return False


def _filter_excluded_questions(
    rows: list[dict[str, Any]],
    blocked_keys: set[str],
    blocked_signatures: list[set[str]],
) -> list[dict[str, Any]]:
    if not blocked_keys and not blocked_signatures:
        return rows
    filtered: list[dict[str, Any]] = []
    skipped = 0
    for row in rows:
        question_text = str(row.get("question", ""))
        question_key = _normalize_question_key(question_text)
        if question_key in blocked_keys or _is_similar_to_excluded_question(question_text, blocked_signatures):
            skipped += 1
            continue
        filtered.append(row)
    if skipped:
        logger.info(
            "Excluded question filter applied | input=%s removed=%s output=%s",
            len(rows),
            skipped,
            len(filtered),
        )
    return filtered


def _normalize_question_row(item: dict[str, Any]) -> dict[str, Any]:
    question = str(item.get("question", "")).strip()
    category = str(item.get("category", "other")).strip().lower() or "other"
    risk = str(item.get("risk", "low")).strip().lower() or "low"
    content_type = str(item.get("content_type", "other")).strip().lower() or "other"
    hints = [
        str(v).strip()
        for v in (item.get("hints", []) if isinstance(item.get("hints"), list) else [])
        if str(v).strip()
    ]
    if risk not in {"low", "medium", "high"}:
        risk = "low"
    return {
        "question": question,
        "category": category,
        "risk": risk,
        "content_type": content_type,
        "hints": hints,
    }


def _extract_package_name(question: str) -> str | None:
    match = PACKAGE_NAME_PATTERN.search(str(question or ""))
    if not match:
        return None
    value = str(match.group(1) or "").strip().lower()
    return value or None


def _question_specificity_score(question: str) -> int:
    value = str(question or "")
    score = 0
    if _extract_package_name(value):
        score += 3
    if NUMBER_PATTERN.search(value):
        score += 1
    if TEMPORAL_HINT_PATTERN.search(value):
        score += 1
    if GENERIC_OLD_NEW_PRICING_PATTERN.search(value):
        score -= 2
    token_count = len(_normalize_question_key(value).split())
    score += min(2, token_count // 8)
    return score


def _question_intent_key(question: str, source_text: str) -> str:
    normalized = _normalize_question_key(question)
    if not normalized:
        return ""
    package = _extract_package_name(question) or "*"
    if (
        _has_multi_period_pricing_context(source_text)
        and PRICING_QUESTION_PATTERN.search(question)
        and (TEMPORAL_HINT_PATTERN.search(question) or PRICE_CHANGE_OPERATION_PATTERN.search(question))
    ):
        return f"pricing.old_new.{package}"
    if DEVICE_SUPPORT_VIA_PATTERN.search(question):
        return "availability.device_support_via"
    return normalized


def _is_specific_old_new_pricing_question(question: str, source_text: str) -> bool:
    if not _has_multi_period_pricing_context(source_text):
        return False
    if not PRICING_QUESTION_PATTERN.search(question):
        return False
    if not _extract_package_name(question):
        return False
    return bool(TEMPORAL_HINT_PATTERN.search(question) or PRICE_CHANGE_OPERATION_PATTERN.search(question))


def _is_generic_old_new_pricing_question(question: str, source_text: str) -> bool:
    if not _has_multi_period_pricing_context(source_text):
        return False
    if not PRICING_QUESTION_PATTERN.search(question):
        return False
    if _extract_package_name(question):
        return False
    return bool(GENERIC_OLD_NEW_PRICING_PATTERN.search(question))


def _drop_redundant_generic_old_new_pricing(
    rows: list[dict[str, Any]],
    source_text: str,
) -> list[dict[str, Any]]:
    has_specific = any(
        _is_specific_old_new_pricing_question(str(item.get("question", "")), source_text)
        for item in rows
        if isinstance(item, dict)
    )
    if not has_specific:
        return rows
    filtered = [
        item
        for item in rows
        if not _is_generic_old_new_pricing_question(str(item.get("question", "")), source_text)
    ]
    return filtered if filtered else rows


def _is_over_specific_device_via_question(question: str, source_text: str) -> bool:
    q = str(question or "")
    if not DEVICE_SUPPORT_VIA_PATTERN.search(q):
        return False
    source = str(source_text or "")
    windows = [part.strip() for part in SENTENCE_SPLIT_PATTERN.split(source) if part.strip()]
    for window in windows:
        if RECEIVER_PATTERN.search(window) and DEVICE_CATEGORY_PATTERN.search(window):
            # Source explicitly binds receiver method + device categories in one statement.
            return False
    return True


def _append_question_with_intent_dedupe(
    cleaned: list[dict[str, Any]],
    seen_keys: set[str],
    seen_intent_best: dict[str, int],
    seen_intent_index: dict[str, int],
    row: dict[str, Any],
    source_text: str,
) -> bool:
    question = str(row.get("question", "")).strip()
    key = _normalize_question_key(question)
    if not key:
        return False
    intent = _question_intent_key(question, source_text) or key
    score = _question_specificity_score(question)
    if key in seen_keys:
        return False
    existing_score = seen_intent_best.get(intent)
    existing_index = seen_intent_index.get(intent)
    if existing_score is not None and existing_index is not None:
        if score > existing_score:
            old_key = _normalize_question_key(str(cleaned[existing_index].get("question", "")))
            if old_key in seen_keys:
                seen_keys.remove(old_key)
            cleaned[existing_index] = row
            seen_keys.add(key)
            seen_intent_best[intent] = score
            return True
        return False
    seen_keys.add(key)
    seen_intent_best[intent] = score
    seen_intent_index[intent] = len(cleaned)
    cleaned.append(row)
    return True


def _merge_questions(primary: list[dict[str, Any]], secondary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for item in [*primary, *secondary]:
        if not isinstance(item, dict):
            continue
        row = _normalize_question_row(item)
        key = _normalize_question_key(str(row.get("question", "")))
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        merged.append(row)
    return merged


def _is_question_grounded(question: str, source_text: str) -> bool:
    question_tokens = _tokenize_grounding(question)
    source_tokens = _tokenize_grounding(source_text)
    if not question_tokens or not source_tokens:
        return False
    overlap = question_tokens & source_tokens
    return len(overlap) >= max(1, GROUNDING_MIN_OVERLAP_TOKENS)


def _has_multi_period_pricing_context(source_text: str) -> bool:
    text = str(source_text or "")
    if not PRICING_QUESTION_PATTERN.search(text):
        return False
    years = {int(value) for value in YEAR_PATTERN.findall(text)}
    if len(years) >= 2:
        return True
    lowered = text.lower()
    return ("old price" in lowered or "old prices" in lowered) and (
        "new price" in lowered or "new prices" in lowered
    )


def _is_temporally_unspecified_pricing_question(question: str, source_text: str) -> bool:
    if not PRICING_QUESTION_PATTERN.search(str(question or "")):
        return False
    if not _has_multi_period_pricing_context(source_text):
        return False
    if TEMPORAL_HINT_PATTERN.search(str(question or "")):
        return False
    return True


def _is_low_signal_meta_question(question: str) -> bool:
    value = str(question or "").strip()
    if not value:
        return True
    return any(pattern.search(value) for pattern in LOW_SIGNAL_META_QUESTION_PATTERNS)


def _normalize_numeric_token(value: str) -> str:
    token = str(value or "").strip().replace(",", ".")
    if "." in token:
        token = token.rstrip("0").rstrip(".")
    return token


def _extract_numeric_tokens(text: str) -> list[str]:
    tokens = [_normalize_numeric_token(raw) for raw in NUMBER_PATTERN.findall(str(text or ""))]
    return [token for token in tokens if token]


def _has_temporal_numeric_alignment(question: str, source_text: str) -> bool:
    q = str(question or "")
    source = str(source_text or "")
    numbers = _extract_numeric_tokens(q)
    if not numbers:
        return True

    has_before = bool(BEFORE_TEMPORAL_PATTERN.search(q))
    has_after = bool(AFTER_TEMPORAL_PATTERN.search(q))
    if not has_before and not has_after:
        return True
    if has_before and has_after:
        # Question explicitly covers multiple periods; keep it and let QA evaluate details.
        return True

    windows = [part.strip().lower() for part in SENTENCE_SPLIT_PATTERN.split(source) if part.strip()]
    if not windows:
        return False

    temporal_pattern = BEFORE_TEMPORAL_PATTERN if has_before else AFTER_TEMPORAL_PATTERN
    for number in numbers:
        aligned = False
        for window in windows:
            if _normalize_numeric_token(number) not in [_normalize_numeric_token(v) for v in NUMBER_PATTERN.findall(window)]:
                continue
            if temporal_pattern.search(window):
                aligned = True
                break
        if not aligned:
            return False
    return True


def _generalize_cardinality_question(question: str, source_text: str) -> str:
    value = str(question or "").strip()
    if not value:
        return value
    if not PACKAGE_SCOPE_PATTERN.search(value):
        return value

    match = CARDINALITY_LEAD_PATTERN.search(value)
    if not match:
        return value

    count_token = str(match.group(1) or "").strip().lower()
    source_lower = str(source_text or "").lower()
    expected_pattern = re.compile(
        CARDINALITY_COUNT_PHRASE_PATTERN_TEMPLATE.format(count=re.escape(count_token)),
        re.IGNORECASE,
    )
    if expected_pattern.search(source_lower):
        return value

    # Convert over-constrained count question into a generalized availability question.
    remainder = CARDINALITY_LEAD_PATTERN.sub("", value).strip()
    remainder = re.sub(r"^\s*(streamplus\s+)?", "", remainder, flags=re.IGNORECASE).strip()
    if not remainder:
        return "What StreamPlus packages are available?"
    if "simultaneous device" in value.lower() or "device limit" in value.lower() or "concurrent" in value.lower():
        return "What StreamPlus packages are available, and what are their simultaneous device limits?"
    return "What StreamPlus packages are available?"


def _top_source_keywords(source_text: str, limit: int = 8) -> list[str]:
    freq: dict[str, int] = {}
    for raw_token in TOKEN_PATTERN.findall(str(source_text or "")):
        token = raw_token.lower()
        if not token or token in STOPWORDS:
            continue
        if token.isdigit():
            continue
        freq[token] = freq.get(token, 0) + 1
    ranked = sorted(freq.items(), key=lambda item: (-item[1], item[0]))
    return [token for token, _ in ranked[: max(1, int(limit))]]


def _parse_questions(content: str, count: int, source_text: str) -> list[dict[str, Any]]:
    data = safe_json_parse(content, allow_array=True)
    items = _extract_questions_items(data)

    cleaned: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    seen_intent_best: dict[str, int] = {}
    seen_intent_index: dict[str, int] = {}
    for item in items:
        question = ""
        row: dict[str, Any] = {"question": ""}

        if isinstance(item, dict):
            row = _normalize_question_row(item)
            question = str(row.get("question", "")).strip()
        elif isinstance(item, str):
            question = item.strip()
            row = _normalize_question_row({"question": question})

        question = _generalize_cardinality_question(question, source_text)
        row["question"] = question

        question_key = _normalize_question_key(question)
        if not question_key or question_key in seen_keys:
            continue
        if not _is_question_grounded(question, source_text):
            logger.debug("Question discarded by grounding filter | question=%s", question)
            continue
        if _is_temporally_unspecified_pricing_question(question, source_text):
            logger.debug("Question discarded by temporal pricing filter | question=%s", question)
            continue
        if not _has_temporal_numeric_alignment(question, source_text):
            logger.debug("Question discarded by temporal-number alignment filter | question=%s", question)
            continue
        if _is_low_signal_meta_question(question):
            logger.debug("Question discarded by low-signal meta filter | question=%s", question)
            continue
        if _is_over_specific_device_via_question(question, source_text):
            logger.debug("Question discarded by device-via specificity filter | question=%s", question)
            continue
        _append_question_with_intent_dedupe(
            cleaned=cleaned,
            seen_keys=seen_keys,
            seen_intent_best=seen_intent_best,
            seen_intent_index=seen_intent_index,
            row=row,
            source_text=source_text,
        )

    if len(cleaned) < count:
        logger.warning(
            "Question generation returned too few items | got=%s expected=%s, using fallback.",
            len(cleaned),
            count,
        )
        fallback_rows = _fallback_questions(source_text, count - len(cleaned))
        for row in fallback_rows:
            question = str(row.get("question", "")).strip()
            question = _generalize_cardinality_question(question, source_text)
            row["question"] = question
            question_key = _normalize_question_key(question)
            if not question_key or question_key in seen_keys:
                continue
            if not _has_temporal_numeric_alignment(question, source_text):
                logger.debug("Fallback question discarded by temporal-number alignment filter | question=%s", question)
                continue
            if _is_low_signal_meta_question(question):
                logger.debug("Fallback question discarded by low-signal meta filter | question=%s", question)
                continue
            if _is_over_specific_device_via_question(question, source_text):
                logger.debug("Fallback question discarded by device-via specificity filter | question=%s", question)
                continue
            _append_question_with_intent_dedupe(
                cleaned=cleaned,
                seen_keys=seen_keys,
                seen_intent_best=seen_intent_best,
                seen_intent_index=seen_intent_index,
                row=row,
                source_text=source_text,
            )
            if len(cleaned) >= count:
                break
        if QGEN_STRICT_ANSWERABLE_ONLY and len(cleaned) < count:
            logger.info(
                "QGen strict mode active | returning fewer questions than requested | requested=%s generated=%s",
                count,
                len(cleaned),
            )

    cleaned = _drop_redundant_generic_old_new_pricing(cleaned, source_text)
    cleaned.sort(key=lambda row: _normalize_question_key(str(row.get("question", ""))))
    return cleaned[:count]

def _extract_questions_items(data: Any) -> list[Any]:
    if isinstance(data, dict):
        val = data.get("questions", [])
        if isinstance(val, list):
            return val
    if isinstance(data, list):
        return data
    return []


def _fallback_questions(source_text: str, count: int) -> list[dict[str, Any]]:
    keywords = _top_source_keywords(source_text, limit=max(4, count * 2))
    multi_period_pricing = _has_multi_period_pricing_context(source_text)
    source = str(source_text or "")

    rows: list[dict[str, Any]] = []
    for keyword in keywords:
        if len(rows) >= count:
            break
        if multi_period_pricing and PRICING_QUESTION_PATTERN.search(keyword + " " + source):
            question = "Which old vs new prices are specified, and what is the effective date?"
            content_type = "pricing"
            hints = [keyword, "effective date", "old vs new"]
            category = "pricing"
        elif PACKAGE_CONTEXT_PATTERN.search(source):
            question = "What StreamPlus packages are available?"
            content_type = "other"
            hints = ["packages", "availability"]
            category = "availability"
        elif RECOVERY_CONTEXT_PATTERN.search(source):
            question = "Which self-service channels are listed for StreamPlus account recovery?"
            content_type = "instructions"
            hints = ["self-service channels", "recovery"]
            category = "procedure"
        elif INVOICE_CONTEXT_PATTERN.search(source) and DATE_CONTEXT_PATTERN.search(source):
            question = "When will invoices reflecting the new prices be issued?"
            content_type = "dates"
            hints = ["invoice", "new prices"]
            category = "terms"
        else:
            continue

        row = {
            "question": question,
            "category": category,
            "risk": "low",
            "content_type": content_type,
            "hints": hints,
        }
        key = _normalize_question_key(question)
        if any(_normalize_question_key(str(existing.get("question", ""))) == key for existing in rows):
            continue
        rows.append(row)

    return rows
