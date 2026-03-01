from __future__ import annotations

import json
from typing import Any

from app.services.metrics_utils import as_optional_float as _as_optional_float


RAGAS_METRICS_ORDER: list[tuple[str, str]] = [
    ("context_recall", "Context Recall"),
    ("context_precision", "Context Precision"),
    ("faithfulness", "Faithfulness"),
    ("factual_correctness", "Factual Correctness"),
    ("answer_relevancy", "Answer Relevancy"),
    ("context_entity_recall", "Context Entity Recall"),
    ("noise_sensitivity", "Noise Sensitivity"),
]

RAGAS_DATASET_QUALITY_ORDER: list[tuple[str, str]] = [
    ("sample_count", "Samples"),
    ("canonical_reference_count", "Canonical Answer References"),
    ("evidence_reference_count", "Evidence Quote References"),
    ("retrieved_context_reference_count", "Retrieved Context References"),
    ("question_fallback_reference_count", "Question Fallback References"),
    ("avg_contexts_per_sample", "Avg Contexts / Sample"),
    ("empty_context_sample_count", "Empty Context Samples"),
    ("samples_with_evidence_count", "Samples With Evidence Citations"),
    ("avg_evidence_citations_per_sample", "Avg Evidence Citations / Sample"),
]


def _safe_json_loads(raw: Any, default: Any) -> Any:
    if isinstance(raw, (dict, list)):
        return raw
    if not isinstance(raw, str):
        return default
    try:
        return json.loads(raw)
    except Exception:
        return default


def _normalize_mode(value: Any) -> str:
    mode = str(value or "parent").strip().lower()
    return mode if mode in {"baseline", "parent"} else "parent"


def _normalize_answer_row(row: dict[str, Any]) -> dict[str, Any]:
    item = dict(row)
    item["mode"] = _normalize_mode(item.get("mode"))
    item["score"] = _as_optional_float(item.get("score"))
    item["confidence"] = _as_optional_float(item.get("confidence"))
    item["effective_score"] = item["score"] if item["score"] is not None else item["confidence"]
    item["used_citations"] = _safe_json_loads(item.get("used_citations_json"), [])
    item["evidence"] = _safe_json_loads(item.get("evidence_json"), [])
    return item


def _extract_category_meta(raw_category: Any) -> dict[str, Any]:
    data = _safe_json_loads(raw_category, {})
    return data if isinstance(data, dict) else {}


def _build_question_rows(
    question_rows: list[dict[str, Any]],
    answer_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    answers_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in answer_rows:
        normalized = _normalize_answer_row(row)
        question_id = str(normalized.get("question_id", "")).strip()
        if not question_id:
            continue
        answers_by_key[(question_id, normalized["mode"])] = normalized

    result: list[dict[str, Any]] = []
    for q in question_rows:
        q_item = dict(q)
        question_id = str(q_item.get("question_id", "")).strip()
        category_meta = _extract_category_meta(q_item.get("category"))

        baseline = answers_by_key.get((question_id, "baseline"))
        parent = answers_by_key.get((question_id, "parent"))
        baseline_score = baseline.get("effective_score") if baseline else None
        parent_score = parent.get("effective_score") if parent else None
        delta = None
        if baseline_score is not None and parent_score is not None:
            delta = parent_score - baseline_score

        result.append(
            {
                "question_id": question_id,
                "question_no": q_item.get("question_no"),
                "question": q_item.get("question_text"),
                "chunk_id": q_item.get("chunk_id"),
                "chunk_order": q_item.get("chunk_order"),
                "category": category_meta.get("category"),
                "risk": category_meta.get("risk"),
                "content_type": category_meta.get("content_type"),
                "baseline_score": baseline_score,
                "parent_score": parent_score,
                "delta": delta,
                "baseline": baseline,
                "parent": parent,
            }
        )

    result.sort(key=lambda item: (int(item.get("chunk_order") or 0), int(item.get("question_no") or 0)))
    return result


def _build_ragas_section(ragas_by_mode: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    baseline = ragas_by_mode.get("baseline")
    parent = ragas_by_mode.get("parent")
    rows = []
    for key, label in RAGAS_METRICS_ORDER:
        baseline_value = _as_optional_float((baseline or {}).get(key))
        parent_value = _as_optional_float((parent or {}).get(key))
        delta = None
        if baseline_value is not None and parent_value is not None:
            delta = parent_value - baseline_value
        rows.append(
            {
                "metric_key": key,
                "metric": label,
                "baseline": baseline_value,
                "parent": parent_value,
                "delta": delta,
            }
        )
    baseline_meta = (baseline or {}).get("dataset_meta") if isinstance(baseline, dict) else None
    parent_meta = (parent or {}).get("dataset_meta") if isinstance(parent, dict) else None
    dataset_rows = []
    for key, label in RAGAS_DATASET_QUALITY_ORDER:
        baseline_value = _as_optional_float((baseline_meta or {}).get(key))
        parent_value = _as_optional_float((parent_meta or {}).get(key))
        delta = None
        if baseline_value is not None and parent_value is not None:
            delta = parent_value - baseline_value
        dataset_rows.append(
            {
                "metric_key": key,
                "metric": label,
                "baseline": baseline_value,
                "parent": parent_value,
                "delta": delta,
            }
        )
    return {
        "baseline": baseline,
        "parent": parent,
        "rows": rows,
        "dataset_rows": dataset_rows,
        "baseline_meta": baseline_meta if isinstance(baseline_meta, dict) else {},
        "parent_meta": parent_meta if isinstance(parent_meta, dict) else {},
    }


def build_comparison_report(
    *,
    question_rows: list[dict[str, Any]],
    answer_rows: list[dict[str, Any]],
    ragas_by_mode: dict[str, dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    normalized_ragas = {"baseline": None, "parent": None}
    if ragas_by_mode:
        for mode in ("baseline", "parent"):
            raw = ragas_by_mode.get(mode)
            normalized_ragas[mode] = dict(raw) if isinstance(raw, dict) else None

    question_table = _build_question_rows(question_rows=question_rows, answer_rows=answer_rows)
    ragas_section = _build_ragas_section(normalized_ragas)

    return {
        "question_rows": question_table,
        "ragas": ragas_section,
    }
