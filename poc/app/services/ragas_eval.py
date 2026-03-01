from __future__ import annotations

import logging
import os
from typing import Any

from app.services.llm_client import get_model_name
from app.services.metrics_utils import as_optional_float as _as_optional_float
from app.services.metrics_utils import fmt_metric as _fmt_metric

logger = logging.getLogger(__name__)
DEFAULT_RAGAS_REFERENCE_MIN_CONFIDENCE = 0.80

RAGAS_METRIC_KEYS: tuple[str, ...] = (
    "context_recall",
    "context_precision",
    "faithfulness",
    "factual_correctness",
    "answer_relevancy",
    "context_entity_recall",
    "noise_sensitivity",
)

_RAGAS_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "context_recall": ("context_recall", "llm_context_recall"),
    "context_precision": ("context_precision", "llm_context_precision_with_reference"),
    "faithfulness": ("faithfulness",),
    "factual_correctness": ("factual_correctness",),
    "answer_relevancy": ("answer_relevancy", "response_relevancy"),
    "context_entity_recall": ("context_entity_recall", "context_entities_recall", "entity_recall"),
    "noise_sensitivity": (
        "noise_sensitivity",
        "noise_sensitivity(mode=relevant)",
        "noise_sensitivity(mode=irrelevant)",
        "noise_sensitivity_relevant",
        "noise_sensitivity_irrelevant",
    ),
}


def evaluate_comparison_mode_ragas(
    *,
    question_records: list[dict[str, Any]],
    mode_answer_rows: list[dict[str, Any]],
    chunk_records: list[dict[str, Any]],
    reference_candidates_by_question_id: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    _disable_unconfigured_langsmith_tracing()

    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import EvaluationDataset, RunConfig, evaluate
        from ragas.dataset_schema import SingleTurnSample
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            ContextEntityRecall,
            ContextPrecision,
            Faithfulness,
            FactualCorrectness,
            LLMContextRecall,
            NoiseSensitivity,
            ResponseRelevancy,
        )
    except Exception as exc:
        raise RuntimeError(
            "RAGAS dependencies are required for comparison runs. Install/update `ragas` and `langchain-openai`."
        ) from exc

    samples, dataset_meta = _build_samples(
        question_records=question_records,
        mode_answer_rows=mode_answer_rows,
        chunk_records=chunk_records,
        reference_candidates_by_question_id=reference_candidates_by_question_id or {},
        sample_factory=SingleTurnSample,
    )
    if not samples:
        raise RuntimeError("RAGAS evaluation cannot run because there are no comparison answers to evaluate.")

    llm_model = os.getenv("OPENAI_RAGAS_MODEL") or get_model_name("OPENAI_QA_MODEL")
    embedding_model = os.getenv("OPENAI_RAGAS_EMBEDDING_MODEL") or get_model_name("OPENAI_EMBEDDING_MODEL")

    timeout_seconds = _env_int("RAGAS_TIMEOUT_SECONDS", 180)
    max_workers = _env_int("RAGAS_MAX_WORKERS", 4)

    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model=llm_model, temperature=0))
    evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=embedding_model))

    dataset = EvaluationDataset(samples=samples)
    logger.info(
        "RAGAS evaluation started | samples=%s llm_model=%s embedding_model=%s max_workers=%s timeout_s=%s gold_refs=%s canonical_refs=%s evidence_refs=%s context_refs=%s question_fallback_refs=%s avg_contexts=%.2f",
        len(samples),
        llm_model,
        embedding_model,
        max_workers,
        timeout_seconds,
        int(dataset_meta.get("gold_reference_count", 0)),
        int(dataset_meta.get("canonical_reference_count", 0)),
        int(dataset_meta.get("evidence_reference_count", 0)),
        int(dataset_meta.get("retrieved_context_reference_count", 0)),
        int(dataset_meta.get("question_fallback_reference_count", 0)),
        float(dataset_meta.get("avg_contexts_per_sample", 0.0)),
    )

    result = evaluate(
        dataset=dataset,
        metrics=[
            LLMContextRecall(),
            ContextPrecision(),
            Faithfulness(),
            FactualCorrectness(language="english"),
            ResponseRelevancy(),
            ContextEntityRecall(),
            NoiseSensitivity(mode="relevant"),
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=RunConfig(max_workers=max_workers, timeout=timeout_seconds),
    )

    metrics = _extract_metric_averages(result)
    missing = [key for key in RAGAS_METRIC_KEYS if metrics.get(key) is None]
    final_metrics: dict[str, Any] = {
        key: (float(metrics[key]) if metrics.get(key) is not None else None)
        for key in RAGAS_METRIC_KEYS
    }
    if missing:
        logger.warning(
            "RAGAS evaluation returned partial metric set | missing_metrics=%s",
            ",".join(missing),
        )
    final_metrics["missing_metrics"] = missing
    final_metrics["metric_set_complete"] = not missing
    final_metrics["dataset_meta"] = dataset_meta
    logger.info(
        "RAGAS evaluation completed | context_recall=%s context_precision=%s faithfulness=%s factual_correctness=%s answer_relevancy=%s context_entity_recall=%s noise_sensitivity=%s complete=%s",
        _fmt_metric(final_metrics["context_recall"]),
        _fmt_metric(final_metrics["context_precision"]),
        _fmt_metric(final_metrics["faithfulness"]),
        _fmt_metric(final_metrics["factual_correctness"]),
        _fmt_metric(final_metrics["answer_relevancy"]),
        _fmt_metric(final_metrics["context_entity_recall"]),
        _fmt_metric(final_metrics["noise_sensitivity"]),
        final_metrics["metric_set_complete"],
    )
    return final_metrics


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _disable_unconfigured_langsmith_tracing() -> None:
    tracing_value = str(os.getenv("LANGCHAIN_TRACING_V2", "")).strip().lower()
    tracing_enabled = tracing_value in {"1", "true", "yes", "on"}
    if not tracing_enabled:
        return
    if os.getenv("LANGCHAIN_API_KEY"):
        return
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    logger.info("LangSmith tracing disabled for RAGAS run because LANGCHAIN_API_KEY is not configured.")


def _build_samples(
    *,
    question_records: list[dict[str, Any]],
    mode_answer_rows: list[dict[str, Any]],
    chunk_records: list[dict[str, Any]],
    reference_candidates_by_question_id: dict[str, dict[str, Any]],
    sample_factory: Any,
) -> tuple[list[Any], dict[str, Any]]:
    question_by_id = {
        str(row.get("id", "")).strip(): str(row.get("question_text", "")).strip()
        for row in question_records
        if str(row.get("id", "")).strip()
    }
    chunk_text_by_id = {
        str(row.get("id", "")).strip(): str(row.get("text", ""))
        for row in chunk_records
        if str(row.get("id", "")).strip()
    }

    min_ref_conf = _env_float("RAGAS_REFERENCE_MIN_CONFIDENCE", DEFAULT_RAGAS_REFERENCE_MIN_CONFIDENCE)
    normalized_refs = _normalize_reference_candidates(reference_candidates_by_question_id)

    samples: list[Any] = []
    context_count_sum = 0
    evidence_citation_count_sum = 0
    samples_with_evidence_count = 0
    empty_context_sample_count = 0
    canonical_reference_count = 0
    gold_reference_count = 0
    evidence_reference_count = 0
    retrieved_context_reference_count = 0
    question_fallback_reference_count = 0
    for answer in mode_answer_rows:
        question_id = str(answer.get("question_id", "")).strip()
        question_text = question_by_id.get(question_id, "")
        answer_text = str(answer.get("answer_text", "")).strip()
        if not question_id or not question_text or not answer_text:
            continue

        evidence_chunk_ids = _evidence_chunk_ids(answer)
        retrieved_contexts = _contexts_from_answer(answer, chunk_text_by_id)
        if not retrieved_contexts:
            # RAGAS metrics are brittle with an empty context list; keep the sample evaluable.
            fallback_context = _fallback_context_from_question(question_id, chunk_text_by_id, answer)
            if fallback_context:
                retrieved_contexts = [fallback_context]

        context_count = len(retrieved_contexts)
        context_count_sum += context_count
        evidence_citation_count_sum += len(evidence_chunk_ids)
        if evidence_chunk_ids:
            samples_with_evidence_count += 1
        if context_count == 0:
            empty_context_sample_count += 1

        reference_text, reference_source = _select_reference_for_question(
            question_id=question_id,
            question_text=question_text,
            normalized_refs=normalized_refs,
            min_confidence=min_ref_conf,
            answer=answer,
        )
        if reference_source == "gold_reference":
            gold_reference_count += 1
        elif reference_source == "canonical_answer":
            canonical_reference_count += 1
        elif reference_source == "evidence_fallback":
            evidence_reference_count += 1
        elif reference_source == "retrieved_context_fallback":
            retrieved_context_reference_count += 1
        else:
            question_fallback_reference_count += 1

        samples.append(
            sample_factory(
                user_input=question_text,
                response=answer_text,
                retrieved_contexts=retrieved_contexts,
                reference=reference_text,
            )
        )

    sample_count = len(samples)
    dataset_meta = {
        "sample_count": sample_count,
        "gold_reference_count": gold_reference_count,
        "canonical_reference_count": canonical_reference_count,
        "evidence_reference_count": evidence_reference_count,
        "retrieved_context_reference_count": retrieved_context_reference_count,
        "question_fallback_reference_count": question_fallback_reference_count,
        "avg_contexts_per_sample": (context_count_sum / sample_count) if sample_count else 0.0,
        "empty_context_sample_count": empty_context_sample_count,
        "samples_with_evidence_count": samples_with_evidence_count,
        "avg_evidence_citations_per_sample": (evidence_citation_count_sum / sample_count) if sample_count else 0.0,
        "reference_min_confidence": float(min_ref_conf),
    }
    return (samples, dataset_meta)


def _normalize_reference_candidates(
    raw_map: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    for question_id, raw in raw_map.items():
        qid = str(question_id or "").strip()
        if not qid or not isinstance(raw, dict):
            continue
        answer_text = str(raw.get("answer_text", raw.get("reference_text", ""))).strip()
        if not answer_text:
            continue
        confidence = _as_optional_float(raw.get("confidence"))
        problem_type = str(raw.get("problem_type", "")).strip().lower()
        normalized[qid] = {
            "answer_text": answer_text,
            "confidence": confidence,
            "problem_type": problem_type,
            "reference_origin": str(raw.get("reference_origin", raw.get("source", ""))).strip().lower(),
        }
    return normalized


def _select_reference_for_question(
    *,
    question_id: str,
    question_text: str,
    normalized_refs: dict[str, dict[str, Any]],
    min_confidence: float,
    answer: dict[str, Any],
) -> tuple[str, str]:
    candidate = normalized_refs.get(question_id)
    if candidate:
        reference_origin = str(candidate.get("reference_origin", "")).strip().lower()
        problem_type = str(candidate.get("problem_type", "")).strip().lower()
        confidence = _as_optional_float(candidate.get("confidence"))
        answer_text = str(candidate.get("answer_text", "")).strip()
        if answer_text and reference_origin == "gold_reference":
            return (answer_text, "gold_reference")
        if (
            answer_text
            and problem_type == "ok"
            and confidence is not None
            and confidence >= min_confidence
        ):
            return (answer_text, "canonical_answer")
    evidence_reference = _reference_from_evidence(answer)
    if evidence_reference:
        return (evidence_reference, "evidence_fallback")
    context_reference = _reference_from_retrieved_contexts(answer)
    if context_reference:
        return (context_reference, "retrieved_context_fallback")
    # Spec-approved fallback until a true gold answer set exists.
    return (question_text, "question_fallback")


def _evidence_chunk_ids(answer: dict[str, Any]) -> list[str]:
    raw_evidence = answer.get("evidence")
    if not isinstance(raw_evidence, list):
        return []
    seen: set[str] = set()
    ordered: list[str] = []
    for item in raw_evidence:
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("chunk_id", "")).strip()
        if chunk_id and chunk_id not in seen:
            seen.add(chunk_id)
            ordered.append(chunk_id)
    return ordered


def _reference_from_evidence(answer: dict[str, Any]) -> str | None:
    raw_evidence = answer.get("evidence")
    if not isinstance(raw_evidence, list):
        return None

    parts: list[str] = []
    for item in raw_evidence:
        if not isinstance(item, dict):
            continue
        quote = str(item.get("quote", "")).strip()
        if not quote:
            continue
        parts.append(_normalize_reference_text(quote))
        if len(parts) >= 2:
            break

    if not parts:
        return None
    return _normalize_reference_text(" ".join(parts), max_chars=520)


def _reference_from_retrieved_contexts(answer: dict[str, Any]) -> str | None:
    raw_contexts = answer.get("retrieved_contexts")
    if not isinstance(raw_contexts, list):
        return None

    parts: list[str] = []
    for ctx in raw_contexts:
        text = str(ctx or "").strip()
        if not text:
            continue
        parts.append(_normalize_reference_text(text, max_chars=260))
        if len(parts) >= 2:
            break

    if not parts:
        return None
    return _normalize_reference_text(" ".join(parts), max_chars=520)


def _normalize_reference_text(text: str, max_chars: int = 420) -> str:
    compact = " ".join(str(text or "").split()).strip()
    if not compact:
        return ""
    return compact[: max_chars - 3] + "..." if len(compact) > max_chars else compact


def _contexts_from_answer(answer: dict[str, Any], chunk_text_by_id: dict[str, str]) -> list[str]:
    raw_evidence = answer.get("evidence")
    evidence = raw_evidence if isinstance(raw_evidence, list) else []

    seen: set[str] = set()
    contexts: list[str] = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("chunk_id", "")).strip()
        quote = str(item.get("quote", "")).strip()
        if chunk_id and chunk_id in chunk_text_by_id and chunk_id not in seen:
            seen.add(chunk_id)
            text = chunk_text_by_id[chunk_id].strip()
            if text:
                contexts.append(text)
                continue
        if quote:
            contexts.append(quote)

    # Secondary fallback: use full retrieval contexts captured during QA execution if provided.
    raw_contexts = answer.get("retrieved_contexts")
    if isinstance(raw_contexts, list):
        for ctx in raw_contexts:
            text = str(ctx or "").strip()
            if text and text not in contexts:
                contexts.append(text)

    return contexts


def _fallback_context_from_question(
    question_id: str,
    chunk_text_by_id: dict[str, str],
    answer: dict[str, Any],
) -> str | None:
    chunk_id = str(answer.get("chunk_id", "")).strip()
    if chunk_id and chunk_id in chunk_text_by_id:
        text = chunk_text_by_id[chunk_id].strip()
        if text:
            return text
    return None


def _extract_metric_averages(result: Any) -> dict[str, float | None]:
    candidates: list[dict[str, Any]] = []

    for attr in ("scores", "aggregate_scores", "_repr_dict"):
        raw = getattr(result, attr, None)
        if isinstance(raw, dict):
            candidates.append(raw)

    for meth in ("to_dict", "dict"):
        fn = getattr(result, meth, None)
        if callable(fn):
            try:
                raw = fn()
            except Exception:
                raw = None
            if isinstance(raw, dict):
                candidates.append(raw)

    to_pandas = getattr(result, "to_pandas", None)
    if callable(to_pandas):
        try:
            df = to_pandas()
            means: dict[str, float] = {}
            for key in _all_ragas_aliases():
                if hasattr(df, "columns") and key in set(getattr(df, "columns", [])):
                    numeric = getattr(df[key], "dropna", lambda: df[key])()
                    try:
                        means[key] = float(numeric.mean())
                    except Exception:
                        pass
            if means:
                candidates.append(means)
        except Exception:
            pass

    flattened = _flatten_metric_candidates(candidates)
    normalized: dict[str, float | None] = {}
    for canonical_key, aliases in _RAGAS_KEY_ALIASES.items():
        value = _first_exact_metric_value(flattened, aliases)
        if value is None:
            value = _prefix_metric_value(flattened, aliases)
        normalized[canonical_key] = value
    return normalized


def _all_ragas_aliases() -> set[str]:
    keys: set[str] = set()
    for aliases in _RAGAS_KEY_ALIASES.values():
        keys.update(aliases)
    return keys


def _flatten_metric_candidates(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for candidate in candidates:
        for key, value in candidate.items():
            normalized_key = str(key).strip()
            if not normalized_key:
                continue
            if isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    nested_name = str(nested_key).strip()
                    if nested_name and nested_name not in merged:
                        merged[nested_name] = nested_value
            if normalized_key not in merged:
                merged[normalized_key] = value
    return merged


def _first_exact_metric_value(flattened: dict[str, Any], aliases: tuple[str, ...]) -> float | None:
    for alias in aliases:
        parsed = _as_optional_float(flattened.get(alias))
        if parsed is not None:
            return parsed
    return None


def _prefix_metric_value(flattened: dict[str, Any], aliases: tuple[str, ...]) -> float | None:
    matches: list[float] = []
    seen_keys: set[str] = set()
    normalized_aliases = [str(alias).strip().lower() for alias in aliases if str(alias).strip()]
    for key, value in flattened.items():
        key_norm = str(key or "").strip().lower()
        if not key_norm or key_norm in seen_keys:
            continue
        for alias in normalized_aliases:
            if key_norm == alias:
                continue
            if key_norm.startswith(alias):
                parsed = _as_optional_float(value)
                if parsed is not None:
                    matches.append(parsed)
                    seen_keys.add(key_norm)
                break
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    return sum(matches) / len(matches)
