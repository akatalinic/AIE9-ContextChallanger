from __future__ import annotations

import logging
import os
from time import perf_counter
from typing import Any

from app.db import (
    clear_document_outputs,
    get_document,
    insert_answer,
    insert_chunks,
    insert_questions,
    list_document_reviewer_exclusion_keys,
    set_document_run_mode,
    set_document_status,
    set_refined_text,
)
from app.config import QA_TOP_K
from app.services.chunk import chunk_text
from app.services.embed_store import delete_document_indexes, index_document
from app.state import JobState, new_job_state
from app.scoring.deterministic import compute_answer_quality_score

from .agents import QAAgent, QGenAgent, SupervisorAgent

logger = logging.getLogger(__name__)
DEFAULT_Q_PER_CHUNK = int(os.getenv("QGEN_QUESTIONS_PER_CHUNK", "5"))


def _to_state_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": str(chunk.get("id", "")).strip(),
        "text": str(chunk.get("text", "")),
        "parent_id": None,
        "metadata": {
            "chunk_order": int(chunk.get("chunk_order", 0)),
        },
    }


def _to_state_question(question: dict[str, Any]) -> dict[str, Any]:
    category_meta = question.get("category", {})
    if not isinstance(category_meta, dict):
        category_meta = {}
    category = str(category_meta.get("category", "technical")).strip().lower()
    if category not in {"pricing", "terms", "availability", "procedure", "technical"}:
        category = "technical"
    risk = str(category_meta.get("risk", "low")).strip().lower()
    if risk not in {"low", "medium", "high"}:
        risk = "low"
    content_type = str(category_meta.get("content_type", "technical")).strip().lower()
    if content_type not in {"pricing", "instructions", "terms", "dates", "technical"}:
        content_type = "technical"
    hints = category_meta.get("hints", [])
    if not isinstance(hints, list):
        hints = []

    return {
        "question_id": str(question.get("id", "")).strip(),
        "chunk_id": str(question.get("chunk_id", "")).strip(),
        "question": str(question.get("question_text", "")),
        "category": category,
        "risk": risk,
        "content_type": content_type,
        "hints": [str(item).strip() for item in hints if str(item).strip()],
    }


def _to_state_answer(question_id: str, qa_mode: str, analysis: dict[str, Any], computed_score: float) -> dict[str, Any]:
    evidence = analysis.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = []
    return {
        "question_id": str(question_id).strip(),
        "mode": qa_mode if qa_mode in {"baseline", "parent"} else "parent",
        "answer": str(analysis.get("answer_text", "")),
        "category": "other",
        "reasoning": str(analysis.get("reasons", "")),
        "used_citations": [
            {"chunk_id": str(item.get("chunk_id", "")).strip()}
            for item in evidence
            if isinstance(item, dict) and str(item.get("chunk_id", "")).strip()
        ],
        # Backward-compatible for deterministic scoring helpers.
        "score": float(computed_score),
    }


def _append_step_result(
    state: JobState,
    *,
    step: str,
    status: str,
    duration_ms: float,
    details: dict[str, Any] | None = None,
) -> None:
    state["step_results"].append(
        {
            "step": step,
            "status": "ok" if status == "ok" else "failed",
            "duration_ms": float(duration_ms),
            "details": dict(details or {}),
        }
    )


def _run_prepare_node(document_id: str, selected_mode: str, state: JobState) -> None:
    state["status"] = "ingesting"
    set_document_status(document_id, "running")
    set_document_run_mode(document_id, selected_mode)
    clear_document_outputs(document_id)
    try:
        delete_document_indexes(document_id)
    except Exception:
        logger.exception("Failed to clear retrieval indexes before rerun | document_id=%s", document_id)
    logger.info(
        "Pipeline step completed | document_id=%s step=prepare selected_qa_mode=%s",
        document_id,
        selected_mode,
    )


def _run_chunk_node(
    *,
    document_id: str,
    raw_text: str,
    state: JobState,
) -> tuple[str, list[dict[str, Any]]]:
    normalized_chunks = [str(item).strip() for item in chunk_text(raw_text) if str(item).strip()]
    if not normalized_chunks:
        raise RuntimeError("Chunking produced no output.")

    # Refine stage is intentionally removed. Keep refined_text aligned with raw text.
    set_refined_text(document_id, raw_text)
    chunk_records = insert_chunks(document_id, normalized_chunks)
    state["chunks"] = [_to_state_chunk(chunk) for chunk in chunk_records]
    state["status"] = "ingested"
    return raw_text, chunk_records


def _run_index_node(document_id: str, source_text: str, chunk_records: list[dict[str, Any]]) -> None:
    index_document(
        document_id,
        chunk_records,
        source_text=source_text,
    )


def _run_qgen_node(
    *,
    supervisor: SupervisorAgent,
    document_id: str,
    chunk_records: list[dict[str, Any]],
    state: JobState,
) -> list[dict[str, Any]]:
    state["status"] = "generating_questions"
    blocked_question_keys = set(list_document_reviewer_exclusion_keys(document_id))
    batches = supervisor.dispatch(
        "qgen",
        chunk_records=chunk_records,
        questions_per_chunk=DEFAULT_Q_PER_CHUNK,
        blocked_question_keys=blocked_question_keys,
    )
    if not isinstance(batches, list):
        raise RuntimeError("QGen agent returned invalid payload.")

    by_chunk_id: dict[str, list[dict[str, Any]]] = {}
    for item in batches:
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("chunk_id", "")).strip()
        questions = item.get("questions", [])
        if not chunk_id or not isinstance(questions, list):
            continue
        by_chunk_id[chunk_id] = questions

    question_records: list[dict[str, Any]] = []
    for chunk in chunk_records:
        chunk_id = str(chunk.get("id", "")).strip()
        if not chunk_id:
            continue
        generated = by_chunk_id.get(chunk_id, [])
        inserted = insert_questions(
            document_id,
            chunk_id,
            generated,
            blocked_question_keys=blocked_question_keys,
        )
        question_records.extend(inserted)
        state["questions"].extend(_to_state_question(question) for question in inserted)
        logger.info(
            "Pipeline step progress | document_id=%s step=qgen chunk_id=%s generated=%s inserted=%s blocked_keys=%s",
            document_id,
            chunk_id,
            len(generated),
            len(inserted),
            len(blocked_question_keys),
        )

    state["status"] = "questions_ready"
    return question_records


def _run_qa_node(
    *,
    supervisor: SupervisorAgent,
    document_id: str,
    selected_mode: str,
    question_records: list[dict[str, Any]],
    state: JobState,
) -> dict[str, int]:
    state["status"] = "qa_parent_running" if selected_mode == "parent" else "qa_baseline_running"
    qa_rows = supervisor.dispatch(
        "qa",
        document_id=document_id,
        question_records=question_records,
        qa_mode=selected_mode,
        top_k=QA_TOP_K,
    )
    if not isinstance(qa_rows, list):
        raise RuntimeError("QA agent returned invalid payload.")

    question_lookup = {str(item.get("id", "")).strip(): item for item in question_records}
    answered = 0
    external_fallback_used_count = 0
    external_evidence_citation_count = 0
    for row in qa_rows:
        if not isinstance(row, dict):
            continue
        question_id = str(row.get("question_id", "")).strip()
        if not question_id:
            continue
        analysis = row.get("analysis", {})
        if not isinstance(analysis, dict):
            analysis = {}

        answer_text = str(analysis.get("answer_text", ""))
        answer_confidence = float(analysis.get("answer_confidence", 0.0))
        classification_confidence = float(analysis.get("classification_confidence", 0.0))
        problem_type = str(analysis.get("problem_type", "missing_info")).strip().lower() or "missing_info"
        reasons = str(analysis.get("reasons", ""))
        suggested_fix = str(analysis.get("suggested_fix", ""))
        evidence = analysis.get("evidence", [])
        if not isinstance(evidence, list):
            evidence = []
        context = row.get("context", [])
        context_chunks = len(context) if isinstance(context, list) else 0
        evidence_count = sum(
            1 for item in evidence if isinstance(item, dict) and str(item.get("chunk_id", "")).strip()
        )
        computed_score = compute_answer_quality_score(
            answer_confidence=answer_confidence,
            classification_confidence=classification_confidence,
            problem_type=problem_type,
            evidence_count=evidence_count,
            context_count=context_chunks,
        )
        external_fallback_used = bool(row.get("external_fallback_used"))
        if external_fallback_used:
            external_fallback_used_count += 1
        external_evidence_for_answer = sum(
            1
            for item in evidence
            if isinstance(item, dict) and str(item.get("chunk_id", "")).strip().startswith("web:")
        )
        external_evidence_citation_count += external_evidence_for_answer

        insert_answer(
            document_id=document_id,
            question_id=question_id,
            answer_text=answer_text,
            confidence=answer_confidence,
            computed_score=computed_score,
            classification_confidence=classification_confidence,
            problem_type=problem_type,
            reasons=reasons,
            suggested_fix=suggested_fix,
            evidence=evidence,
        )

        state_answer = _to_state_answer(
            question_id=question_id,
            qa_mode=selected_mode,
            analysis=analysis,
            computed_score=computed_score,
        )
        if selected_mode == "parent":
            state["answers_parent"].append(state_answer)
        else:
            state["answers_baseline"].append(state_answer)

        logger.info(
            "Pipeline step progress | document_id=%s step=qa mode=%s question_id=%s answer_confidence=%.2f computed_score=%.2f classification_confidence=%.2f problem_type=%s context_chunks=%s external_fallback_used=%s external_evidence_count=%s",
            document_id,
            selected_mode,
            question_id,
            answer_confidence,
            computed_score,
            classification_confidence,
            problem_type,
            context_chunks,
            external_fallback_used,
            external_evidence_for_answer,
        )
        answered += 1

        if question_id not in question_lookup:
            logger.warning(
                "QA output included unknown question_id | document_id=%s question_id=%s",
                document_id,
                question_id,
            )

    state["status"] = "qa_parent_done" if selected_mode == "parent" else "qa_baseline_done"
    return {
        "answered": answered,
        "external_fallback_used": external_fallback_used_count,
        "external_evidence_citations": external_evidence_citation_count,
    }


def _run_finalize_node(document_id: str, state: JobState) -> None:
    state["status"] = "finalizing"
    set_document_status(document_id, "done")
    state["status"] = "completed"


def run_document_pipeline_job(document_id: str) -> JobState:
    pipeline_started = perf_counter()
    logger.info("Pipeline orchestrator started | document_id=%s", document_id)
    doc = get_document(document_id)
    if not doc:
        state = new_job_state(job_id=document_id, source_file="")
        state["status"] = "failed"
        state["error"] = "Document not found."
        logger.warning("Pipeline orchestrator aborted because document was not found | document_id=%s", document_id)
        return state

    state = new_job_state(
        job_id=document_id,
        source_file=str(doc.get("filename", "")).strip(),
    )
    selected_mode = "parent"
    raw_text = str(doc.get("raw_text") or "")
    refined_text = raw_text
    chunk_records: list[dict[str, Any]] = []
    question_records: list[dict[str, Any]] = []

    supervisor = SupervisorAgent(
        qgen_agent=QGenAgent(),
        qa_agent=QAAgent(),
    )

    step_handlers = {
        "prepare": lambda: _run_prepare_node(document_id, selected_mode, state),
        "chunk": lambda: _run_chunk_node(
            document_id=document_id,
            raw_text=raw_text,
            state=state,
        ),
        "index": lambda: _run_index_node(document_id, refined_text, chunk_records),
        "qgen": lambda: _run_qgen_node(
            supervisor=supervisor,
            document_id=document_id,
            chunk_records=chunk_records,
            state=state,
        ),
        "qa": lambda: _run_qa_node(
            supervisor=supervisor,
            document_id=document_id,
            selected_mode=selected_mode,
            question_records=question_records,
            state=state,
        ),
        "finalize": lambda: _run_finalize_node(document_id, state),
    }

    current_step = "prepare"
    try:
        plan = supervisor.build_plan(selected_mode)
        for step in plan:
            current_step = step
            if step not in step_handlers:
                raise RuntimeError(f"Unknown pipeline step from supervisor plan: {step}")

            step_started = perf_counter()
            result = step_handlers[step]()
            step_duration = (perf_counter() - step_started) * 1000

            if step == "chunk":
                refined_text, chunk_records = result  # type: ignore[misc,assignment]
                _append_step_result(
                    state,
                    step=step,
                    status="ok",
                    duration_ms=step_duration,
                    details={
                        "raw_chars": len(raw_text),
                        "stored_refined_chars": len(refined_text),
                        "chunk_count": len(chunk_records),
                        "strategy": "raw_text_chunking",
                    },
                )
                logger.info(
                    "Pipeline step completed | document_id=%s step=chunk raw_chars=%s chunk_count=%s duration_ms=%.2f",
                    document_id,
                    len(raw_text),
                    len(chunk_records),
                    step_duration,
                )
                continue

            if step == "qgen":
                question_records = result  # type: ignore[assignment]
                _append_step_result(
                    state,
                    step=step,
                    status="ok",
                    duration_ms=step_duration,
                    details={
                        "question_count": len(question_records),
                        "agent": supervisor.qgen_agent.name,
                    },
                )
                logger.info(
                    "Pipeline step completed | document_id=%s step=qgen total_questions=%s duration_ms=%.2f",
                    document_id,
                    len(question_records),
                    step_duration,
                )
                continue

            if step == "qa":
                qa_summary = result if isinstance(result, dict) else {}
                answered = int(qa_summary.get("answered", 0))
                external_fallback_used = int(qa_summary.get("external_fallback_used", 0))
                external_evidence_citations = int(qa_summary.get("external_evidence_citations", 0))
                _append_step_result(
                    state,
                    step=step,
                    status="ok",
                    duration_ms=step_duration,
                    details={
                        "mode": selected_mode,
                        "answered": answered,
                        "external_fallback_used": external_fallback_used,
                        "external_evidence_citations": external_evidence_citations,
                        "agent": supervisor.qa_agent.name,
                    },
                )
                logger.info(
                    "Pipeline step completed | document_id=%s step=qa mode=%s answered=%s external_fallback_used=%s external_evidence_citations=%s duration_ms=%.2f",
                    document_id,
                    selected_mode,
                    answered,
                    external_fallback_used,
                    external_evidence_citations,
                    step_duration,
                )
                continue

            _append_step_result(
                state,
                step=step,
                status="ok",
                duration_ms=step_duration,
                details={},
            )
            logger.info(
                "Pipeline step completed | document_id=%s step=%s duration_ms=%.2f",
                document_id,
                step,
                step_duration,
            )

        logger.info(
            "Pipeline orchestrator finished | document_id=%s status=completed total_duration_ms=%.2f",
            document_id,
            (perf_counter() - pipeline_started) * 1000,
        )
    except Exception as exc:
        set_document_status(document_id, "failed")
        state["status"] = "failed"
        state["error"] = str(exc)
        _append_step_result(
            state,
            step=current_step,
            status="failed",
            duration_ms=0.0,
            details={"error": str(exc)},
        )
        logger.exception(
            "Pipeline orchestrator failed | document_id=%s step=%s status=failed total_duration_ms=%.2f",
            document_id,
            current_step,
            (perf_counter() - pipeline_started) * 1000,
        )
    return state
