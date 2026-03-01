from __future__ import annotations

import json
import hashlib
import uuid
from pathlib import Path
from time import perf_counter
from typing import Any

from app.config import QA_TOP_K
from app.db import (
    get_document,
    get_document_chunks,
    normalize_question_key,
)
from app.services.chunk import chunk_text
from app.services.embed_store import index_document, search, search_parent
from app.services.extract import extract_text
from app.services.qa import answer_with_analysis
from app.services.ragas_eval import RAGAS_METRIC_KEYS, evaluate_comparison_mode_ragas


def load_gold_references(gold_file: str | Path) -> list[dict[str, str]]:
    path = Path(gold_file)
    if not path.exists():
        raise FileNotFoundError(f"Gold dataset file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        raw_refs = payload
    elif isinstance(payload, dict):
        refs = payload.get("references", [])
        raw_refs = refs if isinstance(refs, list) else []
    else:
        raw_refs = []

    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw_refs:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", item.get("question_text", ""))).strip()
        reference = str(item.get("reference", item.get("reference_text", ""))).strip()
        source = str(item.get("source", "")).strip()
        if not question or not reference:
            continue
        qkey = normalize_question_key(question)
        if not qkey or qkey in seen:
            continue
        seen.add(qkey)
        normalized.append(
            {
                "question": question,
                "reference": reference,
                "source": source,
            }
        )

    if not normalized:
        raise RuntimeError(f"No valid references found in {path}.")
    return normalized


def _ephemeral_chunks_from_text(text: str) -> list[dict[str, Any]]:
    values = chunk_text(text)
    chunks: list[dict[str, Any]] = []
    for idx, value in enumerate(values):
        chunk_text_value = str(value).strip()
        if not chunk_text_value:
            continue
        chunks.append(
            {
                # Qdrant local mode accepts UUID/int point ids; keep ephemeral ids valid UUID strings.
                "id": str(uuid.uuid4()),
                "chunk_order": idx,
                "text": chunk_text_value,
            }
        )
    return chunks


def _load_or_build_chunks(document_id: str, source_text: str) -> tuple[list[dict[str, Any]], str]:
    chunks = get_document_chunks(document_id)
    if chunks:
        return (chunks, "db")

    ephemeral = _ephemeral_chunks_from_text(source_text)
    if not ephemeral:
        raise RuntimeError(
            "No chunks available for indexing. Document has no persisted chunks and chunking fallback returned empty output."
        )
    return (ephemeral, "ephemeral")


def _extract_text_from_local_file(source_file: str | Path) -> tuple[str, str]:
    path = Path(source_file)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Source document file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        content_type = "application/pdf"
    elif suffix == ".docx":
        content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    else:
        raise RuntimeError(f"Unsupported source file format: {path.suffix}. Use .pdf or .docx")

    content = path.read_bytes()
    text = extract_text(path.name, content_type, content).strip()
    if not text:
        raise RuntimeError(f"No extractable text found in source file: {path}")
    return (text, path.name)


def _ephemeral_document_id_for_file(source_file: str | Path) -> str:
    resolved = str(Path(source_file).resolve())
    digest = hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:16]
    return f"gold-file:{digest}"


def _normalize_modes(mode: str) -> list[str]:
    value = str(mode or "both").strip().lower()
    if value == "both":
        return ["baseline", "parent"]
    if value in {"baseline", "parent"}:
        return [value]
    raise ValueError("mode must be one of: baseline, parent, both")


def _retrieve_for_mode(document_id: str, question_text: str, mode: str, top_k: int) -> list[dict[str, str]]:
    if mode == "parent":
        return search_parent(document_id, question_text, top_k=top_k)
    return search(document_id, question_text, top_k=top_k)


def run_gold_only_ragas_eval(
    *,
    document_id: str | None = None,
    source_file: str | Path | None = None,
    gold_file: str | Path,
    mode: str = "both",
    top_k: int = QA_TOP_K,
) -> dict[str, Any]:
    normalized_document_id = str(document_id or "").strip()
    normalized_source_file = str(source_file or "").strip()
    if not normalized_document_id and not normalized_source_file:
        normalized_source_file = str(Path("templatedata") / "not_a_real_service_OK.docx")

    document_name = ""
    chunk_source = "ephemeral"
    chunk_records: list[dict[str, Any]] = []
    retrieval_document_id = ""
    source_text = ""

    if normalized_document_id:
        doc = get_document(normalized_document_id)
        if not doc:
            raise RuntimeError(f"Document not found: {normalized_document_id}")
        document_name = str(doc.get("filename", ""))
        source_text = str(doc.get("refined_text") or doc.get("raw_text") or "").strip()
        if not source_text:
            raise RuntimeError("Document has no text to evaluate (both refined_text and raw_text are empty).")
        chunk_records, chunk_source = _load_or_build_chunks(normalized_document_id, source_text)
        retrieval_document_id = normalized_document_id
    else:
        source_text, document_name = _extract_text_from_local_file(normalized_source_file)
        chunk_records = _ephemeral_chunks_from_text(source_text)
        if not chunk_records:
            raise RuntimeError("Source file text could not be chunked into retrieval chunks.")
        chunk_source = "file"
        retrieval_document_id = _ephemeral_document_id_for_file(normalized_source_file)

    references = load_gold_references(gold_file)
    question_records: list[dict[str, Any]] = []
    reference_candidates: dict[str, dict[str, Any]] = {}
    for idx, item in enumerate(references, start=1):
        question_id = f"gold:{idx:04d}"
        question_text = str(item.get("question", "")).strip()
        reference_text = str(item.get("reference", "")).strip()
        if not question_text or not reference_text:
            continue
        question_records.append({"id": question_id, "question_text": question_text})
        reference_candidates[question_id] = {
            "answer_text": reference_text,
            "confidence": 1.0,
            "problem_type": "ok",
            "reference_origin": "gold_reference",
        }

    if not question_records:
        raise RuntimeError("Gold dataset contains no usable question/reference pairs.")

    index_document(retrieval_document_id, chunk_records, source_text=source_text)

    modes = _normalize_modes(mode)
    mode_results: dict[str, Any] = {}
    for eval_mode in modes:
        started = perf_counter()
        mode_answers: list[dict[str, Any]] = []
        for question_row in question_records:
            question_id = str(question_row.get("id", "")).strip()
            question_text = str(question_row.get("question_text", "")).strip()
            if not question_id or not question_text:
                continue

            contexts = _retrieve_for_mode(
                document_id=retrieval_document_id,
                question_text=question_text,
                mode=eval_mode,
                top_k=max(1, int(top_k)),
            )
            qa_result = answer_with_analysis(question_text, contexts)
            mode_answers.append(
                {
                    "question_id": question_id,
                    "answer_text": qa_result.answer_text,
                    "evidence": qa_result.evidence,
                    "retrieved_contexts": [
                        str(hit.get("text", "")).strip()
                        for hit in contexts
                        if str(hit.get("text", "")).strip()
                    ],
                }
            )

        metrics = evaluate_comparison_mode_ragas(
            question_records=question_records,
            mode_answer_rows=mode_answers,
            chunk_records=chunk_records,
            reference_candidates_by_question_id=reference_candidates,
        )
        mode_results[eval_mode] = {
            "metrics": metrics,
            "duration_ms": (perf_counter() - started) * 1000,
        }

    deltas: dict[str, float] = {}
    if "baseline" in mode_results and "parent" in mode_results:
        baseline_metrics = mode_results["baseline"]["metrics"]
        parent_metrics = mode_results["parent"]["metrics"]
        for key in RAGAS_METRIC_KEYS:
            baseline_value = baseline_metrics.get(key)
            parent_value = parent_metrics.get(key)
            if baseline_value is None or parent_value is None:
                continue
            deltas[key] = float(parent_value) - float(baseline_value)

    return {
        "document_id": normalized_document_id or None,
        "source_file": normalized_source_file or None,
        "retrieval_document_id": retrieval_document_id,
        "document_filename": document_name,
        "gold_file": str(gold_file),
        "question_count": len(question_records),
        "chunk_count": len(chunk_records),
        "chunk_source": chunk_source,
        "top_k": max(1, int(top_k)),
        "mode": mode,
        "results": mode_results,
        "deltas": deltas,
    }
