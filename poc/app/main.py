from pathlib import Path
import logging
from time import perf_counter

from fastapi import BackgroundTasks, Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.db import (
    DEFAULT_LANGUAGE,
    LANGUAGE_LABELS,
    SUPPORTED_LANGUAGES,
    create_document,
    create_comparison_run,
    delete_comparison_run,
    delete_document,
    delete_document_reviewer_exclusion,
    get_document,
    get_document_comparison_dashboard,
    get_document_chunks,
    get_document_dashboard,
    get_document_parent_chunks,
    get_document_questions,
    list_document_gold_references,
    list_document_reviewer_exclusions,
    normalize_question_key,
    get_translations,
    init_db,
    insert_comparison_answer,
    list_documents,
    set_comparison_run_status,
    set_document_run_mode,
    set_question_reviewer_exclusion,
    upsert_document_gold_references,
    upsert_comparison_ragas,
)
from app.models import ALLOWED_CONTENT_TYPES
from app.logging_config import setup_logging
from app.config import QA_TOP_K
from app.pipeline import run_document_pipeline_job
from app.services.embed_store import delete_document_indexes, index_document, search, search_parent
from app.services.extract import extract_text
from app.services.qa import answer_with_analysis
from app.services.ragas_eval import evaluate_comparison_mode_ragas
from app.scoring import compute_dashboard_readiness


BASE_DIR = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)

app = FastAPI(title="Context Challengers PoC")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def _retrieve_context_for_mode(
    document_id: str,
    question_text: str,
    qa_mode: str,
) -> list[dict[str, str]]:
    if qa_mode == "parent":
        return search_parent(document_id, question_text, top_k=QA_TOP_K)
    return search(document_id, question_text, top_k=QA_TOP_K)


def _build_gold_reference_candidates(
    question_records: list[dict[str, object]],
    gold_rows: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    by_id = {
        str(row.get("id", "")).strip(): row
        for row in question_records
        if str(row.get("id", "")).strip()
    }
    ids_by_key: dict[str, list[str]] = {}
    for row in question_records:
        question_id = str(row.get("id", "")).strip()
        question_text = str(row.get("question_text", "")).strip()
        if not question_id or not question_text:
            continue
        key = normalize_question_key(question_text)
        if not key:
            continue
        ids_by_key.setdefault(key, []).append(question_id)

    candidates: dict[str, dict[str, object]] = {}
    for row in gold_rows:
        if not isinstance(row, dict):
            continue
        reference_text = str(row.get("reference_text", "")).strip()
        if not reference_text:
            continue
        explicit_qid = str(row.get("question_id", "")).strip()
        qkey = str(row.get("question_key", "")).strip()
        matched_ids: list[str] = []
        if explicit_qid and explicit_qid in by_id:
            matched_ids.append(explicit_qid)
        elif qkey and qkey in ids_by_key:
            matched_ids.extend(ids_by_key[qkey])

        for qid in matched_ids:
            candidates[qid] = {
                "answer_text": reference_text,
                "confidence": 1.0,
                "problem_type": "ok",
                "reference_origin": "gold_reference",
            }
    return candidates


def _get_document_or_404(document_id: str, *, warning_message: str) -> dict:
    doc = get_document(document_id)
    if not doc:
        logger.warning("%s | document_id=%s", warning_message, document_id)
        raise HTTPException(status_code=404, detail="Document not found.")
    return doc


@app.on_event("startup")
def startup() -> None:
    setup_logging()
    logger.info("Application startup initiated.")
    init_db()
    logger.info("Database initialized at startup.")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    query_lang = request.query_params.get("lang")
    cookie_lang = request.cookies.get("lang")
    selected_lang = query_lang or cookie_lang or DEFAULT_LANGUAGE
    if selected_lang not in SUPPORTED_LANGUAGES:
        selected_lang = DEFAULT_LANGUAGE
    request.state.lang = selected_lang

    started = perf_counter()
    logger.info("HTTP request started | method=%s path=%s", request.method, request.url.path)
    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (perf_counter() - started) * 1000
        logger.exception("HTTP request failed | method=%s path=%s duration_ms=%.2f", request.method, request.url.path, duration_ms)
        raise
    duration_ms = (perf_counter() - started) * 1000
    logger.info(
        "HTTP request finished | method=%s path=%s status=%s duration_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    if query_lang and query_lang in SUPPORTED_LANGUAGES and cookie_lang != query_lang:
        response.set_cookie("lang", query_lang, max_age=60 * 60 * 24 * 365, samesite="lax")
    return response


@app.get("/")
def home(request: Request):
    docs = list_documents()
    logger.info("Home dashboard loaded | documents=%s", len(docs))
    return _render_template("index.html", request, {"documents": docs})


@app.get("/api/documents/{document_id}/report")
def document_report(document_id: str):
    doc = _get_document_or_404(
        document_id,
        warning_message="Report requested for missing document",
    )
    data = get_document_dashboard(document_id)
    logger.info("JSON report generated | document_id=%s rows=%s", document_id, len(data["rows"]))
    return {"document": doc, "report": data}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    logger.info("Upload started | filename=%s content_type=%s", file.filename, file.content_type)
    content = await file.read()
    if not content:
        logger.warning("Upload failed due to empty file | filename=%s", file.filename)
        raise HTTPException(status_code=400, detail="Empty file.")
    content_type = (file.content_type or "").strip()
    if content_type not in ALLOWED_CONTENT_TYPES and not file.filename.lower().endswith((".pdf", ".docx")):
        logger.warning("Upload rejected due to unsupported file type | filename=%s content_type=%s", file.filename, content_type)
        raise HTTPException(status_code=400, detail="Only PDF and DOCX are supported.")

    try:
        raw_text = extract_text(file.filename, content_type, content)
    except Exception as exc:
        logger.exception("Text extraction failed | filename=%s", file.filename)
        raise HTTPException(status_code=400, detail=f"Extraction failed: {exc}") from exc
    if not raw_text:
        logger.warning("Upload failed because no text was extracted | filename=%s", file.filename)
        raise HTTPException(status_code=400, detail="No text extracted from document.")

    document_id = create_document(file.filename, content_type or "unknown", raw_text)
    logger.info("Upload completed | document_id=%s raw_text_chars=%s", document_id, len(raw_text))
    return RedirectResponse(url=f"/documents/{document_id}", status_code=303)


@app.get("/documents/{document_id}")
def document_dashboard(request: Request, document_id: str):
    doc = _get_document_or_404(
        document_id,
        warning_message="Dashboard requested for missing document",
    )
    data = get_document_dashboard(document_id, include_excluded=True)
    logger.info("Document dashboard loaded | document_id=%s rows=%s", document_id, len(data["rows"]))
    return _render_template(
        "document.html",
        request,
        {
            "document": doc,
            "stats": data,
            "rows": data["rows"],
            "problem_types": ["all", "ok", "missing_info", "contradiction", "ambiguous", "formatting_issue"],
        },
    )


@app.get("/documents/{document_id}/chunks")
def chunks_view(request: Request, document_id: str):
    doc = _get_document_or_404(
        document_id,
        warning_message="Chunks page requested for missing document",
    )
    chunks = get_document_parent_chunks(document_id)
    chunk_source = "parent" if chunks else "parent_missing"
    if not chunks:
        logger.warning(
            "Parent chunks unavailable for chunk page | document_id=%s",
            document_id,
        )

    logger.info(
        "Chunks page loaded | document_id=%s chunks=%s source=%s selected_mode=%s",
        document_id,
        len(chunks),
        chunk_source,
        str(doc.get("selected_qa_mode", "parent")).strip().lower(),
    )
    return _render_template(
        "chunks.html",
        request,
        {
            "document": doc,
            "chunks": chunks,
            "chunk_source": chunk_source,
        },
    )


@app.get("/documents/{document_id}/excluded-questions")
def excluded_questions_view(request: Request, document_id: str):
    doc = _get_document_or_404(
        document_id,
        warning_message="Excluded questions page requested for missing document",
    )
    exclusions = list_document_reviewer_exclusions(document_id)
    logger.info("Excluded questions page loaded | document_id=%s exclusions=%s", document_id, len(exclusions))
    return _render_template(
        "excluded_questions.html",
        request,
        {
            "document": doc,
            "exclusions": exclusions,
        },
    )


@app.post("/documents/{document_id}/excluded-questions/{exclusion_id}/delete")
def delete_excluded_question_key_route(document_id: str, exclusion_id: str):
    _get_document_or_404(
        document_id,
        warning_message="Excluded question key delete requested for missing document",
    )
    deleted = delete_document_reviewer_exclusion(document_id, exclusion_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Exclusion entry not found.")
    return RedirectResponse(url=f"/documents/{document_id}/excluded-questions", status_code=303)


@app.get("/documents/{document_id}/comparison")
def comparison_dashboard(request: Request, document_id: str, run_id: str | None = None):
    doc = _get_document_or_404(
        document_id,
        warning_message="Comparison dashboard requested for missing document",
    )
    comparison_data = get_document_comparison_dashboard(document_id, run_id=run_id)
    logger.info("Comparison dashboard loaded | document_id=%s", document_id)
    return _render_template(
        "comparison.html",
        request,
        {
            "document": doc,
            "comparison_data": comparison_data,
        },
    )


@app.get("/api/documents/{document_id}/comparison")
def comparison_report(document_id: str, run_id: str | None = None):
    doc = _get_document_or_404(
        document_id,
        warning_message="Comparison API requested for missing document",
    )
    return {"document": doc, "comparison": get_document_comparison_dashboard(document_id, run_id=run_id)}


@app.get("/api/documents/{document_id}/gold-references")
def list_gold_references_api(document_id: str):
    _get_document_or_404(
        document_id,
        warning_message="Gold-reference list requested for missing document",
    )
    rows = list_document_gold_references(document_id)
    return {"document_id": document_id, "count": len(rows), "rows": rows}


@app.post("/api/documents/{document_id}/gold-references")
def upsert_gold_references_api(
    document_id: str,
    payload: dict[str, object] = Body(...),
):
    _get_document_or_404(
        document_id,
        warning_message="Gold-reference upsert requested for missing document",
    )

    refs = payload.get("references", [])
    references = list(refs) if isinstance(refs, list) else []
    replace_existing = bool(payload.get("replace_existing", False))
    result = upsert_document_gold_references(
        document_id=document_id,
        references=references,
        replace_existing=replace_existing,
    )
    return result


@app.post("/documents/{document_id}/comparison/run")
def run_comparison(document_id: str, background_tasks: BackgroundTasks):
    _get_document_or_404(
        document_id,
        warning_message="Comparison run requested for missing document",
    )
    run_id = create_comparison_run(document_id)
    logger.info("Comparison run queued | document_id=%s run_id=%s", document_id, run_id)
    background_tasks.add_task(_run_comparison_job, document_id, run_id)
    return RedirectResponse(url=f"/documents/{document_id}/comparison?run_id={run_id}", status_code=303)


@app.post("/documents/{document_id}/comparison-runs/{run_id}/delete")
def delete_comparison_run_route(document_id: str, run_id: str):
    _get_document_or_404(
        document_id,
        warning_message="Comparison run delete requested for missing document",
    )
    delete_comparison_run(document_id, run_id)
    return RedirectResponse(url=f"/documents/{document_id}/comparison", status_code=303)


@app.post("/documents/{document_id}/delete")
def delete_document_route(document_id: str):
    _get_document_or_404(
        document_id,
        warning_message="Delete requested for missing document",
    )
    try:
        delete_document_indexes(document_id)
    except Exception:
        logger.exception("Document delete cleanup failed in retrieval layer | document_id=%s", document_id)
    delete_document(document_id)
    return RedirectResponse(url="/", status_code=303)


@app.post("/documents/{document_id}/questions/{question_id}/exclude")
def exclude_question_route(
    document_id: str,
    question_id: str,
    note: str = Form(...),
):
    _get_document_or_404(
        document_id,
        warning_message="Question exclude requested for missing document",
    )
    normalized_note = " ".join(str(note or "").split()).strip()
    if not normalized_note:
        raise HTTPException(status_code=400, detail="Exclusion note is required.")
    updated = set_question_reviewer_exclusion(
        document_id=document_id,
        question_id=question_id,
        excluded=True,
        note=normalized_note,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Question not found.")
    return RedirectResponse(url=f"/documents/{document_id}", status_code=303)


@app.post("/documents/{document_id}/questions/{question_id}/restore")
def restore_question_route(
    document_id: str,
    question_id: str,
):
    _get_document_or_404(
        document_id,
        warning_message="Question restore requested for missing document",
    )
    updated = set_question_reviewer_exclusion(
        document_id=document_id,
        question_id=question_id,
        excluded=False,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Question not found.")
    return RedirectResponse(url=f"/documents/{document_id}", status_code=303)


def _run_comparison_job(document_id: str, run_id: str) -> None:
    started = perf_counter()
    logger.info("Comparison run started | document_id=%s run_id=%s", document_id, run_id)
    set_comparison_run_status(run_id, "running")

    try:
        doc = get_document(document_id)
        if not doc:
            raise RuntimeError("Document not found.")

        chunk_records = get_document_chunks(document_id)
        question_records = get_document_questions(document_id)
        if not chunk_records:
            raise RuntimeError("No chunks found. Run the main pipeline first.")
        if not question_records:
            raise RuntimeError("No questions found. Run the main pipeline first.")

        # Use trustworthy canonical main-dashboard answers as pseudo-references for RAGAS when available.
        # Fallback remains the question text (per spec) if no suitable canonical answer exists.
        canonical_reference_candidates: dict[str, dict[str, object]] = {}
        try:
            main_dashboard = get_document_dashboard(document_id)
            for row in main_dashboard.get("rows", []):
                question_id = str(row.get("question_id", "")).strip()
                answer_text = str(row.get("answer_text", "")).strip()
                if not question_id or not answer_text:
                    continue
                canonical_reference_candidates[question_id] = {
                    "answer_text": answer_text,
                    "confidence": row.get("confidence"),
                    "problem_type": row.get("problem_type"),
                }
        except Exception:
            logger.exception("Failed to build canonical reference candidates for RAGAS | document_id=%s", document_id)
        gold_rows = list_document_gold_references(document_id)
        gold_candidates = _build_gold_reference_candidates(question_records, gold_rows)
        canonical_reference_candidates.update(gold_candidates)
        logger.info(
            "Gold references resolved for comparison run | document_id=%s resolved_question_ids=%s stored_rows=%s",
            document_id,
            len(gold_candidates),
            len(gold_rows),
        )

        # Rebuild retrieval indexes from persisted chunks/text for this process.
        index_document(
            document_id,
            chunk_records,
            source_text=str(doc.get("refined_text") or doc.get("raw_text") or ""),
        )

        mode_rows: dict[str, list[dict[str, object]]] = {"baseline": [], "parent": []}
        mode_ragas_inputs: dict[str, list[dict[str, object]]] = {"baseline": [], "parent": []}
        for mode in ("baseline", "parent"):
            mode_started = perf_counter()
            answered = 0
            for question in question_records:
                question_id = str(question.get("id", "")).strip()
                question_text = str(question.get("question_text", "")).strip()
                if not question_id or not question_text:
                    continue

                context = _retrieve_context_for_mode(
                    document_id=document_id,
                    question_text=question_text,
                    qa_mode=mode,
                )
                result = answer_with_analysis(question_text, context)
                insert_comparison_answer(
                    run_id=run_id,
                    question_id=question_id,
                    mode=mode,
                    answer_text=result.answer_text,
                    answer_confidence=result.answer_confidence,
                    classification_confidence=result.classification_confidence,
                    problem_type=result.problem_type,
                    reasons=result.reasons,
                    suggested_fix=result.suggested_fix,
                    evidence=result.evidence,
                )
                mode_rows[mode].append(
                    {
                        "question_id": question_id,
                        "answer_text": result.answer_text,
                        "confidence": result.answer_confidence,
                        "classification_confidence": result.classification_confidence,
                        "problem_type": result.problem_type,
                    }
                )
                mode_ragas_inputs[mode].append(
                    {
                        "question_id": question_id,
                        "chunk_id": str(question.get("chunk_id", "")).strip(),
                        "answer_text": result.answer_text,
                        "evidence": result.evidence,
                        "retrieved_contexts": [str(hit.get("text", "")) for hit in context if str(hit.get("text", "")).strip()],
                    }
                )
                answered += 1
                logger.info(
                    "Comparison step progress | document_id=%s run_id=%s mode=%s question_id=%s answer_confidence=%.2f classification_confidence=%.2f problem_type=%s",
                    document_id,
                    run_id,
                    mode,
                    question_id,
                    result.answer_confidence,
                    result.classification_confidence,
                    result.problem_type,
                )
            logger.info(
                "Comparison mode completed | document_id=%s run_id=%s mode=%s answered=%s duration_ms=%.2f",
                document_id,
                run_id,
                mode,
                answered,
                (perf_counter() - mode_started) * 1000,
            )
            ragas_metrics = evaluate_comparison_mode_ragas(
                question_records=question_records,
                mode_answer_rows=mode_ragas_inputs[mode],
                chunk_records=chunk_records,
                reference_candidates_by_question_id=canonical_reference_candidates,
            )
            upsert_comparison_ragas(run_id=run_id, mode=mode, metrics=ragas_metrics)
            logger.info(
                "Comparison RAGAS completed | document_id=%s run_id=%s mode=%s",
                document_id,
                run_id,
                mode,
            )

        baseline_summary = compute_dashboard_readiness(mode_rows["baseline"])
        parent_summary = compute_dashboard_readiness(mode_rows["parent"])
        summary = {
            "baseline": baseline_summary,
            "parent": parent_summary,
            "delta_readiness_score": float(parent_summary.get("readiness_score", 0.0))
            - float(baseline_summary.get("readiness_score", 0.0)),
            "question_count": len(question_records),
            "chunk_count": len(chunk_records),
            "duration_ms": (perf_counter() - started) * 1000,
        }
        set_comparison_run_status(run_id, "completed", summary=summary)
        logger.info(
            "Comparison run completed | document_id=%s run_id=%s baseline_readiness=%.2f parent_readiness=%.2f duration_ms=%.2f",
            document_id,
            run_id,
            float(baseline_summary.get("readiness_score", 0.0)),
            float(parent_summary.get("readiness_score", 0.0)),
            (perf_counter() - started) * 1000,
        )
    except Exception as exc:
        set_comparison_run_status(run_id, "failed", error=str(exc))
        logger.exception(
            "Comparison run failed | document_id=%s run_id=%s duration_ms=%.2f",
            document_id,
            run_id,
            (perf_counter() - started) * 1000,
        )


@app.post("/documents/{document_id}/run")
def run_pipeline(
    document_id: str,
    background_tasks: BackgroundTasks,
):
    _get_document_or_404(
        document_id,
        warning_message="Run requested for missing document",
    )
    selected_mode = set_document_run_mode(document_id, "parent")
    logger.info("Pipeline queued | document_id=%s selected_qa_mode=%s", document_id, selected_mode)
    background_tasks.add_task(_run_pipeline_job, document_id)
    return RedirectResponse(url=f"/documents/{document_id}", status_code=303)


def _run_pipeline_job(document_id: str) -> None:
    state = run_document_pipeline_job(document_id=document_id)
    logger.info(
        "Pipeline job completed via orchestrator | document_id=%s status=%s questions=%s chunks=%s error=%s",
        document_id,
        state.get("status"),
        len(state.get("questions", [])),
        len(state.get("chunks", [])),
        state.get("error"),
    )


def _render_template(name: str, request: Request, context: dict) -> object:
    lang = getattr(request.state, "lang", DEFAULT_LANGUAGE)
    tr = get_translations(lang)
    payload = {
        "request": request,
        "lang": lang,
        "tr": tr,
        "languages": LANGUAGE_LABELS,
    }
    payload.update(context)
    return templates.TemplateResponse(name, payload)
