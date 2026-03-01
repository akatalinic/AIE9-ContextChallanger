from __future__ import annotations

import logging
from time import perf_counter
from typing import Any, Callable

from app.config import QA_TOP_K
from app.db import get_document
from app.services.chunk import chunk_text
from app.services.embed_store import search, search_parent
from app.services.external_search import detect_external_search_hints, search_public_context
from app.services.qa import answer_with_analysis
from app.services.qgen import generate_questions

try:
    from langchain_core.tools import BaseTool, tool
except Exception:  # pragma: no cover - optional dependency fallback
    BaseTool = Any  # type: ignore[misc, assignment]

    def tool(*args: Any, **kwargs: Any):  # type: ignore[no-redef]
        if args and callable(args[0]) and not kwargs:
            return args[0]

        def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return _decorator


logger = logging.getLogger(__name__)


def _tool_name(tool_or_func: BaseTool | Callable[..., Any]) -> str:
    for attr in ("name", "__name__"):
        value = getattr(tool_or_func, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return type(tool_or_func).__name__


def invoke_tool(tool_or_func: BaseTool | Callable[..., Any], **kwargs: Any) -> Any:
    name = _tool_name(tool_or_func)
    started = perf_counter()
    logger.info(
        "Tool invoke started | tool=%s path=%s keys=%s",
        name,
        "langchain_invoke" if hasattr(tool_or_func, "invoke") else "callable",
        ",".join(sorted(str(key) for key in kwargs.keys())),
    )
    if hasattr(tool_or_func, "invoke"):
        try:
            result = tool_or_func.invoke(kwargs)
            logger.info(
                "Tool invoke finished | tool=%s duration_ms=%.2f",
                name,
                (perf_counter() - started) * 1000,
            )
            return result
        except Exception:
            logger.exception(
                "Tool invoke failed | tool=%s duration_ms=%.2f",
                name,
                (perf_counter() - started) * 1000,
            )
            raise
    if callable(tool_or_func):
        try:
            result = tool_or_func(**kwargs)
            logger.info(
                "Tool invoke finished | tool=%s duration_ms=%.2f",
                name,
                (perf_counter() - started) * 1000,
            )
            return result
        except Exception:
            logger.exception(
                "Tool invoke failed | tool=%s duration_ms=%.2f",
                name,
                (perf_counter() - started) * 1000,
            )
            raise
    raise TypeError(f"Unsupported tool object: {type(tool_or_func)}")


@tool("chunk_text_tool")
def chunk_text_tool(text: str) -> list[str]:
    """Split refined/raw text into chunk strings suitable for retrieval indexing."""
    return chunk_text(text)


@tool("generate_questions_tool")
def generate_questions_tool(
    chunk_text_input: str,
    count: int = 5,
    excluded_question_keys: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Generate QA-ready questions for a chunk."""
    blocked = [str(value).strip() for value in (excluded_question_keys or []) if str(value).strip()]
    return generate_questions(
        chunk_text_input,
        count=max(1, int(count)),
        excluded_question_keys=blocked,
    )


@tool("retrieve_context_tool")
def retrieve_context_tool(
    document_id: str,
    question_text: str,
    qa_mode: str = "parent",
    top_k: int = QA_TOP_K,
) -> list[dict[str, Any]]:
    """Retrieve context chunks for a question in baseline or parent mode."""
    mode = str(qa_mode or "parent").strip().lower()
    if mode == "parent":
        return search_parent(document_id, question_text, top_k=max(1, int(top_k)))
    return search(document_id, question_text, top_k=max(1, int(top_k)))


@tool("document_external_links_tool")
def document_external_links_tool(document_id: str) -> dict[str, Any]:
    """Detect whether document text explicitly indicates external-link lookup should be allowed."""
    doc = get_document(document_id)
    if not doc:
        return {"enabled": False, "reason": "document_not_found", "urls": [], "trigger_phrases": []}
    text = str(doc.get("refined_text") or doc.get("raw_text") or "")
    return detect_external_search_hints(text)


@tool("public_search_tool")
def public_search_tool(
    question_text: str,
    reference_urls: list[str] | None = None,
    top_k: int = 3,
) -> list[dict[str, str]]:
    """Run public web search constrained to document-linked domains and return context rows."""
    refs = reference_urls if isinstance(reference_urls, list) else []
    return search_public_context(
        question_text,
        reference_urls=[str(u).strip() for u in refs if str(u).strip()],
        max_results=max(1, int(top_k)),
    )


@tool("answer_with_analysis_tool")
def answer_with_analysis_tool(
    question_text: str,
    context_chunks: list[dict[str, str]],
) -> dict[str, Any]:
    """Answer a question from retrieved context and return structured QA analysis fields."""
    result = answer_with_analysis(question_text, context_chunks)
    return {
        "answer_text": result.answer_text,
        "answer_confidence": float(result.answer_confidence),
        "classification_confidence": float(result.classification_confidence),
        "problem_type": result.problem_type,
        "reasons": result.reasons,
        "suggested_fix": result.suggested_fix,
        "evidence": result.evidence,
    }


@tool("supervisor_plan_tool")
def supervisor_plan_tool(selected_mode: str = "baseline") -> list[str]:
    """Return the pipeline execution plan for the supervisor agent."""
    _ = selected_mode
    return ["prepare", "chunk", "index", "qgen", "qa", "finalize"]
