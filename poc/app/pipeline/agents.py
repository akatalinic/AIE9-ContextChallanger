from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable

from app.config import CONFIG, QA_TOP_K
from app.services.llm_client import get_client
from app.services.llm_helpers import create_chat_completion_with_fallback, safe_json_parse

from .tools import (
    answer_with_analysis_tool,
    document_external_links_tool,
    generate_questions_tool,
    invoke_tool,
    public_search_tool,
    retrieve_context_tool,
    supervisor_plan_tool,
)

logger = logging.getLogger(__name__)
DEFAULT_Q_PER_CHUNK = int(os.getenv("QGEN_QUESTIONS_PER_CHUNK", "5"))
QA_EXTERNAL_MIN_INTERNAL_HITS = int(os.getenv("QA_EXTERNAL_MIN_INTERNAL_HITS", "2"))
QA_EXTERNAL_TOP_K = int(os.getenv("QA_EXTERNAL_TOP_K", "3"))
QA_CONTEXT_PRUNE_MIN_CHARS = int(os.getenv("QA_CONTEXT_PRUNE_MIN_CHARS", "80"))
try:
    QA_TOP1_SCORE_THRESHOLD = float(os.getenv("QA_TOP1_SCORE_THRESHOLD", "0.80"))
except (TypeError, ValueError):
    QA_TOP1_SCORE_THRESHOLD = 0.80
SUPERVISOR_LLM_ENABLED = str(os.getenv("SUPERVISOR_LLM_ENABLED", "1")).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
try:
    SUPERVISOR_TIMEOUT = float(os.getenv("SUPERVISOR_TIMEOUT", "30"))
except (TypeError, ValueError):
    SUPERVISOR_TIMEOUT = 30.0
SUPERVISOR_ALLOWED_STEPS = ("prepare", "chunk", "index", "qgen", "qa", "finalize")
QA_TOOL_ALLOWED_CONTEXT_KEYS = ("chunk_id", "text", "source_url", "title", "snippet")

_WHITESPACE_PATTERN = re.compile(r"\s+")


def _normalize_context_rows(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("chunk_id", "")).strip()
        text = str(item.get("text", "")).strip()
        if not chunk_id or not text:
            continue
        row = {"chunk_id": chunk_id, "text": text}
        source_url = str(item.get("source_url", "")).strip()
        title = str(item.get("title", "")).strip()
        snippet = str(item.get("snippet", "")).strip()
        if source_url:
            row["source_url"] = source_url
        if title:
            row["title"] = title
        if snippet:
            row["snippet"] = snippet
        score = item.get("score")
        try:
            if score is not None:
                row["score"] = float(score)
        except (TypeError, ValueError):
            pass
        chunk_order = item.get("chunk_order")
        try:
            if chunk_order is not None:
                row["chunk_order"] = int(chunk_order)
        except (TypeError, ValueError):
            pass
        rows.append(row)
    rows.sort(
        key=lambda item: (
            -float(item.get("score", -1.0)) if isinstance(item.get("score"), (int, float)) else 1.0,
            int(item.get("chunk_order", 10**9)) if isinstance(item.get("chunk_order"), int) else 10**9,
            str(item.get("chunk_id", "")),
        )
    )
    return rows


def _merge_context_rows(primary: list[dict[str, Any]], fallback: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = list(primary)
    seen_ids = {str(item.get("chunk_id", "")).strip() for item in primary}
    for item in fallback:
        chunk_id = str(item.get("chunk_id", "")).strip()
        if not chunk_id or chunk_id in seen_ids:
            continue
        merged.append(item)
        seen_ids.add(chunk_id)
    return merged


def _normalize_text_for_compare(text: str) -> str:
    compact = _WHITESPACE_PATTERN.sub(" ", str(text or "").strip().lower())
    return compact


def _is_contained_fragment(small: str, big: str) -> bool:
    small_norm = _normalize_text_for_compare(small)
    big_norm = _normalize_text_for_compare(big)
    if not small_norm or not big_norm:
        return False
    if small_norm == big_norm:
        return True
    if len(small_norm) < max(40, QA_CONTEXT_PRUNE_MIN_CHARS):
        return False
    if len(big_norm) <= len(small_norm):
        return False
    if len(big_norm) - len(small_norm) < 25:
        return False
    return small_norm in big_norm


def _prune_contained_context_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) <= 1:
        return rows
    by_length = sorted(rows, key=lambda item: len(str(item.get("text", ""))), reverse=True)
    kept: list[dict[str, Any]] = []
    for candidate in by_length:
        candidate_text = str(candidate.get("text", ""))
        if not candidate_text:
            continue
        contained = any(
            _is_contained_fragment(candidate_text, str(existing.get("text", "")))
            for existing in kept
        )
        if contained:
            continue
        kept.append(candidate)
    # Prefer deterministic order for QA prompt composition.
    kept.sort(
        key=lambda item: (
            -float(item.get("score", -1.0)) if isinstance(item.get("score"), (int, float)) else 1.0,
            int(item.get("chunk_order", 10**9)) if isinstance(item.get("chunk_order"), int) else 10**9,
            str(item.get("chunk_id", "")),
        )
    )
    return kept


def _select_context_by_score_policy(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    if not rows:
        return (rows, "empty")
    top = rows[0]
    top_score = top.get("score")
    if isinstance(top_score, (int, float)) and float(top_score) >= QA_TOP1_SCORE_THRESHOLD:
        return ([top], f"top1(score={float(top_score):.3f}>=threshold={QA_TOP1_SCORE_THRESHOLD:.2f})")
    return (rows, "multi")


def _to_qa_tool_context_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    prepared: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        item: dict[str, str] = {}
        for key in QA_TOOL_ALLOWED_CONTEXT_KEYS:
            value = row.get(key)
            if value is None:
                continue
            text_value = str(value).strip()
            if text_value:
                item[key] = text_value
        if item.get("chunk_id") and item.get("text"):
            prepared.append(item)
    return prepared


def _append_external_citations(
    analysis_row: dict[str, Any],
    external_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    existing_raw = analysis_row.get("evidence", [])
    existing: list[dict[str, str]] = []
    if isinstance(existing_raw, list):
        for item in existing_raw:
            if not isinstance(item, dict):
                continue
            chunk_id = str(item.get("chunk_id", "")).strip()
            quote = str(item.get("quote", "")).strip()
            if chunk_id and quote:
                existing.append({"chunk_id": chunk_id, "quote": quote})

    existing_ids = {item["chunk_id"] for item in existing}
    for row in external_rows:
        chunk_id = str(row.get("chunk_id", "")).strip()
        if not chunk_id or chunk_id in existing_ids:
            continue
        url = str(row.get("source_url", "")).strip()
        title = str(row.get("title", "")).strip()
        snippet = str(row.get("snippet", "")).strip() or str(row.get("text", "")).strip()
        snippet = " ".join(snippet.split())
        if len(snippet) > 180:
            snippet = snippet[:177] + "..."
        quote = snippet
        if title:
            quote = f"{title}: {quote}" if quote else title
        if url:
            quote = f"{quote} ({url})" if quote else url
        quote = quote.strip()
        if not quote:
            continue
        existing.append({"chunk_id": chunk_id, "quote": quote[:240]})
        existing_ids.add(chunk_id)

    analysis_row["evidence"] = existing
    return analysis_row


@dataclass
class QGenAgent:
    name: str = "qgen_agent"
    tools: list[Any] = field(default_factory=lambda: [generate_questions_tool])

    def run(
        self,
        chunk_records: list[dict[str, Any]],
        questions_per_chunk: int = DEFAULT_Q_PER_CHUNK,
        blocked_question_keys: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        logger.info(
            "Agent run started | agent=%s chunk_count=%s questions_per_chunk=%s blocked_keys=%s",
            self.name,
            len(chunk_records),
            max(1, int(questions_per_chunk)),
            len(blocked_question_keys or set()),
        )
        started = perf_counter()
        question_batches: list[dict[str, Any]] = []
        blocked_keys = [str(value).strip() for value in sorted(blocked_question_keys or set()) if str(value).strip()]
        for chunk in chunk_records:
            chunk_id = str(chunk.get("id", "")).strip()
            chunk_text = str(chunk.get("text", ""))
            if not chunk_id or not chunk_text:
                continue
            generated = invoke_tool(
                self.tools[0],
                chunk_text_input=chunk_text,
                count=max(1, int(questions_per_chunk)),
                excluded_question_keys=blocked_keys,
            )
            normalized = list(generated) if isinstance(generated, list) else []
            question_batches.append(
                {
                    "chunk_id": chunk_id,
                    "questions": normalized,
                }
            )
        logger.info(
            "Agent run finished | agent=%s batch_count=%s duration_ms=%.2f",
            self.name,
            len(question_batches),
            (perf_counter() - started) * 1000,
        )
        return question_batches


@dataclass
class QAAgent:
    name: str = "qa_agent"
    tools: list[Any] = field(
        default_factory=lambda: [
            retrieve_context_tool,
            document_external_links_tool,
            public_search_tool,
            answer_with_analysis_tool,
        ]
    )

    def run(
        self,
        *,
        document_id: str,
        question_records: list[dict[str, Any]],
        qa_mode: str,
        top_k: int = QA_TOP_K,
    ) -> list[dict[str, Any]]:
        logger.info(
            "Agent run started | agent=%s document_id=%s mode=%s question_count=%s top_k=%s",
            self.name,
            document_id,
            qa_mode,
            len(question_records),
            max(1, int(top_k)),
        )
        started = perf_counter()
        rows: list[dict[str, Any]] = []
        raw_policy = invoke_tool(self.tools[1], document_id=document_id)
        policy = dict(raw_policy) if isinstance(raw_policy, dict) else {}
        external_enabled = bool(policy.get("enabled"))
        reference_urls = [str(u).strip() for u in (policy.get("urls", []) if isinstance(policy.get("urls"), list) else [])]
        logger.info(
            "QA external fallback policy | document_id=%s enabled=%s reason=%s linked_urls=%s trigger_phrases=%s",
            document_id,
            external_enabled,
            str(policy.get("reason", "")),
            len(reference_urls),
            ",".join(str(v) for v in (policy.get("trigger_phrases", []) if isinstance(policy.get("trigger_phrases"), list) else [])),
        )

        for question in question_records:
            question_id = str(question.get("id", "")).strip()
            question_text = str(question.get("question_text", "")).strip()
            if not question_id or not question_text:
                continue
            context = invoke_tool(
                self.tools[0],
                document_id=document_id,
                question_text=question_text,
                qa_mode=qa_mode,
                top_k=max(1, int(top_k)),
            )
            context_rows = _normalize_context_rows(context)
            context_rows, selection_reason = _select_context_by_score_policy(context_rows)
            logger.debug(
                "QA context score policy | document_id=%s question_id=%s mode=%s selected=%s rows=%s",
                document_id,
                question_id,
                qa_mode,
                selection_reason,
                len(context_rows),
            )
            external_context_rows: list[dict[str, Any]] = []
            if (
                external_enabled
                and len(context_rows) < max(1, QA_EXTERNAL_MIN_INTERNAL_HITS)
                and selection_reason == "multi"
            ):
                external_raw = invoke_tool(
                    self.tools[2],
                    question_text=question_text,
                    reference_urls=reference_urls,
                    top_k=max(1, QA_EXTERNAL_TOP_K),
                )
                external_context_rows = _normalize_context_rows(external_raw)
                context_rows = _merge_context_rows(context_rows, external_context_rows)
            context_rows, selection_reason = _select_context_by_score_policy(context_rows)
            pre_prune_count = len(context_rows)
            context_rows = _prune_contained_context_rows(context_rows)
            if len(context_rows) != pre_prune_count:
                logger.info(
                    "QA context consolidation applied | document_id=%s question_id=%s before=%s after=%s",
                    document_id,
                    question_id,
                    pre_prune_count,
                    len(context_rows),
                )
            tool_context_rows = _to_qa_tool_context_rows(context_rows)
            logger.debug(
                "QA tool context prepared | document_id=%s question_id=%s selected=%s sent_to_tool=%s",
                document_id,
                question_id,
                selection_reason,
                len(tool_context_rows),
            )
            analysis = invoke_tool(
                self.tools[3],
                question_text=question_text,
                context_chunks=tool_context_rows,
            )
            analysis_row = dict(analysis) if isinstance(analysis, dict) else {}
            if external_context_rows:
                analysis_row = _append_external_citations(analysis_row, external_context_rows)
            rows.append(
                {
                    "question_id": question_id,
                    "question_text": question_text,
                    "context": context_rows,
                    "analysis": analysis_row,
                    "external_fallback_used": bool(external_context_rows),
                }
            )
        logger.info(
            "Agent run finished | agent=%s document_id=%s mode=%s answers=%s duration_ms=%.2f",
            self.name,
            document_id,
            qa_mode,
            len(rows),
            (perf_counter() - started) * 1000,
        )
        return rows


@dataclass
class SupervisorAgent:
    qgen_agent: QGenAgent
    qa_agent: QAAgent
    name: str = "supervisor_agent"
    tools: list[Any] = field(default_factory=lambda: [supervisor_plan_tool])

    def _default_plan(self, selected_mode: str) -> list[str]:
        result = invoke_tool(self.tools[0], selected_mode=selected_mode)
        if isinstance(result, list):
            plan = [str(step).strip().lower() for step in result if str(step).strip()]
            if plan:
                return plan
        return list(SUPERVISOR_ALLOWED_STEPS)

    def _validate_plan(self, candidate_steps: list[str]) -> list[str]:
        normalized = [str(step).strip().lower() for step in candidate_steps if str(step).strip()]
        if not normalized:
            return []
        if set(normalized) != set(SUPERVISOR_ALLOWED_STEPS):
            return []
        if len(normalized) != len(SUPERVISOR_ALLOWED_STEPS):
            return []
        for idx, step in enumerate(SUPERVISOR_ALLOWED_STEPS):
            if normalized[idx] != step:
                return []
        return normalized

    def _llm_plan(self, selected_mode: str) -> list[str]:
        client = get_client()
        response = create_chat_completion_with_fallback(
            client=client,
            model=CONFIG.supervisor_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a supervisor agent for a document QA pipeline. "
                        "Return only JSON object with key 'steps' containing a list of pipeline step ids. "
                        "Allowed step ids: prepare, chunk, index, qgen, qa, finalize. "
                        "Include every allowed step exactly once and in valid execution order."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"selected_mode={selected_mode}\n"
                        "available_specialized_agents=qgen_agent,qa_agent\n"
                        "qgen_agent_tools=generate_questions_tool\n"
                        "qa_agent_tools=retrieve_context_tool,document_external_links_tool,public_search_tool,answer_with_analysis_tool\n"
                        "Return JSON only."
                    ),
                },
            ],
            temperature=0.0,
            max_output_tokens=256,
            timeout=SUPERVISOR_TIMEOUT,
            response_format={"type": "json_object"},
            log_label="Supervisor",
        )
        payload = safe_json_parse((response.choices[0].message.content or "").strip())
        raw_steps = payload.get("steps", []) if isinstance(payload, dict) else []
        if not isinstance(raw_steps, list):
            return []
        return self._validate_plan([str(step) for step in raw_steps])

    def build_plan(self, selected_mode: str) -> list[str]:
        logger.info(
            "Supervisor planning started | supervisor=%s selected_mode=%s",
            self.name,
            selected_mode,
        )
        deterministic_plan = self._default_plan(selected_mode)
        if not SUPERVISOR_LLM_ENABLED:
            logger.info(
                "Supervisor planning finished | supervisor=%s planner=deterministic plan=%s",
                self.name,
                ",".join(deterministic_plan),
            )
            return deterministic_plan

        try:
            llm_plan = self._llm_plan(selected_mode)
            if llm_plan:
                logger.info(
                    "Supervisor planning finished | supervisor=%s planner=llm model=%s plan=%s",
                    self.name,
                    CONFIG.supervisor_model,
                    ",".join(llm_plan),
                )
                return llm_plan
            logger.warning(
                "Supervisor LLM plan rejected by validator | supervisor=%s model=%s falling_back=deterministic",
                self.name,
                CONFIG.supervisor_model,
            )
        except Exception:
            logger.exception(
                "Supervisor LLM planning failed | supervisor=%s model=%s falling_back=deterministic",
                self.name,
                CONFIG.supervisor_model,
            )

        logger.info(
            "Supervisor planning finished | supervisor=%s planner=deterministic plan=%s",
            self.name,
            ",".join(deterministic_plan),
        )
        return deterministic_plan

    def dispatch(self, step: str, **kwargs: Any) -> Any:
        step_key = str(step or "").strip().lower()
        handlers: dict[str, Callable[..., Any]] = {
            "qgen": self.qgen_agent.run,
            "qa": self.qa_agent.run,
        }
        if step_key not in handlers:
            raise ValueError(f"Supervisor has no handler for step: {step}")
        handler = handlers[step_key]
        agent_name = getattr(getattr(handler, "__self__", None), "name", handler.__name__)
        logger.info(
            "Supervisor dispatch started | supervisor=%s step=%s agent=%s keys=%s",
            self.name,
            step_key,
            agent_name,
            ",".join(sorted(str(key) for key in kwargs.keys())),
        )
        started = perf_counter()
        try:
            result = handler(**kwargs)
            logger.info(
                "Supervisor dispatch finished | supervisor=%s step=%s agent=%s duration_ms=%.2f",
                self.name,
                step_key,
                agent_name,
                (perf_counter() - started) * 1000,
            )
            return result
        except Exception:
            logger.exception(
                "Supervisor dispatch failed | supervisor=%s step=%s agent=%s duration_ms=%.2f",
                self.name,
                step_key,
                agent_name,
                (perf_counter() - started) * 1000,
            )
            raise
