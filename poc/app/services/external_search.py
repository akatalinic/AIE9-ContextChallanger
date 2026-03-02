from __future__ import annotations

import logging
import os
import re
from typing import Any

from tavily import TavilyClient

logger = logging.getLogger(__name__)

URL_PATTERN = re.compile(r"(https?://[^\s)]+|www\.[^\s)]+)", re.IGNORECASE)
TRIGGER_PHRASES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("for more info", re.compile(r"\bfor\s+more\s+info\b", re.IGNORECASE)),
    ("for more information", re.compile(r"\bfor\s+more\s+information\b", re.IGNORECASE)),
    ("check this link", re.compile(r"\bcheck\s+this\s+link\b", re.IGNORECASE)),
    ("more information on", re.compile(r"\bmore\s+information\s+on\b", re.IGNORECASE)),
    ("learn more", re.compile(r"\blearn\s+more\b", re.IGNORECASE)),
    ("details at", re.compile(r"\bdetails\s+at\b", re.IGNORECASE)),
    ("read more", re.compile(r"\bread\s+more\b", re.IGNORECASE)),
)


def _normalize_url(value: str) -> str:
    raw = str(value or "").strip().rstrip(").,;")
    if not raw:
        return ""
    if raw.lower().startswith(("http://", "https://")):
        return raw
    if raw.lower().startswith("www."):
        return f"https://{raw}"
    return raw


def extract_urls(text: str, limit: int = 20) -> list[str]:
    seen: set[str] = set()
    urls: list[str] = []
    for match in URL_PATTERN.findall(str(text or "")):
        url = _normalize_url(match)
        if not url or url in seen:
            continue
        seen.add(url)
        urls.append(url)
        if len(urls) >= max(1, int(limit)):
            break
    return urls


def detect_external_search_hints(text: str) -> dict[str, Any]:
    content = str(text or "")
    urls = extract_urls(content, limit=25)
    if not urls:
        return {
            "enabled": False,
            "reason": "no_url",
            "urls": [],
            "trigger_phrases": [],
        }

    paragraphs = [p.strip() for p in re.split(r"\n{2,}|\r\n{2,}", content) if p.strip()]
    if not paragraphs:
        paragraphs = [line.strip() for line in content.splitlines() if line.strip()]

    matched_phrases: set[str] = set()
    hinted_urls: list[str] = []
    seen_urls: set[str] = set()
    for paragraph in paragraphs:
        para_urls = extract_urls(paragraph, limit=10)
        if not para_urls:
            continue
        phrase_hits = [label for label, pattern in TRIGGER_PHRASES if pattern.search(paragraph)]
        if not phrase_hits:
            continue
        matched_phrases.update(phrase_hits)
        for url in para_urls:
            if url in seen_urls:
                continue
            seen_urls.add(url)
            hinted_urls.append(url)

    if not matched_phrases:
        for label, pattern in TRIGGER_PHRASES:
            if pattern.search(content):
                matched_phrases.add(label)
        if matched_phrases:
            hinted_urls = urls

    enabled = bool(hinted_urls and matched_phrases)
    return {
        "enabled": enabled,
        "reason": "ok" if enabled else "missing_trigger_phrase",
        "urls": hinted_urls if enabled else [],
        "trigger_phrases": sorted(matched_phrases),
    }


def _normalize_snippet(text: str, max_chars: int = 420) -> str:
    compact = " ".join(str(text or "").split()).strip()
    if not compact:
        return ""
    return compact[: max_chars - 3] + "..." if len(compact) > max_chars else compact


def search_public_context(
    question_text: str,
    *,
    reference_urls: list[str] | None = None,
    max_results: int = 3,
) -> list[dict[str, str]]:
    query = str(question_text or "").strip()
    if not query:
        return []

    api_key = str(os.getenv("TAVILY_API_KEY", "")).strip()
    if not api_key:
        logger.warning("Public extraction skipped — TAVILY_API_KEY is not configured.")
        return []

    urls: list[str] = []
    seen_urls: set[str] = set()
    for raw in reference_urls or []:
        normalized = _normalize_url(str(raw or ""))
        if not normalized or normalized in seen_urls:
            continue
        seen_urls.add(normalized)
        urls.append(normalized)
        if len(urls) >= 20:
            break
    if not urls:
        logger.info("Public extraction skipped — no reference URLs were provided.")
        return []

    extract_depth = str(os.getenv("TAVILY_EXTRACT_DEPTH", "basic")).strip().lower() or "basic"
    if extract_depth not in {"basic", "advanced"}:
        extract_depth = "basic"

    extract_format = str(os.getenv("TAVILY_EXTRACT_FORMAT", "markdown")).strip().lower() or "markdown"
    if extract_format not in {"markdown", "text"}:
        extract_format = "markdown"

    try:
        chunks_per_source = int(os.getenv("TAVILY_CHUNKS_PER_SOURCE", "3"))
    except (TypeError, ValueError):
        chunks_per_source = 3
    chunks_per_source = min(5, max(1, chunks_per_source))

    target_count = max(1, int(max_results))
    client = TavilyClient(api_key=api_key)

    try:
        response = client.extract(
            urls=urls,
            query=query,
            chunks_per_source=chunks_per_source,
            extract_depth=extract_depth,
            format=extract_format,
            include_images=False,
        )
    except Exception as exc:
        logger.warning(
            "Tavily extract failed | query_chars=%s urls=%s error=%s",
            len(query),
            len(urls),
            type(exc).__name__,
        )
        return []

    raw_results = response.get("results", []) if isinstance(response, dict) else []

    rows: list[dict[str, str]] = []
    seen_source_urls: set[str] = set()
    for item in raw_results:
        if len(rows) >= target_count:
            break
        if not isinstance(item, dict):
            continue

        url = _normalize_url(str(item.get("url", "")).strip())
        if not url or url in seen_source_urls:
            continue

        raw_content = str(item.get("raw_content", "")).strip()
        snippet = _normalize_snippet(raw_content, max_chars=420)
        if not snippet:
            continue

        seen_source_urls.add(url)
        rows.append(
            {
                "chunk_id": f"web:{len(rows) + 1}",
                "text": (
                    f"External source URL: {url}\n"
                    f"Snippet: {snippet}"
                ),
                "source_url": url,
                "snippet": snippet,
            }
        )

    logger.info(
        "Public extraction finished | mode=tavily_extract query_chars=%s reference_urls=%s result_count=%s",
        len(query),
        len(urls),
        len(rows),
    )
    return rows
