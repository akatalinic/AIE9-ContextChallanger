from __future__ import annotations

import json
import logging
import os
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

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


def _domain(url: str) -> str:
    parsed = urlparse(_normalize_url(url))
    host = str(parsed.netloc or "").strip().lower()
    if host.startswith("www."):
        host = host[4:]
    return host


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
        # Relaxed fallback: phrase anywhere + URL anywhere still counts.
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
        logger.info("Public search skipped because TAVILY_API_KEY is not configured.")
        return []

    urls = [u for u in (reference_urls or []) if str(u or "").strip()]
    include_domains = [d for d in (_domain(u) for u in urls) if d]
    include_domains = list(dict.fromkeys(include_domains))[:5]

    payload: dict[str, Any] = {
        "api_key": api_key,
        "query": query,
        "max_results": max(1, int(max_results)),
        "search_depth": str(os.getenv("TAVILY_SEARCH_DEPTH", "basic")).strip() or "basic",
        "include_answer": False,
    }
    if include_domains:
        payload["include_domains"] = include_domains

    timeout_seconds = float(os.getenv("TAVILY_TIMEOUT_SECONDS", "20"))
    request = Request(
        url="https://api.tavily.com/search",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=max(1.0, timeout_seconds)) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        details = ""
        try:
            details = exc.read().decode("utf-8")
        except Exception:
            details = ""
        logger.warning(
            "Public search request failed | status=%s detail=%s",
            exc.code,
            details[:300],
        )
        return []
    except URLError as exc:
        logger.warning("Public search request failed | reason=%s", exc.reason)
        return []
    except Exception:
        logger.exception("Public search request failed unexpectedly.")
        return []

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        logger.warning("Public search response was not valid JSON.")
        return []

    raw_results = parsed.get("results", [])
    if not isinstance(raw_results, list):
        return []

    allowed_domains = set(include_domains)
    rows: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for idx, item in enumerate(raw_results, start=1):
        if not isinstance(item, dict):
            continue
        url = _normalize_url(str(item.get("url", "")).strip())
        if not url or url in seen_urls:
            continue
        host = _domain(url)
        if allowed_domains and host not in allowed_domains:
            continue

        title = _normalize_snippet(str(item.get("title", "")).strip(), max_chars=160)
        snippet = _normalize_snippet(str(item.get("content", "")).strip(), max_chars=360)
        if not title and not snippet:
            continue

        seen_urls.add(url)
        rows.append(
            {
                "chunk_id": f"web:{idx}",
                "text": (
                    f"External source URL: {url}\n"
                    f"Title: {title or '(no title)'}\n"
                    f"Snippet: {snippet or '(no snippet)'}"
                ),
                "source_url": url,
                "title": title,
                "snippet": snippet,
            }
        )
        if len(rows) >= max(1, int(max_results)):
            break

    logger.info(
        "Public search finished | query_chars=%s reference_domains=%s result_count=%s",
        len(query),
        len(include_domains),
        len(rows),
    )
    return rows

