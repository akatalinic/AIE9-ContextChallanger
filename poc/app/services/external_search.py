from __future__ import annotations

import json
import logging
import os
import re
from html import unescape
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlparse, urlsplit, urlunsplit
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


def _url_key(url: str) -> str:
    normalized = _normalize_url(url)
    if not normalized:
        return ""
    parsed = urlparse(normalized)
    host = str(parsed.netloc or "").strip().lower()
    if host.startswith("www."):
        host = host[4:]
    path = str(parsed.path or "/").strip()
    if not path:
        path = "/"
    if path != "/":
        path = path.rstrip("/")
    return f"{host}{path}"


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


def _normalize_page_context(text: str, max_chars: int) -> tuple[str, bool]:
    compact = " ".join(str(text or "").split()).strip()
    if not compact:
        return ("", False)
    if len(compact) <= max(200, int(max_chars)):
        return (compact, False)
    limit = max(200, int(max_chars))
    return (compact[: limit - 3] + "...", True)


def _best_matching_snippet(question_text: str, page_text: str, max_chars: int = 360) -> str:
    text = " ".join(str(page_text or "").split()).strip()
    if not text:
        return ""
    tokens = [
        token.lower()
        for token in re.findall(r"[A-Za-z0-9]{4,}", str(question_text or ""))
    ]
    # Keep order stable while removing duplicates.
    tokens = list(dict.fromkeys(tokens))[:12]
    if not tokens:
        return _normalize_snippet(text, max_chars=max_chars)

    candidates = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    if not candidates:
        return _normalize_snippet(text, max_chars=max_chars)

    best = ""
    best_score = -1
    for sentence in candidates[:400]:
        hay = sentence.lower()
        score = sum(1 for token in tokens if token in hay)
        if score > best_score:
            best_score = score
            best = sentence
    if best:
        return _normalize_snippet(best, max_chars=max_chars)
    return _normalize_snippet(text, max_chars=max_chars)


def _reference_url_candidates(url: str) -> list[str]:
    normalized = _normalize_url(url)
    if not normalized:
        return []
    candidates: list[str] = [normalized]

    parsed = urlsplit(normalized)
    host = str(parsed.netloc or "").lower()
    if "wikipedia.org" not in host:
        return candidates

    decoded_path = unquote(parsed.path or "")
    if not decoded_path:
        return candidates
    # Some source texts include Unicode dashes in wiki titles; wiki pages often use ASCII '-'.
    dash_normalized_path = (
        decoded_path.replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2212", "-")
    )
    if dash_normalized_path != decoded_path:
        reencoded_path = quote(dash_normalized_path, safe="/:_(),-'")
        retry = urlunsplit((parsed.scheme, parsed.netloc, reencoded_path, parsed.query, parsed.fragment))
        if retry and retry not in candidates:
            candidates.append(retry)

    return candidates


def _fetch_reference_page_row(
    *,
    url: str,
    question_text: str,
    timeout_seconds: float,
) -> dict[str, str] | None:
    candidates = _reference_url_candidates(url)
    if not candidates:
        return None
    last_reason = "unknown"
    for idx, candidate in enumerate(candidates, start=1):
        request = Request(
            url=candidate,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; ContextChallengerBot/1.0)",
                "Accept": "text/html,application/xhtml+xml",
            },
            method="GET",
        )
        try:
            with urlopen(request, timeout=max(1.0, timeout_seconds)) as response:
                body = response.read()
                content_type = str(response.headers.get("Content-Type", "")).lower()
        except HTTPError as exc:
            last_reason = f"http_{exc.code}"
            # Try next normalized candidate on 404; warn only after all candidates fail.
            if exc.code == 404 and idx < len(candidates):
                continue
            continue
        except URLError as exc:
            last_reason = f"urlerror:{exc.reason}"
            continue
        except Exception as exc:
            last_reason = type(exc).__name__
            continue

        if "html" not in content_type and "text" not in content_type:
            last_reason = f"content_type:{content_type or 'unknown'}"
            continue

        html = body.decode("utf-8", errors="ignore")
        if not html.strip():
            last_reason = "empty_body"
            continue
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
        raw_title = unescape(title_match.group(1)).strip() if title_match else ""

        # Basic HTML to text reduction good enough for snippet extraction.
        stripped = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", html)
        stripped = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", stripped)
        stripped = re.sub(r"(?is)<[^>]+>", " ", stripped)
        text = unescape(" ".join(stripped.split()))
        snippet = _best_matching_snippet(question_text, text, max_chars=360)
        if not snippet and not text.strip():
            last_reason = "no_snippet"
            continue
        if not snippet:
            snippet = _normalize_snippet(text, max_chars=360)

        page_max_chars = int(os.getenv("EXTERNAL_PAGE_MAX_CHARS", "18000"))
        page_context, page_truncated = _normalize_page_context(text, max_chars=page_max_chars)
        if not page_context:
            last_reason = "no_page_context"
            continue

        title = _normalize_snippet(raw_title, max_chars=160) or _normalize_snippet(candidate, max_chars=160)
        truncation_note = " (truncated)" if page_truncated else ""
        return {
            "chunk_id": "",
            "text": (
                f"External source URL: {candidate}\n"
                f"Title: {title or '(no title)'}\n"
                f"Relevant excerpt: {snippet or '(no snippet)'}\n"
                f"Page context{truncation_note}:\n{page_context}"
            ),
            "source_url": candidate,
            "title": title,
            "snippet": snippet,
        }

    logger.warning(
        "Reference URL direct fetch failed | url=%s candidates=%s reason=%s",
        _normalize_url(url),
        len(candidates),
        last_reason,
    )
    return None


def _search_tavily(
    *,
    api_key: str,
    query: str,
    max_results: int,
    include_domains: list[str],
    search_depth: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "api_key": api_key,
        "query": query,
        "max_results": max(1, int(max_results)),
        "search_depth": search_depth,
        "include_answer": False,
    }
    if include_domains:
        payload["include_domains"] = include_domains

    request = Request(
        url="https://api.tavily.com/search",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=max(1.0, timeout_seconds)) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


def _build_query_variants(question_text: str, reference_urls: list[str]) -> list[str]:
    base = str(question_text or "").strip()
    variants: list[str] = [base]
    for url in reference_urls[:2]:
        normalized = _normalize_url(url)
        if not normalized:
            continue
        variants.append(f"{base} {normalized}")
        variants.append(normalized)

    deduped: list[str] = []
    seen: set[str] = set()
    for value in variants:
        query = " ".join(str(value or "").split()).strip()
        if not query:
            continue
        key = query.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(query)
        if len(deduped) >= 6:
            break
    return deduped


def _append_tavily_rows(
    *,
    rows_exact: list[dict[str, str]],
    rows_other: list[dict[str, str]],
    seen_urls: set[str],
    raw_results: Any,
    allowed_domains: set[str],
    reference_url_keys: set[str],
) -> None:
    if not isinstance(raw_results, list):
        return
    for item in raw_results:
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
        row = {
            "chunk_id": "",
            "text": (
                f"External source URL: {url}\n"
                f"Title: {title or '(no title)'}\n"
                f"Snippet: {snippet or '(no snippet)'}"
            ),
            "source_url": url,
            "title": title,
            "snippet": snippet,
        }
        if _url_key(url) in reference_url_keys:
            rows_exact.append(row)
        else:
            rows_other.append(row)


def search_public_context(
    question_text: str,
    *,
    reference_urls: list[str] | None = None,
    max_results: int = 3,
) -> list[dict[str, str]]:
    query = str(question_text or "").strip()
    if not query:
        return []

    urls = [_normalize_url(u) for u in (reference_urls or []) if str(u or "").strip()]
    urls = [u for u in urls if u]
    # Keep order stable and avoid repeated fetches.
    urls = list(dict.fromkeys(urls))
    if not urls:
        logger.info("Public search skipped because no reference URLs were provided.")
        return []

    timeout_seconds = float(os.getenv("TAVILY_TIMEOUT_SECONDS", "20"))
    target_count = max(1, int(max_results))
    rows: list[dict[str, str]] = []
    attempted_urls = 0
    for ref_url in urls:
        attempted_urls += 1
        fetched = _fetch_reference_page_row(
            url=ref_url,
            question_text=query,
            timeout_seconds=timeout_seconds,
        )
        if not fetched:
            continue
        rows.append(fetched)
        if len(rows) >= target_count:
            break

    for idx, row in enumerate(rows, start=1):
        row["chunk_id"] = f"web:{idx}"

    logger.info(
        "Public search finished | mode=reference_url_only query_chars=%s reference_urls=%s attempted_urls=%s fetched_rows=%s result_count=%s",
        len(query),
        len(urls),
        attempted_urls,
        len(rows),
        len(rows),
    )
    return rows

