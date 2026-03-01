import logging
import os
import re


logger = logging.getLogger(__name__)
BULLET_PATTERN = re.compile(r"^\s*(?:[-*]|\u2022|\d+\.)\s+")


def _get_default_chunk_size() -> int:
    raw = os.getenv("CHUNK_SIZE", "1000")
    try:
        return max(200, int(raw))
    except (TypeError, ValueError):
        return 1000


def _get_default_overlap_chars() -> int:
    raw = os.getenv("CHUNK_OVERLAP", "200")
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 200


def _resolve_chunk_bounds(min_len: int | None, max_len: int | None) -> tuple[int, int]:
    default_size = _get_default_chunk_size()
    resolved_min = default_size if min_len is None else max(200, int(min_len))
    resolved_max = default_size if max_len is None else max(200, int(max_len))
    if resolved_max < resolved_min:
        resolved_max = resolved_min
    return (resolved_min, resolved_max)


def _extract_overlap_tail(chunk_text_value: str, overlap_chars: int) -> str:
    if overlap_chars <= 0:
        return ""
    paragraphs = [p.strip() for p in chunk_text_value.split("\n") if p.strip()]
    if not paragraphs:
        return ""

    selected: list[str] = []
    total_chars = 0
    for paragraph in reversed(paragraphs):
        selected.append(paragraph)
        total_chars += len(paragraph) + 1
        if total_chars >= overlap_chars:
            break
    selected.reverse()
    return "\n".join(selected).strip()


def _apply_overlap(chunks: list[str], overlap_chars: int) -> list[str]:
    if overlap_chars <= 0 or len(chunks) <= 1:
        return chunks

    overlapped: list[str] = [chunks[0]]
    for idx in range(1, len(chunks)):
        previous = chunks[idx - 1]
        current = chunks[idx].strip()
        tail = _extract_overlap_tail(previous, overlap_chars)
        if tail and not current.startswith(tail):
            current = f"{tail}\n{current}".strip()
        overlapped.append(current)
    return overlapped


def _merge_small_trailing_chunk(chunks: list[str], min_len: int) -> list[str]:
    if len(chunks) < 2:
        return chunks

    trailing = chunks[-1].strip()
    if len(trailing) >= max(200, int(min_len * 0.6)):
        return chunks

    merged = list(chunks[:-2])
    combined = f"{chunks[-2].rstrip()}\n{trailing}".strip()
    merged.append(combined)
    logger.info(
        "Chunking tail merge applied | trailing_chars=%s min_len=%s",
        len(trailing),
        min_len,
    )
    return merged


def _is_list_header(text: str) -> bool:
    value = str(text or "").strip()
    return bool(value) and value.endswith(":")


def _is_bullet_line(text: str) -> bool:
    return bool(BULLET_PATTERN.match(str(text or "")))


def chunk_text(
    text: str,
    min_len: int | None = None,
    max_len: int | None = None,
    overlap_chars: int | None = None,
) -> list[str]:
    resolved_min, resolved_max = _resolve_chunk_bounds(min_len, max_len)
    resolved_overlap = _get_default_overlap_chars() if overlap_chars is None else max(0, int(overlap_chars))
    logger.info(
        "Chunking started | input_chars=%s min_len=%s max_len=%s overlap_chars=%s",
        len(text),
        resolved_min,
        resolved_max,
        resolved_overlap,
    )
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    groups: list[str] = []
    bucket: list[str] = []

    def flush() -> None:
        if bucket:
            groups.append("\n".join(bucket).strip())
            bucket.clear()

    for para in paragraphs:
        projected = "\n".join(bucket + [para]).strip()
        if len(projected) > resolved_max and len("\n".join(bucket)) >= resolved_min:
            previous = bucket[-1] if bucket else ""
            # Keep list context contiguous: avoid splitting right after a list header
            # when the next paragraph is a bullet item.
            if not (_is_list_header(previous) and _is_bullet_line(para)):
                flush()
        bucket.append(para)

    flush()
    result = [g for g in groups if g]
    normalized = _merge_small_trailing_chunk(result, resolved_min)
    with_overlap = _apply_overlap(normalized, resolved_overlap)
    logger.info(
        "Chunking finished | paragraph_count=%s chunk_count=%s overlap_applied=%s",
        len(paragraphs),
        len(with_overlap),
        resolved_overlap > 0,
    )
    return with_overlap
