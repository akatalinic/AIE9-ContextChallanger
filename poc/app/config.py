from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class AppConfig:
    readiness_threshold: float
    supervisor_model: str
    specialist_model: str
    embedding_model: str
    embedding_dimensions: int
    qdrant_collection_baseline: str
    qdrant_collection_parent: str
    qdrant_url: str
    chunk_size: int
    chunk_overlap: int
    parent_chunk_size: int
    parent_chunk_overlap: int
    child_chunk_size: int
    child_chunk_overlap: int


def load_config() -> AppConfig:
    return AppConfig(
        readiness_threshold=_get_float("READINESS_THRESHOLD", 0.90),
        supervisor_model=os.getenv("SUPERVISOR_MODEL", "gpt-5.2"),
        specialist_model=os.getenv("SPECIALIST_MODEL", "gpt-4.1-mini"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        embedding_dimensions=_get_int("EMBEDDING_DIMENSIONS", 1536),
        qdrant_collection_baseline=os.getenv("QDRANT_COLLECTION_BASELINE", "cc_baseline"),
        qdrant_collection_parent=os.getenv("QDRANT_COLLECTION_PARENT", "cc_parent"),
        qdrant_url=os.getenv("QDRANT_URL", ":memory:"),
        chunk_size=_get_int("CHUNK_SIZE", 500),
        chunk_overlap=_get_int("CHUNK_OVERLAP", 50),
        parent_chunk_size=_get_int("PARENT_CHUNK_SIZE", 2000),
        parent_chunk_overlap=_get_int("PARENT_CHUNK_OVERLAP", 200),
        child_chunk_size=_get_int("CHILD_CHUNK_SIZE", 400),
        child_chunk_overlap=_get_int("CHILD_CHUNK_OVERLAP", 50),
    )


CONFIG = load_config()


READINESS_THRESHOLD = CONFIG.readiness_threshold
QA_TOP_K = _get_int("QA_TOP_K", 12)
