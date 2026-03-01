from __future__ import annotations

import json
import logging
from typing import Any, Iterator, Sequence

from app.db import (
    delete_parent_docstore_items,
    delete_parent_docstore_namespace,
    get_parent_docstore_items,
    list_parent_docstore_keys,
    parent_docstore_namespace,
    upsert_parent_docstore_items,
)

try:
    from langchain_core.documents import Document
except Exception:  # pragma: no cover - dependency installed via uv sync
    Document = Any  # type: ignore[misc, assignment]

try:
    from langchain_core.stores import BaseStore as _BaseStore

    _DocStoreBase = _BaseStore[str, Document]  # type: ignore[index]
except Exception:  # pragma: no cover - dependency installed via uv sync
    class _DocStoreBase:  # type: ignore[no-redef]
        pass


logger = logging.getLogger(__name__)


def _json_safe(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=True, default=str))
    except Exception:
        return {}


class SQLiteParentDocStore(_DocStoreBase):
    """SQLite-backed key-value store for ParentDocumentRetriever parent docs."""

    def __init__(self, document_id: str) -> None:
        self.document_id = str(document_id or "").strip()
        self.namespace = parent_docstore_namespace(self.document_id)

    def mget(self, keys: Sequence[str]) -> list[Document | None]:
        key_list = [str(key or "").strip() for key in keys]
        payloads = get_parent_docstore_items(self.namespace, key_list)
        docs: list[Document | None] = []
        for key in key_list:
            payload = payloads.get(key)
            if not payload:
                docs.append(None)
                continue
            page_content = str(payload.get("page_content", ""))
            metadata = payload.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            try:
                docs.append(Document(page_content=page_content, metadata=metadata))
            except Exception:
                logger.exception(
                    "Failed to deserialize parent doc from SQLite store | document_id=%s key=%s",
                    self.document_id,
                    key,
                )
                docs.append(None)
        return docs

    def mset(self, key_value_pairs: Sequence[tuple[str, Document]]) -> None:
        rows: list[tuple[str, dict[str, Any]]] = []
        for key, doc in key_value_pairs:
            normalized_key = str(key or "").strip()
            if not normalized_key or doc is None:
                continue
            page_content = str(getattr(doc, "page_content", "") or "")
            raw_metadata = getattr(doc, "metadata", {}) or {}
            metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
            metadata.setdefault("document_id", self.document_id)
            metadata.setdefault("parent_doc_id", normalized_key)
            rows.append(
                (
                    normalized_key,
                    {
                        "page_content": page_content,
                        "metadata": _json_safe(metadata),
                    },
                )
            )

        upsert_parent_docstore_items(self.namespace, rows)
        if rows:
            logger.debug(
                "Parent docs stored in SQLite docstore | document_id=%s count=%s",
                self.document_id,
                len(rows),
            )

    def mdelete(self, keys: Sequence[str]) -> None:
        key_list = [str(key or "").strip() for key in keys if str(key or "").strip()]
        delete_parent_docstore_items(self.namespace, key_list)

    def yield_keys(self, prefix: str | None = None) -> Iterator[str]:
        for key in list_parent_docstore_keys(self.namespace, prefix=prefix):
            yield key

    def clear_namespace(self) -> None:
        delete_parent_docstore_namespace(self.namespace)

