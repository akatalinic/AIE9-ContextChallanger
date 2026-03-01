from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Iterable

from app.config import CONFIG

from .llm_client import get_client, get_model_name
from .sqlite_parent_docstore import SQLiteParentDocStore

logger = logging.getLogger(__name__)
PARENT_ID_KEY = "parent_doc_id"

try:
    from qdrant_client import QdrantClient, models as qdrant_models
except Exception:  # pragma: no cover - dependency installed via uv sync
    QdrantClient = None  # type: ignore[assignment]
    qdrant_models = None  # type: ignore[assignment]

try:
    from langchain.retrievers import ParentDocumentRetriever
    from langchain_core.documents import Document
    from langchain_openai import OpenAIEmbeddings
    from langchain_qdrant import QdrantVectorStore
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:  # pragma: no cover - dependency installed via uv sync
    ParentDocumentRetriever = None  # type: ignore[assignment]
    Document = None  # type: ignore[assignment]
    OpenAIEmbeddings = None  # type: ignore[assignment]
    QdrantVectorStore = None  # type: ignore[assignment]
    RecursiveCharacterTextSplitter = None  # type: ignore[assignment]


def _require_qdrant() -> None:
    if QdrantClient is None or qdrant_models is None:
        raise RuntimeError("qdrant-client is not installed. Run `uv sync` to install retrieval dependencies.")


def _require_parent_retriever() -> None:
    if (
        ParentDocumentRetriever is None
        or Document is None
        or OpenAIEmbeddings is None
        or QdrantVectorStore is None
        or RecursiveCharacterTextSplitter is None
    ):
        raise RuntimeError(
            "LangChain parent-retriever dependencies are missing. "
            "Run `uv sync` to install `langchain`, `langchain-qdrant`, and `langchain-text-splitters`."
        )


@lru_cache(maxsize=1)
def _get_qdrant_client() -> Any:
    _require_qdrant()
    target = str(CONFIG.qdrant_url or ":memory:").strip() or ":memory:"
    if target.startswith(("http://", "https://")):
        client = QdrantClient(url=target)
        logger.info("Qdrant client initialized | mode=remote target=%s", target)
        return client

    if target == ":memory:":
        try:
            client = QdrantClient(location=":memory:")
        except TypeError:
            client = QdrantClient(":memory:")
        logger.info("Qdrant client initialized | mode=local-memory")
        return client

    client = QdrantClient(path=target)
    logger.info("Qdrant client initialized | mode=local path=%s", target)
    return client


def _collection_exists(client: Any, collection_name: str) -> bool:
    try:
        return bool(client.collection_exists(collection_name))
    except Exception:
        try:
            collections = client.get_collections()
            return any(getattr(item, "name", "") == collection_name for item in getattr(collections, "collections", []))
        except Exception:
            return False


def _ensure_collection(collection_name: str, vector_size: int) -> None:
    client = _get_qdrant_client()
    if _collection_exists(client, collection_name):
        return
    client.create_collection(
        collection_name=collection_name,
        vectors_config=qdrant_models.VectorParams(
            size=max(1, int(vector_size)),
            distance=qdrant_models.Distance.COSINE,
        ),
    )
    logger.info(
        "Qdrant collection created | collection=%s vector_size=%s",
        collection_name,
        vector_size,
    )


def _payload_filter(document_id: str, payload_key: str) -> Any:
    return qdrant_models.Filter(
        must=[
            qdrant_models.FieldCondition(
                key=payload_key,
                match=qdrant_models.MatchValue(value=str(document_id)),
            )
        ]
    )


def _delete_points_by_document(collection_name: str, document_id: str, *, payload_key: str = "document_id") -> None:
    client = _get_qdrant_client()
    if not _collection_exists(client, collection_name):
        return
    filt = _payload_filter(document_id, payload_key)
    try:
        selector = qdrant_models.FilterSelector(filter=filt)
        client.delete(collection_name=collection_name, points_selector=selector, wait=True)
        return
    except Exception:
        pass
    # Fallback for qdrant-client variants that accept Filter directly.
    client.delete(collection_name=collection_name, points_selector=filt, wait=True)


def _normalize_embed_vectors(vectors: list[list[float]]) -> list[list[float]]:
    normalized: list[list[float]] = []
    for vector in vectors:
        if not vector:
            normalized.append([])
            continue
        norm = sum(float(v) * float(v) for v in vector) ** 0.5
        if norm <= 0:
            normalized.append([0.0 for _ in vector])
        else:
            normalized.append([float(v) / norm for v in vector])
    return normalized


def embed_texts(texts: Iterable[str]) -> list[list[float]]:
    texts_list = [str(text or "") for text in texts]
    if not texts_list:
        return []
    logger.info("Embedding started | item_count=%s", len(texts_list))
    client = get_client()
    model = get_model_name("OPENAI_EMBEDDING_MODEL")
    vectors: list[list[float]] = []
    for idx, text in enumerate(texts_list, start=1):
        result = client.embeddings.create(model=model, input=text)
        vectors.append([float(v) for v in result.data[0].embedding])
        logger.debug("Embedding progress | item=%s chars=%s", idx, len(text))
    normalized = _normalize_embed_vectors(vectors)
    logger.info(
        "Embedding finished | item_count=%s vector_dim=%s",
        len(normalized),
        len(normalized[0]) if normalized else 0,
    )
    return normalized


def _build_parent_source_text(chunk_records: list[dict[str, Any]], source_text: str | None) -> str:
    candidate = str(source_text or "").strip()
    if candidate:
        return candidate
    return "\n\n".join(str(row.get("text", "")).strip() for row in chunk_records if str(row.get("text", "")).strip())


def _get_langchain_embeddings() -> Any:
    _require_parent_retriever()
    model = get_model_name("OPENAI_EMBEDDING_MODEL")
    try:
        return OpenAIEmbeddings(model=model)
    except TypeError:
        # Compatibility across langchain-openai versions.
        return OpenAIEmbeddings(model=model, deployment=model)


def _build_parent_vectorstore() -> Any:
    _require_parent_retriever()
    _ensure_collection(CONFIG.qdrant_collection_parent, CONFIG.embedding_dimensions)
    client = _get_qdrant_client()
    embeddings = _get_langchain_embeddings()
    kwargs = {
        "client": client,
        "collection_name": CONFIG.qdrant_collection_parent,
    }
    try:
        return QdrantVectorStore(embedding=embeddings, **kwargs)
    except TypeError:
        return QdrantVectorStore(embeddings=embeddings, **kwargs)


def _build_parent_retriever(document_id: str, top_k: int) -> Any:
    _require_parent_retriever()
    retriever = ParentDocumentRetriever(
        vectorstore=_build_parent_vectorstore(),
        docstore=SQLiteParentDocStore(document_id=document_id),
        child_splitter=RecursiveCharacterTextSplitter(
            chunk_size=CONFIG.child_chunk_size,
            chunk_overlap=CONFIG.child_chunk_overlap,
        ),
        parent_splitter=RecursiveCharacterTextSplitter(
            chunk_size=CONFIG.parent_chunk_size,
            chunk_overlap=CONFIG.parent_chunk_overlap,
        ),
        search_kwargs={
            "k": max(1, int(top_k)),
            "filter": _payload_filter(document_id, "metadata.document_id"),
        },
        id_key=PARENT_ID_KEY,
    )
    return retriever


def _index_baseline_document(document_id: str, chunks: list[dict[str, Any]]) -> None:
    if not chunks:
        raise ValueError("Cannot build baseline index for empty chunk list.")
    vectors = embed_texts([str(item.get("text", "")) for item in chunks])
    if not vectors:
        raise ValueError("No embeddings generated for baseline index.")

    _ensure_collection(CONFIG.qdrant_collection_baseline, len(vectors[0]))
    _delete_points_by_document(CONFIG.qdrant_collection_baseline, document_id, payload_key="document_id")
    client = _get_qdrant_client()

    points: list[Any] = []
    for row, vector in zip(chunks, vectors):
        chunk_id = str(row.get("id", "")).strip()
        if not chunk_id:
            continue
        text = str(row.get("text", ""))
        chunk_order = int(row.get("chunk_order", 0))
        payload = {
            "document_id": str(document_id),
            "chunk_id": chunk_id,
            "chunk_order": chunk_order,
            "text": text,
        }
        points.append(qdrant_models.PointStruct(id=chunk_id, vector=vector, payload=payload))

    if points:
        client.upsert(collection_name=CONFIG.qdrant_collection_baseline, points=points, wait=True)
    logger.info(
        "Baseline Qdrant index rebuilt | document_id=%s chunk_count=%s",
        document_id,
        len(points),
    )


def _index_parent_document(
    document_id: str,
    chunk_records: list[dict[str, Any]],
    source_text: str | None = None,
) -> None:
    _require_parent_retriever()
    source = _build_parent_source_text(chunk_records, source_text)
    if not source:
        raise ValueError("Cannot build parent retriever index from empty source text.")

    _delete_points_by_document(CONFIG.qdrant_collection_parent, document_id, payload_key="metadata.document_id")
    SQLiteParentDocStore(document_id=document_id).clear_namespace()

    retriever = _build_parent_retriever(document_id=document_id, top_k=5)
    base_doc = Document(
        page_content=source,
        metadata={
            "document_id": str(document_id),
            "retrieval_mode": "parent",
        },
    )
    retriever.add_documents([base_doc])
    logger.info(
        "Parent retriever index rebuilt | document_id=%s parent_chunk_size=%s child_chunk_size=%s",
        document_id,
        CONFIG.parent_chunk_size,
        CONFIG.child_chunk_size,
    )


def index_document(
    document_id: str,
    chunks: list[dict[str, Any]],
    source_text: str | None = None,
) -> None:
    if not chunks:
        raise ValueError("Cannot build retrieval indexes for empty chunk list.")
    logger.info(
        "Retrieval indexing started | document_id=%s chunk_count=%s",
        document_id,
        len(chunks),
    )
    _index_baseline_document(document_id, chunks)
    _index_parent_document(document_id, chunks, source_text=source_text)
    logger.info("Retrieval indexing finished | document_id=%s", document_id)


def search(document_id: str, query: str, top_k: int = 5) -> list[dict[str, Any]]:
    if not str(query or "").strip():
        return []
    logger.info(
        "Baseline Qdrant search started | document_id=%s top_k=%s query_chars=%s",
        document_id,
        top_k,
        len(query),
    )
    _ensure_collection(CONFIG.qdrant_collection_baseline, CONFIG.embedding_dimensions)
    vectors = embed_texts([query])
    if not vectors or not vectors[0]:
        return []
    client = _get_qdrant_client()
    try:
        hits_raw = client.search(
            collection_name=CONFIG.qdrant_collection_baseline,
            query_vector=vectors[0],
            query_filter=_payload_filter(document_id, "document_id"),
            limit=max(1, int(top_k)),
            with_payload=True,
        )
    except AttributeError:
        query_result = client.query_points(
            collection_name=CONFIG.qdrant_collection_baseline,
            query=vectors[0],
            query_filter=_payload_filter(document_id, "document_id"),
            limit=max(1, int(top_k)),
            with_payload=True,
        )
        hits_raw = getattr(query_result, "points", query_result)

    hits: list[dict[str, Any]] = []
    for hit in hits_raw:
        payload = getattr(hit, "payload", None) or {}
        chunk_id = str(payload.get("chunk_id", "")).strip()
        text = str(payload.get("text", ""))
        if chunk_id and text:
            row: dict[str, Any] = {"chunk_id": chunk_id, "text": text}
            score = getattr(hit, "score", None)
            try:
                if score is not None:
                    row["score"] = float(score)
            except (TypeError, ValueError):
                pass
            chunk_order = payload.get("chunk_order")
            try:
                if chunk_order is not None:
                    row["chunk_order"] = int(chunk_order)
            except (TypeError, ValueError):
                pass
            hits.append(row)
    hits.sort(
        key=lambda row: (
            -float(row.get("score", -1.0)) if isinstance(row.get("score"), (int, float)) else 1.0,
            int(row.get("chunk_order", 10**9)) if isinstance(row.get("chunk_order"), int) else 10**9,
            str(row.get("chunk_id", "")),
        )
    )
    logger.info("Baseline Qdrant search finished | document_id=%s hit_count=%s", document_id, len(hits))
    return hits


def search_parent(document_id: str, query: str, top_k: int = 5) -> list[dict[str, str]]:
    if not str(query or "").strip():
        return []
    logger.info(
        "Parent retriever search started | document_id=%s top_k=%s query_chars=%s",
        document_id,
        top_k,
        len(query),
    )
    retriever = _build_parent_retriever(document_id=document_id, top_k=top_k)
    if hasattr(retriever, "invoke"):
        docs = retriever.invoke(query)
    else:
        docs = retriever.get_relevant_documents(query)
    hits: list[dict[str, str]] = []
    seen: set[str] = set()
    for idx, doc in enumerate(docs, start=1):
        text = str(getattr(doc, "page_content", "") or "")
        if not text:
            continue
        metadata = getattr(doc, "metadata", {}) or {}
        if not isinstance(metadata, dict):
            metadata = {}
        chunk_id = str(
            metadata.get(PARENT_ID_KEY)
            or metadata.get("doc_id")
            or f"parent:{document_id}:{idx}"
        ).strip()
        if not chunk_id or chunk_id in seen:
            continue
        seen.add(chunk_id)
        hits.append({"chunk_id": chunk_id, "text": text})
    logger.info("Parent retriever search finished | document_id=%s hit_count=%s", document_id, len(hits))
    return hits


def delete_document_indexes(document_id: str) -> None:
    logger.info("Retrieval index cleanup started | document_id=%s", document_id)
    try:
        _delete_points_by_document(CONFIG.qdrant_collection_baseline, document_id, payload_key="document_id")
    except Exception:
        logger.exception("Failed to delete baseline Qdrant points | document_id=%s", document_id)
    try:
        _delete_points_by_document(CONFIG.qdrant_collection_parent, document_id, payload_key="metadata.document_id")
    except Exception:
        logger.exception("Failed to delete parent Qdrant points | document_id=%s", document_id)
    try:
        SQLiteParentDocStore(document_id=document_id).clear_namespace()
    except Exception:
        logger.exception("Failed to clear SQLite parent docstore namespace | document_id=%s", document_id)
    logger.info("Retrieval index cleanup finished | document_id=%s", document_id)
