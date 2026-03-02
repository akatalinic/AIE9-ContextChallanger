import json
import logging
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from app.reporting import build_comparison_report
from app.scoring import compute_dashboard_readiness

DB_PATH = Path(__file__).resolve().parents[1] / "poc.db"
logger = logging.getLogger(__name__)
SUPPORTED_LANGUAGES = ("hr", "en", "de", "sk", "hu")
DEFAULT_LANGUAGE = "en"
SUPPORTED_QA_MODES = ("baseline", "parent")
LANGUAGE_LABELS = {
    "hr": "Hrvatski",
    "en": "English",
    "de": "Deutsch",
    "sk": "Slovensky",
    "hu": "Magyar",
}

TRANSLATION_SEED: dict[str, dict[str, str]] = {
    "en": {
        "common.home": "Home",
        "common.dashboard": "Dashboard",
        "common.chunks": "Chunks",
        "common.poc": "PoC",
        "common.status": "Status",
        "common.created": "Created",
        "common.action": "Action",
        "common.filters": "Filters",
        "common.search_question": "Search question",
        "common.language": "Language",
        "home.title": "Document validation dashboard",
        "home.subtitle": "Upload PDF/DOCX and run AI pipeline for generated questions, grounded answers, and issue classification.",
        "home.upload_title": "Upload document",
        "home.upload_subtitle": "Supported formats: PDF and DOCX",
        "home.upload_btn": "Upload",
        "home.docs_title": "Documents",
        "home.docs_subtitle": "Track status, generated questions, and problematic answers.",
        "home.empty_docs": "No documents uploaded yet.",
        "home.open_dashboard": "Open dashboard",
        "home.filename": "Filename",
        "home.chunks": "Chunks",
        "home.questions": "Questions",
        "home.problems": "Problems",
        "doc.context_label": "You are in document workspace",
        "doc.id": "Document ID",
        "doc.run_pipeline": "Run pipeline",
        "doc.kpi_chunks": "Chunks",
        "doc.kpi_questions": "Questions",
        "doc.kpi_problematic": "Problematic Answers",
        "doc.kpi_readiness": "Doc Readiness",
        "doc.kpi_avg_confidence": "Avg Confidence",
        "doc.filters_subtitle": "Filter by problem type, confidence threshold, and question text.",
        "doc.problem_type": "Problem type",
        "doc.confidence_lt": "Confidence <",
        "doc.visible_rows": "Visible rows",
        "doc.results_title": "Q&A Results",
        "doc.results_subtitle": "Grounded answers with confidence and remediation guidance.",
        "doc.question": "Question",
        "doc.question_type": "Question Type",
        "doc.answer": "Answer",
        "doc.confidence": "Confidence",
        "doc.computed_score": "Computed Score",
        "doc.problem_type_col": "Problem Type",
        "doc.suggested_fix": "Suggested Fix",
        "doc.evidence": "Evidence",
        "doc.reasons": "Reasons",
        "doc.view": "View",
        "doc.empty_rows": "No Q&A data yet. Run pipeline to generate results.",
        "chunk.context_label": "You are in chunk explorer",
        "chunk.title": "Chunks for",
        "chunk.total": "Total chunks",
        "chunk.back": "Back to dashboard",
        "chunk.empty": "No chunks available. Run pipeline first.",
    },
    "hr": {
        "common.home": "Početna",
        "common.dashboard": "Nadzorna ploča",
        "common.chunks": "Chunkovi",
        "common.poc": "PoC",
        "common.status": "Status",
        "common.created": "Kreirano",
        "common.action": "Akcija",
        "common.filters": "Filteri",
        "common.search_question": "Pretraga pitanja",
        "common.language": "Jezik",
        "home.title": "Nadzor validacije dokumenata",
        "home.subtitle": "Prenesi PDF/DOCX i pokreni AI pipeline za generiranje pitanja, utemeljene odgovore i klasifikaciju problema.",
        "home.upload_title": "Upload dokumenta",
        "home.upload_subtitle": "Podržani formati: PDF i DOCX",
        "home.upload_btn": "Upload",
        "home.docs_title": "Dokumenti",
        "home.docs_subtitle": "Prati status, broj pitanja i problematične odgovore.",
        "home.empty_docs": "Nema učitanih dokumenata.",
        "home.open_dashboard": "Otvori dashboard",
        "home.filename": "Naziv datoteke",
        "home.chunks": "Chunkovi",
        "home.questions": "Pitanja",
        "home.problems": "Problemi",
        "doc.context_label": "Nalazite se u radnom prostoru dokumenta",
        "doc.id": "ID dokumenta",
        "doc.run_pipeline": "Pokreni pipeline",
        "doc.kpi_chunks": "Chunkovi",
        "doc.kpi_questions": "Pitanja",
        "doc.kpi_problematic": "Problematični odgovori",
        "doc.kpi_avg_confidence": "Prosječna pouzdanost",
        "doc.filters_subtitle": "Filtriraj po vrsti problema, pragu confidence i tekstu pitanja.",
        "doc.problem_type": "Vrsta problema",
        "doc.confidence_lt": "Confidence <",
        "doc.visible_rows": "Vidljivi redovi",
        "doc.results_title": "Q&A rezultati",
        "doc.results_subtitle": "Utemeljeni odgovori s confidence vrijednostima i prijedlogom popravka.",
        "doc.question": "Pitanje",
        "doc.answer": "Odgovor",
        "doc.confidence": "Confidence",
        "doc.problem_type_col": "Vrsta problema",
        "doc.suggested_fix": "Predloženi popravak",
        "doc.evidence": "Dokazi",
        "doc.reasons": "Razlozi",
        "doc.view": "Prikaži",
        "doc.empty_rows": "Još nema Q&A podataka. Pokreni pipeline za rezultate.",
        "chunk.context_label": "Nalazite se u pregledu chunkova",
        "chunk.title": "Chunkovi za",
        "chunk.total": "Ukupno chunkova",
        "chunk.back": "Natrag na dashboard",
        "chunk.empty": "Nema chunkova. Prvo pokreni pipeline.",
    },
    "de": {
        "common.home": "Startseite",
        "common.dashboard": "Dashboard",
        "common.chunks": "Chunks",
        "common.poc": "PoC",
        "common.status": "Status",
        "common.created": "Erstellt",
        "common.action": "Aktion",
        "common.filters": "Filter",
        "common.search_question": "Frage suchen",
        "common.language": "Sprache",
        "home.title": "Dashboard zur Dokumentenvalidierung",
        "home.subtitle": "Laden Sie PDF/DOCX hoch und starten Sie die AI-Pipeline für Fragen, Antworten und Problemklassifikation.",
        "home.upload_title": "Dokument hochladen",
        "home.upload_subtitle": "Unterstützte Formate: PDF und DOCX",
        "home.upload_btn": "Hochladen",
        "home.docs_title": "Dokumente",
        "home.docs_subtitle": "Status, Fragen und problematische Antworten verfolgen.",
        "home.empty_docs": "Noch keine Dokumente hochgeladen.",
        "home.open_dashboard": "Dashboard öffnen",
        "home.filename": "Dateiname",
        "home.chunks": "Chunks",
        "home.questions": "Fragen",
        "home.problems": "Probleme",
        "doc.context_label": "Sie befinden sich im Dokument-Arbeitsbereich",
        "doc.id": "Dokument-ID",
        "doc.run_pipeline": "Pipeline starten",
        "doc.kpi_chunks": "Chunks",
        "doc.kpi_questions": "Fragen",
        "doc.kpi_problematic": "Problematische Antworten",
        "doc.kpi_avg_confidence": "Durchschn. Confidence",
        "doc.filters_subtitle": "Nach Problemtyp, Confidence-Schwelle und Fragetext filtern.",
        "doc.problem_type": "Problemtyp",
        "doc.confidence_lt": "Confidence <",
        "doc.visible_rows": "Sichtbare Zeilen",
        "doc.results_title": "Q&A Ergebnisse",
        "doc.results_subtitle": "Belegte Antworten mit Confidence und Korrekturvorschlägen.",
        "doc.question": "Frage",
        "doc.answer": "Antwort",
        "doc.confidence": "Confidence",
        "doc.problem_type_col": "Problemtyp",
        "doc.suggested_fix": "Korrekturvorschlag",
        "doc.evidence": "Belege",
        "doc.reasons": "Begründung",
        "doc.view": "Anzeigen",
        "doc.empty_rows": "Noch keine Q&A-Daten. Starten Sie die Pipeline.",
        "chunk.context_label": "Sie befinden sich im Chunk-Explorer",
        "chunk.title": "Chunks für",
        "chunk.total": "Chunks gesamt",
        "chunk.back": "Zurück zum Dashboard",
        "chunk.empty": "Keine Chunks verfügbar. Pipeline zuerst ausführen.",
    },
    "sk": {
        "common.home": "Domov",
        "common.dashboard": "Dashboard",
        "common.chunks": "Chunky",
        "common.poc": "PoC",
        "common.status": "Stav",
        "common.created": "Vytvorené",
        "common.action": "Akcia",
        "common.filters": "Filtre",
        "common.search_question": "Hľadať otázku",
        "common.language": "Jazyk",
        "home.title": "Dashboard validácie dokumentov",
        "home.subtitle": "Nahrajte PDF/DOCX a spustite AI pipeline pre otázky, odpovede a klasifikáciu problémov.",
        "home.upload_title": "Nahranie dokumentu",
        "home.upload_subtitle": "Podporované formáty: PDF a DOCX",
        "home.upload_btn": "Nahrať",
        "home.docs_title": "Dokumenty",
        "home.docs_subtitle": "Sledujte stav, otázky a problematické odpovede.",
        "home.empty_docs": "Zatiaľ nebol nahraný žiadny dokument.",
        "home.open_dashboard": "Otvoriť dashboard",
        "home.filename": "Názov súboru",
        "home.chunks": "Chunky",
        "home.questions": "Otázky",
        "home.problems": "Problémy",
        "doc.context_label": "Nachádzate sa v pracovnom priestore dokumentu",
        "doc.id": "ID dokumentu",
        "doc.run_pipeline": "Spustiť pipeline",
        "doc.kpi_chunks": "Chunky",
        "doc.kpi_questions": "Otázky",
        "doc.kpi_problematic": "Problematické odpovede",
        "doc.kpi_avg_confidence": "Priemerná confidence",
        "doc.filters_subtitle": "Filtrujte podľa typu problému, confidence prahu a textu otázky.",
        "doc.problem_type": "Typ problému",
        "doc.confidence_lt": "Confidence <",
        "doc.visible_rows": "Viditeľné riadky",
        "doc.results_title": "Q&A výsledky",
        "doc.results_subtitle": "Odpovede s dôkazmi, confidence a návrhmi opráv.",
        "doc.question": "Otázka",
        "doc.answer": "Odpoveď",
        "doc.confidence": "Confidence",
        "doc.problem_type_col": "Typ problému",
        "doc.suggested_fix": "Navrhovaná oprava",
        "doc.evidence": "Dôkazy",
        "doc.reasons": "Dôvody",
        "doc.view": "Zobraziť",
        "doc.empty_rows": "Zatiaľ nie sú Q&A dáta. Spustite pipeline.",
        "chunk.context_label": "Nachádzate sa v prehliadači chunkov",
        "chunk.title": "Chunky pre",
        "chunk.total": "Spolu chunkov",
        "chunk.back": "Späť na dashboard",
        "chunk.empty": "Chunky nie sú dostupné. Najprv spustite pipeline.",
    },
    "hu": {
        "common.home": "Főoldal",
        "common.dashboard": "Dashboard",
        "common.chunks": "Chunkok",
        "common.poc": "PoC",
        "common.status": "Státusz",
        "common.created": "Létrehozva",
        "common.action": "Művelet",
        "common.filters": "Szűrők",
        "common.search_question": "Kérdés keresése",
        "common.language": "Nyelv",
        "home.title": "Dokumentum-validációs dashboard",
        "home.subtitle": "Tölts fel PDF/DOCX fájlt és futtasd az AI pipeline-t kérdés, válasz és hibaklasszifikáció céljából.",
        "home.upload_title": "Dokumentum feltöltése",
        "home.upload_subtitle": "Támogatott formátumok: PDF és DOCX",
        "home.upload_btn": "Feltöltés",
        "home.docs_title": "Dokumentumok",
        "home.docs_subtitle": "Kövesd a státuszt, kérdéseket és problémás válaszokat.",
        "home.empty_docs": "Még nincs feltöltött dokumentum.",
        "home.open_dashboard": "Dashboard megnyitása",
        "home.filename": "Fájlnév",
        "home.chunks": "Chunkok",
        "home.questions": "Kérdések",
        "home.problems": "Problémák",
        "doc.context_label": "A dokumentum munkaterületen vagy",
        "doc.id": "Dokumentum azonosító",
        "doc.run_pipeline": "Pipeline futtatása",
        "doc.kpi_chunks": "Chunkok",
        "doc.kpi_questions": "Kérdések",
        "doc.kpi_problematic": "Problémás válaszok",
        "doc.kpi_avg_confidence": "Átlagos confidence",
        "doc.filters_subtitle": "Szűrés probléma típus, confidence küszöb és kérdésszöveg szerint.",
        "doc.problem_type": "Probléma típusa",
        "doc.confidence_lt": "Confidence <",
        "doc.visible_rows": "Látható sorok",
        "doc.results_title": "Q&A eredmények",
        "doc.results_subtitle": "Forrásolt válaszok confidence értékkel és javítási javaslattal.",
        "doc.question": "Kérdés",
        "doc.answer": "Válasz",
        "doc.confidence": "Confidence",
        "doc.problem_type_col": "Probléma típusa",
        "doc.suggested_fix": "Javasolt javítás",
        "doc.evidence": "Bizonyíték",
        "doc.reasons": "Indoklás",
        "doc.view": "Megnéz",
        "doc.empty_rows": "Még nincs Q&A adat. Futtasd a pipeline-t.",
        "chunk.context_label": "A chunk böngészőben vagy",
        "chunk.title": "Chunkok ehhez:",
        "chunk.total": "Összes chunk",
        "chunk.back": "Vissza a dashboardra",
        "chunk.empty": "Nincsenek chunkok. Előbb futtasd a pipeline-t.",
    },
}


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_QUESTION_KEY_WHITESPACE = re.compile(r"\s+")


def normalize_question_key(text: str) -> str:
    compact = _QUESTION_KEY_WHITESPACE.sub(" ", str(text or "").strip().lower())
    return compact


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(str(row["name"]) == column for row in rows)


def _ensure_documents_mode_column(conn: sqlite3.Connection) -> None:
    if _column_exists(conn, "documents", "selected_qa_mode"):
        return
    conn.execute("ALTER TABLE documents ADD COLUMN selected_qa_mode TEXT NOT NULL DEFAULT 'parent'")
    logger.info("Schema migration applied | table=documents column=selected_qa_mode")


def _ensure_answers_classification_confidence_column(conn: sqlite3.Connection) -> None:
    if _column_exists(conn, "answers", "classification_confidence"):
        return
    conn.execute("ALTER TABLE answers ADD COLUMN classification_confidence REAL")
    logger.info("Schema migration applied | table=answers column=classification_confidence")


def _ensure_answers_computed_score_column(conn: sqlite3.Connection) -> None:
    if _column_exists(conn, "answers", "computed_score"):
        return
    conn.execute("ALTER TABLE answers ADD COLUMN computed_score REAL")
    conn.execute("UPDATE answers SET computed_score = confidence WHERE computed_score IS NULL")
    logger.info("Schema migration applied | table=answers column=computed_score")


def _ensure_questions_reviewer_exclusion_columns(conn: sqlite3.Connection) -> None:
    changed = False
    if not _column_exists(conn, "questions", "reviewer_excluded"):
        conn.execute("ALTER TABLE questions ADD COLUMN reviewer_excluded INTEGER NOT NULL DEFAULT 0")
        changed = True
    if not _column_exists(conn, "questions", "reviewer_excluded_note"):
        conn.execute("ALTER TABLE questions ADD COLUMN reviewer_excluded_note TEXT")
        changed = True
    if not _column_exists(conn, "questions", "reviewer_excluded_at"):
        conn.execute("ALTER TABLE questions ADD COLUMN reviewer_excluded_at TEXT")
        changed = True
    if changed:
        logger.info(
            "Schema migration applied | table=questions columns=reviewer_excluded,reviewer_excluded_note,reviewer_excluded_at"
        )


def _ensure_reviewer_question_exclusion_table(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS reviewer_question_exclusions (
          id TEXT PRIMARY KEY,
          document_id TEXT NOT NULL,
          question_text TEXT,
          question_key TEXT NOT NULL,
          note TEXT,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          UNIQUE(document_id, question_key),
          FOREIGN KEY(document_id) REFERENCES documents(id)
        );

        CREATE INDEX IF NOT EXISTS idx_reviewer_question_exclusions_document
          ON reviewer_question_exclusions (document_id);
        """
    )
    if not _column_exists(conn, "reviewer_question_exclusions", "question_text"):
        conn.execute("ALTER TABLE reviewer_question_exclusions ADD COLUMN question_text TEXT")
        logger.info("Schema migration applied | table=reviewer_question_exclusions column=question_text")


def _ensure_comparison_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS comparison_runs (
          id TEXT PRIMARY KEY,
          document_id TEXT NOT NULL,
          status TEXT NOT NULL,
          created_at TEXT NOT NULL,
          started_at TEXT,
          completed_at TEXT,
          error TEXT,
          summary_json TEXT,
          FOREIGN KEY(document_id) REFERENCES documents(id)
        );

        CREATE TABLE IF NOT EXISTS comparison_answers (
          id TEXT PRIMARY KEY,
          run_id TEXT NOT NULL,
          question_id TEXT NOT NULL,
          mode TEXT NOT NULL,
          answer_text TEXT NOT NULL,
          answer_confidence REAL NOT NULL,
          classification_confidence REAL,
          problem_type TEXT NOT NULL,
          reasons TEXT,
          suggested_fix TEXT,
          evidence_json TEXT,
          created_at TEXT NOT NULL,
          FOREIGN KEY(run_id) REFERENCES comparison_runs(id),
          FOREIGN KEY(question_id) REFERENCES questions(id),
          UNIQUE(run_id, question_id, mode)
        );

        CREATE TABLE IF NOT EXISTS comparison_ragas (
          id TEXT PRIMARY KEY,
          run_id TEXT NOT NULL,
          mode TEXT NOT NULL,
          metrics_json TEXT NOT NULL,
          created_at TEXT NOT NULL,
          FOREIGN KEY(run_id) REFERENCES comparison_runs(id),
          UNIQUE(run_id, mode)
        );

        CREATE INDEX IF NOT EXISTS idx_comparison_runs_document_created
          ON comparison_runs (document_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_comparison_answers_run_mode
          ON comparison_answers (run_id, mode);
        """
    )


def _ensure_retriever_docstore_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS retriever_parent_docs (
          namespace TEXT NOT NULL,
          key TEXT NOT NULL,
          payload_json TEXT NOT NULL,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          PRIMARY KEY(namespace, key)
        );

        CREATE INDEX IF NOT EXISTS idx_retriever_parent_docs_namespace
          ON retriever_parent_docs (namespace);
        """
    )


def _ensure_gold_reference_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS gold_references (
          id TEXT PRIMARY KEY,
          document_id TEXT NOT NULL,
          question_id TEXT,
          question_text TEXT,
          question_key TEXT NOT NULL,
          reference_text TEXT NOT NULL,
          source TEXT,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          FOREIGN KEY(document_id) REFERENCES documents(id),
          FOREIGN KEY(question_id) REFERENCES questions(id),
          UNIQUE(document_id, question_key)
        );

        CREATE INDEX IF NOT EXISTS idx_gold_references_document
          ON gold_references (document_id);
        CREATE INDEX IF NOT EXISTS idx_gold_references_document_question
          ON gold_references (document_id, question_id);
        """
    )


def _enforce_parent_document_mode(conn: sqlite3.Connection) -> None:
    cur = conn.execute(
        "UPDATE documents SET selected_qa_mode = 'parent' WHERE selected_qa_mode IS NULL OR selected_qa_mode != 'parent'"
    )
    if int(cur.rowcount or 0) > 0:
        logger.info("Document mode normalized | updated_rows=%s selected_qa_mode=parent", cur.rowcount)


def normalize_qa_mode(mode: str | None) -> str:
    candidate = str(mode or "parent").strip().lower()
    return candidate if candidate in SUPPORTED_QA_MODES else "parent"


def init_db() -> None:
    logger.info("Initializing SQLite schema | db_path=%s", DB_PATH)
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
              id TEXT PRIMARY KEY,
              filename TEXT NOT NULL,
              content_type TEXT NOT NULL,
              created_at TEXT NOT NULL,
              status TEXT NOT NULL,
              raw_text TEXT,
              refined_text TEXT,
              selected_qa_mode TEXT NOT NULL DEFAULT 'parent'
            );

            CREATE TABLE IF NOT EXISTS chunks (
              id TEXT PRIMARY KEY,
              document_id TEXT NOT NULL,
              chunk_order INTEGER NOT NULL,
              text TEXT NOT NULL,
              FOREIGN KEY(document_id) REFERENCES documents(id)
            );

            CREATE TABLE IF NOT EXISTS questions (
              id TEXT PRIMARY KEY,
              document_id TEXT NOT NULL,
              chunk_id TEXT NOT NULL,
              question_no INTEGER NOT NULL,
              question_text TEXT NOT NULL,
              category TEXT,
              FOREIGN KEY(document_id) REFERENCES documents(id),
              FOREIGN KEY(chunk_id) REFERENCES chunks(id)
            );

            CREATE TABLE IF NOT EXISTS answers (
              id TEXT PRIMARY KEY,
              document_id TEXT NOT NULL,
              question_id TEXT NOT NULL,
              answer_text TEXT NOT NULL,
              confidence REAL NOT NULL,
              computed_score REAL,
              classification_confidence REAL,
              problem_type TEXT NOT NULL,
              reasons TEXT,
              suggested_fix TEXT,
              evidence_json TEXT,
              FOREIGN KEY(document_id) REFERENCES documents(id),
              FOREIGN KEY(question_id) REFERENCES questions(id)
            );

            CREATE TABLE IF NOT EXISTS i18n_translations (
              lang TEXT NOT NULL,
              key TEXT NOT NULL,
              value TEXT NOT NULL,
              PRIMARY KEY (lang, key)
            );
            """
        )
        _ensure_documents_mode_column(conn)
        _ensure_answers_classification_confidence_column(conn)
        _ensure_answers_computed_score_column(conn)
        _ensure_questions_reviewer_exclusion_columns(conn)
        _ensure_reviewer_question_exclusion_table(conn)
        _ensure_comparison_tables(conn)
        _ensure_retriever_docstore_tables(conn)
        _ensure_gold_reference_tables(conn)
        _enforce_parent_document_mode(conn)
        _seed_translations(conn)


def _seed_translations(conn: sqlite3.Connection) -> None:
    for lang, values in TRANSLATION_SEED.items():
        for key, value in values.items():
            conn.execute(
                """
                INSERT INTO i18n_translations (lang, key, value)
                VALUES (?, ?, ?)
                ON CONFLICT(lang, key) DO UPDATE SET value = excluded.value
                """,
                (lang, key, value),
            )
    logger.info("i18n translations seeded | languages=%s keys=%s", len(TRANSLATION_SEED), len(TRANSLATION_SEED["en"]))


def get_translations(lang: str) -> dict[str, str]:
    selected = lang if lang in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE
    with get_conn() as conn:
        fallback_rows = conn.execute(
            "SELECT key, value FROM i18n_translations WHERE lang = ?",
            (DEFAULT_LANGUAGE,),
        ).fetchall()
        rows = conn.execute(
            "SELECT key, value FROM i18n_translations WHERE lang = ?",
            (selected,),
        ).fetchall()

    result = {row["key"]: row["value"] for row in fallback_rows}
    result.update({row["key"]: row["value"] for row in rows})
    logger.debug("Translations fetched | lang=%s key_count=%s", selected, len(result))
    return result


def create_document(filename: str, content_type: str, raw_text: str) -> str:
    document_id = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO documents (id, filename, content_type, created_at, status, raw_text, refined_text, selected_qa_mode)
            VALUES (?, ?, ?, ?, 'uploaded', ?, NULL, ?)
            """,
            (document_id, filename, content_type, now_iso(), raw_text, "parent"),
        )
    logger.info("Document created | document_id=%s filename=%s raw_chars=%s", document_id, filename, len(raw_text))
    return document_id


def list_documents() -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT
              d.*,
              (SELECT COUNT(*) FROM chunks c WHERE c.document_id = d.id) AS chunk_count,
              (SELECT COUNT(*) FROM questions q WHERE q.document_id = d.id AND COALESCE(q.reviewer_excluded, 0) = 0) AS question_count,
              (
                SELECT COUNT(*)
                FROM answers a
                JOIN questions q ON q.id = a.question_id
                WHERE a.document_id = d.id
                  AND a.problem_type != 'ok'
                  AND COALESCE(q.reviewer_excluded, 0) = 0
              ) AS problematic_count
            FROM documents d
            ORDER BY d.created_at DESC
            """
        ).fetchall()
        result = [dict(row) for row in rows]
        for item in result:
            document_id = str(item.get("id", "")).strip()
            if not document_id:
                continue
            parent_count_row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM retriever_parent_docs WHERE namespace = ?",
                (parent_docstore_namespace(document_id),),
            ).fetchone()
            item["chunk_count"] = int((parent_count_row["cnt"] if parent_count_row else 0) or 0)
    logger.debug("Documents listed | count=%s", len(result))
    return result


def get_document(document_id: str) -> dict[str, Any] | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
    return dict(row) if row else None


def set_document_status(document_id: str, status: str) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE documents SET status = ? WHERE id = ?", (status, document_id))
    logger.info("Document status updated | document_id=%s status=%s", document_id, status)


def set_document_run_mode(document_id: str, mode: str) -> str:
    normalized = normalize_qa_mode(mode)
    with get_conn() as conn:
        conn.execute(
            "UPDATE documents SET selected_qa_mode = ? WHERE id = ?",
            (normalized, document_id),
        )
    logger.info("Document run mode updated | document_id=%s selected_qa_mode=%s", document_id, normalized)
    return normalized


def set_refined_text(document_id: str, refined_text: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE documents SET refined_text = ? WHERE id = ?",
            (refined_text, document_id),
        )
    logger.info("Refined text stored | document_id=%s refined_chars=%s", document_id, len(refined_text))


def parent_docstore_namespace(document_id: str) -> str:
    return f"parent:{str(document_id).strip()}"


def upsert_parent_docstore_items(namespace: str, items: list[tuple[str, dict[str, Any]]]) -> None:
    normalized_namespace = str(namespace or "").strip()
    if not normalized_namespace or not items:
        return
    timestamp = now_iso()
    with get_conn() as conn:
        for key, payload in items:
            normalized_key = str(key or "").strip()
            if not normalized_key:
                continue
            conn.execute(
                """
                INSERT INTO retriever_parent_docs (namespace, key, payload_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(namespace, key) DO UPDATE SET
                  payload_json = excluded.payload_json,
                  updated_at = excluded.updated_at
                """,
                (
                    normalized_namespace,
                    normalized_key,
                    json.dumps(payload or {}, ensure_ascii=True),
                    timestamp,
                    timestamp,
                ),
            )


def get_parent_docstore_items(namespace: str, keys: list[str]) -> dict[str, dict[str, Any]]:
    normalized_namespace = str(namespace or "").strip()
    normalized_keys = [str(key or "").strip() for key in keys if str(key or "").strip()]
    if not normalized_namespace or not normalized_keys:
        return {}

    placeholders = ",".join("?" for _ in normalized_keys)
    with get_conn() as conn:
        rows = conn.execute(
            f"""
            SELECT key, payload_json
            FROM retriever_parent_docs
            WHERE namespace = ?
              AND key IN ({placeholders})
            """,
            (normalized_namespace, *normalized_keys),
        ).fetchall()

    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row["key"])
        raw = row["payload_json"] or "{}"
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {}
        result[key] = payload if isinstance(payload, dict) else {}
    return result


def delete_parent_docstore_items(namespace: str, keys: list[str]) -> None:
    normalized_namespace = str(namespace or "").strip()
    normalized_keys = [str(key or "").strip() for key in keys if str(key or "").strip()]
    if not normalized_namespace or not normalized_keys:
        return

    placeholders = ",".join("?" for _ in normalized_keys)
    with get_conn() as conn:
        conn.execute(
            f"""
            DELETE FROM retriever_parent_docs
            WHERE namespace = ?
              AND key IN ({placeholders})
            """,
            (normalized_namespace, *normalized_keys),
        )


def delete_parent_docstore_namespace(namespace: str) -> None:
    normalized_namespace = str(namespace or "").strip()
    if not normalized_namespace:
        return
    with get_conn() as conn:
        conn.execute(
            "DELETE FROM retriever_parent_docs WHERE namespace = ?",
            (normalized_namespace,),
        )


def list_parent_docstore_keys(namespace: str, prefix: str | None = None) -> list[str]:
    normalized_namespace = str(namespace or "").strip()
    if not normalized_namespace:
        return []

    with get_conn() as conn:
        if prefix:
            rows = conn.execute(
                """
                SELECT key
                FROM retriever_parent_docs
                WHERE namespace = ? AND key LIKE ?
                ORDER BY key ASC
                """,
                (normalized_namespace, f"{prefix}%"),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT key
                FROM retriever_parent_docs
                WHERE namespace = ?
                ORDER BY key ASC
                """,
                (normalized_namespace,),
            ).fetchall()
    return [str(row["key"]) for row in rows]


def clear_document_outputs(document_id: str) -> None:
    with get_conn() as conn:
        _delete_comparison_runs_for_document(conn, document_id)
        conn.execute("DELETE FROM answers WHERE document_id = ?", (document_id,))
        conn.execute("DELETE FROM questions WHERE document_id = ?", (document_id,))
        conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        conn.execute(
            "DELETE FROM retriever_parent_docs WHERE namespace = ?",
            (parent_docstore_namespace(document_id),),
        )
    logger.info("Previous outputs cleared | document_id=%s", document_id)


def _delete_comparison_runs_for_document(conn: sqlite3.Connection, document_id: str) -> None:
    conn.execute(
        """
        DELETE FROM comparison_ragas
        WHERE run_id IN (SELECT id FROM comparison_runs WHERE document_id = ?)
        """,
        (document_id,),
    )
    conn.execute(
        """
        DELETE FROM comparison_answers
        WHERE run_id IN (SELECT id FROM comparison_runs WHERE document_id = ?)
        """,
        (document_id,),
    )
    conn.execute("DELETE FROM comparison_runs WHERE document_id = ?", (document_id,))


def delete_document(document_id: str) -> None:
    with get_conn() as conn:
        _delete_comparison_runs_for_document(conn, document_id)
        conn.execute("DELETE FROM gold_references WHERE document_id = ?", (document_id,))
        conn.execute("DELETE FROM reviewer_question_exclusions WHERE document_id = ?", (document_id,))
        conn.execute("DELETE FROM answers WHERE document_id = ?", (document_id,))
        conn.execute("DELETE FROM questions WHERE document_id = ?", (document_id,))
        conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        conn.execute(
            "DELETE FROM retriever_parent_docs WHERE namespace = ?",
            (parent_docstore_namespace(document_id),),
        )
        conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
    logger.info("Document deleted | document_id=%s", document_id)


def get_document_questions(document_id: str, include_excluded: bool = False) -> list[dict[str, Any]]:
    with get_conn() as conn:
        if include_excluded:
            rows = conn.execute(
                """
                SELECT
                  q.id,
                  q.chunk_id,
                  q.question_no,
                  q.question_text,
                  q.category,
                  COALESCE(q.reviewer_excluded, 0) AS reviewer_excluded,
                  q.reviewer_excluded_note,
                  q.reviewer_excluded_at,
                  c.chunk_order
                FROM questions q
                JOIN chunks c ON c.id = q.chunk_id
                WHERE q.document_id = ?
                ORDER BY c.chunk_order ASC, q.question_no ASC
                """,
                (document_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT
                  q.id,
                  q.chunk_id,
                  q.question_no,
                  q.question_text,
                  q.category,
                  COALESCE(q.reviewer_excluded, 0) AS reviewer_excluded,
                  q.reviewer_excluded_note,
                  q.reviewer_excluded_at,
                  c.chunk_order
                FROM questions q
                JOIN chunks c ON c.id = q.chunk_id
                WHERE q.document_id = ?
                  AND COALESCE(q.reviewer_excluded, 0) = 0
                ORDER BY c.chunk_order ASC, q.question_no ASC
                """,
                (document_id,),
            ).fetchall()
    return [dict(row) for row in rows]


def list_document_reviewer_exclusion_keys(document_id: str) -> list[str]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT question_key
            FROM reviewer_question_exclusions
            WHERE document_id = ?
            ORDER BY updated_at DESC
            """,
            (document_id,),
        ).fetchall()
    keys: list[str] = []
    for row in rows:
        key = normalize_question_key(str(row["question_key"] or ""))
        if key:
            keys.append(key)
    return keys


def list_document_reviewer_exclusions(document_id: str) -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT
              id,
              document_id,
              question_text,
              question_key,
              note,
              created_at,
              updated_at
            FROM reviewer_question_exclusions
            WHERE document_id = ?
            ORDER BY updated_at DESC, created_at DESC
            """,
            (document_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def delete_document_reviewer_exclusion(document_id: str, exclusion_id: str) -> bool:
    with get_conn() as conn:
        cur = conn.execute(
            """
            DELETE FROM reviewer_question_exclusions
            WHERE document_id = ?
              AND id = ?
            """,
            (document_id, exclusion_id),
        )
    deleted = int(cur.rowcount or 0) > 0
    if deleted:
        logger.info(
            "Reviewer exclusion deleted | document_id=%s exclusion_id=%s",
            document_id,
            exclusion_id,
        )
    return deleted


def set_question_reviewer_exclusion(
    document_id: str,
    question_id: str,
    *,
    excluded: bool,
    note: str | None = None,
) -> bool:
    normalized_note = " ".join(str(note or "").split()).strip()
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT
              id,
              question_text
            FROM questions q
            WHERE q.document_id = ?
              AND q.id = ?
            """,
            (document_id, question_id),
        ).fetchone()
        if not row:
            return False
        question_text = str(row["question_text"] or "").strip()
        question_key = normalize_question_key(question_text)
        timestamp = now_iso()
        if excluded:
            conn.execute(
                """
                UPDATE questions
                SET reviewer_excluded = 1,
                    reviewer_excluded_note = ?,
                    reviewer_excluded_at = ?
                WHERE document_id = ?
                  AND id = ?
                """,
                (
                    normalized_note or "Excluded by reviewer.",
                    timestamp,
                    document_id,
                    question_id,
                ),
            )
            if question_key:
                conn.execute(
                    """
                    INSERT INTO reviewer_question_exclusions (
                      id, document_id, question_text, question_key, note, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(document_id, question_key) DO UPDATE SET
                      question_text = excluded.question_text,
                      note = excluded.note,
                      updated_at = excluded.updated_at
                    """,
                    (
                        str(uuid.uuid4()),
                        document_id,
                        question_text,
                        question_key,
                        normalized_note or "Excluded by reviewer.",
                        timestamp,
                        timestamp,
                    ),
                )
        else:
            conn.execute(
                """
                UPDATE questions
                SET reviewer_excluded = 0,
                    reviewer_excluded_note = NULL,
                    reviewer_excluded_at = NULL
                WHERE document_id = ?
                  AND id = ?
                """,
                (document_id, question_id),
            )
            if question_key:
                conn.execute(
                    """
                    DELETE FROM reviewer_question_exclusions
                    WHERE document_id = ?
                      AND question_key = ?
                    """,
                    (document_id, question_key),
                )
    logger.info(
        "Question reviewer exclusion updated | document_id=%s question_id=%s excluded=%s",
        document_id,
        question_id,
        excluded,
    )
    return True


def list_document_gold_references(document_id: str) -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT
              gr.id,
              gr.document_id,
              gr.question_id,
              gr.question_text,
              gr.question_key,
              gr.reference_text,
              gr.source,
              gr.created_at,
              gr.updated_at
            FROM gold_references gr
            WHERE gr.document_id = ?
            ORDER BY gr.updated_at DESC
            """,
            (document_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def upsert_document_gold_references(
    document_id: str,
    references: list[dict[str, Any]],
    *,
    replace_existing: bool = False,
) -> dict[str, Any]:
    if replace_existing:
        with get_conn() as conn:
            conn.execute("DELETE FROM gold_references WHERE document_id = ?", (document_id,))

    inserted = 0
    skipped = 0
    with get_conn() as conn:
        question_rows = conn.execute(
            """
            SELECT id, question_text
            FROM questions
            WHERE document_id = ?
            """,
            (document_id,),
        ).fetchall()
        question_text_by_id = {
            str(row["id"]).strip(): str(row["question_text"] or "").strip()
            for row in question_rows
            if str(row["id"] or "").strip()
        }

        for item in references:
            if not isinstance(item, dict):
                skipped += 1
                continue
            raw_question_id = str(item.get("question_id", "")).strip()
            raw_question_text = str(item.get("question", item.get("question_text", ""))).strip()
            reference_text = str(item.get("reference", item.get("reference_text", ""))).strip()
            source = str(item.get("source", "")).strip() or None
            if not reference_text:
                skipped += 1
                continue

            resolved_question_id = raw_question_id if raw_question_id in question_text_by_id else None
            question_text = question_text_by_id.get(resolved_question_id or "", "") or raw_question_text
            question_key = normalize_question_key(question_text)
            if not question_key:
                skipped += 1
                continue

            conn.execute(
                """
                INSERT INTO gold_references (
                  id, document_id, question_id, question_text, question_key, reference_text, source, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id, question_key) DO UPDATE SET
                  question_id = excluded.question_id,
                  question_text = excluded.question_text,
                  reference_text = excluded.reference_text,
                  source = excluded.source,
                  updated_at = excluded.updated_at
                """,
                (
                    str(uuid.uuid4()),
                    document_id,
                    resolved_question_id,
                    question_text,
                    question_key,
                    reference_text,
                    source,
                    now_iso(),
                    now_iso(),
                ),
            )
            inserted += 1

    logger.info(
        "Gold references upserted | document_id=%s inserted=%s skipped=%s replace_existing=%s",
        document_id,
        inserted,
        skipped,
        replace_existing,
    )
    return {
        "document_id": document_id,
        "inserted": inserted,
        "skipped": skipped,
        "replace_existing": replace_existing,
        "total": len(list_document_gold_references(document_id)),
    }


def insert_chunks(document_id: str, chunks: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with get_conn() as conn:
        for idx, text in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO chunks (id, document_id, chunk_order, text)
                VALUES (?, ?, ?, ?)
                """,
                (chunk_id, document_id, idx, text),
            )
            records.append({"id": chunk_id, "chunk_order": idx, "text": text})
    logger.info("Chunks inserted | document_id=%s chunk_count=%s", document_id, len(records))
    return records


def insert_questions(
    document_id: str,
    chunk_id: str,
    questions: list[dict[str, Any]],
    *,
    blocked_question_keys: set[str] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    skipped_duplicates = 0
    skipped_blocked = 0
    blocked_keys = {
        normalize_question_key(str(value or ""))
        for value in (blocked_question_keys or set())
        if normalize_question_key(str(value or ""))
    }
    with get_conn() as conn:
        existing_rows = conn.execute(
            """
            SELECT question_text
            FROM questions
            WHERE document_id = ?
            """,
            (document_id,),
        ).fetchall()
        existing_keys = {
            normalize_question_key(str(row["question_text"] or ""))
            for row in existing_rows
            if normalize_question_key(str(row["question_text"] or ""))
        }
        batch_keys: set[str] = set()
        current_max_row = conn.execute(
            """
            SELECT COALESCE(MAX(question_no), 0) AS max_question_no
            FROM questions
            WHERE document_id = ? AND chunk_id = ?
            """,
            (document_id, chunk_id),
        ).fetchone()
        next_question_no = int((current_max_row["max_question_no"] if current_max_row else 0) or 0)

        for item in questions:
            question = str(item.get("question", "")).strip()
            category = str(item.get("category", "other")).strip().lower()
            risk = str(item.get("risk", "low")).strip().lower()
            content_type = str(item.get("content_type", "other")).strip().lower()
            hints = item.get("hints", [])
            if not isinstance(hints, list):
                hints = []
            hints_clean = [str(v).strip() for v in hints if str(v).strip()]
            category_payload = {
                "category": category or "other",
                "risk": risk if risk in {"low", "medium", "high"} else "low",
                "content_type": content_type or "other",
                "hints": hints_clean,
            }
            if not question:
                continue
            question_key = normalize_question_key(question)
            if not question_key:
                continue
            if question_key in blocked_keys:
                skipped_blocked += 1
                continue
            if question_key in existing_keys or question_key in batch_keys:
                skipped_duplicates += 1
                continue

            question_id = str(uuid.uuid4())
            next_question_no += 1
            conn.execute(
                """
                INSERT INTO questions (id, document_id, chunk_id, question_no, question_text, category)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    question_id,
                    document_id,
                    chunk_id,
                    next_question_no,
                    question,
                    json.dumps(category_payload, ensure_ascii=True),
                ),
            )
            existing_keys.add(question_key)
            batch_keys.add(question_key)
            records.append(
                {
                    "id": question_id,
                    "chunk_id": chunk_id,
                    "question_no": next_question_no,
                    "question_text": question,
                    "category": category_payload,
                }
            )
    logger.info(
        "Questions inserted | document_id=%s chunk_id=%s question_count=%s skipped_duplicates=%s skipped_blocked=%s",
        document_id,
        chunk_id,
        len(records),
        skipped_duplicates,
        skipped_blocked,
    )
    return records


def insert_answer(
    document_id: str,
    question_id: str,
    answer_text: str,
    confidence: float,
    computed_score: float | None,
    classification_confidence: float | None,
    problem_type: str,
    reasons: str,
    suggested_fix: str,
    evidence: list[dict[str, Any]],
) -> str:
    answer_id = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO answers (
              id, document_id, question_id, answer_text, confidence, computed_score, classification_confidence, problem_type, reasons, suggested_fix, evidence_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                answer_id,
                document_id,
                question_id,
                answer_text,
                confidence,
                computed_score,
                classification_confidence,
                problem_type,
                reasons,
                suggested_fix,
                json.dumps(evidence, ensure_ascii=True),
            ),
        )
    logger.debug(
        "Answer inserted | document_id=%s question_id=%s answer_confidence=%.2f computed_score=%s classification_confidence=%s problem_type=%s",
        document_id,
        question_id,
        confidence,
        computed_score,
        classification_confidence,
        problem_type,
    )
    return answer_id


def create_comparison_run(document_id: str) -> str:
    run_id = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO comparison_runs (id, document_id, status, created_at, started_at, completed_at, error, summary_json)
            VALUES (?, ?, 'queued', ?, NULL, NULL, NULL, NULL)
            """,
            (run_id, document_id, now_iso()),
        )
    logger.info("Comparison run created | document_id=%s run_id=%s", document_id, run_id)
    return run_id


def set_comparison_run_status(
    run_id: str,
    status: str,
    *,
    error: str | None = None,
    summary: dict[str, Any] | None = None,
) -> None:
    normalized = str(status or "").strip().lower() or "queued"
    started_at = None
    completed_at = None
    if normalized == "running":
        started_at = now_iso()
    if normalized in {"completed", "failed"}:
        completed_at = now_iso()
    summary_json = json.dumps(summary, ensure_ascii=True) if summary is not None else None

    with get_conn() as conn:
        if normalized == "running":
            conn.execute(
                """
                UPDATE comparison_runs
                SET status = ?, started_at = COALESCE(started_at, ?), error = NULL
                WHERE id = ?
                """,
                (normalized, started_at, run_id),
            )
        elif normalized == "completed":
            conn.execute(
                """
                UPDATE comparison_runs
                SET status = ?, completed_at = ?, error = NULL, summary_json = COALESCE(?, summary_json)
                WHERE id = ?
                """,
                (normalized, completed_at, summary_json, run_id),
            )
        elif normalized == "failed":
            conn.execute(
                """
                UPDATE comparison_runs
                SET status = ?, completed_at = ?, error = ?, summary_json = COALESCE(?, summary_json)
                WHERE id = ?
                """,
                (normalized, completed_at, error, summary_json, run_id),
            )
        else:
            conn.execute(
                "UPDATE comparison_runs SET status = ? WHERE id = ?",
                (normalized, run_id),
            )
    logger.info("Comparison run status updated | run_id=%s status=%s", run_id, normalized)


def insert_comparison_answer(
    run_id: str,
    question_id: str,
    mode: str,
    answer_text: str,
    answer_confidence: float,
    classification_confidence: float | None,
    problem_type: str,
    reasons: str,
    suggested_fix: str,
    evidence: list[dict[str, Any]],
) -> str:
    normalized_mode = normalize_qa_mode(mode)
    answer_id = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO comparison_answers (
              id, run_id, question_id, mode, answer_text, answer_confidence, classification_confidence,
              problem_type, reasons, suggested_fix, evidence_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, question_id, mode) DO UPDATE SET
              answer_text = excluded.answer_text,
              answer_confidence = excluded.answer_confidence,
              classification_confidence = excluded.classification_confidence,
              problem_type = excluded.problem_type,
              reasons = excluded.reasons,
              suggested_fix = excluded.suggested_fix,
              evidence_json = excluded.evidence_json,
              created_at = excluded.created_at
            """,
            (
                answer_id,
                run_id,
                question_id,
                normalized_mode,
                answer_text,
                float(answer_confidence),
                classification_confidence,
                problem_type,
                reasons,
                suggested_fix,
                json.dumps(evidence, ensure_ascii=True),
                now_iso(),
            ),
        )
    logger.debug(
        "Comparison answer inserted | run_id=%s question_id=%s mode=%s answer_confidence=%.2f problem_type=%s",
        run_id,
        question_id,
        normalized_mode,
        answer_confidence,
        problem_type,
    )
    return answer_id


def upsert_comparison_ragas(
    run_id: str,
    mode: str,
    metrics: dict[str, Any],
) -> str:
    normalized_mode = normalize_qa_mode(mode)
    row_id = str(uuid.uuid4())
    payload = dict(metrics)
    payload["mode"] = normalized_mode
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO comparison_ragas (id, run_id, mode, metrics_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(run_id, mode) DO UPDATE SET
              metrics_json = excluded.metrics_json,
              created_at = excluded.created_at
            """,
            (
                row_id,
                run_id,
                normalized_mode,
                json.dumps(payload, ensure_ascii=True),
                now_iso(),
            ),
        )
    logger.info("Comparison RAGAS metrics stored | run_id=%s mode=%s", run_id, normalized_mode)
    return row_id


def delete_comparison_run(document_id: str, run_id: str) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            DELETE FROM comparison_ragas
            WHERE run_id = ?
              AND EXISTS (SELECT 1 FROM comparison_runs r WHERE r.id = ? AND r.document_id = ?)
            """,
            (run_id, run_id, document_id),
        )
        conn.execute(
            """
            DELETE FROM comparison_answers
            WHERE run_id = ?
              AND EXISTS (SELECT 1 FROM comparison_runs r WHERE r.id = ? AND r.document_id = ?)
            """,
            (run_id, run_id, document_id),
        )
        conn.execute(
            "DELETE FROM comparison_runs WHERE id = ? AND document_id = ?",
            (run_id, document_id),
        )
    logger.info("Comparison run deleted | document_id=%s run_id=%s", document_id, run_id)


def _parse_summary_json(raw: Any) -> dict[str, Any]:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def list_comparison_runs(document_id: str) -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM comparison_runs
            WHERE document_id = ?
            ORDER BY created_at DESC
            """,
            (document_id,),
        ).fetchall()
    result: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["summary"] = _parse_summary_json(item.get("summary_json"))
        result.append(item)
    return result


def _build_comparison_run_report(document_id: str, run_id: str) -> dict[str, Any] | None:
    with get_conn() as conn:
        run_row = conn.execute(
            "SELECT * FROM comparison_runs WHERE id = ? AND document_id = ?",
            (run_id, document_id),
        ).fetchone()
        if not run_row:
            return None

        question_rows = [
            dict(row)
            for row in conn.execute(
                """
                SELECT
                  q.id AS question_id,
                  q.question_no,
                  q.question_text,
                  q.category,
                  q.chunk_id,
                  c.chunk_order
                FROM questions q
                JOIN chunks c ON c.id = q.chunk_id
                WHERE q.document_id = ?
                ORDER BY c.chunk_order ASC, q.question_no ASC
                """,
                (document_id,),
            ).fetchall()
        ]
        answer_rows = [
            dict(row)
            for row in conn.execute(
                """
                SELECT
                  ca.question_id,
                  ca.mode,
                  ca.answer_text,
                  ca.answer_confidence AS confidence,
                  ca.classification_confidence,
                  ca.problem_type,
                  ca.reasons,
                  ca.suggested_fix,
                  ca.evidence_json
                FROM comparison_answers ca
                WHERE ca.run_id = ?
                """,
                (run_id,),
            ).fetchall()
        ]
        ragas_rows = conn.execute(
            "SELECT mode, metrics_json FROM comparison_ragas WHERE run_id = ?",
            (run_id,),
        ).fetchall()

    ragas_by_mode: dict[str, dict[str, Any] | None] = {"baseline": None, "parent": None}
    for row in ragas_rows:
        mode = str(row["mode"] or "").strip().lower()
        if mode not in ragas_by_mode:
            continue
        parsed = _parse_summary_json(row["metrics_json"])
        ragas_by_mode[mode] = parsed or None

    report = build_comparison_report(
        question_rows=question_rows,
        answer_rows=answer_rows,
        ragas_by_mode=ragas_by_mode,
    )
    run_dict = dict(run_row)
    run_dict["summary"] = _parse_summary_json(run_dict.get("summary_json"))
    return {
        "run": run_dict,
        "report": report,
    }


def get_document_comparison_dashboard(document_id: str, run_id: str | None = None) -> dict[str, Any]:
    runs = list_comparison_runs(document_id)
    selected_id = run_id or (runs[0]["id"] if runs else None)
    selected = _build_comparison_run_report(document_id, selected_id) if selected_id else None
    return {
        "runs": runs,
        "selected_run_id": selected_id,
        "selected": selected,
    }


def get_document_chunks(document_id: str) -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_order ASC",
            (document_id,),
        ).fetchall()
    result = [dict(row) for row in rows]
    logger.debug("Chunks fetched | document_id=%s chunk_count=%s", document_id, len(result))
    return result


def get_document_parent_chunks(document_id: str) -> list[dict[str, Any]]:
    namespace = parent_docstore_namespace(document_id)
    keys = list_parent_docstore_keys(namespace)
    if not keys:
        logger.debug("Parent chunks fetched | document_id=%s chunk_count=0", document_id)
        return []

    payloads = get_parent_docstore_items(namespace, keys)
    rows: list[dict[str, Any]] = []
    for idx, key in enumerate(keys):
        payload = payloads.get(key, {})
        if not isinstance(payload, dict):
            continue
        text = str(payload.get("page_content", "")).strip()
        if not text:
            continue
        raw_meta = payload.get("metadata", {})
        metadata = raw_meta if isinstance(raw_meta, dict) else {}

        raw_order = metadata.get("chunk_order", idx)
        try:
            chunk_order = int(raw_order)
        except (TypeError, ValueError):
            chunk_order = idx

        rows.append(
            {
                "id": str(key),
                "chunk_order": chunk_order,
                "text": text,
                "source": "parent",
                "metadata": metadata,
            }
        )

    rows.sort(key=lambda row: (int(row.get("chunk_order", 0)), str(row.get("id", ""))))
    logger.debug("Parent chunks fetched | document_id=%s chunk_count=%s", document_id, len(rows))
    return rows


def get_document_dashboard(document_id: str, include_excluded: bool = False) -> dict[str, Any]:
    with get_conn() as conn:
        stats = conn.execute(
            """
            SELECT
              (SELECT COUNT(*) FROM chunks c WHERE c.document_id = ?) AS chunk_count,
              (SELECT COUNT(*) FROM questions q WHERE q.document_id = ? AND COALESCE(q.reviewer_excluded, 0) = 0) AS question_count,
              (SELECT COUNT(*) FROM questions q WHERE q.document_id = ? AND COALESCE(q.reviewer_excluded, 0) = 1) AS excluded_count,
              (
                SELECT COUNT(*)
                FROM answers a
                JOIN questions q ON q.id = a.question_id
                WHERE a.document_id = ?
                  AND a.problem_type != 'ok'
                  AND COALESCE(q.reviewer_excluded, 0) = 0
              ) AS problematic_count,
              (
                SELECT AVG(a.confidence)
                FROM answers a
                JOIN questions q ON q.id = a.question_id
                WHERE a.document_id = ?
                  AND COALESCE(q.reviewer_excluded, 0) = 0
              ) AS avg_confidence
            """,
            (document_id, document_id, document_id, document_id, document_id),
        ).fetchone()
        parent_count_row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM retriever_parent_docs WHERE namespace = ?",
            (parent_docstore_namespace(document_id),),
        ).fetchone()
        parent_chunk_count = int((parent_count_row["cnt"] if parent_count_row else 0) or 0)

        if include_excluded:
            rows = conn.execute(
                """
                SELECT
                  q.id AS question_id,
                  q.question_no,
                  q.question_text,
                  q.category,
                  q.chunk_id,
                  COALESCE(q.reviewer_excluded, 0) AS reviewer_excluded,
                  q.reviewer_excluded_note,
                  q.reviewer_excluded_at,
                  c.chunk_order,
                  a.answer_text,
                  a.confidence,
                  a.computed_score,
                  a.classification_confidence,
                  a.problem_type,
                  a.reasons,
                  a.suggested_fix,
                  a.evidence_json
                FROM questions q
                JOIN chunks c ON c.id = q.chunk_id
                LEFT JOIN answers a ON a.question_id = q.id
                WHERE q.document_id = ?
                ORDER BY c.chunk_order ASC, q.question_no ASC
                """,
                (document_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT
                  q.id AS question_id,
                  q.question_no,
                  q.question_text,
                  q.category,
                  q.chunk_id,
                  COALESCE(q.reviewer_excluded, 0) AS reviewer_excluded,
                  q.reviewer_excluded_note,
                  q.reviewer_excluded_at,
                  c.chunk_order,
                  a.answer_text,
                  a.confidence,
                  a.computed_score,
                  a.classification_confidence,
                  a.problem_type,
                  a.reasons,
                  a.suggested_fix,
                  a.evidence_json
                FROM questions q
                JOIN chunks c ON c.id = q.chunk_id
                LEFT JOIN answers a ON a.question_id = q.id
                WHERE q.document_id = ?
                  AND COALESCE(q.reviewer_excluded, 0) = 0
                ORDER BY c.chunk_order ASC, q.question_no ASC
                """,
                (document_id,),
            ).fetchall()

    answers = []
    for row in rows:
        item = dict(row)
        raw = item.get("evidence_json") or "[]"
        try:
            evidence = json.loads(raw)
        except json.JSONDecodeError:
            evidence = []
        item["evidence"] = evidence
        raw_category = item.get("category") or ""
        try:
            item["category_meta"] = json.loads(raw_category) if raw_category else {}
        except json.JSONDecodeError:
            item["category_meta"] = {}
        question_type = str(item["category_meta"].get("content_type", "technical")).strip().lower()
        item["question_type"] = question_type or "technical"
        computed_score = item.get("computed_score")
        if computed_score is None:
            item["computed_score"] = float(item.get("confidence") or 0.0)
        else:
            try:
                item["computed_score"] = float(computed_score)
            except (TypeError, ValueError):
                item["computed_score"] = float(item.get("confidence") or 0.0)
        answers.append(item)

    readiness = compute_dashboard_readiness(answers)

    resolved_chunk_count = parent_chunk_count

    result = {
        "chunk_count": resolved_chunk_count,
        "question_count": int(stats["question_count"] or 0),
        "excluded_count": int(stats["excluded_count"] or 0),
        "problematic_count": int(stats["problematic_count"] or 0),
        "avg_confidence": float(stats["avg_confidence"] or 0.0),
        "avg_answer_confidence": float(stats["avg_confidence"] or 0.0),
        "rows": answers,
        **readiness,
    }
    logger.debug(
        "Dashboard computed | document_id=%s chunks=%s questions=%s problematic=%s readiness=%.2f rows=%s",
        document_id,
        result["chunk_count"],
        result["question_count"],
        result["problematic_count"],
        result["readiness_score"],
        len(result["rows"]),
    )
    return result
