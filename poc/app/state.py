from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, TypedDict

try:
    from langgraph.graph.message import add_messages
except Exception:  # pragma: no cover - optional dependency for future refactor
    def add_messages(existing: list[Any], new: list[Any]) -> list[Any]:
        return (existing or []) + (new or [])

try:
    from langchain_core.messages import BaseMessage
except Exception:  # pragma: no cover - optional dependency for future refactor
    BaseMessage = Any


JobStatus = Literal[
    "created",
    "ingesting",
    "ingested",
    "generating_questions",
    "questions_ready",
    "qa_baseline_running",
    "qa_baseline_done",
    "qa_parent_running",
    "qa_parent_done",
    "ragas_baseline_running",
    "ragas_baseline_done",
    "ragas_parent_running",
    "ragas_parent_done",
    "finalizing",
    "completed",
    "failed",
]

QuestionCategory = Literal["pricing", "terms", "availability", "procedure", "technical"]
QuestionRisk = Literal["low", "medium", "high"]
QuestionContentType = Literal["pricing", "instructions", "terms", "dates", "technical"]
AnswerCategory = Literal["pricing", "technical", "procedure", "administration", "other"]
RetrievalMode = Literal["baseline", "parent"]
StepStatus = Literal["ok", "failed"]


class ChunkRecord(TypedDict):
    chunk_id: str
    text: str
    parent_id: Optional[str]
    metadata: dict[str, Any]


class QuestionRecord(TypedDict):
    question_id: str
    chunk_id: str
    question: str
    category: QuestionCategory
    risk: QuestionRisk
    content_type: QuestionContentType
    hints: list[str]


class AnswerRecord(TypedDict):
    question_id: str
    mode: RetrievalMode
    answer: str
    category: AnswerCategory
    reasoning: str
    used_citations: list[dict[str, str]]
    score: float


class RagasResult(TypedDict):
    mode: RetrievalMode
    context_recall: float
    faithfulness: float
    factual_correctness: float
    answer_relevancy: float
    context_entity_recall: float
    noise_sensitivity: float


class StepTrace(TypedDict):
    step: str
    status: StepStatus
    duration_ms: float
    details: dict[str, Any]


class JobState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    job_id: str
    status: JobStatus
    source_file: str
    chunks: list[ChunkRecord]
    questions: list[QuestionRecord]
    answers_baseline: list[AnswerRecord]
    answers_parent: list[AnswerRecord]
    ragas_baseline: Optional[RagasResult]
    ragas_parent: Optional[RagasResult]
    readiness_pass: Optional[bool]
    error: Optional[str]
    step_results: list[StepTrace]


def new_job_state(job_id: str, source_file: str) -> JobState:
    return {
        "messages": [],
        "job_id": job_id,
        "status": "created",
        "source_file": source_file,
        "chunks": [],
        "questions": [],
        "answers_baseline": [],
        "answers_parent": [],
        "ragas_baseline": None,
        "ragas_parent": None,
        "readiness_pass": None,
        "error": None,
        "step_results": [],
    }
