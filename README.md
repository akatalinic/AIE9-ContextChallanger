# Context Challengers PoC

Agentic RAG PoC for pre-release documentation quality validation before chatbot rollout.

## What This Project Does

- Upload internal PDF/DOCX documentation
- Run a supervised pipeline: extract -> chunk -> question generation -> retrieval-grounded QA
- Classify answer quality (`ok`, `missing_info`, `ambiguous`, `contradiction`, `formatting_issue`)
- Show evidence citations and deterministic document readiness KPI
- Compare retrieval strategies with RAGAS metrics (baseline vs parent)
- Run gold-dataset evaluation from CLI

## Repository Layout

- `poc/` - main application (FastAPI + Jinja + pipeline + DB + services)
- `poc/doc/CertChallengeSubmission.md` - rubric-aligned submission writeup

## Quickstart

### 1) Prerequisites

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/)

### 2) Install dependencies

```bash
cd poc
uv venv --python 3.12      
uv sync
```

### 3) Configure environment

Create `poc/.env` from `poc/.env.example` and set required keys:

- `OPENAI_API_KEY`
- Optional for gated external fallback: `TAVILY_API_KEY`

Do not commit real secrets.

### 4) Run the app

```bash
cd poc
uv run uvicorn app.main:app --reload
```

Open:

- `http://127.0.0.1:8000/`

## Gold Evaluation (CLI)

Example run:

```bash
cd poc
uv run python -m app.cli.gold_eval --source-file templatedata/not_a_real_service_OK.docx --gold-file goldendataset.json --mode both --top-k 5 --output-json reports/gold_eval_topk5_detailed_refs.json
```

Results are written to `poc/reports/`.

## Current Retrieval Strategy

- Main pipeline default: **parent retrieval**
- Baseline dense retrieval remains available for comparison/evaluation

## Main Docs

- Implementation log: `poc/README.md`
- Submission writeup: `poc/doc/CertChallengeSubmission.md`
- Gold dataset: `poc/goldendataset.json`

## Notes

- `.env` is git-ignored. Keep all API keys private.
- If LangSmith tracing is enabled without a valid key, tracing calls may fail while app functionality still works.
