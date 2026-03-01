from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.config import QA_TOP_K
from app.logging_config import setup_logging
from app.services.metrics_utils import fmt_metric as _fmt_metric
from app.services.ragas_eval import RAGAS_METRIC_KEYS


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run gold-only RAGAS evaluation directly from goldendataset.json, "
            "without dependency on generated questions."
        )
    )
    parser.add_argument(
        "--document-id",
        default="",
        help="Optional document id from SQLite (`documents.id`).",
    )
    parser.add_argument(
        "--source-file",
        default="",
        help=(
            "Optional local source document path (.pdf/.docx). "
            "If omitted together with --document-id, defaults to `templatedata/not_a_real_service_OK.docx`."
        ),
    )
    parser.add_argument(
        "--gold-file",
        default="goldendataset.json",
        help="Path to JSON file with `references` question/reference pairs.",
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "parent", "both"],
        default="both",
        help="Retrieval mode to evaluate.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=QA_TOP_K,
        help="Retrieved context chunks per question.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path for full JSON output.",
    )
    return parser


def _print_summary(result: dict[str, Any]) -> None:
    print("Gold-only RAGAS evaluation")
    print(f"document_id: {result.get('document_id')}")
    print(f"source_file: {result.get('source_file')}")
    print(f"retrieval_document_id: {result.get('retrieval_document_id')}")
    print(f"filename: {result.get('document_filename')}")
    print(f"gold_file: {result.get('gold_file')}")
    print(
        "questions/chunks: "
        f"{int(result.get('question_count', 0))}/{int(result.get('chunk_count', 0))} "
        f"(chunk_source={result.get('chunk_source')})"
    )
    print()

    modes = result.get("results", {})
    baseline = modes.get("baseline", {}).get("metrics", {}) if isinstance(modes, dict) else {}
    parent = modes.get("parent", {}).get("metrics", {}) if isinstance(modes, dict) else {}

    if baseline and parent:
        header = f"{'Metric':<24} {'Baseline':>10} {'Parent':>10} {'Delta':>10}"
        print(header)
        print("-" * len(header))
        deltas = result.get("deltas", {})
        for key in RAGAS_METRIC_KEYS:
            label = key.replace("_", " ").title()
            b = baseline.get(key)
            p = parent.get(key)
            d = deltas.get(key)
            print(f"{label:<24} {_fmt_metric(b):>10} {_fmt_metric(p):>10} {_fmt_metric(d):>10}")
        print()
        return

    for mode, payload in modes.items() if isinstance(modes, dict) else []:
        print(f"{mode.title()} metrics:")
        metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
        for key in RAGAS_METRIC_KEYS:
            label = key.replace("_", " ").title()
            print(f"  {label:<22} {_fmt_metric(metrics.get(key))}")
        print()


def main() -> int:
    setup_logging()
    args = _build_parser().parse_args()
    from app.services.gold_eval import run_gold_only_ragas_eval

    result = run_gold_only_ragas_eval(
        document_id=str(args.document_id or "").strip() or None,
        source_file=str(args.source_file or "").strip() or None,
        gold_file=args.gold_file,
        mode=args.mode,
        top_k=max(1, int(args.top_k)),
    )
    _print_summary(result)

    output_path = str(args.output_json or "").strip()
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Saved full result JSON to: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
