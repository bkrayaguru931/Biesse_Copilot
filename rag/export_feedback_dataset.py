"""Export reviewed copilot recommendations for offline RAG evaluation/curation.

This does not automatically fine-tune a model. It produces an auditable JSONL
dataset so only reviewed, useful examples are promoted into an evaluation set
or an approved training workflow.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = PROJECT_ROOT / "data" / "biesse_auth.db"
DEFAULT_OUTPUT = PROJECT_ROOT / "knowledge_base" / "processed" / "recommendation_feedback.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--helpful-only", action="store_true")
    args = parser.parse_args()

    if not args.database.exists():
        raise SystemExit(f"Feedback database not found: {args.database}")

    where_clause = "WHERE rating = 1" if args.helpful_only else ""
    with sqlite3.connect(args.database) as connection:
        rows = connection.execute(
            f"""
            SELECT suggestion_id, document_key, source, rating, conversation,
                   suggestion_json, created_at
            FROM recommendation_feedback
            {where_clause}
            ORDER BY created_at ASC
            """
        ).fetchall()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for suggestion_id, document_key, source, rating, conversation, suggestion_json, created_at in rows:
            output.write(json.dumps({
                "id": suggestion_id,
                "conversation": conversation,
                "source": source,
                "document_key": document_key,
                "recommendation": json.loads(suggestion_json),
                "label": "helpful" if rating == 1 else "not_helpful",
                "created_at": created_at,
            }, ensure_ascii=False) + "\n")

    print(f"Exported {len(rows)} feedback records to {args.output}")


if __name__ == "__main__":
    main()
