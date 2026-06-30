from __future__ import annotations

import argparse
import json
import sys

from src.rag_pipeline import MedicalRAGPipeline


def _configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def print_result(result: dict) -> None:
    if result.get("type"):
        print(result.get("message", ""))
        return
    print(result.get("answer", ""))
    if result.get("disclaimer"):
        print("\n---")
        print(result["disclaimer"])
    if result.get("sources"):
        print("\nSources:")
        for source in result["sources"]:
            print(
                f"[{source['index']}] {source.get('title') or source.get('source')} "
                f"- {source.get('url', '')}"
            )


def main() -> None:
    _configure_stdout()
    parser = argparse.ArgumentParser(description="Medical RAG Assistant")
    parser.add_argument("--ingest", action="store_true", help="Ingest data into stores")
    parser.add_argument("--query", type=str, help="Ask one question")
    parser.add_argument("--emergency", action="store_true", help="Set emergency flag")
    parser.add_argument("--json", action="store_true", help="Print raw JSON result")
    args = parser.parse_args()

    pipeline = MedicalRAGPipeline()

    if args.ingest:
        print("Ingesting data...")
        stats = pipeline.ingest_data()
        print(f"Done: {stats}")
        return

    if args.query:
        result = pipeline.process_query(args.query, isEmergency=args.emergency)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        else:
            print_result(result)
        return

    print("Medical Assistant CLI. Type 'quit' to exit. Prefix with '!emergency ' to toggle emergency flag.")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"quit", "exit"}:
            break
        is_emerg = False
        if question.startswith("!emergency "):
            is_emerg = True
            question = question[len("!emergency "):].strip()
        result = pipeline.process_query(question, isEmergency=is_emerg)
        print("\nAssistant:")
        print_result(result)


if __name__ == "__main__":
    main()
