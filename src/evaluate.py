from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from sentence_transformers import SentenceTransformer

from answer import build_context, call_deepseek

ROOT = Path(__file__).resolve().parents[1]
QA_FILE = ROOT / "evaluation" / "qa_pairs.csv"
RESULTS_DIR = ROOT / "evaluation" / "results"
QDRANT_PATH = ROOT / "qdrant_data"
COLLECTION_NAME = "qdrant_docs"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

STRATEGIES = ["fixed_500_50", "structure_800_100"]
REFUSAL_TEXT = (
    "The provided Qdrant documentation does not contain enough information "
    "to answer this question."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Qdrant RAG system")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    return parser.parse_args()


def extract_citations(answer: str) -> list[int]:
    return sorted({int(match) for match in re.findall(r"\[(\d+)\]", answer)})


def keyword_coverage(answer: str, keywords: object) -> float | None:
    if pd.isna(keywords) or not str(keywords).strip():
        return None

    keyword_list = [
        keyword.strip().lower()
        for keyword in str(keywords).split(";")
        if keyword.strip()
    ]
    if not keyword_list:
        return None

    answer_lower = answer.lower()
    matched = sum(1 for keyword in keyword_list if keyword in answer_lower)
    return matched / len(keyword_list)


def citation_validity(answer: str, result_count: int) -> float:
    citations = extract_citations(answer)
    if not citations:
        return 0.0

    valid_citations = [
        citation for citation in citations if 1 <= citation <= result_count
    ]
    return len(valid_citations) / len(citations)


def main() -> int:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    qa = pd.read_csv(QA_FILE)
    qa["answerable"] = qa["answerable"].astype(bool)

    questions = qa["question"].tolist()
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print(f"Embedding {len(questions)} questions...")
    question_embeddings = model.encode(
        questions,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    client = QdrantClient(path=str(QDRANT_PATH))

    detailed_rows = []

    for strategy in STRATEGIES:
        print(f"Evaluating strategy: {strategy}")

        query_filter = Filter(
            must=[
                FieldCondition(
                    key="chunk_strategy",
                    match=MatchValue(value=strategy),
                )
            ]
        )

        for index, row in qa.iterrows():
            question = row["question"]
            answerable = bool(row["answerable"])
            expected_source_url = row.get("expected_source_url")
            expected_keywords = row.get("expected_keywords")

            started_at = time.perf_counter()

            try:
                response = client.query_points(
                    collection_name=COLLECTION_NAME,
                    query=question_embeddings[index].tolist(),
                    query_filter=query_filter,
                    limit=args.top_k,
                )
                results = response.points
            except Exception as exc:
                print(f"Retrieval failed for {row['question_id']}: {exc}")
                results = []

            retrieved_source_urls = [
                result.payload.get("source_url", "")
                for result in results
            ]
            retrieved_chunk_ids = [
                result.payload.get("chunk_id", "")
                for result in results
            ]

            if results:
                context = build_context(results)
                try:
                    answer = call_deepseek(question, context)
                except Exception as exc:
                    answer = f"ERROR: {exc}"
            else:
                answer = REFUSAL_TEXT

            if answerable:
                hit_at_k = (
                    expected_source_url in retrieved_source_urls
                    if not pd.isna(expected_source_url)
                    else False
                )
                refusal_correct = None
                coverage = keyword_coverage(answer, expected_keywords)
                validity = citation_validity(answer, len(results))

                if coverage is None:
                    coverage = 0.0

                answer_correctness = 0.7 * coverage + 0.3 * validity
            else:
                hit_at_k = None
                refusal_correct = REFUSAL_TEXT in answer
                coverage = None
                validity = None
                answer_correctness = 1.0 if refusal_correct else 0.0

            elapsed = time.perf_counter() - started_at

            detailed_rows.append(
                {
                    "question_id": row["question_id"],
                    "strategy": strategy,
                    "question": question,
                    "answerable": answerable,
                    "expected_source_url": expected_source_url,
                    "retrieved_source_urls": " | ".join(retrieved_source_urls),
                    "retrieved_chunk_ids": " | ".join(retrieved_chunk_ids),
                    "hit_at_k": hit_at_k,
                    "answer": answer,
                    "refusal_correct": refusal_correct,
                    "keyword_coverage": coverage,
                    "citation_validity": validity,
                    "answer_correctness": answer_correctness,
                    "elapsed_seconds": elapsed,
                }
            )

            print(
                f"  {row['question_id']} | "
                f"hit={hit_at_k} | "
                f"correctness={answer_correctness:.2f} | "
                f"{elapsed:.2f}s"
            )

    client.close()

    detailed_df = pd.DataFrame(detailed_rows)
    detailed_path = RESULTS_DIR / "evaluation_details.csv"
    detailed_df.to_csv(detailed_path, index=False, encoding="utf-8")

    summary_rows = []

    for strategy, group in detailed_df.groupby("strategy"):
        answerable_group = group[group["answerable"]]
        unanswerable_group = group[~group["answerable"]]

        summary_rows.append(
            {
                "strategy": strategy,
                "questions": len(group),
                "answerable_questions": len(answerable_group),
                "unanswerable_questions": len(unanswerable_group),
                "retrieval_hit_at_5": answerable_group["hit_at_k"].mean(),
                "answerable_answer_correctness": answerable_group[
                    "answer_correctness"
                ].mean(),
                "refusal_accuracy": unanswerable_group["refusal_correct"].mean(),
                "keyword_coverage": answerable_group["keyword_coverage"].mean(),
                "citation_validity": answerable_group["citation_validity"].mean(),
                "overall_answer_correctness": group["answer_correctness"].mean(),
                "avg_latency_seconds": group["elapsed_seconds"].mean(),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "evaluation_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")

    print()
    print("Evaluation summary:")
    print(summary_df.to_string(index=False))
    print()
    print(f"Saved detailed results to {detailed_path.relative_to(ROOT).as_posix()}")
    print(f"Saved summary to {summary_path.relative_to(ROOT).as_posix()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())