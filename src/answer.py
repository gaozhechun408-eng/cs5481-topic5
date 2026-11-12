from __future__ import annotations

import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
QDRANT_PATH = ROOT / "qdrant_data"
COLLECTION_NAME = "qdrant_docs"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

VALID_STRATEGIES = {"fixed_500_50", "structure_800_100"}


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Answer questions using Qdrant documentation")
    parser.add_argument("--query", required=True, help="Natural language question")
    parser.add_argument(
        "--strategy",
        required=True,
        choices=sorted(VALID_STRATEGIES),
        help="Chunking strategy to use",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    return parser.parse_args()


def retrieve_chunks(query: str, strategy: str, top_k: int):
    client = QdrantClient(path=str(QDRANT_PATH))
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    query_vector = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()

    query_filter = Filter(
        must=[
            FieldCondition(
                key="chunk_strategy",
                match=MatchValue(value=strategy),
            )
        ]
    )

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k,
    )

    client.close()
    return response.points


def build_context(results) -> str:
    blocks = []

    for index, result in enumerate(results, start=1):
        payload = result.payload
        text = payload.get("text", "")

        blocks.append(
            f"[{index}]\n"
            f"Title: {payload.get('title', '')}\n"
            f"Section: {payload.get('section_path', '')}\n"
            f"Source URL: {payload.get('source_url', '')}\n"
            f"Chunk ID: {payload.get('chunk_id', '')}\n"
            f"Content:\n{text}"
        )

    return "\n\n".join(blocks)


def call_deepseek(query: str, context: str) -> str:
    load_dotenv(ROOT / ".env")

    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4.1-flash")

    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is missing. Please create .env first.")

    system_prompt = (
        "You are a technical Q&A assistant for Qdrant documentation. "
        "You must answer only using the numbered context provided by the user. "
        "Cite every factual statement with bracketed source numbers such as [1] or [2]. "
        "Do not use outside knowledge. "
        "If the context is insufficient, reply exactly: "
        "\"The provided Qdrant documentation does not contain enough information to answer this question.\" "
        "Do not guess. Keep the answer concise and technical."
    )

    user_prompt = (
        f"Question: {query}\n\n"
        f"Context:\n{context}\n\n"
        "Using only the context above, answer the question. "
        "Cite sources using [1], [2], etc. "
        "If the context is insufficient, say the documentation does not contain enough information."
    )

    response = requests.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 1200,
        },
        timeout=60,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"DeepSeek API failed with status {response.status_code}: {response.text}"
        )

    data = response.json()
    answer = data["choices"][0]["message"].get("content", "").strip()

    if not answer:
        raise RuntimeError("DeepSeek returned an empty answer.")

    return answer


def print_sources(results):
    print()
    print("Sources:")
    for index, result in enumerate(results, start=1):
        payload = result.payload
        print(
            f"[{index}] {payload.get('title', '')} - "
            f"{payload.get('section_path', '')}"
        )
        print(f"    URL: {payload.get('source_url', '')}")
        print(f"    Chunk ID: {payload.get('chunk_id', '')}")
        print(f"    Similarity score: {result.score:.4f}")


def main() -> int:
    args = parse_args()

    results = retrieve_chunks(
        query=args.query,
        strategy=args.strategy,
        top_k=args.top_k,
    )

    print(f"Query: {args.query}")
    print(f"Strategy: {args.strategy}")
    print(f"Retrieved chunks: {len(results)}")

    if not results:
        print(
            "The provided Qdrant documentation does not contain enough information "
            "to answer this question."
        )
        return 1

    context = build_context(results)
    answer = call_deepseek(args.query, context)

    print()
    print("Answer:")
    print(answer)

    print_sources(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())