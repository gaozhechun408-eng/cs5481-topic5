from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
QDRANT_PATH = ROOT / "qdrant_data"
COLLECTION_NAME = "qdrant_docs"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

VALID_STRATEGIES = {"fixed_500_50", "structure_800_100"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search Qdrant documentation chunks")
    parser.add_argument("--query", required=True, help="Natural language query")
    parser.add_argument(
        "--strategy",
        required=True,
        choices=sorted(VALID_STRATEGIES),
        help="Chunking strategy to filter on",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    client = QdrantClient(path=str(QDRANT_PATH))
    model = SentenceTransformer(MODEL_NAME)

    query_vector = model.encode(
        args.query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()

    query_filter = Filter(
        must=[
            FieldCondition(
                key="chunk_strategy",
                match=MatchValue(value=args.strategy),
            )
        ]
    )

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=query_filter,
        limit=args.top_k,
    )

    results = response.points

    print(f"Query: {args.query}")
    print(f"Strategy: {args.strategy}")
    print(f"Results: {len(results)}")

    if not results:
        print("No results found.")
        client.close()
        return 1

    for rank, result in enumerate(results, start=1):
        payload = result.payload
        text = payload.get("text", "")

        print()
        print(f"[{rank}] score={result.score:.4f}")
        print(f"chunk_id: {payload.get('chunk_id')}")
        print(f"title: {payload.get('title')}")
        print(f"section: {payload.get('section_path')}")
        print(f"source: {payload.get('source_url')}")
        print(f"tokens: {payload.get('token_count')}")
        print("text:")
        print(textwrap.shorten(text, width=600, placeholder="..."))

    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())