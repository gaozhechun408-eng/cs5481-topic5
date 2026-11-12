from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
CHUNKS_FILE = ROOT / "data" / "processed" / "chunks.parquet"
QDRANT_PATH = ROOT / "qdrant_data"
COLLECTION_NAME = "qdrant_docs"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
VECTOR_SIZE = 384
ENCODE_BATCH_SIZE = 32
UPSERT_BATCH_SIZE = 64


def main() -> int:
    chunks = pd.read_parquet(CHUNKS_FILE)
    if chunks.empty:
        raise RuntimeError("chunks.parquet is empty.")

    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(QDRANT_PATH))

    existing_collections = [
        collection.name
        for collection in client.get_collections().collections
    ]

    if COLLECTION_NAME in existing_collections:
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Embedding {len(chunks)} chunks...")
    embeddings = model.encode(
        chunks["text"].tolist(),
        batch_size=ENCODE_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    if embeddings.shape[0] != len(chunks):
        raise RuntimeError(
            f"Embedding count mismatch: {embeddings.shape[0]} vs {len(chunks)}"
        )

    for start in range(0, len(chunks), UPSERT_BATCH_SIZE):
        end = min(start + UPSERT_BATCH_SIZE, len(chunks))
        batch = chunks.iloc[start:end]
        vectors = embeddings[start:end]

        points = []
        for point_id, (row, vector) in enumerate(
            zip(batch.to_dict("records"), vectors), start=start
        ):
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    payload=row,
                )
            )

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )

    total_points = client.count(
        collection_name=COLLECTION_NAME,
        exact=True,
    ).count

    print(f"Qdrant collection '{COLLECTION_NAME}' now has {total_points} points.")
    print("Chunk strategy counts:")
    print(chunks["chunk_strategy"].value_counts().to_string())

    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())