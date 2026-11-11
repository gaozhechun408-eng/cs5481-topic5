from __future__ import annotations

import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SECTIONS_FILE = ROOT / "data" / "processed" / "sections.parquet"
CHUNKS_FILE = ROOT / "data" / "processed" / "chunks.parquet"

TOKEN_RE = re.compile(r"\S+")
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

FIXED_STRATEGY = "fixed_500_50"
STRUCTURE_STRATEGY = "structure_800_100"

FIXED_MAX_TOKENS = 500
FIXED_OVERLAP = 50

STRUCTURE_MAX_TOKENS = 800
STRUCTURE_OVERLAP = 100


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def token_count(text: str) -> int:
    return len(tokenize(text))


def sliding_chunks(
    tokens: list[str], max_tokens: int, overlap: int
) -> Iterable[list[str]]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap < 0 or overlap >= max_tokens:
        raise ValueError("overlap must be between 0 and max_tokens - 1")

    if len(tokens) <= max_tokens:
        yield tokens
        return

    step = max_tokens - overlap
    start = 0

    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        yield tokens[start:end]

        if end == len(tokens):
            break

        start += step


def parent_path(section_path: str, title: str) -> str:
    parts = [part.strip() for part in section_path.split(">") if part.strip()]
    if len(parts) <= 1:
        return title
    return " > ".join(parts[:-1])


def make_chunk(
    section: dict,
    strategy: str,
    chunk_index: int,
    tokens: list[str],
    section_path: str,
    anchor: str,
) -> dict:
    text = " ".join(tokens)

    return {
        "chunk_id": f"{section['doc_id']}::{strategy}::{chunk_index:04d}",
        "doc_id": section["doc_id"],
        "title": section["title"],
        "source_url": section["source_url"],
        "section_path": section_path,
        "anchor": anchor,
        "chunk_strategy": strategy,
        "chunk_index": chunk_index,
        "text": text,
        "token_count": len(tokens),
        "char_count": len(text),
        "embedding_model": EMBEDDING_MODEL,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def build_fixed_chunks(sections: pd.DataFrame) -> list[dict]:
    chunks: list[dict] = []

    for section in sections.to_dict("records"):
        tokens = tokenize(section["text"])
        for index, window in enumerate(
            sliding_chunks(tokens, FIXED_MAX_TOKENS, FIXED_OVERLAP)
        ):
            chunks.append(
                make_chunk(
                    section=section,
                    strategy=FIXED_STRATEGY,
                    chunk_index=index,
                    tokens=window,
                    section_path=section["section_path"],
                    anchor=section["anchor"],
                )
            )

    return chunks


def build_structure_chunks(sections: pd.DataFrame) -> list[dict]:
    chunks: list[dict] = []

    sections = sections.copy()
    sections["parent_path"] = sections.apply(
        lambda row: parent_path(row["section_path"], row["title"]), axis=1
    )

    grouped = sections.groupby(["doc_id", "parent_path"], sort=False)

    for (_, parent), group in grouped:
        group = group.sort_values("order_index")
        group_records = group.to_dict("records")
        if not group_records:
            continue

        representative = group_records[0]
        current_tokens: list[str] = []
        current_anchor = representative["anchor"]
        chunk_index = 0

        for section in group_records:
            section_tokens = tokenize(section["text"])
            if not section_tokens:
                continue

            if len(section_tokens) > STRUCTURE_MAX_TOKENS:
                if current_tokens:
                    chunks.append(
                        make_chunk(
                            section=representative,
                            strategy=STRUCTURE_STRATEGY,
                            chunk_index=chunk_index,
                            tokens=current_tokens,
                            section_path=parent,
                            anchor=current_anchor,
                        )
                    )
                    chunk_index += 1
                    current_tokens = []

                for window in sliding_chunks(
                    section_tokens, STRUCTURE_MAX_TOKENS, STRUCTURE_OVERLAP
                ):
                    chunks.append(
                        make_chunk(
                            section=section,
                            strategy=STRUCTURE_STRATEGY,
                            chunk_index=chunk_index,
                            tokens=window,
                            section_path=section["section_path"],
                            anchor=section["anchor"],
                        )
                    )
                    chunk_index += 1

                continue

            if current_tokens and (
                len(current_tokens) + len(section_tokens) > STRUCTURE_MAX_TOKENS
            ):
                chunks.append(
                    make_chunk(
                        section=representative,
                        strategy=STRUCTURE_STRATEGY,
                        chunk_index=chunk_index,
                        tokens=current_tokens,
                        section_path=parent,
                        anchor=current_anchor,
                    )
                )
                chunk_index += 1

                available_space = STRUCTURE_MAX_TOKENS - len(section_tokens)
                prefix_length = min(STRUCTURE_OVERLAP, max(0, available_space))
                prefix = (
                    current_tokens[-prefix_length:] if prefix_length > 0 else []
                )
                current_tokens = prefix + section_tokens
                current_anchor = section["anchor"]
            else:
                if not current_tokens:
                    current_anchor = section["anchor"]
                current_tokens.extend(section_tokens)

        if current_tokens:
            chunks.append(
                make_chunk(
                    section=representative,
                    strategy=STRUCTURE_STRATEGY,
                    chunk_index=chunk_index,
                    tokens=current_tokens,
                    section_path=parent,
                    anchor=current_anchor,
                )
            )

    return chunks


def main() -> int:
    sections = pd.read_parquet(SECTIONS_FILE)

    fixed_chunks = build_fixed_chunks(sections)
    structure_chunks = build_structure_chunks(sections)
    all_chunks = fixed_chunks + structure_chunks

    chunks_df = pd.DataFrame(all_chunks)

    chunks_df["chunk_index"] = chunks_df.groupby(
        ["doc_id", "chunk_strategy"]
    ).cumcount()

    chunks_df["chunk_id"] = (
        chunks_df["doc_id"]
        + "::"
        + chunks_df["chunk_strategy"]
        + "::"
        + chunks_df["chunk_index"].map("{:04d}".format)
    )

    chunks_df.to_parquet(CHUNKS_FILE, index=False)

    summary = (
        chunks_df.groupby("chunk_strategy")
        .agg(
            chunks=("chunk_id", "count"),
            avg_tokens=("token_count", "mean"),
            avg_chars=("char_count", "mean"),
        )
        .round(1)
    )

    print(f"Saved {len(chunks_df)} chunks to data/processed/chunks.parquet")
    print(summary.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())