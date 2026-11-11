from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
from bs4 import BeautifulSoup, NavigableString, Tag

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_FILE = ROOT / "data" / "manifest.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
SECTIONS_FILE = PROCESSED_DIR / "sections.parquet"
DOCUMENTS_FILE = PROCESSED_DIR / "documents.parquet"

BLOCK_TAGS = {
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "p",
    "pre",
    "blockquote",
    "ul",
    "ol",
    "table",
}

REMOVE_SELECTORS = (
    "script",
    "style",
    "noscript",
    "template",
    "nav",
    "footer",
    "aside",
    "form",
    "button",
    "svg",
    "img",
    ".copy-markdown",
    ".copy-button",
    ".header-anchor",
    ".theme-doc-toc",
    ".docs-toc",
    ".pagination",
    ".visually-hidden",
)


def normalize_inline(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_block(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:80]


def remove_noise(article: Tag) -> None:
    for selector in REMOVE_SELECTORS:
        for element in article.select(selector):
            element.decompose()


def iter_blocks(element: Tag) -> Iterable[Tag]:
    for child in element.children:
        if isinstance(child, NavigableString):
            continue
        if not isinstance(child, Tag):
            continue
        if child.name in {"script", "style", "noscript", "template"}:
            continue
        if child.name in BLOCK_TAGS:
            yield child
        else:
            yield from iter_blocks(child)


def block_to_text(element: Tag) -> str:
    if element.name == "pre":
        return normalize_block(element.get_text())

    if element.name in {"ul", "ol"}:
        items = []
        for index, li in enumerate(element.find_all("li", recursive=False), start=1):
            item_text = normalize_inline(li.get_text(" ", strip=True))
            prefix = "- " if element.name == "ul" else f"{index}. "
            items.append(prefix + item_text)
        return "\n".join(items)

    if element.name == "table":
        rows = []
        for tr in element.find_all("tr"):
            cells = [
                normalize_inline(cell.get_text(" ", strip=True))
                for cell in tr.find_all(["th", "td"])
            ]
            rows.append(" | ".join(cells))
        return "\n".join(rows)

    return normalize_inline(element.get_text(" ", strip=True))


def parse_document(html: str, doc_id: str, title: str, source_url: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")

    article = (
        soup.find("article", class_="documentation-article")
        or soup.find("article")
        or soup.find("main")
        or soup.body
    )
    if article is None:
        raise RuntimeError(f"No article element found for {doc_id}")

    remove_noise(article)

    heading_stack: list[tuple[int, str, str]] = []
    current_section_path = ["Preamble"]
    current_anchor = ""
    current_blocks: list[str] = []
    sections: list[dict] = []

    def flush_section() -> None:
        text = "\n\n".join(current_blocks).strip()
        if not text:
            return

        section_index = len(sections)
        sections.append(
            {
                "section_id": f"{doc_id}::{section_index:04d}",
                "doc_id": doc_id,
                "title": title,
                "source_url": source_url,
                "section_path": " > ".join(current_section_path),
                "anchor": current_anchor,
                "order_index": section_index,
                "text": text,
                "char_count": len(text),
            }
        )
        current_blocks.clear()

    for block in iter_blocks(article):
        if block.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            flush_section()

            level = int(block.name[1])
            heading_text = normalize_inline(block.get_text(" ", strip=True))
            anchor = block.get("id") or slugify(heading_text)

            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()

            heading_stack.append((level, heading_text, anchor))
            current_section_path = [item[1] for item in heading_stack]
            current_anchor = anchor
            current_blocks.append(f"{'#' * level} {heading_text}")
        else:
            text = block_to_text(block)
            if text:
                current_blocks.append(text)

    flush_section()

    if not sections:
        raise RuntimeError(f"No content extracted from {doc_id}")

    return sections


def main() -> int:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST_FILE)
    section_rows: list[dict] = []
    document_rows: list[dict] = []

    for row in manifest.itertuples(index=False):
        file_path = ROOT / row.file_path
        html = file_path.read_text(encoding="utf-8")

        sections = parse_document(
            html=html,
            doc_id=row.doc_id,
            title=row.title,
            source_url=row.source_url,
        )
        section_rows.extend(sections)

        document_rows.append(
            {
                "doc_id": row.doc_id,
                "title": row.title,
                "source_url": row.source_url,
                "section_count": len(sections),
                "text": "\n\n".join(section["text"] for section in sections),
            }
        )

    sections_df = pd.DataFrame(section_rows)
    documents_df = pd.DataFrame(document_rows)

    sections_df.to_parquet(SECTIONS_FILE, index=False)
    documents_df.to_parquet(DOCUMENTS_FILE, index=False)

    print(f"Parsed {len(documents_df)} documents into {len(sections_df)} sections.")
    print(f"Saved {SECTIONS_FILE.relative_to(ROOT).as_posix()}")
    print(f"Saved {DOCUMENTS_FILE.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())