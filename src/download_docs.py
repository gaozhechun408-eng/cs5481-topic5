from __future__ import annotations

import hashlib
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
SOURCES_FILE = ROOT / "docs" / "sources.txt"
RAW_DIR = ROOT / "data" / "raw"
MANIFEST_FILE = ROOT / "data" / "manifest.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0 Safari/537.36 CS5481/1.0"
    )
}


def make_doc_id(url: str) -> str:
    parsed = urlparse(url)
    parts = [part for part in parsed.path.split("/") if part and part != "documentation"]
    if not parts:
        parts = ["docs"]

    slug = "-".join(parts[-4:])
    slug = re.sub(r"[^a-z0-9]+", "-", slug.lower()).strip("-")
    slug = slug[:80].rstrip("-") or "docs"
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    return f"qdrant-{slug}-{digest}"


def extract_metadata(html: str, fallback_title: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "lxml")

    title = ""
    og_title = soup.find("meta", attrs={"property": "og:title"})
    if og_title and og_title.get("content"):
        title = og_title["content"].strip()
    if not title and soup.title:
        title = soup.title.get_text(strip=True)
    title = re.sub(r"\s+", " ", title).strip() or fallback_title

    published_date = ""
    for attr in ("property", "name"):
        meta = (
            soup.find("meta", attrs={attr: "article:modified_time"})
            or soup.find("meta", attrs={attr: "og:updated_time"})
        )
        if meta and meta.get("content"):
            published_date = meta["content"].strip()
            break

    return title, published_date


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(HEADERS)

    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    urls = [
        line.strip()
        for line in SOURCES_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not urls:
        raise RuntimeError("docs/sources.txt is empty.")

    rows = []
    failures = []
    session = build_session()

    for url in tqdm(urls, desc="Downloading Qdrant docs", unit="page"):
        doc_id = make_doc_id(url)
        file_path = RAW_DIR / f"{doc_id}.html"

        try:
            if file_path.exists() and file_path.stat().st_size > 0:
                html = file_path.read_text(encoding="utf-8")
            else:
                response = session.get(url, timeout=30)
                response.raise_for_status()
                response.encoding = response.apparent_encoding or "utf-8"
                html = response.text
                file_path.write_text(html, encoding="utf-8")

            title, published_date = extract_metadata(html, doc_id)
            rows.append(
                {
                    "doc_id": doc_id,
                    "title": title,
                    "source_url": url,
                    "publisher": "Qdrant",
                    "published_date": published_date,
                    "file_path": file_path.relative_to(ROOT).as_posix(),
                    "file_type": "html",
                    "language": "en",
                    "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
            )
        except Exception as exc:
            failures.append((url, str(exc)))

    columns = [
        "doc_id",
        "title",
        "source_url",
        "publisher",
        "published_date",
        "file_path",
        "file_type",
        "language",
        "downloaded_at",
    ]
    pd.DataFrame(rows, columns=columns).to_csv(
        MANIFEST_FILE, index=False, encoding="utf-8"
    )

    print(f"Processed {len(rows)} pages; saved manifest to data/manifest.csv")

    if failures:
        print("Failed pages:")
        for url, error in failures:
            print(f"  - {url}: {error}")
        return 1

    print("All pages processed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())