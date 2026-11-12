# CS5481 Topic 5 — Qdrant Documentation QA

A retrieval-augmented question answering system built on the official Qdrant documentation. It retrieves relevant documentation chunks from a local Qdrant index, generates citation-aware answers with DeepSeek, and refuses to answer when the corpus does not contain enough information.

## Features

- 30 curated Qdrant documentation pages
- Two chunking strategies: `fixed_500_50` and `structure_800_100`
- Local Qdrant vector database
- `BAAI/bge-small-en-v1.5` embeddings
- Citation-aware answers with source links
- Refusal behavior for out-of-corpus questions
- Streamlit demo interface

## Quickstart

Activate the environment:

```powershell
conda activate cs5481qdrant
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Create `.env` in the project root:

```text
DEEPSEEK_API_KEY=your-api-key
DEEPSEEK_MODEL=deepseek-flash
```

Rebuild the pipeline:

```powershell
python src\download_docs.py
```

```powershell
python src\parse_docs.py
```

```powershell
python src\chunking.py
```

```powershell
python src\embed_and_store.py
```

Run the QA interface:

```powershell
streamlit run src\app.py
```

## Evaluation

Run:

```powershell
python src\evaluate.py --top-k 5
```

The benchmark contains 30 QA pairs: 20 answerable and 10 unanswerable.

| Strategy | Hit@5 | Answer correctness | Refusal accuracy | Overall | Avg latency |
|---|---:|---:|---:|---:|---:|
| `fixed_500_50` | 1.000 | 0.725 | 1.000 | 0.817 | 1.99 s |
| `structure_800_100` | 1.000 | 0.796 | 1.000 | 0.864 | 2.20 s |

Default strategy in the UI: `structure_800_100`.

## Project Structure

```text
docs/
data/
src/
evaluation/
qdrant_data/
```

`qdrant_data/` stores the local Qdrant index and can be rebuilt by rerunning `src\embed_and_store.py`.