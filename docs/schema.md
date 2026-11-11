# Data Schema

## manifest.csv

One row per source document.

| Field          | Type   | Description                      |
| -------------- | ------ | -------------------------------- |
| doc_id         | string | Stable document ID               |
| title          | string | Document title                   |
| source_url     | string | Original URL                     |
| publisher      | string | Publisher, e.g. Qdrant           |
| published_date | string | Publication or last-updated date |
| file_path      | string | Local raw file path              |
| file_type      | string | html or pdf                      |
| language       | string | Document language                |
| downloaded_at  | string | Download timestamp               |

## chunks.parquet

One row per chunk.

| Field           | Type    | Description                       |
| --------------- | ------- | --------------------------------- |
| chunk_id        | string  | Unique chunk ID                   |
| doc_id          | string  | Parent document ID                |
| title           | string  | Document title                    |
| source_url      | string  | Original URL                      |
| section_path    | string  | Heading path                      |
| anchor          | string  | Section anchor, if available      |
| chunk_strategy  | string  | fixed_500_50 or structure_800_100 |
| chunk_index     | integer | Chunk order inside the document   |
| text            | string  | Chunk text                        |
| token_count     | integer | Approximate token count           |
| embedding_model | string  | Embedding model name              |
| created_at      | string  | Creation timestamp                |