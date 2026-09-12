"""Postgres + pgvector access: `documents` (one row per file) and `chunks` (the
embedded pieces), with JSONB metadata so new filter keys need no migration.
"""

import json
import logging
from contextlib import asynccontextmanager
from typing import Any

import asyncpg

from .config import settings

logger = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None


SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id           BIGSERIAL PRIMARY KEY,
    title        TEXT NOT NULL,
    source       TEXT,
    content_type TEXT NOT NULL DEFAULT 'application/pdf',
    -- Content hash: re-uploading the same file updates this row rather than
    -- duplicating every chunk into search results.
    sha256       TEXT NOT NULL UNIQUE,
    page_count   INTEGER,
    metadata     JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal     INTEGER NOT NULL,
    page        INTEGER,
    text        TEXT NOT NULL,
    embedding   vector(:DIMS) NOT NULL,
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    -- Generated, so keyword search can never drift out of sync with the text.
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    UNIQUE (document_id, ordinal)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_metadata_idx ON chunks USING gin (metadata);
CREATE INDEX IF NOT EXISTS documents_metadata_idx ON documents USING gin (metadata);
"""


async def get_pool() -> asyncpg.Pool:
    """Lazily create the connection pool, so one code path serves stdio,
    uvicorn, and tests alike. Kept small because Cloud Run cold-starts often."""
    global _pool
    if _pool is None:
        if not settings.database_url:
            raise RuntimeError("MCP_DATABASE_URL is not set — document tools are disabled.")
        async def init_connection(conn: asyncpg.Connection) -> None:
            # asyncpg hands back JSONB as a raw string; without this codec every
            # metadata field arrives as text and fails model validation.
            await conn.set_type_codec(
                "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
            )

        _pool = await asyncpg.create_pool(
            settings.database_url,
            init=init_connection,
            min_size=0,
            max_size=5,
            # Neon's pooler runs pgbouncer in transaction mode, where reusing
            # prepared statements raises DuplicatePreparedStatement under load.
            statement_cache_size=0,
            command_timeout=30,
        )
        logger.info("database pool created")
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


@asynccontextmanager
async def connection():
    pool = await get_pool()
    async with pool.acquire() as conn:
        yield conn


async def init_schema() -> None:
    """Apply the schema. Idempotent, so it is safe to run on every deploy."""
    async with connection() as conn:
        # Plain replace, not str.format: SQL is full of braces.
        await conn.execute(SCHEMA_SQL.replace(":DIMS", str(settings.embedding_dimensions)))
    logger.info("schema applied")


def as_dict(value: Any) -> dict[str, Any]:
    """Coerce a JSONB column to a dict.

    The pool registers a jsonb codec, but a connection that misses it returns
    raw text, and spreading a string raises deep inside a tool where the model
    only sees "Error executing tool". Cheap here, and it cannot regress.
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            logger.warning("metadata column was not valid JSON: %.60s", value)
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def to_vector_literal(embedding: list[float]) -> str:
    """pgvector's text input format. asyncpg has no codec for the vector type,
    so values cross the wire as strings and are cast with `$n::vector`."""
    return "[" + ",".join(f"{x:.7g}" for x in embedding) + "]"


async def upsert_document(
    *,
    title: str,
    source: str | None,
    content_type: str,
    sha256: str,
    page_count: int | None,
    metadata: dict[str, Any],
) -> tuple[int, bool]:
    """Insert or update a document by content hash, returning (id, is_new).
    Re-uploading the same bytes reuses the row instead of duplicating it."""
    async with connection() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO documents (title, source, content_type, sha256, page_count, metadata)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb)
            ON CONFLICT (sha256) DO UPDATE
                SET title = EXCLUDED.title,
                    source = EXCLUDED.source,
                    page_count = EXCLUDED.page_count,
                    metadata = EXCLUDED.metadata
            RETURNING id, (xmax = 0) AS is_new
            """,
            title, source, content_type, sha256, page_count, json.dumps(metadata),
        )
        return row["id"], row["is_new"]


async def replace_chunks(document_id: int, chunks: list[dict[str, Any]]) -> int:
    """Atomically swap in a document's chunks. Delete-then-insert in one
    transaction, so a concurrent search never sees a half-indexed document."""
    async with connection() as conn, conn.transaction():
        await conn.execute("DELETE FROM chunks WHERE document_id = $1", document_id)
        await conn.executemany(
            """
                INSERT INTO chunks (document_id, ordinal, page, text, embedding, metadata)
                VALUES ($1, $2, $3, $4, $5::vector, $6::jsonb)
                """,
            [
                (
                    document_id,
                    c["ordinal"],
                    c.get("page"),
                    c["text"],
                    to_vector_literal(c["embedding"]),
                    json.dumps(c.get("metadata", {})),
                )
                for c in chunks
            ],
        )
    return len(chunks)


async def search_chunks(
    *,
    embedding: list[float],
    query_text: str,
    limit: int = 8,
    metadata_filter: dict[str, Any] | None = None,
    document_ids: list[int] | None = None,
    semantic_only: bool = False,
) -> list[dict[str, Any]]:
    """Hybrid search fusing vector similarity with keyword relevance.

    Vectors miss exact terms like tickers and keywords miss paraphrase, so
    Reciprocal Rank Fusion combines them by rank alone.
    """
    vec = to_vector_literal(embedding)
    # Over-fetch per arm so fusion has room to reorder.
    candidates = max(limit * 4, 20)
    meta_json = json.dumps(metadata_filter) if metadata_filter else None

    sql = """
    WITH filtered AS (
        SELECT c.id, c.document_id, c.ordinal, c.page, c.text, c.embedding, c.tsv,
               d.title, d.source, d.metadata AS doc_metadata
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE ($3::jsonb IS NULL OR d.metadata @> $3::jsonb)
          AND ($4::bigint[] IS NULL OR c.document_id = ANY($4::bigint[]))
    ),
    vector_hits AS (
        SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> $1::vector) AS rank,
               1 - (embedding <=> $1::vector) AS similarity
        FROM filtered
        ORDER BY embedding <=> $1::vector
        LIMIT $5
    ),
    keyword_hits AS (
        SELECT id, ROW_NUMBER() OVER (
                   ORDER BY ts_rank_cd(tsv, websearch_to_tsquery('english', $2)) DESC
               ) AS rank
        FROM filtered
        WHERE $6 = FALSE
          AND websearch_to_tsquery('english', $2) @@ tsv
        ORDER BY ts_rank_cd(tsv, websearch_to_tsquery('english', $2)) DESC
        LIMIT $5
    ),
    fused AS (
        SELECT COALESCE(v.id, k.id) AS id,
               -- RRF with k=60 (the original paper's constant), which damps the
               -- top ranks so neither arm dominates.
               COALESCE(1.0 / (60 + v.rank), 0) + COALESCE(1.0 / (60 + k.rank), 0) AS score,
               v.similarity
        FROM vector_hits v
        FULL OUTER JOIN keyword_hits k ON k.id = v.id
    )
    SELECT f.id, f.score, f.similarity, c.document_id, c.ordinal, c.page, c.text,
           c.metadata AS chunk_metadata, d.title, d.source, d.metadata AS doc_metadata
    FROM fused f
    JOIN chunks c ON c.id = f.id
    JOIN documents d ON d.id = c.document_id
    ORDER BY f.score DESC
    LIMIT $7
    """
    async with connection() as conn:
        rows = await conn.fetch(
            sql, vec, query_text, meta_json, document_ids, candidates, semantic_only, limit
        )
    return [dict(r) for r in rows]


async def list_documents(
    *, limit: int = 50, metadata_filter: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    meta_json = json.dumps(metadata_filter) if metadata_filter else None
    async with connection() as conn:
        rows = await conn.fetch(
            """
            SELECT d.id, d.title, d.source, d.content_type, d.page_count,
                   d.metadata, d.created_at,
                   (SELECT count(*) FROM chunks c WHERE c.document_id = d.id) AS chunk_count
            FROM documents d
            WHERE ($1::jsonb IS NULL OR d.metadata @> $1::jsonb)
            ORDER BY d.created_at DESC
            LIMIT $2
            """,
            meta_json, limit,
        )
    return [dict(r) for r in rows]


async def get_chunk_window(chunk_id: int, before: int = 1, after: int = 1) -> list[dict[str, Any]]:
    """Fetch a chunk plus its neighbours, so a citation can quote fairly.
    Neighbours come from the same document by ordinal."""
    async with connection() as conn:
        rows = await conn.fetch(
            """
            WITH target AS (SELECT document_id, ordinal FROM chunks WHERE id = $1)
            SELECT c.id, c.ordinal, c.page, c.text, d.title, d.source, d.metadata AS doc_metadata
            FROM chunks c
            JOIN target t ON c.document_id = t.document_id
            JOIN documents d ON d.id = c.document_id
            WHERE c.ordinal BETWEEN t.ordinal - $2 AND t.ordinal + $3
            ORDER BY c.ordinal
            """,
            chunk_id, before, after,
        )
    return [dict(r) for r in rows]


async def delete_document(document_id: int) -> bool:
    async with connection() as conn:
        result = await conn.execute("DELETE FROM documents WHERE id = $1", document_id)
    return result.endswith("1")


async def corpus_stats() -> dict[str, Any]:
    async with connection() as conn:
        row = await conn.fetchrow(
            """
            SELECT (SELECT count(*) FROM documents) AS documents,
                   (SELECT count(*) FROM chunks) AS chunks,
                   (SELECT max(created_at) FROM documents) AS latest
            """
        )
    return dict(row)
