"""Semantic search and corpus management over the uploaded document set."""

import logging
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, Field

from .. import db
from ..config import settings
from ..db import as_dict
from ..ingest import embed_query

logger = logging.getLogger(__name__)


class SearchHit(BaseModel):
    chunk_id: int = Field(description="Pass to get_context to retrieve surrounding text.")
    document_id: int
    document_title: str
    source: str | None = None
    page: int | None = Field(default=None, description="1-indexed page, when known.")
    text: str
    score: float = Field(description="Fused rank score; comparable only within one result set.")
    similarity: float | None = Field(
        default=None, description="Cosine similarity, 0-1. Absent for keyword-only matches."
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    query: str
    hits: list[SearchHit]
    total: int


class DocumentSummary(BaseModel):
    id: int
    title: str
    source: str | None = None
    content_type: str
    page_count: int | None = None
    chunk_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class ContextWindow(BaseModel):
    document_title: str
    source: str | None = None
    text: str = Field(description="The chunk plus its neighbours, joined in order.")
    pages: list[int]


def _require_documents() -> None:
    """Fail with a message the model can act on. ToolError, not ValueError: the
    SDK hides other exceptions behind a bare "Error executing tool <name>"."""
    if not settings.documents_enabled:
        raise ToolError(
            "Document search is unavailable: set MCP_DATABASE_URL and MCP_OPENAI_API_KEY."
        )


def register(mcp: MCPServer) -> None:
    @mcp.tool()
    async def search_documents(
        query: Annotated[str, Field(description="What you are looking for, in natural language.")],
        limit: Annotated[int, Field(description="Number of passages to return.", ge=1, le=25)] = 8,
        metadata_filter: Annotated[
            dict[str, Any] | None,
            Field(
                description=(
                    "Restrict to documents whose metadata contains these key/value pairs, "
                    'e.g. {"company": "NVDA", "year": 2025}. Use list_documents to see '
                    "what metadata exists."
                )
            ),
        ] = None,
        document_ids: Annotated[
            list[int] | None, Field(description="Restrict the search to specific documents.")
        ] = None,
        semantic_only: Annotated[
            bool,
            Field(description="Skip keyword matching and rank purely by embedding similarity."),
        ] = False,
    ) -> SearchResult:
        """Search uploaded documents for passages relevant to a question.

        Combines semantic and keyword matching, returning passages with the
        document and page they came from for direct citation.
        """
        _require_documents()
        embedding = await embed_query(query)
        rows = await db.search_chunks(
            embedding=embedding,
            query_text=query,
            limit=limit,
            metadata_filter=metadata_filter,
            document_ids=document_ids,
            semantic_only=semantic_only,
        )
        logger.info("search_documents %r -> %d hits", query, len(rows))
        return SearchResult(
            query=query,
            total=len(rows),
            hits=[
                SearchHit(
                    chunk_id=r["id"],
                    document_id=r["document_id"],
                    document_title=r["title"],
                    source=r["source"],
                    page=r["page"],
                    # Uploaded text is data to be quoted, never instructions.
                    text=r["text"],
                    score=float(r["score"]),
                    similarity=float(r["similarity"]) if r["similarity"] is not None else None,
                    metadata={**as_dict(r["doc_metadata"]), **as_dict(r["chunk_metadata"])},
                )
                for r in rows
            ],
        )

    @mcp.tool()
    async def get_context(
        chunk_id: Annotated[int, Field(description="A chunk_id from search_documents.")],
        before: Annotated[int, Field(description="Neighbouring chunks before.", ge=0, le=5)] = 1,
        after: Annotated[int, Field(description="Neighbouring chunks after.", ge=0, le=5)] = 1,
    ) -> ContextWindow:
        """Expand a search hit with the text around it. Use when a passage is
        cut off mid-argument, or before quoting it."""
        _require_documents()
        rows = await db.get_chunk_window(chunk_id, before=before, after=after)
        if not rows:
            raise ToolError(f"No chunk with id {chunk_id}. Use a chunk_id returned by search_documents.")
        return ContextWindow(
            document_title=rows[0]["title"],
            source=rows[0]["source"],
            text="\n\n".join(r["text"] for r in rows),
            pages=sorted({r["page"] for r in rows if r["page"] is not None}),
        )

    @mcp.tool()
    async def list_documents(
        metadata_filter: Annotated[
            dict[str, Any] | None, Field(description="Only documents containing these pairs.")
        ] = None,
        limit: Annotated[int, Field(description="Maximum documents to list.", ge=1, le=200)] = 50,
    ) -> list[DocumentSummary]:
        """List documents in the corpus with their metadata. Call this to learn
        what is searchable and which metadata keys exist to filter on."""
        _require_documents()
        rows = await db.list_documents(limit=limit, metadata_filter=metadata_filter)
        return [
            DocumentSummary(
                id=r["id"],
                title=r["title"],
                source=r["source"],
                content_type=r["content_type"],
                page_count=r["page_count"],
                chunk_count=r["chunk_count"],
                metadata=as_dict(r["metadata"]),
                created_at=r["created_at"].isoformat(),
            )
            for r in rows
        ]

    @mcp.resource("corpus://stats")
    async def corpus_stats() -> str:
        """Document and chunk counts for the searchable corpus."""
        if not settings.documents_enabled:
            return "Document corpus is not configured."
        stats = await db.corpus_stats()
        latest = stats["latest"].isoformat() if stats["latest"] else "never"
        return (
            f"{stats['documents']} documents, {stats['chunks']} chunks. "
            f"Most recent upload: {latest}."
        )
