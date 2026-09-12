import pytest
from starlette.testclient import TestClient

from mcp_server.app import create_app
from mcp_server.server import mcp

# --- registration ---------------------------------------------------------


async def test_tools_are_registered():
    names = {tool.name for tool in await mcp.list_tools()}
    assert {"web_search", "search_documents", "get_context", "list_documents"} <= names


async def test_search_documents_advertises_metadata_filter():
    """The filter must reach the model through the schema, not just exist."""
    tool = next(t for t in await mcp.list_tools() if t.name == "search_documents")
    assert "metadata_filter" in tool.input_schema["properties"]


# --- HTTP surface ---------------------------------------------------------


@pytest.fixture
def client(monkeypatch):
    from mcp_server import app as app_module

    monkeypatch.setattr(app_module.settings, "auth_token", "test-token")
    # The context manager runs the lifespan that starts the session manager's
    # task group; without it every /mcp request 500s.
    with TestClient(create_app()) as test_client:
        yield test_client


def test_health_is_open_and_reports_capabilities(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    # Unconfigured in tests; the point is that the probe says so.
    assert body["documents"] is False
    assert body["web_search"] is False


def test_mcp_endpoint_rejects_missing_token(client):
    response = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"})
    assert response.status_code == 401


def test_mcp_endpoint_rejects_wrong_token(client):
    response = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
        headers={"Authorization": "Bearer wrong"},
    )
    assert response.status_code == 401


def test_initialize_with_valid_token(client):
    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0"},
            },
        },
        headers={
            "Authorization": "Bearer test-token",
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        },
    )
    assert response.status_code == 200
    assert "mcp-server" in response.text


def test_upload_requires_auth(client):
    """An open ingestion endpoint would let anyone poison the cited corpus."""
    response = client.post("/documents", files={"file": ("x.txt", b"hello", "text/plain")})
    assert response.status_code == 401


def test_upload_reports_unconfigured(client):
    response = client.post(
        "/documents",
        files={"file": ("x.txt", b"hello", "text/plain")},
        headers={"Authorization": "Bearer test-token"},
    )
    assert response.status_code == 503
    assert "MCP_DATABASE_URL" in response.json()["error"]


# --- text processing (no network, no database) ----------------------------


def test_clean_text_dehyphenates_across_line_breaks():
    from mcp_server.ingest import clean_text

    # PDF extraction splits words at line ends, making them unsearchable.
    assert "revenue" in clean_text("total rev-\nenue grew")


def test_clean_text_preserves_paragraph_breaks():
    from mcp_server.ingest import clean_text

    cleaned = clean_text("First para line one\nline two\n\nSecond para")
    assert "line one line two" in cleaned
    assert "\n\n" in cleaned


def test_chunk_text_splits_on_paragraphs():
    from mcp_server.ingest import chunk_text

    text = "\n\n".join(f"Paragraph {i} " + "word " * 200 for i in range(5))
    chunks = chunk_text(text, max_tokens=256, overlap_tokens=32)
    assert len(chunks) > 1
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_chunk_text_splits_oversized_paragraph():
    """A huge paragraph must still be broken up, not sent past the token limit."""
    from mcp_server.ingest import chunk_text

    chunks = chunk_text("word " * 3000, max_tokens=128, overlap_tokens=16)
    assert len(chunks) > 1
    assert all(c.text for c in chunks)


def test_chunk_pages_keeps_ordinals_contiguous_and_tracks_pages():
    from mcp_server.ingest import chunk_pages

    chunks = chunk_pages([(1, "alpha " * 400), (2, "beta " * 400)])
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert {c.page for c in chunks} == {1, 2}


def test_vector_literal_format():
    from mcp_server.db import to_vector_literal

    assert to_vector_literal([1.0, -0.5, 0.25]) == "[1,-0.5,0.25]"


# --- web search -----------------------------------------------------------


async def test_web_search_requires_key(monkeypatch):
    from mcp_server import config

    monkeypatch.setattr(config.settings, "tavily_api_key", "")
    with pytest.raises(Exception, match="MCP_TAVILY_API_KEY"):
        await mcp.call_tool("web_search", {"query": "anything"})


async def test_web_search_parses_results(monkeypatch):
    import httpx

    from mcp_server import config

    monkeypatch.setattr(config.settings, "tavily_api_key", "fake-key")

    captured = {}

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, json=None):
            captured["url"] = url
            captured["payload"] = json
            return httpx.Response(
                200,
                json={
                    "answer": "Synthesized answer.",
                    "results": [
                        {
                            "title": "Result one",
                            "url": "https://example.com/a",
                            "content": "Relevant passage.",
                            "score": 0.93,
                        }
                    ],
                },
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    result = await mcp.call_tool("web_search", {"query": "nvidia revenue", "max_results": 3})
    payload = str(result)

    assert "Result one" in payload
    assert captured["payload"]["query"] == "nvidia revenue"
    assert captured["payload"]["max_results"] == 3
    # The key travels in the body, never in the URL or a log line.
    assert "fake-key" not in captured["url"]


async def test_document_tools_report_unconfigured(monkeypatch):
    from mcp_server import config

    monkeypatch.setattr(config.settings, "database_url", "")
    with pytest.raises(Exception, match="MCP_DATABASE_URL"):
        await mcp.call_tool("search_documents", {"query": "anything"})


def test_as_dict_handles_both_dict_and_json_string():
    """JSONB arrives as a dict with the codec and as text without it; spreading
    a string raises where the model only sees 'Error executing tool'."""
    from mcp_server.db import as_dict

    assert as_dict({"a": 1}) == {"a": 1}
    assert as_dict('{"a": 1}') == {"a": 1}
    assert as_dict(None) == {}
    assert as_dict("") == {}
    assert as_dict("not json") == {}
    assert as_dict("[1,2]") == {}  # valid JSON, wrong shape
