from __future__ import annotations

import httpx2
import pytest

from diskovod.http_client import PublicHTTPResponse
from diskovod.web_access import MAX_FETCH_BYTES, fetch_url, search_web


class RecordingHTTP:
    def __init__(self, responses: list[PublicHTTPResponse]):
        self.responses = responses
        self.calls: list[tuple[str, int]] = []

    async def get(self, url: str, *, max_bytes: int, timeout=None) -> PublicHTTPResponse:
        del timeout
        self.calls.append((url, max_bytes))
        return self.responses.pop(0)


def response(url: str, body: str) -> PublicHTTPResponse:
    return PublicHTTPResponse(
        url=url,
        status_code=200,
        headers=httpx2.Headers({"Content-Type": "text/html; charset=utf-8"}),
        content=body.encode(),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_web_operations_reuse_the_injected_bounded_http_client() -> None:
    http = RecordingHTTP(
        [
            response("https://example.test/final", "<h1>Example</h1><p>Readable body</p>"),
            response(
                "https://html.duckduckgo.com/html/?q=example",
                '<a class="result__a" href="https://result.test/page">Search result</a>',
            ),
        ]
    )

    fetched = await fetch_url(http, "https://example.test/start")
    results = await search_web(http, "example")

    assert fetched["url"] == "https://example.test/final"
    assert "Example" in fetched["text"]
    assert "Readable body" in fetched["text"]
    assert results == [{"title": "Search result", "url": "https://result.test/page"}]
    assert http.calls == [
        ("https://example.test/start", MAX_FETCH_BYTES),
        ("https://html.duckduckgo.com/html/?q=example", MAX_FETCH_BYTES),
    ]
