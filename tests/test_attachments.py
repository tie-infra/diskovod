from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import httpx2
import pytest

from diskovod.attachments import AttachmentRepository
from diskovod.http_client import PublicHTTPResponse
from diskovod.store import Store


class Attachment(SimpleNamespace):
    async def read(self, *, use_cached: bool = False) -> bytes:
        raise AssertionError("Discord's HTTP client must not download attachments")


class RecordingHTTP:
    def __init__(self, body: bytes):
        self.body = body
        self.calls: list[tuple[str, int]] = []

    async def get(self, url: str, *, max_bytes: int, timeout=None) -> PublicHTTPResponse:
        del timeout
        self.calls.append((url, max_bytes))
        return PublicHTTPResponse(
            url=url,
            status_code=200,
            headers=httpx2.Headers({"Content-Type": "text/plain"}),
            content=self.body,
            encoding="utf-8",
        )


@pytest.mark.asyncio
async def test_attachment_is_content_addressed_and_searchable_per_chat(tmp_path: Path):
    body = b"The launch code name is blue heron."
    http = RecordingHTTP(body)
    store = Store(tmp_path / "diskovod.sqlite3", "x" * 32)
    repository = AttachmentRepository(store.database, http)
    attachment = Attachment(
        id="attachment-1",
        filename="notes.txt",
        content_type="text/plain",
        size=len(body),
        url="https://cdn.example/notes.txt",
        description=None,
        body=body,
    )

    captured = await repository.capture(
        [attachment],
        channel_id="chat-a",
        message_id="message-1",
    )

    digest = captured[0]["sha256"]
    assert captured[0]["text"] == body.decode()
    assert http.calls == [("https://cdn.example/notes.txt", len(body))]
    assert (tmp_path / "attachments" / digest[:2] / digest).read_bytes() == body
    assert (await repository.search("chat-a", "blue heron"))[0]["filename"] == "notes.txt"
    assert await repository.search("chat-b", "blue heron") == []
    assert await repository.manifest() == [
        {"sha256": digest, "size": len(body), "storage_path": f"{digest[:2]}/{digest}"}
    ]
    await store.aclose()
