from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from diskovod.attachments import AttachmentRepository


class Attachment(SimpleNamespace):
    async def read(self, *, use_cached: bool = False) -> bytes:
        assert use_cached is True
        return self.body


@pytest.mark.asyncio
async def test_attachment_is_content_addressed_and_searchable_per_chat(tmp_path: Path):
    repository = AttachmentRepository(tmp_path / "diskovod.sqlite3")
    body = b"The launch code name is blue heron."
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
    assert (tmp_path / "attachments" / digest[:2] / digest).read_bytes() == body
    assert repository.search("chat-a", "blue heron")[0]["filename"] == "notes.txt"
    assert repository.search("chat-b", "blue heron") == []
    assert repository.manifest() == [
        {"sha256": digest, "size": len(body), "storage_path": f"{digest[:2]}/{digest}"}
    ]
    repository.close()
