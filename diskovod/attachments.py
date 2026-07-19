from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Iterable

import httpx2

from .http_client import PublicHTTP, PublicNetworkError
from .models import (
    MAX_INLINE_TEXT_BYTES,
    MAX_INLINE_TEXT_CHARACTERS,
    MAX_NATIVE_ATTACHMENT_BYTES,
    discord_attachment_metadata,
    is_text_attachment,
)
from .persistence import AsyncSQLite

CHUNK_CHARACTERS = 4_000
MAX_EXTRACTED_CHARACTERS = 200_000
log = logging.getLogger(__name__)


class AttachmentRepository:
    """Content-addressed attachment objects and bounded per-chat lexical retrieval."""

    def __init__(
        self,
        database: AsyncSQLite,
        http: PublicHTTP,
        object_root: Path | None = None,
    ):
        self.database = database
        database_path = database.path
        self.http = http
        self.object_root = object_root or database_path.parent / "attachments"
        self.object_root.mkdir(parents=True, exist_ok=True)

    async def capture(
        self,
        values: Iterable[Any],
        *,
        channel_id: str,
        message_id: str,
    ) -> list[dict[str, Any]]:
        captured = discord_attachment_metadata(values)
        inline_bytes = 0
        for metadata in captured:
            size = int(metadata.get("size") or 0)
            if size <= 0 or size > MAX_NATIVE_ATTACHMENT_BYTES:
                continue
            try:
                response = await self.http.get(
                    str(metadata.get("url") or ""),
                    max_bytes=size,
                    timeout=httpx2.Timeout(60, connect=8),
                )
            except Exception as error:
                code = str(error) if isinstance(error, (PublicNetworkError, httpx2.HTTPError)) else ""
                metadata["ingestion_error"] = code or type(error).__name__
                log.warning("Could not download Discord attachment %s: %s", metadata["filename"], error)
                continue
            if response.status_code < 200 or response.status_code >= 300:
                metadata["ingestion_error"] = f"http_status_{response.status_code}"
                continue
            body = response.content
            if len(body) != size:
                metadata["ingestion_error"] = "size_mismatch"
                continue
            remaining = MAX_INLINE_TEXT_BYTES - inline_bytes
            if (
                remaining > 0
                and is_text_attachment(
                    str(metadata.get("filename") or ""),
                    str(metadata.get("content_type") or ""),
                )
                and len(body) <= remaining
                and b"\0" not in body
            ):
                text = body.decode("utf-8", errors="replace").strip()
                if text:
                    metadata["text"] = text[:MAX_INLINE_TEXT_CHARACTERS]
                    inline_bytes += len(body)
            digest = await self._store_capture(
                channel_id,
                message_id,
                metadata,
                body,
            )
            metadata["sha256"] = digest
        return captured

    async def _store_capture(
        self,
        channel_id: str,
        message_id: str,
        metadata: dict[str, Any],
        body: bytes,
    ) -> str:
        media_type = str(metadata.get("content_type") or "")
        digest = hashlib.sha256(body).hexdigest()
        relative = Path(digest[:2]) / digest
        target = self.object_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
            with temporary.open("xb") as stream:
                stream.write(body)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(target)
        async with self.database.transaction() as connection:
            await connection.execute(
                """
                INSERT OR IGNORE INTO attachment_objects(
                  sha256, size, media_type, storage_path, created_at
                ) VALUES(?, ?, ?, ?, ?)
                """,
                (digest, len(body), media_type, str(relative), time.time()),
            )
            await connection.execute(
                """
                INSERT OR IGNORE INTO attachment_references(
                  channel_id, message_id, attachment_id, filename,
                  object_sha256, metadata, created_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    channel_id,
                    message_id,
                    str(metadata.get("id") or digest),
                    str(metadata.get("filename") or "attachment")[:255],
                    digest,
                    json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
                    time.time(),
                ),
            )
            if text := metadata.get("text"):
                bounded = str(text)[:MAX_EXTRACTED_CHARACTERS]
                now = time.time()
                existing = await (
                    await connection.execute(
                        "SELECT id FROM attachment_artifacts WHERE object_sha256=? AND kind='text'",
                        (digest,),
                    )
                ).fetchone()
                if existing:
                    return digest
                cursor = await connection.execute(
                    """
                    INSERT INTO attachment_artifacts(
                      object_sha256, kind, state, content, metadata, created_at, updated_at
                    ) VALUES(?, 'text', 'ready', ?, ?, ?, ?)
                    """,
                    (
                        digest,
                        bounded,
                        json.dumps({"filename": metadata.get("filename")}, ensure_ascii=False),
                        now,
                        now,
                    ),
                )
                artifact_id = int(cursor.lastrowid)
                for index, start in enumerate(range(0, len(bounded), CHUNK_CHARACTERS)):
                    content = bounded[start : start + CHUNK_CHARACTERS]
                    chunk = await connection.execute(
                        "INSERT INTO attachment_chunks(artifact_id, chunk_index, content, metadata) "
                        "VALUES(?, ?, ?, ?)",
                        (artifact_id, index, content, "{}"),
                    )
                    await connection.execute(
                        "INSERT INTO attachment_chunks_fts(rowid, content) VALUES(?, ?)",
                        (int(chunk.lastrowid), content),
                    )
        return digest

    async def search(self, channel_id: str, query: str, limit: int = 5) -> list[dict[str, Any]]:
        terms = " ".join(f'"{part}"' for part in query.replace('"', " ").split() if part)[:500]
        if not terms:
            return []
        async with self.database.transaction() as connection:
            rows = await (
                await connection.execute(
                    """
                SELECT r.message_id, r.attachment_id, r.filename, r.object_sha256,
                       c.chunk_index, snippet(attachment_chunks_fts, 0, '[', ']', ' … ', 30) excerpt
                FROM attachment_chunks_fts
                JOIN attachment_chunks c ON c.id=attachment_chunks_fts.rowid
                JOIN attachment_artifacts a ON a.id=c.artifact_id
                JOIN attachment_references r ON r.object_sha256=a.object_sha256
                WHERE r.channel_id=? AND attachment_chunks_fts MATCH ?
                ORDER BY bm25(attachment_chunks_fts) LIMIT ?
                """,
                    (channel_id, terms, max(1, min(limit, 10))),
                )
            ).fetchall()
        return [dict(row) for row in rows]

    async def manifest(self) -> list[dict[str, Any]]:
        async with self.database.transaction() as connection:
            rows = await (
                await connection.execute(
                    "SELECT sha256, size, storage_path FROM attachment_objects ORDER BY sha256"
                )
            ).fetchall()
        return [dict(row) for row in rows]
