from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from .models import MAX_NATIVE_ATTACHMENT_BYTES, capture_discord_attachments
from .persistence import SQLITE_BUSY_TIMEOUT_MS, initialize_target_schema

CHUNK_CHARACTERS = 4_000
MAX_EXTRACTED_CHARACTERS = 200_000


class AttachmentRepository:
    """Content-addressed attachment objects and bounded per-chat lexical retrieval."""

    def __init__(self, database_path: Path, object_root: Path | None = None):
        self.database_path = database_path
        self.object_root = object_root or database_path.parent / "attachments"
        self.object_root.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(database_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._connection:
            initialize_target_schema(self._connection)
            self._connection.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    async def capture(
        self,
        values: Iterable[Any],
        *,
        channel_id: str,
        message_id: str,
    ) -> list[dict[str, Any]]:
        attachments = list(values)
        captured = await capture_discord_attachments(attachments)
        by_id = {str(getattr(item, "id", "")): item for item in attachments}
        for metadata in captured:
            source = by_id.get(str(metadata.get("id") or ""))
            size = int(metadata.get("size") or 0)
            if source is None or size <= 0 or size > MAX_NATIVE_ATTACHMENT_BYTES:
                continue
            try:
                body = await source.read(use_cached=True)
            except Exception as error:
                metadata["ingestion_error"] = type(error).__name__
                continue
            if len(body) != size:
                metadata["ingestion_error"] = "size_mismatch"
                continue
            digest = self.put_object(body, str(metadata.get("content_type") or ""))
            metadata["sha256"] = digest
            self.add_reference(channel_id, message_id, metadata, digest)
            if text := metadata.get("text"):
                self.index_text(digest, str(text), {"filename": metadata.get("filename")})
        return captured

    def put_object(self, body: bytes, media_type: str) -> str:
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
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT OR IGNORE INTO attachment_objects(
                  sha256, size, media_type, storage_path, created_at
                ) VALUES(?, ?, ?, ?, ?)
                """,
                (digest, len(body), media_type, str(relative), time.time()),
            )
        return digest

    def add_reference(
        self,
        channel_id: str,
        message_id: str,
        metadata: dict[str, Any],
        digest: str,
    ) -> None:
        with self._lock, self._connection:
            self._connection.execute(
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

    def index_text(self, digest: str, text: str, metadata: dict[str, Any]) -> None:
        bounded = text[:MAX_EXTRACTED_CHARACTERS]
        now = time.time()
        with self._lock, self._connection:
            existing = self._connection.execute(
                "SELECT id FROM attachment_artifacts WHERE object_sha256=? AND kind='text'",
                (digest,),
            ).fetchone()
            if existing:
                return
            cursor = self._connection.execute(
                """
                INSERT INTO attachment_artifacts(
                  object_sha256, kind, state, content, metadata, created_at, updated_at
                ) VALUES(?, 'text', 'ready', ?, ?, ?, ?)
                """,
                (
                    digest,
                    bounded,
                    json.dumps(metadata, ensure_ascii=False),
                    now,
                    now,
                ),
            )
            artifact_id = int(cursor.lastrowid)
            for index, start in enumerate(range(0, len(bounded), CHUNK_CHARACTERS)):
                content = bounded[start : start + CHUNK_CHARACTERS]
                chunk = self._connection.execute(
                    "INSERT INTO attachment_chunks(artifact_id, chunk_index, content, metadata) "
                    "VALUES(?, ?, ?, ?)",
                    (artifact_id, index, content, "{}"),
                )
                self._connection.execute(
                    "INSERT INTO attachment_chunks_fts(rowid, content) VALUES(?, ?)",
                    (int(chunk.lastrowid), content),
                )

    def search(self, channel_id: str, query: str, limit: int = 5) -> list[dict[str, Any]]:
        terms = " ".join(f'"{part}"' for part in query.replace('"', " ").split() if part)[:500]
        if not terms:
            return []
        with self._lock:
            rows = self._connection.execute(
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
            ).fetchall()
        return [dict(row) for row in rows]

    def manifest(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT sha256, size, storage_path FROM attachment_objects ORDER BY sha256"
            ).fetchall()
        return [dict(row) for row in rows]
