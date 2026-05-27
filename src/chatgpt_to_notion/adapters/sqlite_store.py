"""SQLite persistence layer for image generations and API caches."""

import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ..domain.models import ChatGPTImageGeneration
from ..shared import constants

_lock = threading.Lock()


def _get_db_path() -> Path:
    db_dir = Path(constants.OUTPUT_PATH).resolve()
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "chatgpt.db"


def _get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or _get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path | None = None) -> None:
    conn = _get_connection(db_path)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS image_generations (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            message_id TEXT NOT NULL,
            asset_pointer TEXT NOT NULL,
            url TEXT NOT NULL,
            prompt TEXT DEFAULT '',
            uploaded_at TEXT DEFAULT '',
            account TEXT DEFAULT ''
        );

        CREATE INDEX IF NOT EXISTS idx_generations_account
            ON image_generations(account);
        CREATE INDEX IF NOT EXISTS idx_generations_created_at
            ON image_generations(created_at);
        CREATE INDEX IF NOT EXISTS idx_generations_uploaded_at
            ON image_generations(uploaded_at);

        CREATE TABLE IF NOT EXISTS cache_data_sources (
            db_id TEXT PRIMARY KEY,
            data_sources_json TEXT NOT NULL,
            expires_at TEXT NOT NULL
        );
        """
    )
    conn.commit()
    conn.close()


def upsert_generations(
    account: str,
    generations: list[ChatGPTImageGeneration],
    db_path: Path | None = None,
) -> None:
    if not generations:
        return
    with _lock:
        conn = _get_connection(db_path)
        try:
            conn.executemany(
                """
                INSERT INTO image_generations (
                    id, created_at, conversation_id, message_id,
                    asset_pointer, url, prompt, account
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    created_at=excluded.created_at,
                    conversation_id=excluded.conversation_id,
                    message_id=excluded.message_id,
                    asset_pointer=excluded.asset_pointer,
                    url=excluded.url,
                    prompt=excluded.prompt,
                    account=excluded.account
                """,
                [
                    (
                        generation.id,
                        generation.created_at,
                        generation.conversation_id,
                        generation.message_id,
                        generation.asset_pointer,
                        generation.url,
                        generation.prompt,
                        account,
                    )
                    for generation in generations
                ],
            )
            conn.commit()
        finally:
            conn.close()


def get_generations(
    account: str,
    *,
    include_uploaded: bool = False,
    keep_days: int | None = None,
    timezone_name: str | None = None,
    ids_filter: set[str] | None = None,
    db_path: Path | None = None,
) -> list[ChatGPTImageGeneration]:
    with _lock:
        conn = _get_connection(db_path)
        try:
            query = "SELECT * FROM image_generations WHERE account = ?"
            params: list[Any] = [account]

            if ids_filter:
                placeholders = ",".join("?" for _ in ids_filter)
                query += f" AND id IN ({placeholders})"
                params.extend(ids_filter)

            if not include_uploaded:
                query += " AND uploaded_at = ''"

            if keep_days is not None:
                from zoneinfo import ZoneInfo

                tz = (
                    ZoneInfo(timezone_name)
                    if timezone_name
                    else datetime.now().astimezone().tzinfo
                )
                cutoff = datetime.now(tz) - timedelta(days=keep_days)
                query += " AND created_at >= ?"
                params.append(cutoff.isoformat())

            query += " ORDER BY created_at ASC"

            rows = conn.execute(query, params).fetchall()
            return [
                ChatGPTImageGeneration(
                    created_at=row["created_at"],
                    id=row["id"],
                    conversation_id=row["conversation_id"],
                    message_id=row["message_id"],
                    asset_pointer=row["asset_pointer"],
                    url=row["url"],
                    prompt=row["prompt"] or "",
                )
                for row in rows
            ]
        finally:
            conn.close()


def get_uploaded_ids(account: str, db_path: Path | None = None) -> set[str]:
    with _lock:
        conn = _get_connection(db_path)
        try:
            rows = conn.execute(
                "SELECT id FROM image_generations "
                "WHERE account = ? AND uploaded_at != ''",
                (account,),
            ).fetchall()
            return {row["id"] for row in rows}
        finally:
            conn.close()


def get_existing_ids(
    account: str, ids: set[str], db_path: Path | None = None
) -> set[str]:
    if not ids:
        return set()
    with _lock:
        conn = _get_connection(db_path)
        try:
            placeholders = ",".join("?" for _ in ids)
            rows = conn.execute(
                "SELECT id FROM image_generations "
                f"WHERE account = ? AND id IN ({placeholders})",
                (account, *ids),
            ).fetchall()
            return {row["id"] for row in rows}
        finally:
            conn.close()


def mark_uploaded(
    account: str,
    generation_ids: set[str],
    db_path: Path | None = None,
) -> None:
    if not generation_ids:
        return
    uploaded_at = datetime.now(timezone.utc).isoformat()
    with _lock:
        conn = _get_connection(db_path)
        try:
            placeholders = ",".join("?" for _ in generation_ids)
            conn.execute(
                "UPDATE image_generations SET uploaded_at = ? "
                f"WHERE account = ? AND id IN ({placeholders})",
                (uploaded_at, account, *generation_ids),
            )
            conn.commit()
        finally:
            conn.close()


def delete_old_generations(
    account: str,
    keep_days: int = 2,
    db_path: Path | None = None,
) -> int:
    cutoff = datetime.now(timezone.utc) - timedelta(days=keep_days)
    with _lock:
        conn = _get_connection(db_path)
        try:
            cursor = conn.execute(
                "DELETE FROM image_generations WHERE account = ? AND created_at < ?",
                (account, cutoff.isoformat()),
            )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()


def get_cached_data_sources(
    db_id: str,
    db_path: Path | None = None,
) -> list[dict[str, Any]] | None:
    with _lock:
        conn = _get_connection(db_path)
        try:
            row = conn.execute(
                "SELECT data_sources_json, expires_at "
                "FROM cache_data_sources WHERE db_id = ?",
                (db_id,),
            ).fetchone()
            if row is None:
                return None
            expires_at = datetime.fromisoformat(row["expires_at"])
            if datetime.now(timezone.utc) > expires_at:
                conn.execute("DELETE FROM cache_data_sources WHERE db_id = ?", (db_id,))
                conn.commit()
                return None
            return json.loads(row["data_sources_json"])
        finally:
            conn.close()


def set_cached_data_sources(
    db_id: str,
    sources: list[dict[str, Any]],
    ttl_days: int = 1,
    db_path: Path | None = None,
) -> None:
    expires_at = (datetime.now(timezone.utc) + timedelta(days=ttl_days)).isoformat()
    with _lock:
        conn = _get_connection(db_path)
        try:
            conn.execute(
                """
                INSERT INTO cache_data_sources (db_id, data_sources_json, expires_at)
                VALUES (?, ?, ?)
                ON CONFLICT(db_id) DO UPDATE SET
                    data_sources_json=excluded.data_sources_json,
                    expires_at=excluded.expires_at
                """,
                (db_id, json.dumps(sources), expires_at),
            )
            conn.commit()
        finally:
            conn.close()


def has_generations(account: str, db_path: Path | None = None) -> bool:
    with _lock:
        conn = _get_connection(db_path)
        try:
            row = conn.execute(
                "SELECT 1 FROM image_generations WHERE account = ? LIMIT 1",
                (account,),
            ).fetchone()
            return row is not None
        finally:
            conn.close()


def get_activity_stats(
    account: str,
    date_start: str,
    date_end: str,
    cooldown_threshold: str,
    db_path: Path | None = None,
) -> dict[str, Any] | None:
    with _lock:
        conn = _get_connection(db_path)
        try:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN created_at > ? THEN 1 ELSE 0 END) as active_count,
                    MIN(CASE WHEN created_at > ? THEN created_at END) as first_active,
                    MAX(CASE WHEN created_at > ? THEN created_at END) as last_active
                FROM image_generations
                WHERE account = ?
                  AND created_at >= ?
                  AND created_at < ?
                """,
                (
                    cooldown_threshold,
                    cooldown_threshold,
                    cooldown_threshold,
                    account,
                    date_start,
                    date_end,
                ),
            ).fetchone()
            if row is None or row["total"] == 0:
                return None
            return {
                "total": row["total"],
                "active_count": row["active_count"] or 0,
                "first_active": row["first_active"],
                "last_active": row["last_active"],
            }
        finally:
            conn.close()


def count_recent_generations(
    account: str, since: str, db_path: Path | None = None
) -> int:
    with _lock:
        conn = _get_connection(db_path)
        try:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM image_generations "
                "WHERE account = ? AND created_at >= ?",
                (account, since),
            ).fetchone()
            return row["cnt"] if row else 0
        finally:
            conn.close()


async def async_upsert_generations(
    account: str,
    generations: list[ChatGPTImageGeneration],
    db_path: Path | None = None,
) -> None:
    upsert_generations(account, generations, db_path)


async def async_mark_uploaded(
    account: str,
    generation_ids: set[str],
    db_path: Path | None = None,
) -> None:
    mark_uploaded(account, generation_ids, db_path)


async def async_set_cached_data_sources(
    db_id: str,
    sources: list[dict[str, Any]],
    ttl_days: int = 1,
    db_path: Path | None = None,
) -> None:
    set_cached_data_sources(db_id, sources, ttl_days, db_path)
