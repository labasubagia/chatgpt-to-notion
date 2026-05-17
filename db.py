"""SQLite persistence layer for image generations and API caches."""

import asyncio
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from models import ChatGPTImageGeneration

_lock = asyncio.Lock()


def _get_db_path() -> Path:
    from util import OUTPUT_PATH

    db_dir = Path(OUTPUT_PATH).resolve()
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "chatgpt.db"


def _get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or _get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
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

        CREATE TABLE IF NOT EXISTS cache_fetch_generations (
            account TEXT PRIMARY KEY,
            generations_json TEXT NOT NULL,
            fetched_at TEXT NOT NULL,
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
                    g.id,
                    g.created_at,
                    g.conversation_id,
                    g.message_id,
                    g.asset_pointer,
                    g.url,
                    g.prompt,
                    account,
                )
                for g in generations
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
    db_path: Path | None = None,
) -> list[ChatGPTImageGeneration]:
    conn = _get_connection(db_path)
    try:
        query = "SELECT * FROM image_generations WHERE account = ?"
        params: list[Any] = [account]

        if not include_uploaded:
            query += " AND (uploaded_at IS NULL OR uploaded_at = '')"

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


def get_uploaded_ids(
    account: str,
    db_path: Path | None = None,
) -> set[str]:
    conn = _get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT id FROM image_generations WHERE account = ? AND uploaded_at != ''",
            (account,),
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
    conn = _get_connection(db_path)
    try:
        placeholders = ",".join("?" for _ in generation_ids)
        conn.execute(
            f"""
            UPDATE image_generations
            SET uploaded_at = ?
            WHERE account = ? AND id IN ({placeholders})
            """,
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
    conn = _get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT data_sources_json, expires_at"
            " FROM cache_data_sources WHERE db_id = ?",
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


def get_cached_fetch_generations(
    account: str,
    db_path: Path | None = None,
) -> list[ChatGPTImageGeneration] | None:
    conn = _get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT generations_json, expires_at"
            " FROM cache_fetch_generations WHERE account = ?",
            (account,),
        ).fetchone()
        if row is None:
            return None
        expires_at = datetime.fromisoformat(row["expires_at"])
        if datetime.now(timezone.utc) > expires_at:
            conn.execute(
                "DELETE FROM cache_fetch_generations WHERE account = ?",
                (account,),
            )
            conn.commit()
            return None
        data = json.loads(row["generations_json"])
        return [ChatGPTImageGeneration(**item) for item in data]
    finally:
        conn.close()


def set_cached_fetch_generations(
    account: str,
    generations: list[ChatGPTImageGeneration],
    ttl_days: int = 1,
    db_path: Path | None = None,
) -> None:
    expires_at = (datetime.now(timezone.utc) + timedelta(days=ttl_days)).isoformat()
    fetched_at = datetime.now(timezone.utc).isoformat()
    conn = _get_connection(db_path)
    try:
        conn.execute(
            """
            INSERT INTO cache_fetch_generations (
                account, generations_json, fetched_at, expires_at
            )
            VALUES (?, ?, ?, ?)
            ON CONFLICT(account) DO UPDATE SET
                generations_json=excluded.generations_json,
                fetched_at=excluded.fetched_at,
                expires_at=excluded.expires_at
            """,
            (
                account,
                json.dumps([g.model_dump() for g in generations]),
                fetched_at,
                expires_at,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def reset_fetch_cache(
    account: str,
    db_path: Path | None = None,
) -> None:
    conn = _get_connection(db_path)
    try:
        conn.execute(
            "DELETE FROM cache_fetch_generations WHERE account = ?",
            (account,),
        )
        conn.commit()
    finally:
        conn.close()


async def async_upsert_generations(
    account: str,
    generations: list[ChatGPTImageGeneration],
    db_path: Path | None = None,
) -> None:
    async with _lock:
        upsert_generations(account, generations, db_path)


async def async_mark_uploaded(
    account: str,
    generation_ids: set[str],
    db_path: Path | None = None,
) -> None:
    async with _lock:
        mark_uploaded(account, generation_ids, db_path)


async def async_set_cached_data_sources(
    db_id: str,
    sources: list[dict[str, Any]],
    ttl_days: int = 1,
    db_path: Path | None = None,
) -> None:
    async with _lock:
        set_cached_data_sources(db_id, sources, ttl_days, db_path)


async def async_set_cached_fetch_generations(
    account: str,
    generations: list[ChatGPTImageGeneration],
    ttl_days: int = 1,
    db_path: Path | None = None,
) -> None:
    async with _lock:
        set_cached_fetch_generations(account, generations, ttl_days, db_path)


async def async_reset_fetch_cache(
    account: str,
    db_path: Path | None = None,
) -> None:
    async with _lock:
        reset_fetch_cache(account, db_path)
