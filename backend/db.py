from __future__ import annotations

from typing import Any

import asyncpg

from backend.core.config import get_settings


class Database:
    def __init__(self) -> None:
        self.pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        if self.pool is not None:
            return
        settings = get_settings()
        self.pool = await asyncpg.create_pool(
            dsn=settings.postgres_dsn,
            min_size=1,
            max_size=10,
            command_timeout=10,
        )

    async def disconnect(self) -> None:
        if self.pool is not None:
            await self.pool.close()
            self.pool = None

    async def fetchrow(self, query: str, *args: Any) -> asyncpg.Record | None:
        if self.pool is None:
            return None
        async with self.pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetch(self, query: str, *args: Any) -> list[asyncpg.Record]:
        if self.pool is None:
            return []
        async with self.pool.acquire() as conn:
            return await conn.fetch(query, *args)


_db = Database()


def get_db() -> Database:
    return _db
