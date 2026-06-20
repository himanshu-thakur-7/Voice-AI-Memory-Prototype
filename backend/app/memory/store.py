"""Persistence for the affective memory layer.

Two things live here:
  • AffectiveStore — the caller's latest emotional state (read pre-call, written post-call).
    Postgres-backed in production, in-memory fallback so the prototype runs with no DB.
  • make_mem0() — constructs a Mem0 client wired to pgvector (semantic) and, when enabled,
    FalkorDB (per-user graph isolation). Returns None if Mem0 isn't installed/configured,
    in which case the contradiction engine uses its deterministic in-process resolver.
"""

from __future__ import annotations

import abc
from functools import lru_cache

from app.config import Settings
from app.logging import get_logger
from app.memory.schemas import AffectiveState

log = get_logger(__name__)


class AffectiveStore(abc.ABC):
    @abc.abstractmethod
    async def get_state(self, tenant_id: str, user_id: str) -> AffectiveState | None: ...

    @abc.abstractmethod
    async def upsert_state(self, state: AffectiveState) -> None: ...


class InMemoryAffectiveStore(AffectiveStore):
    def __init__(self) -> None:
        self._mem: dict[tuple[str, str], AffectiveState] = {}

    async def get_state(self, tenant_id: str, user_id: str) -> AffectiveState | None:
        return self._mem.get((tenant_id, user_id))

    async def upsert_state(self, state: AffectiveState) -> None:
        self._mem[(state.tenant_id, state.user_id)] = state


class PgAffectiveStore(AffectiveStore):
    """Postgres-backed store (table defined in backend/sql/init.sql)."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    async def get_state(self, tenant_id: str, user_id: str) -> AffectiveState | None:
        import psycopg  # lazy

        async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
            row = await (await conn.execute(
                "SELECT emotion, valence, arousal, confidence, features, paralinguistics "
                "FROM affective_state WHERE tenant_id=%s AND user_id=%s",
                (tenant_id, user_id),
            )).fetchone()
        if not row:
            return None
        return AffectiveState(
            tenant_id=tenant_id, user_id=user_id, emotion=row[0],
            valence=row[1], arousal=row[2], confidence=row[3],
            features=row[4] or {}, paralinguistics=row[5] or {},
        )

    async def upsert_state(self, state: AffectiveState) -> None:
        import json

        import psycopg  # lazy

        async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
            await conn.execute(
                "INSERT INTO affective_state "
                "(tenant_id, user_id, emotion, valence, arousal, confidence, features, "
                "paralinguistics, updated_at) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s, now()) "
                "ON CONFLICT (tenant_id, user_id) DO UPDATE SET "
                "emotion=EXCLUDED.emotion, valence=EXCLUDED.valence, arousal=EXCLUDED.arousal, "
                "confidence=EXCLUDED.confidence, features=EXCLUDED.features, "
                "paralinguistics=EXCLUDED.paralinguistics, updated_at=now()",
                (state.tenant_id, state.user_id, state.emotion.value, state.valence,
                 state.arousal, state.confidence, json.dumps(state.features),
                 json.dumps(state.paralinguistics)),
            )
            await conn.commit()


@lru_cache
def get_affective_store(_dsn: str, _use_pg: bool) -> AffectiveStore:
    if _use_pg:
        try:
            import psycopg  # noqa: F401

            log.info("affective_store.postgres")
            return PgAffectiveStore(_dsn)
        except ImportError as e:
            log.warning("affective_store.pg_unavailable", reason=str(e))
    log.info("affective_store.in_memory")
    return InMemoryAffectiveStore()


def affective_store(settings: Settings) -> AffectiveStore:
    # Use Postgres only when the memory extra is installed; otherwise in-memory.
    use_pg = _has_psycopg()
    return get_affective_store(settings.database_url, use_pg)


def _has_psycopg() -> bool:
    try:
        import psycopg  # noqa: F401

        return True
    except ImportError:
        return False


def make_mem0(settings: Settings):
    """Build a Mem0 client (pgvector + optional FalkorDB graph), or None if unavailable."""
    try:
        from mem0 import Memory  # type: ignore
    except ImportError:
        log.warning("mem0.unavailable", hint="pip install '.[memory]' to enable")
        return None

    config: dict = {
        "vector_store": {
            "provider": "pgvector",
            "config": {
                "connection_string": settings.database_url,
                "collection_name": "voiceai_mem",
            },
        },
    }
    if settings.mem0_graph_enabled:
        # FalkorDB gives per-user graph isolation (mem0_<user_id>) — see
        # docs.falkordb.com/agentic-memory/mem0.html
        config["graph_store"] = {
            "provider": "falkordb",
            "config": {"host": settings.falkordb_host, "port": settings.falkordb_port},
        }
    try:
        return Memory.from_config(config)
    except Exception as e:  # noqa: BLE001 - any backend wiring error → fall back
        log.warning("mem0.init_failed", reason=str(e))
        return None
