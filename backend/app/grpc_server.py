"""gRPC Orchestrator server: receives CallContext from the Go scheduler before the
media socket opens, so the /media loop already knows who is calling (and their affective
hint) on the very first turn.

Gated on generated stubs: if `make proto` hasn't run, the server stays disabled and the
session loop falls back to Twilio's <Parameter> values — the prototype still works, just
without the gRPC pre-seed optimization.
"""

from __future__ import annotations

import asyncio

from app.config import Settings
from app.logging import get_logger

log = get_logger(__name__)


class ContextStore:
    """callSid → seeded context. In-memory by default; swap for Redis in production
    (the .env has REDIS_URL) so multiple backend replicas share the seed."""

    def __init__(self) -> None:
        self._mem: dict[str, dict] = {}
        self._lock = asyncio.Lock()

    async def put(self, call_sid: str, ctx: dict) -> None:
        async with self._lock:
            self._mem[call_sid] = ctx

    async def get(self, call_sid: str) -> dict | None:
        async with self._lock:
            return self._mem.get(call_sid)


context_store = ContextStore()


async def start_grpc_server(settings: Settings):
    """Start the async gRPC server, or return None if stubs aren't generated."""
    try:
        import grpc  # type: ignore

        from app.pb import orchestrator_pb2 as pb  # type: ignore
        from app.pb import orchestrator_pb2_grpc as pb_grpc  # type: ignore
    except ImportError as e:
        log.warning("grpc.disabled", reason=str(e), hint="run `make proto` to enable")
        return None

    class Orchestrator(pb_grpc.OrchestratorServicer):
        async def RegisterCall(self, request, context):  # noqa: N802
            c = request.context
            await context_store.put(c.call_sid, {
                "from": c.from_,
                "to": c.to,
                "tenant_id": c.tenant_id,
                "user_id": c.user_id,
                "affective_hint": c.affective_hint,
                "engine": c.engine,
            })
            log.info("grpc.register_call", call=c.call_sid, tenant=c.tenant_id, engine=c.engine)
            return pb.RegisterCallResponse(media_ws_url="", accepted=True, reason="ok")

        async def Heartbeat(self, request_iterator, context):  # noqa: N802
            async for _ in request_iterator:
                yield pb.HeartbeatResponse(active_sessions=0, session_pressure=0.0)

    server = grpc.aio.server()
    pb_grpc.add_OrchestratorServicer_to_server(Orchestrator(), server)
    server.add_insecure_port(settings.grpc_listen_addr)
    await server.start()
    log.info("grpc.listening", addr=settings.grpc_listen_addr)
    return server
