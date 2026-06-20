"""Module 2 entrypoint: FastAPI app + the /media WebSocket session loop.

The loop is engine-agnostic. It:
  1. waits for Twilio's `start` event and captures streamSid + customParameters;
  2. builds the CallContext (merging anything the Go scheduler seeded over gRPC);
  3. runs the pre-call prosody injection (Module 3a);
  4. spins up reader + speaker + engine as concurrent asyncio tasks;
  5. on hang-up, fires the post-call worker (Module 3b) with the captured caller audio.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from app.config import Settings, get_settings
from app.engines.base import INBOUND_EOF, CallContext, Speaker, VoiceEngine
from app.engines.cascade import CascadeEngine
from app.engines.realtime import RealtimeEngine
from app.grpc_server import context_store, start_grpc_server
from app.logging import configure, get_logger
from app.memory.precall import resolve_prosody
from app.providers.factory import make_llm, make_stt, make_tts
from app.telephony.twilio_protocol import decode_media, parse_event, parse_start
from app.workers.postcall import schedule_postcall

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure(settings.log_level)
    log.info("backend.start", engine=settings.engine, stt=settings.stt_provider)
    grpc_server = await start_grpc_server(settings)  # None if stubs not generated
    try:
        yield
    finally:
        if grpc_server is not None:
            await grpc_server.stop(grace=2.0)
        log.info("backend.stop")


app = FastAPI(title="Voice AI Backend", lifespan=lifespan)

# Ringg-styled test console (dashboard at GET / + /api/*).
from app.webui import router as webui_router  # noqa: E402

app.include_router(webui_router)


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.websocket("/media")
async def media(ws: WebSocket) -> None:
    await ws.accept()
    settings = get_settings()

    start = await _await_start(ws)
    if start is None:
        await ws.close()
        return

    ctx = await _build_context(settings, start)
    await resolve_prosody(ctx, settings)  # Module 3a: pick prosody + system prompt
    log.info("media.session", call=ctx.call_sid, engine=ctx.engine, prosody=ctx.prosody.label)

    inbound: asyncio.Queue = asyncio.Queue()
    speaker = Speaker(ws.send_text, ctx.stream_sid)
    engine = _make_engine(settings, ctx)

    caller_audio = bytearray()  # captured µ-law for post-call paralinguistic analysis

    async def reader() -> None:
        try:
            while True:
                event, msg = parse_event(await ws.receive_text())
                if event == "media":
                    frame = decode_media(msg)
                    caller_audio.extend(frame)
                    await inbound.put(frame)
                elif event == "mark":
                    speaker.ack_mark(msg.get("mark", {}).get("name", ""))
                elif event == "stop":
                    break
        except WebSocketDisconnect:
            log.info("media.disconnect", call=ctx.call_sid)
        finally:
            await inbound.put(INBOUND_EOF)

    speaker_task = asyncio.create_task(speaker.run())
    reader_task = asyncio.create_task(reader())
    try:
        await engine.run(inbound, speaker, ctx)
    finally:
        reader_task.cancel()
        speaker_task.cancel()
        # Module 3b — runs in the background; never blocks call teardown.
        schedule_postcall(
            ctx=ctx,
            caller_ulaw=bytes(caller_audio),
            transcript=getattr(engine, "history", []),
            settings=settings,
        )
        try:
            await ws.close()
        except RuntimeError:
            pass


# ── helpers ──────────────────────────────────────────────────────────────────


async def _await_start(ws: WebSocket):
    """Consume frames until Twilio's `start`. Returns the parsed StartInfo (or None)."""
    while True:
        try:
            event, msg = parse_event(await ws.receive_text())
        except WebSocketDisconnect:
            return None
        if event == "start":
            return parse_start(msg)
        if event == "stop":
            return None
        # 'connected' (and any stray frames) are ignored until 'start' arrives.


async def _build_context(settings: Settings, start) -> CallContext:
    params = start.custom_parameters or {}
    # Prefer context the Go scheduler seeded over gRPC (keyed by callSid); fall back to
    # the <Parameter> values Twilio delivered in the start event.
    seeded = await context_store.get(start.call_sid) or {}
    return CallContext(
        call_sid=start.call_sid,
        stream_sid=start.stream_sid,
        from_number=seeded.get("from", ""),
        to_number=seeded.get("to", params.get("to", "")),
        tenant_id=seeded.get("tenant_id", params.get("tenant", "default")),
        user_id=seeded.get("user_id", params.get("userId", start.call_sid)),
        engine=seeded.get("engine", params.get("engine", settings.engine)),
        affective_hint=seeded.get("affective_hint", ""),
        custom=params,
    )


def _make_engine(settings: Settings, ctx: CallContext) -> VoiceEngine:
    if ctx.engine == "realtime":
        return RealtimeEngine(settings)
    return CascadeEngine(settings, make_stt(settings), make_llm(settings), make_tts(settings))
