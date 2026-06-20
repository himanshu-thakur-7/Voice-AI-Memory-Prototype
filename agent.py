"""LiveKit worker entry — the Cognitive Voice AI Agent.

Run:
    python agent.py dev       # local dev (livekit-agents CLI handles registration)

Per LiveKit 1.x: an entrypoint coroutine is invoked once per job/room. We:
  1) Connect to the room and wait for the human participant.
  2) Parse user_id / tenant_id out of participant.metadata (JSON) — falls back to identity.
  3) Pre-call: read the cognitive graph (Step 2; ``NullGraph`` stub until then) and
     build the system prompt + ElevenLabs voice/settings + optional empathy greeting.
  4) Build ``AgentSession(vad, stt, llm, tts)`` with the per-call config and start it.
  5) If a proactive greeting was produced, ``session.say()`` it; otherwise let the user
     speak first (we don't generate a generic opener).
  6) On participant_disconnected, schedule the post-call pipeline (Step 3 wires the real
     audio path and graph writes — the stub here just logs).
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import structlog
from dotenv import load_dotenv

from config import Settings, get_settings
from memory.post_call import schedule_post_call
from memory.pre_call import NullGraph, build_precall_context
from memory.schemas import ParticipantContext

if TYPE_CHECKING:
    from memory.graph_engine import CognitiveGraph

load_dotenv()

log = structlog.get_logger(__name__)

# We import livekit lazily inside ``entrypoint`` so this module can be unit-tested without
# the livekit-agents extras installed (CI / smoke tests stay fast).


# ── helpers ──────────────────────────────────────────────────────────────────


def parse_participant_context(room_name: str, participant) -> ParticipantContext:  # noqa: ANN001
    """Pull user_id / tenant_id out of participant.metadata (a JSON string by convention).

    Resilient to: missing metadata, non-JSON metadata, missing keys. The participant's
    LiveKit ``identity`` is the fallback user_id, ``"default"`` is the fallback tenant.
    """
    md_raw = getattr(participant, "metadata", "") or ""
    md: dict = {}
    if md_raw:
        try:
            md = json.loads(md_raw)
            if not isinstance(md, dict):
                md = {}
        except json.JSONDecodeError:
            log.warning("livekit.metadata.invalid_json", raw=md_raw[:200])

    identity = getattr(participant, "identity", "") or "anonymous"
    return ParticipantContext(
        room=room_name,
        participant_identity=identity,
        user_id=str(md.get("user_id") or identity),
        tenant_id=str(md.get("tenant_id") or "default"),
        metadata=md,
    )


async def _build_graph(settings: Settings) -> CognitiveGraph | NullGraph:
    """Return a connected ``CognitiveGraph``; fall back to ``NullGraph`` if the neo4j
    driver isn't installed or the database is unreachable, so the worker can still boot
    against an empty cognitive layer (useful for a no-deps demo / CI smoke test).

    The union type lets the caller use ``isinstance(graph, NullGraph)`` to gate post-call
    writes — when narrowed away, mypy knows it's a real ``CognitiveGraph``.
    """
    try:
        from memory.graph_engine import CognitiveGraph
    except ImportError:
        log.warning("graph.unavailable", reason="neo4j driver not installed")
        return NullGraph()
    graph = CognitiveGraph(settings)
    try:
        await graph.connect()
    except Exception as e:  # noqa: BLE001
        log.warning("graph.unavailable", reason=str(e))
        return NullGraph()
    return graph


# ── entrypoint ───────────────────────────────────────────────────────────────


async def entrypoint(ctx) -> None:  # noqa: ANN001 — JobContext (lazy import)
    from livekit.agents import Agent, AgentSession
    from livekit.plugins import deepgram, elevenlabs, openai, silero

    settings = get_settings()

    await ctx.connect()
    log.info("livekit.connected", room=ctx.room.name)

    # Wait for the human caller.
    participant = await ctx.wait_for_participant()
    pctx = parse_participant_context(ctx.room.name, participant)
    log.info("participant.joined",
             identity=pctx.participant_identity, user=pctx.user_id, tenant=pctx.tenant_id)

    # Pre-call: cognitive read + dynamic config.
    graph = await _build_graph(settings)
    pre = await build_precall_context(graph, pctx, settings)

    # Audio capture — write the caller's mic track to a WAV for post-call analysis.
    from audio_capture import CallerAudioRecorder

    recorder = CallerAudioRecorder()
    recorder.attach(ctx.room, target_identity=pctx.participant_identity)

    # Build the per-call pipeline.
    vad = ctx.proc.userdata.get("vad") or silero.VAD.load()
    stt = deepgram.STT(model=settings.deepgram_model) if settings.has_deepgram else None
    llm = openai.LLM(model=settings.llm_model) if settings.has_openai else None
    tts = (
        elevenlabs.TTS(
            voice=elevenlabs.Voice(
                id=pre.voice_id,
                settings=elevenlabs.VoiceSettings(
                    stability=pre.voice_settings.stability,
                    similarity_boost=pre.voice_settings.similarity_boost,
                    style=pre.voice_settings.style,
                    speed=pre.voice_settings.speed,
                    use_speaker_boost=pre.voice_settings.use_speaker_boost,
                ),
            ),
        )
        if settings.has_elevenlabs
        else None
    )

    agent = Agent(instructions=pre.system_prompt)
    session = AgentSession(vad=vad, stt=stt, llm=llm, tts=tts)
    await session.start(agent=agent, room=ctx.room)

    # Proactive Empathy: open with the remembered event when present.
    if pre.greeting:
        log.info("proactive.greeting", text=pre.greeting)
        await session.say(pre.greeting)

    # Adaptive Verbosity signal — tally user-on-agent interruptions for the post-call
    # writer. LiveKit's exact event name moves between minor versions, so we wrap each
    # ``session.on(...)`` registration in try/except. If none fire, interruption_count
    # stays 0 and ConversationStyle stays NORMAL — graceful degradation.
    interruption_counter = {"n": 0}
    _attach_interruption_hooks(session, interruption_counter)

    # Post-call hook: fire-and-forget so the disconnect handler returns immediately.
    @ctx.room.on("participant_disconnected")
    def _on_disc(p) -> None:  # noqa: ANN001
        if getattr(p, "identity", None) != pctx.participant_identity:
            return
        log.info("participant.left", identity=getattr(p, "identity", "?"))
        if isinstance(graph, NullGraph):
            log.info("postcall.skip", reason="no graph backend connected")
            return
        wav_path = recorder.stop_and_save()
        history = _capture_history(session)
        schedule_post_call(
            graph=graph, pctx=pctx, audio_path=wav_path,
            transcript_turns=history,
            interruption_count=interruption_counter["n"],
            settings=settings,
        )


def _attach_interruption_hooks(session, counter: dict[str, int]) -> None:  # noqa: ANN001
    """Best-effort wiring of LiveKit session events to an interruption counter.

    The 1.x ``AgentSession`` exposes ``.on(event, cb)`` for various lifecycle hooks but
    the exact event names ('speech_interrupted', 'agent_state_changed', etc.) vary by
    minor version. We try the names we know about and silently skip ones the installed
    version doesn't emit — never raising into the caller. Worst case: the counter stays
    zero and the agent doesn't flip to IMPATIENT, which is the safe default.
    """
    def _bump_on(event_name: str, predicate=lambda *_, **__: True) -> None:  # noqa: ANN001
        try:
            @session.on(event_name)
            def _cb(*args, **kwargs) -> None:  # noqa: ANN001
                if predicate(*args, **kwargs):
                    counter["n"] += 1
                    log.info("session.interruption", event=event_name, total=counter["n"])
        except Exception as e:  # noqa: BLE001 — event simply doesn't exist; ignore
            log.debug("session.hook_unavailable", event=event_name, err=str(e))

    # Each of these is a "user spoke during agent's turn" signal in some version:
    _bump_on("user_speech_committed")
    _bump_on("speech_interrupted")
    # Fallback: state transitions where the agent was speaking when the user started.
    _bump_on(
        "agent_state_changed",
        predicate=lambda ev=None, **__: bool(
            ev and getattr(ev, "old_state", "") == "speaking"
            and getattr(ev, "new_state", "") == "listening"
            and getattr(ev, "interrupted", False)
        ),
    )


def _capture_history(session) -> list[dict[str, str]]:  # noqa: ANN001
    """Pull the chat history off the session in a shape post_call understands.

    ``AgentSession.history`` is a ``ChatContext``-shaped object; the items() / messages
    attribute names move between versions. We try the common shapes and degrade to ``[]``
    so the post-call writer can still record the affective state from audio alone.
    """
    history = getattr(session, "history", None)
    if history is None:
        return []
    items = getattr(history, "items", None) or getattr(history, "messages", None) or []
    out: list[dict[str, str]] = []
    for item in items:
        role = getattr(item, "role", "") or ""
        # Items may store content as str OR list[str] (multimodal).
        raw_content = getattr(item, "content", "") or getattr(item, "text", "") or ""
        if isinstance(raw_content, list):
            raw_content = " ".join(str(part) for part in raw_content if part)
        out.append({"role": str(role), "content": str(raw_content)})
    return out


def _prewarm(proc) -> None:  # noqa: ANN001 — JobProcess
    """Load Silero once per worker process so the first call doesn't pay the cost."""
    from livekit.plugins import silero

    proc.userdata["vad"] = silero.VAD.load()


def main() -> None:
    """Entry: ``python agent.py dev`` (registers with LIVEKIT_URL via env)."""
    # Ensure the livekit CLI sees the same env vars even when we ran via dotenv above.
    for k in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
        if k not in os.environ:
            os.environ.setdefault(k, getattr(get_settings(), k.lower(), ""))

    from livekit.agents import WorkerOptions, cli

    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=_prewarm))


if __name__ == "__main__":
    main()
