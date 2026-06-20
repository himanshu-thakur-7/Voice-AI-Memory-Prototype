"""Module 3b — the post-call background worker.

Triggered when the media WebSocket closes. It must never block call teardown, so the
session loop fires it and walks away. It:
  1. extracts the caller's affective state from the captured audio and upserts it
     (so the NEXT call's pre-call prosody is right) — closing the affective feedback loop;
  2. extracts factual assertions from the transcript;
  3. runs them through the Contradiction Engine (Mem0 / Mem0g), overwriting stale facts.

In production this would be a durable queue (Redis/Celery) so analysis survives a backend
restart; here it's an asyncio task. ``process_postcall`` is also directly awaitable, which
is what the tests and the simulator use.
"""

from __future__ import annotations

import asyncio

from app.config import Settings
from app.engines.base import CallContext
from app.logging import get_logger
from app.memory.affective import extract_affect
from app.memory.contradiction import ContradictionEngine, extract_assertions
from app.memory.paralinguistics import run_paralinguistics
from app.memory.schemas import MemoryOp
from app.memory.store import affective_store

log = get_logger(__name__)

_background_tasks: set[asyncio.Task] = set()


def schedule_postcall(
    *, ctx: CallContext, caller_ulaw: bytes, transcript: list[dict], settings: Settings
) -> None:
    """Fire-and-forget the post-call analysis. Keeps a strong ref so the task isn't GC'd."""
    task = asyncio.create_task(process_postcall(ctx, caller_ulaw, transcript, settings))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def process_postcall(
    ctx: CallContext, caller_ulaw: bytes, transcript: list[dict], settings: Settings
) -> dict:
    """Run the full post-call pipeline. Returns a summary (used by tests/simulator)."""
    log.info("postcall.start", call=ctx.call_sid, audio_bytes=len(caller_ulaw),
             turns=len(transcript))

    # 1) Affective extraction → persist for the next call's pre-call prosody.
    #    Prefer the rich SenseVoice+openSMILE engine (acoustic emotion, biometrics, and the
    #    LLM contradiction pass); fall back to the dependency-free heuristic when its heavy
    #    deps aren't installed, so the worker always produces a state.
    state = await run_paralinguistics(caller_ulaw, ctx, settings)
    used_rich = state is not None
    if state is None:
        state = await extract_affect(caller_ulaw, ctx)
    await affective_store(settings).upsert_state(state)

    # 2) Factual assertions. When the rich extractor produced its own (acoustic) transcript,
    #    feed that to the contradiction engine alongside the conversation history.
    turns = list(transcript)
    para_transcript = state.paralinguistics.get("transcript")
    if para_transcript:
        turns.append({"role": "user", "content": para_transcript})
    assertions = extract_assertions(turns, settings)

    # 3) Contradiction engine: overwrite/retire stale facts.
    engine = ContradictionEngine(settings)
    decisions = await engine.reconcile(
        assertions, tenant_id=ctx.tenant_id, user_id=ctx.user_id, call_sid=ctx.call_sid
    )

    summary = {
        "call_sid": ctx.call_sid,
        "emotion": state.emotion.value,
        "final_affective_state": state.paralinguistics.get("final_affective_state", ""),
        "detected_events": state.paralinguistics.get("detected_events", []),
        "extractor": "paralinguistics" if used_rich else "heuristic",
        "valence": state.valence,
        "arousal": state.arousal,
        "assertions": len(assertions),
        "ops": {op.value: sum(1 for d in decisions if d.op is op) for op in MemoryOp},
    }
    log.info("postcall.done", **summary)
    return summary
