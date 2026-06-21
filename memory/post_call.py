"""Post-call orchestration — runs after participant_disconnected.

Order, end-to-end:

  1. **Acoustic analysis** (rich SenseVoice+openSMILE if available, librosa otherwise) →
     ``pitch_variance``, ``rms_energy``, ``acoustic_affect``, transcript, biometrics.
  2. **Fact extraction** — LLM triplets over the (SenseVoice or LiveKit) transcript.
  3. **Graph apply_facts** — sarcasm floor (positive text + agitated voice → trust 0.2)
     and contradiction decay (UPDATE → old rel × 0.3) land here.
  4. **Conversation style** — interruption_count vs ``settings.impatience_threshold`` →
     IMPATIENT.
  5. **Affective state + negative event** — mapped from the rich engine's outputs and
     persisted on ``(:User)`` so the next call's pre-call read picks it up.

This whole pipeline runs in a background task (``schedule_post_call``) so the LiveKit
event loop never blocks. The two heavy/sync steps (acoustic analysis and the OpenAI
extraction call) are pushed into ``asyncio.to_thread``.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import structlog

from config import Settings
from memory.acoustic_engine import (
    AcousticResult,
    analyze_audio,
    map_emotion,
    valence_arousal,
)
from memory.graph_engine import CognitiveGraph
from memory.schemas import (
    AffectiveState,
    ConversationStyle,
    NegativeEvent,
    ParticipantContext,
)

log = structlog.get_logger(__name__)


# A task set keeps fire-and-forget tasks alive until they finish (otherwise the GC may
# collect them while the asyncio loop still has work for them). Ported pattern from
# ``archive/backend/app/workers/postcall.py``.
_BACKGROUND: set[asyncio.Task] = set()


# ── main pipeline ───────────────────────────────────────────────────────────


async def process_post_call(
    graph: CognitiveGraph,
    pctx: ParticipantContext,
    audio_path: str | None,
    transcript_turns: list[dict[str, str]],
    interruption_count: int,
    settings: Settings,
) -> dict[str, Any]:
    """Run the full post-call pipeline. Returns a summary dict (useful for tests/observability).

    All input shapes are intentionally loose so the LiveKit-side wiring can hand whatever
    it has — empty audio_path, empty transcript, zero interruptions all degrade
    gracefully to a no-op rather than crashing.
    """
    started = time.monotonic()
    log.info("postcall.start", user=pctx.user_id, tenant=pctx.tenant_id,
             room=pctx.room, has_audio=bool(audio_path), turns=len(transcript_turns),
             interruptions=interruption_count)

    # 1) Acoustic analysis — heavy/sync, push off the loop.
    acoustic: AcousticResult
    if audio_path:
        acoustic = await asyncio.to_thread(analyze_audio, audio_path, settings)
    else:
        acoustic = AcousticResult()
    log.info("postcall.acoustic", engine=acoustic.engine,
             pitch_variance=acoustic.pitch_variance, rms=acoustic.rms_energy,
             affect=acoustic.acoustic_affect,
             final=acoustic.final_affective_state or acoustic.base_acoustic_emotion)

    # 2) Pick the transcript that will feed fact extraction. Prefer the rich engine's
    #    transcript (it has speech-emotion-aware punctuation); fall back to LiveKit STT.
    transcript_text = acoustic.transcript or _join_transcript(transcript_turns)
    if not transcript_text.strip():
        log.warning("postcall.no_transcript", reason="empty audio + empty STT history")

    # 3) Fact extraction via LLM (sync OpenAI) — push off the loop. The caller_name keeps
    # the extracted subjects aligned with whatever the seed set (e.g. "Anil"), so the
    # resolver can detect contradictions instead of forking subjects into User/Anil.
    caller_name = _caller_display_name(pctx)
    triplets = await asyncio.to_thread(
        graph.extract_facts_to_triplets, transcript_text, caller_name,
    )
    log.info("postcall.facts.extracted", n=len(triplets), caller_name=caller_name)

    # 4) Apply facts: this is where the sarcasm floor + contradiction decay fire.
    affective_state_label = (
        acoustic.final_affective_state or acoustic.base_acoustic_emotion or "neutral"
    )
    ops = await graph.apply_facts(
        user_id=pctx.user_id,
        new_triplets=triplets,
        affective_state=affective_state_label,
        acoustic_affect=acoustic.acoustic_affect,
    )

    # 5) Conversation style — Adaptive Verbosity signal.
    style = (
        ConversationStyle.IMPATIENT
        if interruption_count >= settings.impatience_threshold
        else ConversationStyle.NORMAL
    )

    # 6) Severe negative event — Proactive Empathy seed for the NEXT call.
    last_event = _maybe_negative_event(acoustic, transcript_text)

    # 7) Map paralinguistics → coarse AffectiveState; persist.
    # CRITICAL: only persist when we actually have a *meaningful* signal. We treat librosa's
    # "neutral" read as "no information" — librosa often can't differentiate calm-content
    # from quiet-mic from short utterance, so a neutral classification across a whole call
    # is more likely "didn't detect anything" than "user genuinely is now calm". If we
    # wrote that, we'd silently clobber the seeded (or previously-detected) state. So we
    # only persist when the acoustic engine ACTUALLY detected agitated/subdued, OR when
    # there's semantic/behavioral signal from elsewhere.
    has_audio_signal = bool(audio_path) and acoustic.acoustic_affect != "neutral"
    has_semantic_signal = bool(acoustic.final_affective_state)
    has_behavioral_signal = interruption_count > 0
    if has_audio_signal or has_semantic_signal or has_behavioral_signal:
        emotion = map_emotion(acoustic.base_acoustic_emotion, acoustic.final_affective_state)
        valence, arousal = valence_arousal(emotion, acoustic.acoustic_biometrics)
        state = AffectiveState(
            tenant_id=pctx.tenant_id, user_id=pctx.user_id, emotion=emotion,
            valence=valence, arousal=arousal,
            confidence=0.75 if acoustic.engine == "rich" else 0.55,
            features={
                "pitch_variance": acoustic.pitch_variance,
                "rms_energy": acoustic.rms_energy,
                "interruption_count": float(interruption_count),
            },
            paralinguistics=acoustic.to_paralinguistics_dict(),
        )
        await graph.record_affective_state(state, style, last_negative_event=last_event)
    else:
        log.info("postcall.affective.skipped",
                 reason="no audio, no semantic, no behavioral signal — preserving prior state")
        emotion = None   # for the summary below

    elapsed = round((time.monotonic() - started) * 1000, 1)
    summary = {
        "user_id": pctx.user_id, "tenant_id": pctx.tenant_id,
        "engine": acoustic.engine,
        "emotion": emotion.value if emotion is not None else "preserved",
        "acoustic_affect": acoustic.acoustic_affect,
        "final_affective_state": acoustic.final_affective_state,
        "conversation_style": style.value,
        "interruption_count": interruption_count,
        "facts_extracted": len(triplets),
        "ops_applied": len(ops),
        "negative_event_kind": last_event.kind if last_event else None,
        "elapsed_ms": elapsed,
    }
    log.info("postcall.done", **summary)
    return summary


def schedule_post_call(
    *,
    graph: CognitiveGraph,
    pctx: ParticipantContext,
    audio_path: str | None,
    transcript_turns: list[dict[str, str]],
    interruption_count: int,
    settings: Settings,
) -> asyncio.Task:
    """Fire-and-forget the post-call pipeline. The disconnect handler that calls this
    returns immediately; the work happens in the background."""
    task = asyncio.create_task(
        process_post_call(
            graph, pctx, audio_path, transcript_turns, interruption_count, settings,
        ),
        name=f"postcall:{pctx.user_id}:{pctx.room}",
    )
    _BACKGROUND.add(task)
    task.add_done_callback(_drop_task)
    return task


def _drop_task(task: asyncio.Task) -> None:
    _BACKGROUND.discard(task)
    if exc := task.exception():
        log.error("postcall.task_crashed", err=str(exc))


# ── derivations ─────────────────────────────────────────────────────────────


def _join_transcript(turns: list[dict[str, str]]) -> str:
    """LiveKit-style chat history → newline-joined `role: content` for the LLM."""
    out: list[str] = []
    for turn in turns:
        role = (turn.get("role") or "").lower()
        content = (turn.get("content") or "").strip()
        if content and role in ("user", "assistant", "system"):
            speaker = "User" if role == "user" else "Agent" if role == "assistant" else role
            out.append(f"{speaker}: {content}")
    return "\n".join(out)


def _caller_display_name(pctx: ParticipantContext) -> str:
    """Pick a stable subject name for fact extraction. Priority:
      1. participant.metadata["display_name"] (the web UI can set this per-persona)
      2. participant.metadata["caller_name"]
      3. heuristic from user_id (e.g. "demo-anil" → "Anil")
      4. fallback "User"
    """
    md = pctx.metadata or {}
    for key in ("display_name", "caller_name"):
        v = md.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    raw = (pctx.user_id or "").strip()
    if raw.startswith("demo-"):
        return raw[len("demo-"):].capitalize() or "User"
    return raw.capitalize() if raw else "User"


# Phrases that flag the user mentioning a severe negative event — used as a guardrail so
# we don't surface every neutral memory as Proactive Empathy. Tuned to match the brief's
# canonical examples (flight cancellation, loss/death, getting fired, breakup, illness).
_NEGATIVE_KIND_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Loss",        ("died", "passed away", "lost my", "death of", "funeral")),
    ("Flight",      ("flight was cancelled", "flight cancelled", "missed my flight",
                     "delayed for hours", "stuck at the airport")),
    ("Job",         ("got fired", "was let go", "lost my job", "laid off")),
    ("Breakup",     ("broke up", "got divorced", "left me", "separation")),
    ("Illness",     ("diagnosed with", "in the hospital", "surgery", "chemo")),
)


def _maybe_negative_event(
    acoustic: AcousticResult, transcript: str
) -> NegativeEvent | None:
    """Build a ``NegativeEvent`` when the rich engine read a grief/defeat-like state,
    OR when the transcript contains one of the canonical negative-event phrases.

    The summary is the most informative-looking user sentence containing the trigger, so
    the next call's Proactive Empathy greeting reads naturally (e.g. *"Last time we
    spoke, your flight to Bangalore was cancelled — has that been sorted?"*).
    """
    final = (acoustic.final_affective_state or "").lower()
    severity_from_state = any(
        k in final for k in ("grief", "defeat", "mourn", "despair", "loss", "masking_grief")
    )

    kind: str | None = None
    sentence: str | None = None
    lowered = transcript.lower()
    for k, hints in _NEGATIVE_KIND_HINTS:
        for hint in hints:
            idx = lowered.find(hint)
            if idx != -1:
                kind = k
                sentence = _extract_sentence(transcript, idx)
                break
        if kind:
            break

    if kind is None and severity_from_state:
        kind = "Affect"
        sentence = _first_sentence(transcript) or acoustic.llm_reasoning

    if kind is None or not sentence:
        return None

    return NegativeEvent(
        kind=kind,
        summary=sentence.strip()[:280],
        emotion=map_emotion(acoustic.base_acoustic_emotion, acoustic.final_affective_state),
    )


def _extract_sentence(text: str, char_idx: int) -> str:
    """Return the sentence (period-delimited) of ``text`` containing the character at
    ``char_idx``. Used to lift a natural-sounding summary out of the transcript."""
    # Find the sentence boundaries before and after the hit.
    boundaries = ".!?\n"
    start = max((text.rfind(b, 0, char_idx) for b in boundaries), default=-1) + 1
    end = char_idx
    nearest = len(text)
    for b in boundaries:
        i = text.find(b, char_idx)
        if i != -1:
            nearest = min(nearest, i)
    end = nearest
    return text[start:end].strip()


def _first_sentence(text: str) -> str:
    for b in ".!?\n":
        i = text.find(b)
        if i > 0:
            return text[:i].strip()
    return text.strip()
