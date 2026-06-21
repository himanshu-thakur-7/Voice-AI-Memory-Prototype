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
    # Log a preview so when fact extraction returns 0 we can tell whether the issue is
    # an empty/garbage transcript vs. an over-strict extraction prompt.
    log.info("postcall.transcript",
             chars=len(transcript_text),
             preview=transcript_text[:300].replace("\n", " | "))

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

    # 5) Conversation style — Adaptive Verbosity signal. Two paths to "impatient":
    #    (a) literal interruptions above the threshold (the brief's original rule), OR
    #    (b) acoustic_affect == "agitated" (an angry caller IS impatient even when they
    #        don't talk over the agent — and LiveKit's interruption counter is unreliable
    #        anyway because event-name surface area drifts between versions).
    impatient_by_acoustic = acoustic.acoustic_affect == "agitated"
    impatient_by_interruptions = interruption_count >= settings.impatience_threshold
    style = (
        ConversationStyle.IMPATIENT
        if (impatient_by_interruptions or impatient_by_acoustic)
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
        emotion = map_emotion(
            acoustic.base_acoustic_emotion,
            acoustic.final_affective_state,
            acoustic.acoustic_affect,    # critical: librosa-only path needs this
        )
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
                     "delayed for hours", "stuck at the airport",
                     "flight got delayed", "flight is delayed", "flight delayed",
                     "flight was delayed", "got delayed today")),
    ("Job",         ("got fired", "was let go", "lost my job", "laid off")),
    ("Breakup",     ("broke up", "got divorced", "left me", "separation")),
    ("Illness",     ("diagnosed with", "in the hospital", "surgery", "chemo")),
    ("Service",     ("service outage", "outage", "billing issue", "wrong charge",
                     "broken", "doesn't work", "isn't working", "not working")),
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

    # Cold-start path: with the librosa-only engine, ``final_affective_state`` is empty
    # but ``acoustic_affect`` may be "agitated". If neither a phrase hint nor a rich
    # signal fired, *cautiously* fall back to a generic Affect event so the NEXT call's
    # proactive empathy can reference what happened — BUT only when the candidate
    # sentence is actually substantive. Without this guard, a call where the caller just
    # said "Hello, Hello, Hello" (because TTS failed and they were checking if anyone
    # was there) creates a bogus NegativeEvent with summary="Hello", which renders as
    # "Hi again. Last time we spoke, Hello — has that been sorted?" on the next call.
    if kind is None and acoustic.acoustic_affect == "agitated":
        candidate = _first_user_sentence(transcript) or _first_sentence(transcript)
        if candidate and _is_substantive(candidate):
            kind = "Affect"
            sentence = candidate

    if kind is None or not sentence:
        return None

    return NegativeEvent(
        kind=kind,
        summary=_polish_event_summary(sentence.strip())[:280],
        emotion=map_emotion(
            acoustic.base_acoustic_emotion,
            acoustic.final_affective_state,
            acoustic.acoustic_affect,
        ),
    )


# Used by _polish_event_summary to drop chatty leads ("So my flight ...") and a couple of
# two-word leads ("I mean ..."). Keep this list short — it's run on every event.
_LEADING_FILLERS = {
    "so", "yeah", "yes", "um", "uh", "like", "well", "ok", "okay", "right",
    "alright", "actually", "basically", "hmm", "huh",
}
_LEADING_TWO_WORD_FILLERS = {"i mean", "you know", "to be honest", "you see"}

# First-person → second-person word swaps for the proactive empathy template.
# We polish summaries at WRITE time (post-call), not at READ time, because the polished
# form is the canonical phrasing — every render of the greeting then matches.
_PERSON_SWAPS: tuple[tuple[str, str], ...] = (
    (r"\bI'm\b", "you're"), (r"\bI've\b", "you've"),
    (r"\bI'd\b", "you'd"),  (r"\bI'll\b", "you'll"),
    (r"\bI\b",  "you"),     (r"\bi\b",  "you"),
    (r"\bmy\b", "your"),    (r"\bMy\b", "your"),
    (r"\bme\b", "you"),     (r"\bmine\b", "yours"),
)


def _polish_event_summary(raw: str) -> str:
    """Turn a verbatim user sentence into a clean, agent-friendly second-person
    description suitable for *"Last time we spoke, {summary} — has that been sorted?"*.

    Cleanups (deterministic, no LLM call):
      1. Strip leading conversational fillers: "So my flight ..." → "my flight ..."
      2. First → second person: "my flight got delayed" → "your flight got delayed"
      3. Lowercase the first word so it reads as the middle of a sentence.

    Examples:
      "So my flight got delayed today"  →  "your flight got delayed today"
      "Yeah I'm really angry about it"  →  "you're really angry about it"
      "the credits never arrived"       →  "the credits never arrived"  (unchanged)
    """
    import re

    s = raw.strip()
    if not s:
        return s

    # 1. Strip leading fillers (single-word and two-word).
    while True:
        tokens = s.split()
        if not tokens:
            return s
        two_word = (tokens[0] + " " + tokens[1]).lower().rstrip(",.!?:") if len(tokens) >= 2 else ""
        one_word = tokens[0].lower().rstrip(",.!?:")
        if two_word in _LEADING_TWO_WORD_FILLERS:
            s = " ".join(tokens[2:])
            continue
        if one_word in _LEADING_FILLERS:
            s = " ".join(tokens[1:])
            continue
        break

    if not s:
        return raw  # all-filler input — fall back to original rather than emit empty

    # 2. First → second person word swaps.
    for pattern, replacement in _PERSON_SWAPS:
        s = re.sub(pattern, replacement, s)

    # 3. Lowercase first character (the summary is interpolated mid-sentence).
    return s[0].lower() + s[1:]


def _first_user_sentence(transcript: str) -> str | None:
    """Pull the first 'User: ...' line from the joined transcript so the next call's
    greeting paraphrases what the CALLER actually said, not the agent's opener."""
    for line in transcript.splitlines():
        if line.startswith("User:"):
            sentence = _first_sentence(line[len("User:"):].strip())
            if sentence:
                return sentence
    return None


# Conversational filler tokens that should NEVER become NegativeEvent summaries — getting
# "Hi again. Last time we spoke, Hello — has that been sorted?" reads as broken to a user.
_FILLER_SET = frozenset({
    "hello", "hi", "hey", "yeah", "yes", "no", "ok", "okay", "um", "uh", "hmm",
    "right", "sure", "thanks", "thank you", "bye", "goodbye", "what", "huh",
})


def _is_substantive(sentence: str) -> bool:
    """Heuristic: a sentence is substantive enough to anchor a future greeting if it has
    at least three real words AND isn't just stacked conversational fillers.

    Examples that PASS: "my flight got delayed today", "the credits never posted".
    Examples that FAIL: "Hello", "Hi", "Hello hello", "Yeah ok", "Bye".
    """
    cleaned = sentence.strip().lower()
    cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in cleaned)
    tokens = [t for t in cleaned.split() if t]
    if len(tokens) < 3:
        return False
    # If 80%+ of tokens are fillers, it's not substantive.
    fillers = sum(1 for t in tokens if t in _FILLER_SET)
    return fillers / len(tokens) < 0.8


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
