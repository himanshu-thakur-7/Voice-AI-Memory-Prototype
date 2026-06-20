"""End-to-end loop tests using the mock providers (no API keys)."""

from __future__ import annotations

import asyncio
import json

from app.config import Settings
from app.engines.base import INBOUND_EOF, CallContext, Speaker
from app.engines.cascade import CascadeEngine
from app.main import media
from app.providers.llm_openai import MockLLM
from app.providers.stt_base import MockSTT
from app.providers.tts_elevenlabs import MockTTS
from app.workers.postcall import process_postcall
from tests.util import FakeTwilioWS, build_script, utterance


async def test_cascade_full_turn_produces_audio_and_history():
    sent: list[str] = []

    async def send(t: str) -> None:
        sent.append(t)

    speaker = Speaker(send, "MZ1", pace=False)
    engine = CascadeEngine(Settings(), MockSTT(), MockLLM(), MockTTS())

    inbound: asyncio.Queue = asyncio.Queue()
    for frame in utterance():
        inbound.put_nowait(frame)
    inbound.put_nowait(INBOUND_EOF)

    speaker_task = asyncio.create_task(speaker.run())
    await asyncio.wait_for(
        engine.run(inbound, speaker, CallContext(call_sid="CA1", stream_sid="MZ1")), timeout=10
    )
    await asyncio.sleep(0.05)  # let the speaker flush queued audio
    speaker_task.cancel()

    events = [json.loads(s)["event"] for s in sent]
    assert "media" in events            # greeting + reply audio reached "Twilio"
    assert "mark" in events             # playback bookmarks emitted

    roles = [m["role"] for m in engine.history]
    assert "user" in roles              # MockSTT endpointed a turn
    assert roles.count("assistant") >= 2  # greeting + at least one reply


async def test_media_websocket_loop_runs_end_to_end():
    script = build_script("MZ1", "CA1", utterance(), {"engine": "cascade", "tenant": "t", "userId": "u"})
    ws = FakeTwilioWS(script)
    await asyncio.wait_for(media(ws), timeout=10)
    await asyncio.sleep(0.05)  # allow the fire-and-forget post-call task to finish
    assert "media" in ws.events()


async def test_postcall_extracts_emotion_and_resolves_contradiction():
    uid = "user-postcall-iso-1"  # unique to avoid the shared resolver from other tests
    ctx = CallContext(call_sid="CA1", tenant_id="t", user_id=uid)
    audio = b"".join(utterance(speech=900))

    s1 = await process_postcall(ctx, audio, [{"role": "user", "content": "my plan is pro"}], Settings())
    assert s1["assertions"] == 1
    assert s1["ops"]["ADD"] == 1
    assert s1["emotion"]  # an emotion label was produced from the audio

    # A later call where the same fact changed must resolve to UPDATE (overwrite stale).
    s2 = await process_postcall(
        ctx, audio, [{"role": "user", "content": "my plan is enterprise"}], Settings()
    )
    assert s2["ops"]["UPDATE"] == 1


async def test_realtime_mock_mode_runs():
    from app.engines.realtime import RealtimeEngine

    sent: list[str] = []

    async def send(t: str) -> None:
        sent.append(t)

    speaker = Speaker(send, "MZ1", pace=False)
    engine = RealtimeEngine(Settings())  # no OpenAI key → mock mode

    inbound: asyncio.Queue = asyncio.Queue()
    for frame in utterance():
        inbound.put_nowait(frame)
    inbound.put_nowait(INBOUND_EOF)

    speaker_task = asyncio.create_task(speaker.run())
    await asyncio.wait_for(
        engine.run(inbound, speaker, CallContext(call_sid="CA1", stream_sid="MZ1", engine="realtime")),
        timeout=10,
    )
    await asyncio.sleep(0.05)
    speaker_task.cancel()
    assert any('"media"' in s for s in sent)
