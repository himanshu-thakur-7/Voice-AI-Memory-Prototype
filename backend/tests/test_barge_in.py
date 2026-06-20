"""VAD endpointing + the Speaker's barge-in (clear + mid-chunk abandon)."""

from __future__ import annotations

import asyncio

from app.engines.base import Speaker
from app.telephony.audio import FRAME_BYTES
from app.vad.turn import SPEECH_END, SPEECH_START, EnergyVAD
from tests.util import frames_for


def test_vad_detects_turn_start_and_end():
    vad = EnergyVAD()
    events = [vad.update(f) for f in
             frames_for(120, kind="silence") + frames_for(200, kind="speech")
             + frames_for(620, kind="silence")]
    assert SPEECH_START in events
    assert SPEECH_END in events
    # start must come before end
    assert events.index(SPEECH_START) < events.index(SPEECH_END)


async def test_speaker_clear_interrupts_and_flushes():
    sent: list[str] = []

    async def send(t: str) -> None:
        sent.append(t)

    speaker = Speaker(send, "MZ1", pace=True)
    total_frames = 200  # 4 seconds of audio queued
    await speaker.play(b"\x10" * (FRAME_BYTES * total_frames))

    run_task = asyncio.create_task(speaker.run())
    await asyncio.sleep(0.05)      # let a couple of 20 ms frames go out
    await speaker.clear()          # barge-in!
    await asyncio.sleep(0.02)
    run_task.cancel()

    media_sent = sum(1 for s in sent if '"media"' in s)
    assert any('"clear"' in s for s in sent), "barge-in must send Twilio a clear"
    assert media_sent < total_frames, "remaining audio must be abandoned on barge-in"
    assert not speaker.speaking.is_set()


async def test_speaker_plays_and_marks():
    sent: list[str] = []

    async def send(t: str) -> None:
        sent.append(t)

    speaker = Speaker(send, "MZ1", pace=False)
    run_task = asyncio.create_task(speaker.run())
    await speaker.play(b"\x10" * (FRAME_BYTES * 3))
    await speaker.mark("turn:1")
    await asyncio.sleep(0.05)
    run_task.cancel()

    assert sum(1 for s in sent if '"media"' in s) == 3
    assert any('"mark"' in s and "turn:1" in s for s in sent)
