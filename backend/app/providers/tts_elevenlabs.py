"""TTS leg: ElevenLabs Flash v2.5 streaming, emitting µ-law 8 kHz for Twilio.

Two facts from the research shape this file:
  • ``output_format=ulaw_8000`` is emitted directly → the audio drops onto Twilio Media
    Streams with ZERO resampling on the return leg.
  • ``voice_settings`` (our prosody) are LOCKED at socket init on a single-context stream.
    To change prosody per utterance we therefore open a fresh context each utterance — one
    of the two ElevenLabs-sanctioned ways to do dynamic prosody (the other being the
    multi-stream-input socket, noted below as the lower-latency upgrade).

Ref: https://elevenlabs.io/docs/api-reference/text-to-speech/v-1-text-to-speech-voice-id-stream-input
"""

from __future__ import annotations

import abc
import asyncio
import base64
import json
import math
from collections.abc import AsyncIterator

from app.engines.base import ProsodyProfile
from app.metrics import TurnTimer
from app.telephony.audio import pcm16_to_ulaw


class TTS(abc.ABC):
    name = "base"

    @abc.abstractmethod
    def synthesize(
        self, text: str, prosody: ProsodyProfile, timer: TurnTimer
    ) -> AsyncIterator[bytes]:
        """Yield µ-law 8 kHz audio chunks for ``text`` (Twilio-ready, no resample)."""
        raise NotImplementedError


class ElevenLabsTTS(TTS):
    name = "elevenlabs"

    def __init__(self, api_key: str, voice_id: str, model: str = "eleven_flash_v2_5") -> None:
        self._api_key = api_key
        self._voice_id = voice_id
        self._model = model

    async def synthesize(
        self, text: str, prosody: ProsodyProfile, timer: TurnTimer
    ) -> AsyncIterator[bytes]:
        import websockets  # lazy

        url = (
            f"wss://api.elevenlabs.io/v1/text-to-speech/{self._voice_id}/stream-input"
            f"?model_id={self._model}&output_format=ulaw_8000&auto_mode=true"
        )
        async with websockets.connect(
            url, additional_headers={"xi-api-key": self._api_key}
        ) as ws:
            # init — voice_settings carry the prosody profile (locked for this context).
            await ws.send(json.dumps({
                "text": " ",
                "voice_settings": {
                    "stability": prosody.stability,
                    "style": prosody.style,
                    "similarity_boost": prosody.similarity_boost,
                    "speed": prosody.speed,
                },
            }))
            await ws.send(json.dumps({"text": text + " ", "flush": True}))
            await ws.send(json.dumps({"text": ""}))  # close the context

            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("audio"):
                    timer.mark_tts_first_byte()
                    yield base64.b64decode(msg["audio"])
                if msg.get("isFinal"):
                    break


class MockTTS(TTS):
    """Generates a faint tone of text-proportional length so the loop produces audible,
    correctly-timed µ-law without an ElevenLabs key."""

    name = "mock"

    def __init__(self, *, tone_hz: int = 220) -> None:
        self._tone_hz = tone_hz

    async def synthesize(
        self, text: str, prosody: ProsodyProfile, timer: TurnTimer
    ) -> AsyncIterator[bytes]:
        words = max(1, len(text.split()))
        total_ms = min(6000, max(400, int(words * 180 / max(0.7, prosody.speed))))
        chunk_ms = 120
        for start in range(0, total_ms, chunk_ms):
            await asyncio.sleep(0.01)  # simulate streamed first-byte
            timer.mark_tts_first_byte()
            yield self._tone(min(chunk_ms, total_ms - start))

    def _tone(self, ms: int) -> bytes:
        n = 8000 * ms // 1000
        pcm = bytearray(n * 2)
        amp = 6000
        for i in range(n):
            s = int(amp * math.sin(2 * math.pi * self._tone_hz * i / 8000))
            s &= 0xFFFF
            pcm[2 * i] = s & 0xFF
            pcm[2 * i + 1] = (s >> 8) & 0xFF
        return pcm16_to_ulaw(bytes(pcm))
